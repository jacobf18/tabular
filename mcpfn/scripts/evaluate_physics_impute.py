"""Add the frozen MatrixFormer-Impute checkpoint to the four saved physics pilots."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

import evaluate_cylinder_completion as shared
import evaluate_mri_completion as synthetic
import evaluate_quantum_tomography as quantum
import evaluate_real_mri_completion as real
from recsys_evaluation_common import atomic_npz


ROOT = Path(__file__).resolve().parents[1]
IMPUTE_SHA = 'c0ddf75274c8ff92ca5fd22c1263369ffa7e55eb7cb85fe80961b0afa101f7a9'
PARENTS = {
    'cylinder': ROOT / 'runs/physics_cylinder_20261027/pilot',
    'quantum': ROOT / 'runs/physics_quantum_20261027/pilot',
    'synthetic_mri': ROOT / 'runs/physics_mri_20261027/pilot',
    'real_mri': Path('/mnt/matrixformer_scratch/physics_mri_real_20261027/full_resolution'),
}
COUNTS = {'cylinder': 18, 'quantum': 18, 'synthetic_mri': 12, 'real_mri': 12}


def parent_artifact(benchmark, record_path):
    if benchmark == 'cylinder':
        return record_path.parent.parent / 'predictions' / record_path.with_suffix('.npz').name
    return record_path.with_suffix('.npz')


def register(output):
    if output.exists():
        raise ValueError('Use a fresh output directory; parent experiments are never overwritten')
    if shared.sha256(shared.GENERAL) != IMPUTE_SHA or shared.sha256(shared.SINGLE) != shared.SINGLE_SHA256:
        raise ValueError('Unexpected specialist or combined checkpoint')
    cases, inputs = [], {}
    for benchmark, parent in PARENTS.items():
        audit = json.loads((parent / 'audit.json').read_text())
        assert audit.get('passed') or audit.get('status') in ('pass', 'passed')
        records = sorted((parent / 'cases').glob('*.json'))
        assert len(records) == COUNTS[benchmark]
        paths = [parent / name for name in ('manifest.json', 'audit.json', 'per_case.csv', 'summary.csv')]
        for record_path in records:
            record = json.loads(record_path.read_text())
            artifact = parent_artifact(benchmark, record_path)
            expected = record['prediction_sha256' if benchmark == 'cylinder' else 'arrays_sha256']
            assert shared.sha256(artifact) == expected
            paths.extend((record_path, artifact))
            cases.append(dict(benchmark=benchmark, case_id=record['case_id'], parent_record=str(record_path),
                              parent_artifact=str(artifact)))
        if benchmark == 'cylinder':
            paths.append(parent / 'dataset.npz')
        elif benchmark == 'quantum':
            paths.extend(sorted((parent / 'states').glob('*.npz')))
        elif benchmark == 'synthetic_mri':
            paths.extend(parent / name for name in ('image.npy', 'raw_kspace.npy'))
        else:
            paths.append(parent / 'phantom_data.mat')
        inputs.update({str(path): shared.sha256(path) for path in paths})
    sources = [Path(__file__).resolve(), Path(shared.__file__).resolve(), Path(synthetic.__file__).resolve(),
               Path(quantum.__file__).resolve(), Path(real.__file__).resolve(),
               ROOT / 'scripts/recsys_evaluation_common.py', ROOT / 'src/tabimpute/model/model_new_stable.py',
               ROOT / 'src/tabimpute/model/bar_distribution.py', ROOT / 'src/tabimpute/data/borders.pt']
    output.mkdir(parents=True)
    for source in sources:
        target = output / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    protocol = dict(created=datetime.now(timezone.utc).isoformat(), cases=cases,
                    method='MatrixFormer-Impute', checkpoint=str(shared.GENERAL), checkpoint_sha256=IMPUTE_SHA,
                    combined_checkpoint=str(shared.SINGLE), combined_checkpoint_sha256=shared.SINGLE_SHA256,
                    config_sha256={str(path.with_name('config.json')): shared.sha256(path.with_name('config.json'))
                                   for path in (shared.GENERAL, shared.SINGLE)},
                    input_sha256=inputs, source_sha256={str(path): shared.sha256(path) for path in sources},
                    parents={name: str(path) for name, path in PARENTS.items()}, expected_cases=60,
                    training=False, parameter_selection=False, new_masks=False, new_measurements=False,
                    orientation='standard; no transpose averaging or expert routing',
                    normalization='Original observed-only global normalization per representation',
                    context='Full matrices except real MRI: original maximum 1024x1024 tiles, same pairing and permutations',
                    uncertainty='Same cylinder held-out indices and all hidden quantum coefficients; none for MRI',
                    classical_baselines='Reuse archived predictions and scores; no baseline refitting',
                    replay='Re-evaluate combined model on first lexicographic case of each benchmark before specialist evaluation',
                    torch=torch.__version__, numpy=np.__version__)
    shared.write_json(output / 'protocol.json', protocol)
    print(f'Registered 60 cases and {len(inputs)} frozen inputs at {output}', flush=True)


def verify(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    for name, expected in protocol['input_sha256'].items():
        if shared.sha256(name) != expected:
            raise ValueError(f'Parent input changed: {name}')
    for section in ('source_sha256', 'config_sha256'):
        for name, expected in protocol[section].items():
            if shared.sha256(name) != expected:
                raise ValueError(f'Frozen dependency changed: {name}')
    assert shared.sha256(protocol['checkpoint']) == protocol['checkpoint_sha256']
    assert shared.sha256(protocol['combined_checkpoint']) == protocol['combined_checkpoint_sha256']
    return protocol


@torch.inference_mode()
def forward_native(matrix, model, distribution):
    normalized, mean, scale = shared.normalize_observed(matrix)
    values = torch.as_tensor(normalized.astype(np.float32), device='cuda')[None]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        encoded = model(values, return_embeddings=True)[0]
    features = encoded.reshape(-1, encoded.shape[-1])
    point = torch.empty(matrix.size, device='cuda', dtype=torch.float32)
    for start in range(0, matrix.size, 2048):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model.decoder(features[start:start + 2048])
        point[start:start + len(logits)] = distribution.mean(logits.float())
    prediction = point.cpu().numpy().reshape(matrix.shape).astype(np.float64) * scale + mean
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite neural prediction')
    return prediction, features


@torch.inference_mode()
def native_uncertainty(model, features, distribution, matrix, truth, indices):
    _, mean, scale = shared.normalize_observed(matrix)
    target = torch.as_tensor(((truth.ravel()[indices] - mean) / scale).astype(np.float32), device='cuda')
    levels = (torch.arange(199, device='cuda') + .5) / 199
    bounds = torch.tensor([.025, .05, .25, .75, .95, .975], device='cuda')
    all_levels = torch.cat([levels, bounds])
    saved = {key: [] for key in ('crps', 'nll', 'quantiles', 'std')}
    for start in range(0, len(indices), 1024):
        selected = torch.as_tensor(indices[start:start + 1024], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model.decoder(features[selected])
        probabilities = logits.float().softmax(-1)
        quantiles = shared.full_support_quantiles(probabilities, distribution.borders, all_levels)
        log_probability = probabilities.clamp_min(1e-38).log()
        current = target[start:start + len(selected)]
        fields = dict(crps=shared.quantile_crps(quantiles[:, :199], current, levels) * scale,
                      nll=distribution(log_probability, current) + math.log(scale),
                      quantiles=quantiles[:, 199:] * scale + mean,
                      std=distribution.variance(log_probability).clamp_min(0).sqrt() * scale)
        for key, values in fields.items():
            saved[key].append(values.cpu().numpy())
    arrays = {key: np.concatenate(parts) for key, parts in saved.items()}
    scores = uncertainty_scores(arrays, truth.ravel()[indices], scale)
    return scores, arrays


def uncertainty_scores(arrays, target, scale):
    scores = dict(crps=float(arrays['crps'].mean()), normalized_crps=float(arrays['crps'].mean() / scale),
                  nll=float(arrays['nll'].mean()), uncertainty_cells=len(target),
                  uncertainty_type='native full-support bars')
    for coverage, lower, upper in ((50, 2, 3), (90, 1, 4), (95, 0, 5)):
        quantiles = arrays['quantiles']
        scores[f'coverage_{coverage}'] = float(((target >= quantiles[:, lower]) & (target <= quantiles[:, upper])).mean())
        scores[f'width_{coverage}'] = float((quantiles[:, upper] - quantiles[:, lower]).mean())
    return scores


def quantum_scores(truth_density, truth, matrix, prediction, operators):
    completed = quantum.complete_coefficients(matrix, prediction)
    raw_density = quantum.density_from_coefficients(completed, operators)
    density = quantum.project_density(raw_density)
    scores = quantum.reconstruction_metrics(truth_density, truth, density, matrix, operators)
    missing = ~np.isfinite(matrix)
    scores.update(preprojection_hidden_rmse=float(np.sqrt(np.mean((completed[missing] - truth[missing]) ** 2))),
                  projection_frobenius=float(np.linalg.norm(density - raw_density)),
                  raw_hidden_rmse=float(np.sqrt(np.mean((prediction[missing] - truth[missing]) ** 2))))
    return scores, dict(completed=completed, raw_density=raw_density, density=density)


def load_reference(benchmark):
    parent = PARENTS[benchmark]
    if benchmark == 'cylinder':
        with np.load(parent / 'dataset.npz') as saved:
            return dict(truth=saved['truth'])
    if benchmark == 'quantum':
        manifest = json.loads((parent / 'manifest.json').read_text())
        _, operators = quantum.pauli_basis(manifest['qubits'])
        return dict(operators=operators)
    if benchmark == 'synthetic_mri':
        truth = np.load(parent / 'raw_kspace.npy')
        return dict(truth=truth, image=np.load(parent / 'image.npy'), lift=synthetic.SLift(truth.shape[0]))
    truth = real.load_phantom(parent / 'phantom_data.mat')
    return dict(truth=truth, lift=real.MultiCoilSLift(truth.shape[:2], truth.shape[2]))


def evaluate(case, model, distribution, reference, uncertainty=True):
    benchmark = case['benchmark']
    parent = PARENTS[benchmark]
    record = json.loads(Path(case['parent_record']).read_text())
    with np.load(case['parent_artifact']) as saved:
        if benchmark in ('cylinder', 'quantum'):
            if benchmark == 'cylinder':
                truth = reference['truth']
                matrix = np.where(saved['missing'], np.nan, truth)
                indices = saved['uncertainty_indices']
            else:
                matrix, indices = saved['matrix'], saved['hidden_indices']
                with np.load(parent / 'states' / f"{record['state_id']}.npz") as state:
                    truth, density = state['coefficients'], state['density']
            prediction, features = forward_native(matrix, model, distribution)
            _, mean, scale = shared.normalize_observed(matrix)
            if benchmark == 'cylinder':
                scores = shared.point_metrics(truth, prediction, ~np.isfinite(matrix), scale)
                arrays = dict(prediction=prediction, missing=saved['missing'], uncertainty_indices=indices)
            else:
                scores, arrays = quantum_scores(density, truth, matrix, prediction, reference['operators'])
                arrays.update(prediction=prediction, matrix=matrix, counts=saved['counts'], uncertainty_indices=indices)
            if uncertainty:
                native, native_arrays = native_uncertainty(model, features, distribution, matrix, truth, indices)
                scores.update(native)
                if benchmark == 'quantum':
                    bounds = native_arrays['quantiles']
                    scores['interval_90_outside_physical_range'] = float(((bounds[:, 1] < -1) | (bounds[:, 4] > 1)).mean())
                arrays.update({f'uncertainty_{key}': value for key, value in native_arrays.items()})
            return {'MatrixFormer-Impute': scores}, arrays, dict(observed_mean=mean, observed_std=scale)
        acquired = saved['acquired']
        truth, lift = reference['truth'], reference['lift']
        measured = np.where(acquired if benchmark == 'synthetic_mri' else acquired[..., None],
                            truth, complex(np.nan, np.nan))
        if benchmark == 'synthetic_mri':
            raw_input, lifted_input = saved['raw_input'], saved['lifted_input']
            expected_raw, expected_lifted = synthetic.observed_inputs(measured, lift)
            np.testing.assert_array_equal(raw_input, expected_raw)
            np.testing.assert_array_equal(lifted_input, expected_lifted)
            raw, _, raw_details = synthetic.neural_complete(raw_input, model, distribution)
            lifted, _, lifted_details = synthetic.neural_complete(lifted_input, model, distribution)
        else:
            raw_input, lifted_input = real.observed_inputs(measured, acquired, lift)
            raw, raw_details = real.neural_complete(raw_input, model, distribution, False)
            lifted, lifted_details = real.neural_complete(lifted_input, model, distribution, True)
        width = raw.shape[1] // 2
        raw_kspace = (raw[:, :width] + 1j * raw[:, width:]).reshape(truth.shape)
        predictions = dict(raw=synthetic.preserve_measurements(raw_kspace, measured),
                           lifted=synthetic.preserve_measurements(lift.inverse(lifted), measured))
        scores = {}
        for key, prediction in predictions.items():
            method = 'MatrixFormer-Impute raw' if key == 'raw' else 'MatrixFormer-Impute S'
            scores[method] = (synthetic.score_reconstruction(reference['image'], truth, prediction, acquired)
                              if benchmark == 'synthetic_mri' else real.score_reconstruction(truth, prediction, acquired))
        arrays = dict(acquired=acquired, **{f'{key}_kspace': prediction for key, prediction in predictions.items()})
        return scores, arrays, dict(raw_inference=raw_details, lifted_inference=lifted_details)


def check_replay(case, arrays):
    comparisons = []
    with np.load(case['parent_artifact']) as saved:
        if case['benchmark'] == 'cylinder':
            pairs = [('prediction', 'prediction_4')]
        elif case['benchmark'] == 'quantum':
            pairs = [('prediction', 'single_raw_prediction'), ('density', 'single_density')]
        else:
            pairs = [('raw_kspace', 'raw_kspace'), ('lifted_kspace', 'lifted_kspace')]
        for current_key, parent_key in pairs:
            current, expected = arrays[current_key], saved[parent_key]
            scale = max(float(np.max(np.abs(expected))), 1e-12)
            np.testing.assert_allclose(current, expected, rtol=1e-6, atol=scale * 1e-7)
            comparisons.append(dict(key=current_key, max_absolute_difference=float(np.max(np.abs(current - expected))),
                                    bit_identical=bool(np.array_equal(current.astype(expected.dtype), expected))))
    return dict(case_id=case['case_id'], passed=True, comparisons=comparisons)


def execute(output, benchmarks, limit):
    protocol = verify(output)
    torch.set_num_threads(4)
    torch.manual_seed(20261027)
    torch.backends.cuda.enable_cudnn_sdp(False)
    models, distribution = shared.load_models({'impute': shared.GENERAL, 'combined': shared.SINGLE})
    versions = {name: [parameter._version for parameter in model.parameters()] for name, model in models.items()}
    total_completed = 0
    for benchmark in benchmarks:
        reference = load_reference(benchmark)
        cases = [case for case in protocol['cases'] if case['benchmark'] == benchmark]
        directory = output / benchmark
        (directory / 'cases').mkdir(parents=True, exist_ok=True)
        replay_path = directory / 'combined_replay.json'
        if not replay_path.exists():
            print(f'Replaying original combined model: {benchmark}/{cases[0]["case_id"]}', flush=True)
            _, arrays, _ = evaluate(cases[0], models['combined'], distribution, reference, uncertainty=False)
            shared.write_json(replay_path, check_replay(cases[0], arrays))
            del arrays
        for case in cases:
            record_path = directory / 'cases' / f"{case['case_id']}.json"
            artifact = record_path.with_suffix('.npz')
            if record_path.exists():
                record = json.loads(record_path.read_text())
                assert shared.sha256(artifact) == record['arrays_sha256']
                continue
            shared.write_json(output / 'status.json', dict(status='running', benchmark=benchmark, case_id=case['case_id']))
            print(f'Running MatrixFormer-Impute: {benchmark}/{case["case_id"]}', flush=True)
            started = time.perf_counter()
            metrics, arrays, details = evaluate(case, models['impute'], distribution, reference)
            atomic_npz(artifact, **arrays)
            parent_record = json.loads(Path(case['parent_record']).read_text())
            metadata = {key: parent_record[key] for key in ('case_id', 'pattern', 'family', 'state_id', 'seed', 'rate') if key in parent_record}
            record = dict(**metadata, benchmark=benchmark, metrics=metrics, details=details,
                          arrays_sha256=shared.sha256(artifact), elapsed_seconds=time.perf_counter() - started,
                          parent_record_sha256=shared.sha256(case['parent_record']))
            for name, model in models.items():
                assert versions[name] == [parameter._version for parameter in model.parameters()]
                assert not model.training
            shared.write_json(record_path, record)
            records = [json.loads(path.read_text()) for path in sorted((directory / 'cases').glob('*.json'))]
            rows = [dict(**{key: saved[key] for key in ('case_id', 'pattern', 'family', 'state_id', 'seed', 'rate') if key in saved},
                         method=method, **scores) for saved in records for method, scores in saved['metrics'].items()]
            pd.DataFrame(rows).to_csv(directory / 'per_case.csv', index=False)
            primary = {'cylinder': 'nrmse', 'quantum': 'fidelity', 'synthetic_mri': 'complex_image_nrmse', 'real_mri': 'rss_image_nrmse'}[benchmark]
            print(json.dumps(dict(case=case['case_id'], seconds=record['elapsed_seconds'],
                                  scores={method: values.get(primary) for method, values in metrics.items()})), flush=True)
            del arrays
            total_completed += 1
            if limit and total_completed >= limit:
                shared.write_json(output / 'status.json', dict(status='partial', new_cases=total_completed))
                return
    completed = sum(len(list((output / name / 'cases').glob('*.json'))) for name in PARENTS)
    shared.write_json(output / 'status.json', dict(status='inference_complete' if completed == 60 else 'partial',
                                                  completed=completed, expected=60, model_parameters_unchanged=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmarks', nargs='+', choices=list(PARENTS), default=list(PARENTS))
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.action == 'plan':
        register(args.output)
    else:
        execute(args.output, args.benchmarks, args.limit)
