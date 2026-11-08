"""Observed-only adaptation of frozen MatrixFormer-Impute on flow and tomography."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.linalg import eigh
import torch

import evaluate_cylinder_completion as shared
import evaluate_quantum_tomography as quantum
from evaluate_physics_impute import forward_native
from recsys_evaluation_common import atomic_npz


ARCHIVE = Path('/mnt/matrixformer_scratch/physics_impute_20261106')
CYLINDER = shared.ROOT / 'runs/physics_cylinder_20261027/pilot'
QUANTUM = shared.ROOT / 'runs/physics_quantum_20261027/pilot'
FLOW_TRANSFORMS = ('global', 'row_center', 'row_scale', 'row_scale_lags')
QUANTUM_TRANSFORMS = ('global', 'identity_free_unfoldings')
RANKS = (2, 4, 8, 16, 32)
STRENGTHS = (0., .0001, .001, .01, .1, 1.)
PROJECTION_STEPS = 40


def row_transform(matrix, scale_rows):
    observed = np.isfinite(matrix)
    _, global_mean, global_scale = shared.normalize_observed(matrix)
    counts = observed.sum(axis=1, keepdims=True)
    means = np.divide(np.nansum(matrix, axis=1, keepdims=True), counts,
                      out=np.full((len(matrix), 1), global_mean), where=counts > 0)
    residual = np.where(observed, matrix - means, 0.)
    scales = np.sqrt((residual ** 2).sum(axis=1, keepdims=True) / np.maximum(counts, 1))
    scales = np.maximum(scales, .05 * global_scale) if scale_rows else np.ones_like(means)
    return (matrix - means) / scales, means, scales


def lag_transform(matrix):
    padded = np.pad(matrix, ((0, 0), (1, 1)), constant_values=np.nan)
    return np.concatenate([padded[:, offset:offset + matrix.shape[1]] for offset in range(3)], axis=1)


def undo_lags(prediction, columns):
    sums = np.zeros((len(prediction), columns))
    counts = np.zeros(columns)
    for offset, block in zip((-1, 0, 1), np.split(prediction, 3, axis=1)):
        source = np.arange(columns)
        target = source + offset
        valid = (target >= 0) & (target < columns)
        sums[:, target[valid]] += block[:, source[valid]]
        counts[target[valid]] += 1
    return sums / counts


def predict_flow(matrix, transform, predict):
    if transform == 'global':
        return predict(matrix)
    values, means, scales = row_transform(matrix, transform != 'row_center')
    if transform == 'row_scale_lags':
        result = undo_lags(predict(lag_transform(values)), matrix.shape[1])
    else:
        result = predict(values)
    return result * scales + means


def low_rank_refine(matrix, initial, rank, steps=PROJECTION_STEPS):
    observed = np.isfinite(matrix)
    completed = np.where(observed, matrix, initial).copy()
    if not np.isfinite(completed).all() or not 0 < rank <= min(matrix.shape):
        raise ValueError('Invalid low-rank completion input')
    for _ in range(steps):
        previous = completed[~observed].copy()
        _, basis = eigh(completed.T @ completed,
                        subset_by_index=(matrix.shape[1] - rank, matrix.shape[1] - 1))
        projected = (completed @ basis) @ basis.T
        completed[~observed] = projected[~observed]
        change = np.linalg.norm(completed[~observed] - previous)
        if change <= 1e-5 * max(np.linalg.norm(previous), 1e-12):
            break
    return completed


def predict_quantum(matrix, transform, predict):
    if transform == 'global':
        return predict(matrix)
    values = matrix.copy()
    values.flat[0] = np.nan
    tensor = values.reshape((4,) * 4)
    predictions = []
    for axes in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):
        result = predict(tensor.transpose(axes).reshape(16, 16))
        predictions.append(result.reshape((4,) * 4).transpose(np.argsort(axes)).reshape(16, 16))
    return np.mean(predictions, axis=0)


def physical_refine(matrix, prior, strength, operators):
    observed = np.flatnonzero(np.isfinite(matrix.ravel()))
    observed = observed[observed != 0]
    hidden = np.flatnonzero(~np.isfinite(matrix.ravel()))
    density = cp.Variable((operators.shape[-1], operators.shape[-1]), hermitian=True)
    design = operators.transpose(0, 2, 1).reshape(len(operators), -1)
    coefficients = cp.real(design @ cp.vec(density, order='C'))
    objective = cp.sum_squares(coefficients[observed] - matrix.ravel()[observed]) / len(observed)
    if strength:
        objective += strength * cp.sum_squares(coefficients[hidden] - prior.ravel()[hidden]) / len(hidden)
    problem = cp.Problem(cp.Minimize(objective), [density >> 0, cp.trace(density) == 1])
    problem.solve(solver='SCS', eps=1e-6, max_iters=50000, verbose=False)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or density.value is None:
        raise RuntimeError(f'Physical refinement failed: {problem.status}')
    result = quantum.project_density(np.asarray(density.value))
    diagnostics = dict(status=problem.status, objective=float(problem.value), iterations=int(problem.solver_stats.num_iters))
    return result, diagnostics


def validation_error(prediction, matrix, validation):
    return float(np.sqrt(np.mean((prediction[validation] - matrix[validation]) ** 2)))


def select_flow(matrix, validation, predict):
    reduced = matrix.copy()
    reduced[validation] = np.nan
    trials = []
    for transform in (*FLOW_TRANSFORMS, 'row_mean_control'):
        if transform == 'row_mean_control':
            _, means, _ = row_transform(reduced, False)
            initial = np.broadcast_to(means, matrix.shape)
        else:
            initial = predict_flow(reduced, transform, predict)
            trials.append(dict(transform=transform, rank=None,
                               validation_rmse=validation_error(initial, matrix, validation)))
        for rank in RANKS:
            candidate = low_rank_refine(reduced, initial, rank)
            trials.append(dict(transform=transform, rank=rank,
                               validation_rmse=validation_error(candidate, matrix, validation)))
    selected = {}
    neural = [trial for trial in trials if trial['transform'] != 'row_mean_control']
    selected['MF-Impute input transform'] = min([trial for trial in neural if trial['rank'] is None], key=lambda trial: trial['validation_rmse'])
    selected['MF-Impute + low-rank refinement'] = min(neural, key=lambda trial: trial['validation_rmse'])
    selected['Low-rank refinement control'] = min([trial for trial in trials if trial['transform'] == 'row_mean_control'], key=lambda trial: trial['validation_rmse'])
    full_predictions = {name: predict_flow(matrix, name, predict) for name in FLOW_TRANSFORMS}
    _, means, _ = row_transform(matrix, False)
    full_predictions['row_mean_control'] = np.broadcast_to(means, matrix.shape)
    predictions = {f'MF-Impute input: {name}': full_predictions[name] for name in FLOW_TRANSFORMS}
    for method, trial in selected.items():
        initial = full_predictions[trial['transform']]
        predictions[method] = low_rank_refine(matrix, initial, trial['rank']) if trial['rank'] else initial
    return predictions, dict(trials=trials, selected=selected)


def select_quantum(matrix, validation, predict, operators):
    reduced = matrix.copy()
    reduced[validation] = np.nan
    reduced_predictions = {name: predict_quantum(reduced, name, predict) for name in QUANTUM_TRANSFORMS}
    trials, input_trials = [], []
    for transform, initial in reduced_predictions.items():
        completed = quantum.complete_coefficients(reduced, initial)
        density = quantum.project_density(quantum.density_from_coefficients(completed, operators))
        input_trials.append(dict(transform=transform, validation_rmse=validation_error(quantum.pauli_coefficients(density, operators), matrix, validation)))
    reduced_predictions['zero_control'] = np.zeros_like(matrix)
    for transform, initial in reduced_predictions.items():
        for strength in STRENGTHS:
            density, diagnostics = physical_refine(reduced, np.clip(initial, -1, 1), strength, operators)
            trials.append(dict(transform=transform, strength=strength, solver=diagnostics,
                               validation_rmse=validation_error(quantum.pauli_coefficients(density, operators), matrix, validation)))
    selected = {
        'MF-Impute input transform': min(input_trials, key=lambda trial: trial['validation_rmse']),
        'MF-Impute + PSD refinement': min([trial for trial in trials if trial['transform'] != 'zero_control'], key=lambda trial: trial['validation_rmse']),
        'Zero-prior + PSD refinement': min([trial for trial in trials if trial['transform'] == 'zero_control'], key=lambda trial: trial['validation_rmse']),
    }
    full_predictions = {name: predict_quantum(matrix, name, predict) for name in QUANTUM_TRANSFORMS}
    full_predictions['zero_control'] = np.zeros_like(matrix)
    densities = {}
    for name in QUANTUM_TRANSFORMS:
        completed = quantum.complete_coefficients(matrix, full_predictions[name])
        densities[f'MF-Impute input: {name}'] = quantum.project_density(quantum.density_from_coefficients(completed, operators))
    densities['MF-Impute input transform'] = densities[f"MF-Impute input: {selected['MF-Impute input transform']['transform']}"]
    for method in ('MF-Impute + PSD refinement', 'Zero-prior + PSD refinement'):
        trial = selected[method]
        densities[method], trial['refit_solver'] = physical_refine(matrix, np.clip(full_predictions[trial['transform']], -1, 1), trial['strength'], operators)
    return densities, dict(trials=trials, input_trials=input_trials, selected=selected)


def register(output):
    if (output / 'protocol.json').exists():
        raise FileExistsError('Use a fresh output directory or run the frozen campaign')
    inputs = []
    for parent in (CYLINDER, QUANTUM):
        inputs.extend(path for path in parent.rglob('*') if path.suffix in ('.json', '.npz', '.csv'))
    inputs.extend(ARCHIVE / benchmark / 'comparison_per_case.csv' for benchmark in ('cylinder', 'quantum'))
    inputs.append(shared.GENERAL.with_name('config.json'))
    sources = [Path(__file__), Path(shared.__file__), Path(quantum.__file__),
               Path(__file__).with_name('evaluate_physics_impute.py')]
    (output / 'source_snapshot').mkdir(parents=True, exist_ok=True)
    for source in sources:
        shutil.copy2(source, output / 'source_snapshot' / source.name)
    shared.write_json(output / 'protocol.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(), benchmarks=['cylinder', 'quantum'], cases=36,
        checkpoint=str(shared.GENERAL), checkpoint_sha256=shared.sha256(shared.GENERAL),
        source_sha256={str(path.resolve()): shared.sha256(path) for path in sources},
        input_sha256={str(path.resolve()): shared.sha256(path) for path in inputs},
        flow_transforms=FLOW_TRANSFORMS, quantum_transforms=QUANTUM_TRANSFORMS,
        library_versions=dict(torch=torch.__version__, numpy=np.__version__, cvxpy=cp.__version__),
        ranks=RANKS, projection_steps=PROJECTION_STEPS, strengths=STRENGTHS,
        selection='RMSE on 10% hidden observed entries; refit on all observations; no test-target selection',
        cylinder_validation='Original archived baseline validation mask',
        quantum_validation='Seed 20261106 + state seed; 10% nonidentity observations, no support repair',
        adaptations='Frozen checkpoint, no gradients, no training; exploratory pilot, not independent confirmation',
        uncertainty='Not propagated through nonlinear refinement; no new interval claims',
        controls='Same low-rank rank grid from row means; same PSD strength grid with a zero prior'))


def summarize(output):
    records = [json.loads(path.read_text()) for path in sorted((output / 'cases').glob('*.json'))]
    rows = []
    for record in records:
        for method, metrics in record['metrics'].items():
            rows.append(dict(**record['metadata'], benchmark=record['benchmark'], method=method, **metrics))
    pd.DataFrame(rows).to_csv(output / 'per_case.csv', index=False)
    frames = [pd.DataFrame(rows)]
    for benchmark in ('cylinder', 'quantum'):
        frames.append(pd.read_csv(ARCHIVE / benchmark / 'comparison_per_case.csv').assign(benchmark=benchmark))
    comparison = pd.concat(frames, ignore_index=True)
    comparison.to_csv(output / 'comparison_per_case.csv', index=False)
    summaries = []
    for benchmark, metric in (('cylinder', 'nrmse'), ('quantum', 'fidelity')):
        frame = comparison[comparison.benchmark == benchmark]
        overall = frame.groupby('method')[metric].agg(['mean', 'std', 'count']).reset_index()
        overall['benchmark'], overall['metric'] = benchmark, metric
        summaries.append(overall)
        groups = ['pattern', 'rate', 'method'] if benchmark == 'cylinder' else ['family', 'rate', 'method']
        frame.groupby(groups)[metric].agg(['mean', 'std', 'count']).to_csv(output / f'{benchmark}_by_condition.csv')
    pd.concat(summaries, ignore_index=True).to_csv(output / 'overall.csv', index=False)


def run(output, benchmarks):
    protocol = json.loads((output / 'protocol.json').read_text())
    for section in ('source_sha256', 'input_sha256'):
        for path, digest in protocol[section].items():
            assert shared.sha256(path) == digest, f'Changed {path}'
    assert shared.sha256(protocol['checkpoint']) == protocol['checkpoint_sha256']
    torch.set_num_threads(4)
    torch.manual_seed(20261106)
    torch.backends.cuda.enable_cudnn_sdp(False)
    models, distribution = shared.load_models({'impute': shared.GENERAL})
    model = models['impute']
    versions = [parameter._version for parameter in model.parameters()]

    def predict(matrix):
        prediction, features = forward_native(matrix, model, distribution)
        del features
        return prediction

    _, operators = quantum.pauli_basis(4)
    (output / 'cases').mkdir(parents=True, exist_ok=True)
    for benchmark in benchmarks:
        parent = CYLINDER if benchmark == 'cylinder' else QUANTUM
        for parent_path in sorted((parent / 'cases').glob('*.json')):
            record = json.loads(parent_path.read_text())
            case_id = record['case_id']
            destination = output / 'cases' / f'{benchmark}_{case_id}.json'
            if destination.exists():
                saved = json.loads(destination.read_text())
                assert shared.sha256(destination.with_suffix('.npz')) == saved['arrays_sha256']
                continue
            shared.write_json(output / 'status.json', dict(status='running', benchmark=benchmark, case_id=case_id))
            print(f'Running {benchmark}/{case_id}', flush=True)
            started = time.perf_counter()
            if benchmark == 'cylinder':
                with np.load(parent / 'dataset.npz') as data:
                    truth = data['truth']
                with np.load(parent / 'predictions' / f'{case_id}.npz') as data:
                    matrix = np.where(data['missing'], np.nan, truth)
                    validation = data['validation']
                predictions, decisions = select_flow(matrix, validation, predict)
                _, _, scale = shared.normalize_observed(matrix)
                metrics = {method: shared.point_metrics(truth, prediction, ~np.isfinite(matrix), scale) for method, prediction in predictions.items()}
            else:
                with np.load(parent_path.with_suffix('.npz')) as data:
                    matrix = data['matrix']
                observed = np.flatnonzero(np.isfinite(matrix.ravel()))
                observed = observed[observed != 0]
                selected = np.random.default_rng(20261106 + record['seed']).choice(observed, size=max(1, len(observed) // 10), replace=False)
                validation = np.zeros_like(matrix, dtype=bool)
                validation.ravel()[selected] = True
                predictions, decisions = select_quantum(matrix, validation, predict, operators)
                with np.load(parent / 'states' / f"{record['state_id']}.npz") as data:
                    truth, density = data['coefficients'], data['density']
                metrics = {method: quantum.reconstruction_metrics(density, truth, prediction, matrix, operators) for method, prediction in predictions.items()}
            assert not (validation & ~np.isfinite(matrix)).any()
            assert versions == [parameter._version for parameter in model.parameters()] and not model.training
            method_keys = {method: f'prediction_{index}' for index, method in enumerate(predictions)}
            arrays = {method_keys[method]: prediction for method, prediction in predictions.items()}
            atomic_npz(destination.with_suffix('.npz'), matrix=matrix, validation=validation, **arrays)
            metadata = {key: record[key] for key in ('case_id', 'seed', 'rate', 'pattern', 'family', 'state_id') if key in record}
            shared.write_json(destination, dict(benchmark=benchmark, metadata=metadata, decisions=decisions,
                              metrics=metrics, prediction_keys=method_keys, arrays_sha256=shared.sha256(destination.with_suffix('.npz')),
                              seconds=time.perf_counter() - started))
            primary = 'nrmse' if benchmark == 'cylinder' else 'fidelity'
            print(json.dumps(dict(case=case_id, scores={method: scores[primary] for method, scores in metrics.items()})), flush=True)
            summarize(output)
    completed = len(list((output / 'cases').glob('*.json')))
    shared.write_json(output / 'status.json', dict(status='complete' if completed == 36 else 'partial', cases=completed, expected=36))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmarks', nargs='+', choices=['cylinder', 'quantum'], default=['cylinder', 'quantum'])
    args = parser.parse_args()
    if args.action == 'plan':
        register(args.output)
    else:
        run(args.output, args.benchmarks)
