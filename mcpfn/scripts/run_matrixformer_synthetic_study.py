"""Run a frozen single-model capability, oracle-neighbor and attention study."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time
import warnings

import numpy as np
import torch

from recsys_evaluation_common import atomic_npz, sha256, write_csv, write_json
import synthetic_matrixformer_study as study


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path('/mnt/checkpoints_200/single_model_goal_20261005/four_draw_v57_user_stop_100/best_validation_checkpoint.pth')
CHECKPOINT_SHA = '0d0ecb1c61be86e63a2b959004f858fd9ab80fbc71e934fed439eac0c1039f49'


def source_paths():
    return [Path(__file__), ROOT / 'scripts/synthetic_matrixformer_study.py',
            ROOT / 'scripts/synthetic_low_rank_benchmark.py', ROOT / 'scripts/recsys_evaluation_common.py',
            ROOT / 'src/tabimpute/interface.py', ROOT / 'src/tabimpute/routing.py',
            ROOT / 'src/tabimpute/model/model_new_stable.py', ROOT / 'src/tabimpute/model/bar_distribution.py',
            ROOT / 'src/tabimpute/model/prediction_distribution.py', ROOT / 'src/tabimpute/data/borders.pt']


def register(output, profile):
    if output.exists():
        raise ValueError('Plan requires a new directory')
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Frozen checkpoint hash mismatch')
    output.mkdir(parents=True)
    specification = study.protocol(profile)
    write_json(output / 'protocol.json', specification)
    sources = {}
    for path in source_paths():
        relative = path.relative_to(ROOT)
        target = output / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        sources[str(relative)] = sha256(path)
    shutil.copyfile(CHECKPOINT.parent / 'config.json', output / 'model_config.json')
    write_json(output / 'manifest.json', dict(created=datetime.now(timezone.utc).isoformat(),
               checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA, source_sha256=sources,
               protocol_sha256=sha256(output / 'protocol.json'), config_sha256=sha256(output / 'model_config.json'),
               git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               torch=torch.__version__, numpy=np.__version__, standard_orientation=True,
               model='MatrixFormer (combined V57 checkpoint)', no_neural_updates=True))
    print(json.dumps(dict(planned_cases=len(specification['cases']), output=str(output))), flush=True)


class Predictor:
    def __init__(self, device='cuda'):
        from tabimpute.interface import ImputePFN
        config = json.loads((CHECKPOINT.parent / 'config.json').read_text())['model']
        self.imputer = ImputePFN(device=device, entry_wise_features=False, checkpoint_path=str(CHECKPOINT),
                                json_config={'config': config}, model_family='new_stable', preprocessors=None,
                                max_num_rows=None, max_num_cols=None, prediction_stat='mean',
                                precision_mode='amp_bfloat16' if device == 'cuda' else 'float32')
        self.versions = [parameter._version for parameter in self.imputer.model.parameters()]

    def predict(self, context, means=None, scales=None):
        with torch.inference_mode():
            prediction = self.imputer.impute(context.copy(), means=means, stds=scales).astype(np.float32)
        observed = np.isfinite(context)
        prediction[observed] = context[observed]
        return prediction

    def capture(self, context, targets, means, scales):
        from tabimpute.model.model_new_stable import _apply_rope_to_subset
        collected, maps, handles = {}, {}, []

        def callback(layer, axis):
            def collect(module, arguments, kwargs):
                source = arguments[0]
                encoder = self.imputer.model.feature_encoder
                special_rows, special_columns = encoder.last_num_cls_rows, encoder.last_num_cls_cols
                if axis == 'row':
                    context_indices = targets[:, 1] + special_columns
                    query_indices = targets[:, 0] + special_rows
                else:
                    context_indices = targets[:, 0] + special_rows
                    query_indices = targets[:, 1] + special_columns
                selected = source[torch.as_tensor(context_indices, device=source.device)]
                queries = module._reshape_heads(module.q_proj(selected))
                keys = module._reshape_heads(module.k_proj(selected))
                start = kwargs.get('rope_start_index', 0)
                if module.rope is not None:
                    cosine, sine = module.rope.get_cos_sin(queries.size(2) - start, queries.device, queries.dtype)
                    queries = _apply_rope_to_subset(queries, cosine, sine, start, module.rotary_dim)
                    keys = _apply_rope_to_subset(keys, cosine, sine, start, module.rotary_dim)
                positions = torch.as_tensor(query_indices, device=source.device)
                chosen = queries.gather(2, positions[:, None, None, None].expand(
                    -1, queries.shape[1], 1, queries.shape[3])).squeeze(2)
                with torch.autocast(device_type=source.device.type, enabled=False):
                    weights = torch.softmax(torch.einsum('qhd,qhkd->qhk', chosen.float(), keys.float())
                                            * module.head_dim ** -.5, dim=-1)
                    heatmap = torch.softmax(queries[0].float() @ keys[0].float().transpose(-2, -1)
                                            * module.head_dim ** -.5, dim=-1).mean(0)
                collected[layer, axis] = weights.cpu().numpy()
                maps[layer, axis] = heatmap.cpu().numpy()
            return collect

        for layer, block in enumerate(self.imputer.model.transformer_blocks):
            for axis, module in (('row', block.self_attention_between_datapoints),
                                 ('column', block.self_attention_between_features)):
                handles.append(module.register_forward_pre_hook(callback(layer, axis), with_kwargs=True))
        try:
            prediction = self.predict(context, means, scales)
        finally:
            for handle in handles:
                handle.remove()
        return prediction, collected, maps

    def check_unchanged(self):
        if self.versions != [parameter._version for parameter in self.imputer.model.parameters()]:
            raise ValueError('Model state changed')


def interventions(predictor, context, targets, attention, data, case, means, scales, baseline):
    records = []
    for query, target in enumerate(targets[:4]):
        row, column = map(int, target)
        for axis in ('row', 'column'):
            coordinate, other = (row, column) if axis == 'row' else (column, row)
            oriented = context if axis == 'row' else context.T
            eligible = np.flatnonzero(np.isfinite(oriented[:, other]))
            eligible = eligible[eligible != coordinate]
            count = min(3, len(eligible))
            if not count:
                continue
            weights = np.mean([array[query].mean(0) for (layer, direction), array in attention.items()
                               if direction == axis], axis=0)
            special = len(weights) - oriented.shape[0]
            selected = eligible[np.argsort(-weights[special:][eligible], kind='stable')[:count]]
            rng = np.random.default_rng([study.MASTER_SEED, case['seed'], query, 0 if axis == 'row' else 1])
            arms = [('attention', 0, selected)]
            for repeat in range(3):
                arms.append(('random', repeat, study.matched_random_donors(eligible, selected, coordinate, rng)))
            if case['family'] == 'clusters':
                labels = data[f'{axis}_labels']
                same = eligible[labels[eligible] == labels[coordinate]]
                if len(same) >= count:
                    arms.append(('oracle', 0, rng.choice(same, count, replace=False)))
            for arm, repeat, donors in arms:
                changed = context.copy()
                if axis == 'row':
                    changed[donors, column] = np.nan
                else:
                    changed[row, donors] = np.nan
                prediction = float(predictor.predict(changed, means, scales)[row, column])
                original = float(baseline[row, column])
                truth = float(data['truth'][row, column])
                records.append(dict(query=query, row=row, column=column, axis=axis, arm=arm, repeat=repeat,
                                    removed=len(donors), donor_indices=json.dumps(donors.tolist()),
                                    prediction=prediction, original_prediction=original, truth=truth,
                                    prediction_shift=abs(prediction - original),
                                    squared_error_change=(prediction - truth) ** 2 - (original - truth) ** 2))
    return records


def attention_diagnostics(predictor, data, case, path, means, scales, baseline):
    rng = np.random.default_rng([study.MASTER_SEED, case['seed'], 92])
    missing = np.argwhere(~data['observed'])
    targets = missing[rng.choice(len(missing), min(8, len(missing)), replace=False)]
    captured, attention, maps = predictor.capture(data['context'], targets, means, scales)
    difference = float(np.max(np.abs(captured - baseline)))
    if not np.array_equal(captured, baseline):
        raise ValueError(f'Read-only capture changed predictions: {difference}')
    records = []
    for (layer, axis), weights in attention.items():
        np.testing.assert_allclose(weights.sum(-1), 1, atol=1e-6)
        records += [dict(layer=layer, axis=axis, **record)
                    for record in study.attention_statistics(weights, targets, data, axis)]
    write_csv(path / 'attention.csv', records)
    atomic_npz(path / 'attention.npz', targets=targets,
               **{f'{axis}_{layer}': array for (layer, axis), array in attention.items()},
               **{f'map_{axis}_{layer}': array for (layer, axis), array in maps.items()})
    removal = interventions(predictor, data['context'], targets, attention, data, case, means, scales, baseline)
    write_csv(path / 'interventions.csv', removal)
    write_json(path / 'attention_audit.json', dict(capture_max_difference=difference,
               capture_bit_identical=True, queries=targets.tolist(), layers=8, heads=16,
               normalization='Observed-column statistics frozen at original context during donor removals',
               special_tokens='Reported separately; not renormalized away from full attention',
               selection='All heads/layers, first four of eight seeded queries for interventions'))


def execute(output, case_limit=None):
    specification = json.loads((output / 'protocol.json').read_text())
    manifest = json.loads((output / 'manifest.json').read_text())
    if sha256(CHECKPOINT) != CHECKPOINT_SHA or sha256(output / 'protocol.json') != manifest['protocol_sha256']:
        raise ValueError('Checkpoint or protocol changed')
    for relative, expected in manifest['source_sha256'].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f'Source changed since registration: {relative}')
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError(f'GPU is occupied; do not overlap existing work: {active}')
    predictor = Predictor()
    attempted = 0
    for index, case in enumerate(specification['cases']):
        path = output / 'cases' / case['case_id']
        if (path / 'complete.json').exists() or (path / 'failure.json').exists():
            continue
        if case_limit is not None and attempted >= case_limit:
            break
        attempted += 1
        path.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            data = study.generate(case)
            atomic_npz(path / 'inputs.npz', **data)
            means = np.nanmean(data['context'].astype(float), axis=0)
            scales = np.nanstd(data['context'].astype(float), axis=0)
            means = np.where(np.isfinite(means), means, 0.)
            scales = np.where(np.isfinite(scales), scales, 1.)
            matrixformer = predictor.predict(data['context'], means, scales)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                predictions, tuning = study.fit_baselines(data['context'], data, case)
            predictions['matrixformer'] = matrixformer
            records = [dict(**case, method=method, **study.scores(prediction, data))
                       for method, prediction in predictions.items()]
            atomic_npz(path / 'predictions.npz', **predictions)
            write_json(path / 'scores.json', records)
            write_json(path / 'tuning.json', dict(**tuning, warnings=[str(warning.message) for warning in caught]))
            if study.selected_attention_case(case):
                attention_diagnostics(predictor, data, case, path, means, scales, matrixformer)
            predictor.check_unchanged()
            write_json(path / 'complete.json', dict(seconds=time.perf_counter() - started,
                       sha256={file.name: sha256(file) for file in path.iterdir() if file.is_file()}))
            print(json.dumps(dict(index=index + 1, total=len(specification['cases']), case=case,
                                  seconds=round(time.perf_counter() - started, 2),
                                  scores={record['method']: round(record['rmse'], 4) for record in records})), flush=True)
        except Exception as error:
            write_json(path / 'failure.json', dict(case=case, error=repr(error)))
            raise
    paths = [output / 'cases' / case['case_id'] for case in specification['cases']]
    write_json(output / 'status.json', dict(expected=len(paths), complete=sum((path / 'complete.json').exists() for path in paths),
               failed=sum((path / 'failure.json').exists() for path in paths), checkpoint_unchanged=sha256(CHECKPOINT) == CHECKPOINT_SHA))


def audit(output):
    specification = json.loads((output / 'protocol.json').read_text())
    rows, attention, removal = [], [], []
    import pandas as pd
    for case in specification['cases']:
        path = output / 'cases' / case['case_id']
        complete = json.loads((path / 'complete.json').read_text())
        for name, expected in complete['sha256'].items():
            if sha256(path / name) != expected:
                raise ValueError(f'Artifact hash mismatch: {path / name}')
        generated = study.generate(case)
        with np.load(path / 'inputs.npz') as stored:
            for name, expected in generated.items():
                np.testing.assert_array_equal(stored[name], expected)
        saved_scores = json.loads((path / 'scores.json').read_text())
        with np.load(path / 'predictions.npz') as predictions:
            expected_methods = set(study.METHOD_NAMES)
            if case['family'] != 'clusters':
                expected_methods.remove('oracle_block')
            if case['family'] == 'noise':
                expected_methods.remove('oracle_neighbors')
            if set(predictions.files) != expected_methods or {row['method'] for row in saved_scores} != expected_methods:
                raise ValueError('Missing method')
            for row in saved_scores:
                for name, value in study.scores(predictions[row['method']], generated).items():
                    if value != row[name]:
                        raise ValueError(f'Recomputed metric differs: {name}')
        rows.extend(saved_scores)
        if study.selected_attention_case(case):
            attention.append(pd.read_csv(path / 'attention.csv').assign(**case))
            removal.append(pd.read_csv(path / 'interventions.csv').assign(**case))
            audit_record = json.loads((path / 'attention_audit.json').read_text())
            if not audit_record['capture_bit_identical']:
                raise ValueError('Attention changed predictions')
    pd.DataFrame(rows).to_csv(output / 'scores.csv', index=False)
    pd.concat(attention, ignore_index=True).to_csv(output / 'attention.csv', index=False)
    pd.concat(removal, ignore_index=True).to_csv(output / 'interventions.csv', index=False)
    write_json(output / 'audit.json', dict(cases=len(specification['cases']), method_results=len(rows),
               attention_cases=len(attention), all_inputs_regenerated=True, all_predictions_rescored=True,
               all_artifact_hashes_verified=True, all_attention_captures_bit_identical=True,
               checkpoint_unchanged=sha256(CHECKPOINT) == CHECKPOINT_SHA,
               source_sha256=sha256(Path(__file__))))
    print((output / 'audit.json').read_text(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run', 'audit'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=('smoke', 'main'), default='main')
    parser.add_argument('--case-limit', type=int)
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    if arguments.command == 'plan':
        register(arguments.output, arguments.profile)
    elif arguments.command == 'run':
        execute(arguments.output, arguments.case_limit)
    else:
        audit(arguments.output)


if __name__ == '__main__':
    main()
