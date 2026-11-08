"""Real-benchmark retention gates relative to the unchanged combined MatrixFormer."""

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import evaluate_cylinder_completion as shared
import evaluate_single_matrixformer_tabular as tabular
import evaluate_single_matrixformer_llm as llm
import evaluate_single_matrixformer_recsys as recsys


ARCHIVE = Path('/mnt/matrixformer_scratch/single_model_goal_20261005/four_draw_v57_user_eval_20261020_v2')
PROTOCOLS = shared.ROOT / 'runs/single_model_goal_20261005'
ROOT = Path('/mnt/matrixformer_scratch/scientific_prior_training_20261108')
LIMITS = dict(overall_relative_error=.02, pattern_relative_error=.05,
              dataset_relative_error=.10, llm_r2_drop=.01)


def read(path):
    return json.loads(Path(path).read_text())


def selected_cases(cases, stage, full):
    if full:
        return [case for case in cases if stage != 'recsys' or case['dataset'] != '10M']
    if stage == 'tabular':
        selected = [case for case in cases if case['suite'] == 'missbench']
        groups = defaultdict(list)
        for case in cases:
            if case['suite'] == 'uci' and case['missingness'] == .3:
                groups[case['dataset'], case['pattern']].append(case)
        selected += [min(group, key=lambda case: case['case_id']) for group in groups.values()]
        return sorted(selected, key=lambda case: case['case_id'])
    if stage == 'llm':
        return [case for case in cases if case['fold'] in (0, 5) and case['budget'] in (1, 8, 15)]
    return [case for case in cases if case['dataset'] in ('100K', 'netflix')]


def register(output, full=False):
    output.mkdir(parents=True, exist_ok=False)
    stages = {}
    hashes = {}
    for stage in ('tabular', 'llm', 'recsys'):
        path = PROTOCOLS / f'{stage}_protocol_v1/protocol.json'
        protocol = read(path)
        hashes[str(path)] = shared.sha256(path)
        manifest = read(ARCHIVE / stage / 'manifest.json')
        if shared.SINGLE_SHA256 not in manifest['candidate_sha256'].values():
            raise ValueError('Archive is not the original combined model')
        cases = selected_cases(protocol['cases'], stage, full)
        for case in cases:
            if stage == 'tabular':
                record_path = ARCHIVE / stage / 'cases' / f'{case["case_id"]}.json'
                prediction_path = ARCHIVE / stage / 'predictions' / f'{case["case_id"]}.npz'
            elif stage == 'llm':
                record_path = ARCHIVE / stage / 'predictions' / f'{case["case_id"]}.json'
                prediction_path = record_path.with_suffix('.npz')
                hashes[case['fold_path']] = shared.sha256(case['fold_path'])
                data = llm.BENCH / 'data' / f'{case["dataset"]}.matrix.npz'
                hashes[str(data)] = shared.sha256(data)
            else:
                record_path = ARCHIVE / stage / case['dataset'] / 'student.json'
                prediction_path = record_path.with_suffix('.npz')
                hashes[case['split']] = shared.sha256(case['split'])
                hashes[case['normalization']] = case['normalization_sha256']
            parent = read(record_path)
            assert shared.sha256(prediction_path) == parent['prediction_sha256']
            hashes[str(record_path)] = shared.sha256(record_path)
            hashes[str(prediction_path)] = parent['prediction_sha256']
            case['parent_record'] = str(record_path)
            case['parent_prediction'] = str(prediction_path)
        stages[stage] = cases
    sources = [Path(__file__), Path(tabular.__file__), Path(llm.__file__), Path(recsys.__file__)]
    shared.write_json(output / 'protocol.json', dict(
        stages=stages, input_sha256=hashes, source_sha256={str(path.resolve()): shared.sha256(path) for path in sources},
        limits=LIMITS, full=full, anchor_sha256=shared.SINGLE_SHA256,
        interpretation='Previously inspected real benchmarks used as development retention gates; no untouched-test claim',
        scope='All MissBench cases; UCI .3 missingness, first lexical repeat per dataset/pattern; LLM folds 0/5 and budgets 1/8/15, all methods; full 100K/Netflix. Final full gate expands UCI/LLM and adds MovieLens1M. MovieLens10M excluded.',
    ))


def gate_checks(rows, limits=LIMITS):
    frame = pd.DataFrame(rows)
    checks = []

    def add(name, before, after, margin, difference=False):
        change = before - after if difference else after / max(before, 1e-12) - 1
        checks.append(dict(name=name, parent=float(before), candidate=float(after),
                           change=float(change), limit=margin, passed=bool(change <= margin)))

    for suite in ('missbench', 'uci'):
        selected = frame[(frame.stage == 'tabular') & (frame.suite == suite)]
        by_dataset = selected.groupby('dataset')[['parent_error', 'error']].mean()
        add(suite, by_dataset.parent_error.mean(), by_dataset.error.mean(), limits['overall_relative_error'])
        for pattern, group in selected.groupby('pattern'):
            means = group.groupby('dataset')[['parent_error', 'error']].mean().mean()
            add(f'{suite}/{pattern}', means.parent_error, means.error, limits['pattern_relative_error'])
        for dataset, row in by_dataset.iterrows():
            add(f'{suite}/dataset/{dataset}', row.parent_error, row.error, limits['dataset_relative_error'])
    for dataset, group in frame[frame.stage == 'llm'].groupby('dataset'):
        add(f'llm/{dataset}/rmse', group.parent_error.mean(), group.error.mean(), limits['overall_relative_error'])
        add(f'llm/{dataset}/r2', group.parent_r2.mean(), group.r2.mean(), limits['llm_r2_drop'], difference=True)
    for row in frame[frame.stage == 'recsys'].itertuples():
        add(f'recsys/{row.dataset}/rmse', row.parent_error, row.error, limits['overall_relative_error'])
        add(f'recsys/{row.dataset}/mae', row.parent_mae, row.mae, limits['overall_relative_error'])
    if not all(np.isfinite(record['change']) for record in checks):
        raise ValueError('Incomplete or nonfinite retention metrics')
    return checks


def evaluate(checkpoint, protocol_path, output):
    checkpoint, output = Path(checkpoint), Path(output)
    protocol = read(protocol_path / 'protocol.json')
    for field in ('source_sha256', 'input_sha256'):
        tabular.verify_hashes(protocol[field])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'predictions').mkdir()
    shared.write_json(output / 'manifest.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=shared.sha256(checkpoint),
                      config_sha256=shared.sha256(checkpoint.with_name('config.json')),
                      protocol_sha256=shared.sha256(protocol_path / 'protocol.json')))
    torch.set_num_threads(2)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(.45)
    originals = {suite: tabular.load_original(path, 'compute_normalized_rmse_columnwise') for suite, path in tabular.SCORERS.items()}
    wrapper = tabular.make_model(checkpoint)
    expert = tabular.TabularExpert(wrapper)
    rows = []

    def save_prediction(stage, case_id, **arrays):
        path = output / 'predictions' / stage / f'{case_id}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return shared.sha256(path)

    def record(row):
        rows.append(row)
        with (output / 'records.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        shared.write_json(output / 'status.json', dict(state='evaluating', records=len(rows), updated=time.time()))

    for case in protocol['stages']['tabular']:
        missing, truth = tabular.load_case(case)
        expert.suite = case['suite']
        prediction = expert.impute(missing.copy(order='K'))
        score, _ = tabular.checked_score(truth, prediction, missing, case['suite'], originals[case['suite']])
        parent = read(case['parent_record'])
        previous = tabular.prediction_grid(missing, Path(case['parent_prediction']), 'prediction')
        parent_score, _ = tabular.checked_score(truth, previous, missing, case['suite'], originals[case['suite']])
        np.testing.assert_allclose(parent_score, parent['new_nrmse'], rtol=1e-6)
        digest = save_prediction('tabular', case['case_id'], prediction=prediction[np.isnan(missing)], shape=missing.shape)
        record(dict(stage='tabular', suite=case['suite'], dataset=case['dataset'], pattern=case['pattern'],
                    case_id=case['case_id'], error=score, parent_error=parent_score, prediction_sha256=digest))
    del expert, wrapper
    gc.collect()
    torch.cuda.empty_cache()
    wrapper = tabular.make_model(checkpoint)
    datasets = {}
    for case in protocol['stages']['llm']:
        dataset = case['dataset']
        if dataset not in datasets:
            with np.load(llm.BENCH / 'data' / f'{dataset}.matrix.npz') as stored:
                datasets[dataset] = stored['B'].astype(np.float64), stored['O'].astype(bool)
        frozen = dict(np.load(case['fold_path']))
        context, test, mask, truth = llm.setting_inputs(*datasets[dataset], frozen, case['method'], case['budget'])
        prediction = llm.predict_setting(wrapper, context, test, mask, frozen['mean'], frozen['scale'])
        score = llm.checked_metrics(truth, prediction)
        parent = read(case['parent_record'])['student']
        previous = np.load(case['parent_prediction'])
        np.testing.assert_allclose(previous['target_z'], truth, rtol=1e-10, atol=1e-10)
        rescored = llm.checked_metrics(truth, previous['student_z'])
        np.testing.assert_allclose([rescored['rmse'], rescored['r2']], [parent['rmse'], parent['r2']], rtol=1e-10)
        digest = save_prediction('llm', case['case_id'], prediction=prediction, target=truth)
        record(dict(stage='llm', suite='llm', dataset=dataset, case_id=case['case_id'], error=score['rmse'],
                    parent_error=parent['rmse'], r2=score['r2'], parent_r2=parent['r2'], prediction_sha256=digest))
    del wrapper, datasets
    gc.collect()
    torch.cuda.empty_cache()
    wrapper = tabular.make_model(checkpoint, streamed=True)
    for case in protocol['stages']['recsys']:
        split = recsys.load_split(case['split'])
        observed = recsys.observed_matrix(split)
        normalizer = dict(np.load(case['normalization']))
        prediction, counts = recsys.predict_queries(wrapper, observed, split['test_users'], split['test_items'], normalizer, case['rating_range'])
        score = recsys.audit_scores(split, prediction)
        parent = read(case['parent_record'])['metrics']
        recsys.audit_scores(split, np.load(case['parent_prediction'])['prediction'], parent)
        digest = save_prediction('recsys', case['dataset'], prediction=prediction, counts=counts)
        record(dict(stage='recsys', suite='recsys', dataset=case['dataset'], case_id=case['dataset'],
                    error=score['RMSE'], parent_error=parent['RMSE'], mae=score['MAE'], parent_mae=parent['MAE'], prediction_sha256=digest))
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()
    assert len(rows) == sum(len(cases) for cases in protocol['stages'].values())
    pd.DataFrame(rows).to_csv(output / 'per_case.csv', index=False)
    checks = gate_checks(rows, protocol['limits'])
    pd.DataFrame(checks).to_csv(output / 'checks.csv', index=False)
    result = dict(complete=True, passed=all(check['passed'] for check in checks), records=len(rows),
                  full=protocol['full'], failures=[check for check in checks if not check['passed']])
    shared.write_json(output / 'gate.json', result)
    shared.write_json(output / 'status.json', dict(state='complete', **result, updated=time.time()))
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['register', 'evaluate'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    if args.stage == 'register':
        register(args.output, args.full)
    else:
        evaluate(args.checkpoint, args.protocol, args.output)
