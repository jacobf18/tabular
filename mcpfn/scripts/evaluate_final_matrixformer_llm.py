"""Evaluate one checkpoint on all frozen LLM-completion settings, with resumable audits."""

import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from evaluate_matched_llm import BENCH, DATASETS, METHODS, OUTPUT as REFERENCE, verify_frozen
from evaluate_single_matrixformer_llm import checked_metrics, frozen_settings, predict_setting
from evaluate_two_expert_router import make_model
from recsys_evaluation_common import atomic_npz, sha256, write_json


def kernel_ridge(visible, frozen, query):
    standardized = np.clip((visible - frozen['mean']) / frozen['scale'], -10., 10.)
    prediction = np.zeros_like(visible, dtype=float)
    for row_index, row in enumerate(standardized):
        revealed = np.flatnonzero(np.isfinite(row))
        if revealed.size:
            gram = frozen['correlation'][np.ix_(revealed, revealed)] + .01 * np.eye(revealed.size)
            prediction[row_index] = frozen['correlation'][:, revealed] @ np.linalg.solve(gram, row[revealed])
    return prediction[query]


def fold_rows(specification, scores):
    rows = []
    for method, metrics in scores.items():
        relative = np.sqrt(metrics['ss_res'] / metrics['ss_tot']) if metrics['ss_tot'] > 0 else np.nan
        rows.append(dict(dataset=specification['dataset'], selection=specification['method'],
                         budget=specification['budget'], fold=specification['fold'], method=method,
                         rmse=metrics['rmse'], relative_rmse=relative, n_eval=metrics['n_eval']))
    return rows


def verify_prediction(path, specification, truth, query, manifest_hash):
    record = json.loads(path.with_suffix('.json').read_text())
    if record['manifest_sha256'] != manifest_hash or record['prediction_sha256'] != sha256(path):
        raise ValueError('Changed prediction or evaluation manifest')
    for key, value in specification.items():
        if record[key] != value:
            raise ValueError(f'Changed setting metadata: {key}')
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['target_z'], truth)
        np.testing.assert_array_equal(saved['query_rows'], np.where(query)[0])
        np.testing.assert_array_equal(saved['query_columns'], np.where(query)[1])
        scores = dict(matrixformer=checked_metrics(truth, saved['student_z']),
                      kernel_ridge=checked_metrics(truth, saved['kernel_ridge_z']))
    if scores != record['scores']:
        raise ValueError('Saved metrics do not match prediction rescore')
    return record


def summarize(output, records, complete=False):
    table = pd.DataFrame([row for record in records for row in fold_rows(record, record['scores'])])
    if table.empty:
        return
    keys = ['dataset', 'selection', 'budget', 'fold', 'method']
    if table.duplicated(keys).any() or not np.isfinite(table[['rmse', 'relative_rmse']]).all().all():
        raise ValueError('Duplicated or nonfinite fold scores')
    if complete:
        expected = {(dataset, selection, budget, fold, method) for dataset in DATASETS
                    for selection in METHODS for budget in range(1, 16) for fold in range(10)
                    for method in ('matrixformer', 'kernel_ridge')}
        if set(table[keys].itertuples(index=False, name=None)) != expected:
            raise ValueError('Incomplete LLM evaluation inventory')
    table.to_csv(output / 'per_fold.csv', index=False)
    panels = table.groupby(['dataset', 'selection', 'budget', 'method']).agg(
        rmse=('rmse', 'mean'), relative_rmse=('relative_rmse', 'mean'), folds=('fold', 'nunique'))
    panels.to_csv(output / 'panels.csv')


def run(args):
    import torch

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint = args.checkpoint.resolve()
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError('Checkpoint differs from the requested model')
    verify_frozen(REFERENCE)
    sources = [Path(__file__).resolve(), Path(__file__).with_name('evaluate_single_matrixformer_llm.py'),
               Path(__file__).with_name('evaluate_two_expert_router.py'),
               *sorted((Path(__file__).resolve().parents[1] / 'src/tabimpute').rglob('*.py'))]
    manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=args.checkpoint_sha256,
                    config_sha256=sha256(checkpoint.parent / 'config.json'),
                    frozen_index_sha256=sha256(REFERENCE / 'frozen_index.json'),
                    source_sha256={str(path): sha256(path) for path in sources},
                    data_sha256={dataset: sha256(BENCH / 'data' / f'{dataset}.matrix.npz') for dataset in DATASETS},
                    context_rows=3800, test_batch_rows=200, folds=10, budgets=list(range(1, 16)),
                    datasets=list(DATASETS), selections=list(METHODS), seed=42,
                    inference='Single checkpoint, standard orientation, FP32 weights, BF16 autocast, predictive mean',
                    scoring='Context-standardized targets and predictions clipped to [-10, 10]; all query entries',
                    kernel_ridge='Frozen correlation, ridge 0.01, revealed standardized inputs clipped to [-10, 10]')
    manifest_path = output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Refusing to resume with changed checkpoint, sources or inputs')
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    manifest_hash = sha256(manifest_path)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.enable_cudnn_sdp(True)
    torch.cuda.set_per_process_memory_fraction(.9)
    model = None
    records = []
    new_settings = 0
    started = time.monotonic()
    for specification, frozen, inputs in frozen_settings():
        context, visible, query, truth = inputs
        path = output / 'predictions' / (specification['case_id'] + '.npz')
        if path.with_suffix('.json').exists():
            record = verify_prediction(path, specification, truth, query, manifest_hash)
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_allclose(saved['kernel_ridge_z'], kernel_ridge(visible, frozen, query), rtol=1e-12, atol=1e-12)
        else:
            if args.limit and new_settings >= args.limit:
                break
            if model is None:
                model = make_model(checkpoint)
            begin = time.monotonic()
            with torch.inference_mode():
                prediction = predict_setting(model, context, visible, query, frozen['mean'], frozen['scale'])
            baseline = kernel_ridge(visible, frozen, query)
            scores = dict(matrixformer=checked_metrics(truth, prediction),
                          kernel_ridge=checked_metrics(truth, baseline))
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_npz(path, target_z=truth, student_z=prediction, kernel_ridge_z=baseline,
                       query_rows=np.where(query)[0], query_columns=np.where(query)[1])
            record = {**specification, 'scores': scores, 'seconds': time.monotonic()-begin,
                      'manifest_sha256': manifest_hash, 'prediction_sha256': sha256(path)}
            write_json(path.with_suffix('.json'), record)
            verify_prediction(path, specification, truth, query, manifest_hash)
            new_settings += 1
            print(json.dumps(dict(case_id=specification['case_id'], completed=len(records)+1,
                                  total=1800, seconds=record['seconds'],
                                  rmse=scores['matrixformer']['rmse'])), flush=True)
        records.append(record)
        write_json(output / 'status.json', dict(state='running', completed=len(records), total=1800,
                                                new_settings=new_settings, elapsed_seconds=time.monotonic()-started,
                                                last_case=specification['case_id']))
        if len(records) % 45 == 0:
            summarize(output, records)
    complete = len(records) == 1800
    summarize(output, records, complete=complete)
    if complete:
        write_json(output / 'audit.json', dict(complete=True, settings=1800, folds=10, panels=180,
                                               all_predictions_rescored=True, shared_frozen_queries=True,
                                               manifest_sha256=manifest_hash,
                                               per_fold_sha256=sha256(output / 'per_fold.csv')))
    write_json(output / 'status.json', dict(state='complete' if complete else 'limited',
                                            completed=len(records), total=1800,
                                            elapsed_seconds=time.monotonic()-started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
