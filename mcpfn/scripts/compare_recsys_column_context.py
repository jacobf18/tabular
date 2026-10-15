"""Probe wider item contexts on training observations, then evaluate one fixed width."""

import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix

from compare_recsys_checkpoint import DATASETS, audit_scores
from evaluate_large_recsys import predict_blocks, rating_statistics
from evaluate_matrixformer_recsys_ratings import build_observed, normalization_parameters
from recsys_evaluation_common import REPO, atomic_npz, sha256, write_csv, write_json
from tabimpute.streamed_inference import StreamedImputePFN


RUNS = REPO / 'runs/benchmark_evaluations'
REFERENCE = RUNS / 'matrixformer_large_trained_recsys_20260930'
OUTPUT = RUNS / 'matrixformer_recsys_wider_columns_20261001'
DEPENDENCY = RUNS / 'matrixformer_large_trained_llm_20261001/status.json'
NORMALIZATIONS = ('global', 'column')


def read_json(path):
    return json.loads(path.read_text())


def verify_hashes(hashes):
    for filename, expected in hashes.items():
        if sha256(filename) != expected:
            raise ValueError(f'Input or source changed: {filename}')


def select_columns(probes, candidates, memory_fraction):
    if sorted(probe['columns'] for probe in probes) != sorted(candidates):
        raise ValueError('Require exactly one capacity result per candidate')
    feasible = []
    for probe in probes:
        if probe['status'] not in ('ok', 'oom'):
            raise ValueError('Unexpected capacity failure; do not silently select around it')
        if probe['status'] == 'ok':
            if probe['peak_gpu_allocated_bytes'] > memory_fraction * probe['total_gpu_bytes']:
                raise ValueError('Successful probe exceeded the memory budget')
            feasible.append(probe['columns'])
    if not feasible:
        raise ValueError('None of the wider contexts fit the memory budget')
    return max(feasible)


def load_training(split_path):
    with np.load(split_path, allow_pickle=False) as stored:
        return {key: stored[key] for key in ('shape', 'train_users', 'train_items', 'train_ratings')}


def sparse_training(split):
    train = csr_matrix((split['train_ratings'], (split['train_users'], split['train_items'])),
                       shape=tuple(split['shape']))
    if train.nnz != len(split['train_ratings']) or split['train_ratings'].min() <= 0:
        raise ValueError('Expected positive, unique training ratings')
    return train


def setup_model(manifest, rows, columns):
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(manifest['memory_fraction'])
    config = read_json(Path(manifest['checkpoint']).parent / 'config.json')
    return StreamedImputePFN(device='cuda', checkpoint_path=manifest['checkpoint'],
                            json_config={'config': config['model']}, max_num_rows=rows,
                            max_num_cols=columns, precision_mode='amp_bfloat16', prediction_stat='mean')


def probe(output, columns):
    import torch

    manifest = read_json(output / 'manifest.json')
    reference = read_json(REFERENCE / '10M/global/metrics.json')
    split = load_training(Path(reference['split']))
    observed = sparse_training(split)[:reference['block_rows'], :columns].toarray().astype(np.float64)
    if observed.shape != (reference['block_rows'], columns):
        raise ValueError('Capacity probe must exercise the entire requested context')
    observed[observed == 0] = np.nan
    means, stds = rating_statistics(split['train_items'], split['train_ratings'].astype(np.float64),
                                    int(split['shape'][1]), 'global')
    model = setup_model(manifest, reference['block_rows'], columns)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    record = dict(columns=columns, rows=reference['block_rows'], training_only=True,
                  total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory)
    try:
        completed = model.impute(observed, means=means[:columns], stds=stds[:columns])
        torch.cuda.synchronize()
        visible = np.isfinite(observed)
        if not np.isfinite(completed).all() or not np.array_equal(completed[visible], observed[visible]):
            raise ValueError('Capacity probe produced invalid predictions or changed observations')
        record.update(status='ok', finite=True, observations_preserved=True)
    except torch.OutOfMemoryError as error:
        record.update(status='oom', error=str(error))
    record.update(seconds=time.perf_counter() - started, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
    write_json(output / f'probes/columns_{columns}.json', record)


def evaluate(output, dataset, normalization):
    import torch

    manifest = read_json(output / 'manifest.json')
    columns = read_json(output / 'selection.json')['columns']
    reference_case = REFERENCE / dataset / normalization
    previous = read_json(reference_case / 'metrics.json')
    with np.load(previous['split'], allow_pickle=False) as stored:
        split = dict(stored)
    with np.load(reference_case / 'predictions.npz', allow_pickle=False) as stored:
        audit_scores(split, stored['prediction'], previous)
    dense = DATASETS[dataset][3] == 'dense'
    train = build_observed(split) if dense else sparse_training(split)
    expected = (normalization_parameters(train, normalization) if dense else
                rating_statistics(split['train_items'], split['train_ratings'].astype(np.float64), train.shape[1], normalization))
    with np.load(reference_case / 'normalization.npz', allow_pickle=False) as stored:
        means, stds = stored['means'].copy(), stored['stds'].copy()
    for computed, reused in zip(expected, (means, stds)):
        np.testing.assert_allclose(computed, reused, rtol=1e-7, atol=1e-7)
    case = output / dataset / normalization
    case.mkdir(parents=True, exist_ok=True)
    model = setup_model(manifest, previous['block_rows'], columns)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    context = dict(dataset=dataset, normalization=normalization, columns=columns)
    write_json(output / 'status.json', dict(state='evaluating', **context))
    if dense:
        completed = model.impute(train.copy(), means=means, stds=stds)
        if not np.isfinite(completed).all():
            raise ValueError('Nonfinite completed matrix')
        np.testing.assert_array_equal(completed[split['train_users'], split['train_items']], split['train_ratings'])
        prediction = completed[split['test_users'], split['test_items']]
    else:
        prediction, counts = predict_blocks(model, train, split['test_users'], split['test_items'], means, stds,
                                             output / 'status.json', context)
        atomic_npz(case / 'query_coverage.npz', counts=counts)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    prediction = np.clip(prediction, split['train_ratings'].min(), split['train_ratings'].max())
    scores = audit_scores(split, prediction)
    atomic_npz(case / 'predictions.npz', prediction=prediction)
    atomic_npz(case / 'normalization.npz', means=means, stds=stds)
    write_json(case / 'metrics.json', dict(dataset=previous['dataset'], normalization=normalization, **scores,
               checkpoint_sha256=manifest['checkpoint_sha256'], split=previous['split'], split_sha256=previous['split_sha256'],
               block_rows=previous['block_rows'], block_cols=columns, effective_block_cols=min(columns, train.shape[1]),
               previous_block_cols=previous['block_cols'], previous_RMSE=previous['RMSE'], previous_MAE=previous['MAE'],
               test_ratings=len(prediction), train_ratings=len(split['train_ratings']), seconds=elapsed,
               peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
               predictions_sha256=sha256(case / 'predictions.npz'), normalization_sha256=sha256(case / 'normalization.npz')))
    write_json(case / 'audit.json', dict(reference_and_new_predictions_independently_rescored=True,
               normalization_reused_exactly=True, same_checkpoint=True, same_split=True, same_row_limit=True,
               all_users_and_items_retained=True, selection_uses_training_only_capacity=True))


def summarize(output, manifest):
    rows = []
    for dataset in manifest['datasets']:
        for normalization in NORMALIZATIONS:
            case = output / dataset / normalization
            if not (case / 'metrics.json').exists():
                continue
            record = read_json(case / 'metrics.json')
            verify_hashes({str(case / name): record[field] for name, field in
                           (('predictions.npz', 'predictions_sha256'), ('normalization.npz', 'normalization_sha256'))})
            with np.load(record['split'], allow_pickle=False) as stored:
                split = dict(stored)
            with np.load(case / 'predictions.npz', allow_pickle=False) as stored:
                audit_scores(split, stored['prediction'], record)
            rows.append(dict(dataset=record['dataset'], normalization=normalization, rows=record['block_rows'],
                             columns=record['block_cols'], effective_columns=record['effective_block_cols'],
                             previous_RMSE=record['previous_RMSE'], new_RMSE=record['RMSE'],
                             delta_RMSE=record['RMSE'] - record['previous_RMSE'], previous_MAE=record['previous_MAE'],
                             new_MAE=record['MAE'], delta_MAE=record['MAE'] - record['previous_MAE']))
    if rows:
        write_csv(output / 'comparison.csv', rows)
    lines = ['# Wider recommender column contexts', '', f'Completed {len(rows)}/{2 * len(manifest["datasets"])} cases.',
             'Same larger-matrix checkpoint, frozen splits, user-row limits, observed-training normalization arrays, and scoring.',
             'Global normalization remains primary; column normalization is a control. All users/items are retained.',
             'Only the maximum item-column context changes. Width selection uses training-only memory probes, not held-out errors.',
             'These public splits were previously inspected; the wider contexts exceed the 1000-column training maximum.', '',
             '| Dataset | Normalization | Rows | Column limit | RMSE, 1000 cols | RMSE, wider | MAE, 1000 cols | MAE, wider |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['normalization']} | {row['rows']} | {row['columns']} | {row['previous_RMSE']:.6f} | {row['new_RMSE']:.6f} | {row['previous_MAE']:.6f} | {row['new_MAE']:.6f} |")
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return len(rows)


def campaign(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference_manifest = read_json(REFERENCE / 'manifest.json')
    checkpoint = Path(reference_manifest['checkpoint'])
    if sha256(checkpoint) != reference_manifest['checkpoint_sha256']:
        raise ValueError('Reference checkpoint changed')
    sources = [Path(__file__).resolve(), REPO / 'src/tabimpute/streamed_inference.py',
               *sorted((REPO / 'src/tabimpute/model').rglob('*.py')),
               *[Path(__file__).with_name(name) for name in ('compare_recsys_checkpoint.py', 'evaluate_large_recsys.py',
                  'evaluate_matrixformer_recsys_ratings.py', 'recsys_evaluation_common.py')]]
    inputs = {str(path): sha256(path) for path in (checkpoint, checkpoint.parent / 'config.json', REFERENCE / 'manifest.json')}
    reference_cases = {(dataset, normalization) for dataset in arguments.datasets for normalization in NORMALIZATIONS}
    reference_cases.add(('10M', 'global'))
    for dataset, normalization in sorted(reference_cases):
        case = REFERENCE / dataset / normalization
        record = read_json(case / 'metrics.json')
        if record['checkpoint_sha256'] != reference_manifest['checkpoint_sha256'] or record['block_cols'] != 1000:
            raise ValueError('Require the same new checkpoint and original 1000-column context')
        inputs.update({record['split']: record['split_sha256'], str(case / 'metrics.json'): sha256(case / 'metrics.json'),
                       str(case / 'predictions.npz'): record['predictions_sha256'],
                       str(case / 'normalization.npz'): record['normalization_sha256']})
    manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=reference_manifest['checkpoint_sha256'],
                    reference=str(REFERENCE), datasets=arguments.datasets, candidates=arguments.candidates,
                    memory_fraction=0.8, probe='First 10000 MovieLens 10M users; training observations only; fixed candidate widths',
                    selection='Largest passing candidate within 80 percent GPU memory; freeze before any new held-out scoring',
                    input_sha256=inputs, source_sha256={str(path): sha256(path) for path in sources})
    if (output / 'manifest.json').exists():
        if read_json(output / 'manifest.json') != manifest:
            raise ValueError('Refusing resume with changed sources, inputs, or protocol')
    else:
        write_json(output / 'manifest.json', manifest)
        for path in sources:
            destination = output / 'sources' / path.relative_to(REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    verify_hashes(inputs)
    try:
        while True:
            dependency = read_json(arguments.after)
            active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
            if dependency['state'] in ('complete', 'failed') and not active:
                break
            write_json(output / 'status.json', dict(state='queued', waiting_for=str(arguments.after),
                       dependency_state=dependency['state'], active_gpu_pids=active, updated_unix=time.time()))
            time.sleep(30)
        def worker(mode, *extra):
            verify_hashes(manifest['source_sha256'])
            with (output / ('_'.join((mode, *map(str, extra))).replace('--', '') + '.log')).open('a') as stream:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--mode', mode, '--output', str(output),
                                *map(str, extra)], stdout=stream, stderr=subprocess.STDOUT, check=True)
        for columns in arguments.candidates:
            if not (output / f'probes/columns_{columns}.json').exists():
                write_json(output / 'status.json', dict(state='probing', columns=columns, rows=10000))
                worker('probe', '--columns', columns)
        probes = [read_json(output / f'probes/columns_{columns}.json') for columns in arguments.candidates]
        columns = select_columns(probes, arguments.candidates, manifest['memory_fraction'])
        selection = dict(columns=columns, candidates=arguments.candidates, memory_fraction=manifest['memory_fraction'],
                         criterion=manifest['selection'], probes=probes)
        if (output / 'selection.json').exists() and read_json(output / 'selection.json') != selection:
            raise ValueError('Frozen width selection changed')
        write_json(output / 'selection.json', selection)
        summarize(output, manifest)
        for dataset in arguments.datasets:
            for normalization in NORMALIZATIONS:
                if not (output / dataset / normalization / 'metrics.json').exists():
                    worker('evaluate', '--dataset', dataset, '--normalization', normalization)
                verify_hashes(inputs)
                verify_hashes(manifest['source_sha256'])
                completed = summarize(output, manifest)
                print(f'Completed {completed}/{2 * len(arguments.datasets)} cases; width={columns}', flush=True)
        write_json(output / 'status.json', dict(state='complete', completed=completed, columns=columns))
    except Exception as error:
        write_json(output / 'status.json', dict(state='failed', error=repr(error), updated_unix=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('campaign', 'probe', 'evaluate'), default='campaign')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--datasets', nargs='+', choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument('--candidates', nargs='+', type=int, default=[2000, 3000, 4000])
    parser.add_argument('--after', type=Path, default=DEPENDENCY)
    parser.add_argument('--columns', type=int)
    parser.add_argument('--dataset', choices=tuple(DATASETS))
    parser.add_argument('--normalization', choices=NORMALIZATIONS)
    arguments = parser.parse_args()
    if arguments.mode == 'campaign':
        if sorted(set(arguments.candidates)) != arguments.candidates or min(arguments.candidates) <= 1000:
            parser.error('Candidate widths must be unique, ascending, and greater than 1000')
        campaign(arguments)
    elif arguments.mode == 'probe':
        if arguments.columns is None:
            parser.error('--columns is required for a probe')
        probe(arguments.output.resolve(), arguments.columns)
    else:
        if arguments.dataset is None or arguments.normalization is None:
            parser.error('--dataset and --normalization are required for evaluation')
        evaluate(arguments.output.resolve(), arguments.dataset, arguments.normalization)


if __name__ == '__main__':
    main()
