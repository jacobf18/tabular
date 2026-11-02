"""Frozen, scale-preserving orientation ablation of the combined MatrixFormer.

Standard predictions are verified from the October 20 archive. Only transposed
inference is new; averaging is performed on aligned predictions, never metrics.
The evaluation archive and paper are read-only inputs to this campaign.
"""

import argparse
import fcntl
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

import evaluate_single_matrixformer_llm as llm
import evaluate_single_matrixformer_recsys as recsys
import evaluate_single_matrixformer_tabular as tabular
from evaluate_chunked_uci import configure_windows
from evaluate_dwip_missbench_uci import standardize_uci
from recsys_evaluation_common import atomic_npz, sha256, write_json
from tabimpute.streamed_inference import overlapping_spans


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path('/mnt/checkpoints_200/single_model_goal_20261005/four_draw_v57_user_eval_20261020')
CHECKPOINT = Path('/mnt/checkpoints_200/single_model_goal_20261005/four_draw_v57_user_stop_100/best_validation_checkpoint.pth')
CHECKPOINT_SHA = '0d0ecb1c61be86e63a2b959004f858fd9ab80fbc71e934fed439eac0c1039f49'
OUTPUT = Path('/mnt/matrixformer_scratch/single_matrixformer_orientation_20261102')
STAGES = ('tabular', 'llm', 'recsys')
VARIANTS = ('standard', 'transpose', 'average')


def read(path):
    return json.loads(Path(path).read_text())


def verify_file(path, expected):
    if sha256(Path(path)) != expected:
        raise ValueError(f'Frozen artifact changed: {path}')


def archive_paths(stage, case):
    if stage == 'tabular':
        return (ARCHIVE / stage / 'cases' / (case['case_id'] + '.json'),
                ARCHIVE / stage / 'predictions' / (case['case_id'] + '.npz'))
    if stage == 'llm':
        path = ARCHIVE / stage / 'predictions' / (case['case_id'] + '.npz')
        return path.with_suffix('.json'), path
    path = ARCHIVE / stage / case['dataset'] / 'student.npz'
    return path.with_suffix('.json'), path


def case_key(stage, case):
    return case['dataset'] if stage == 'recsys' else case['case_id']


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        return load_manifest(output)
    verify_file(CHECKPOINT, CHECKPOINT_SHA)
    protocols, hashes, originals = {}, {}, {}
    for stage in STAGES:
        source = ARCHIVE / stage / 'manifest.json'
        manifest = read(source)
        if manifest['candidate_sha256'].get(str(CHECKPOINT)) != CHECKPOINT_SHA:
            raise ValueError('Wrong archived model')
        protocol_path = Path(manifest['protocol']) / 'protocol.json'
        verify_file(protocol_path, manifest['protocol_sha256'])
        protocols[stage] = read(protocol_path)
        hashes[str(source)] = sha256(source)
        hashes[str(protocol_path)] = sha256(protocol_path)
        originals[stage] = {}
        for case in protocols[stage]['cases']:
            record_path, prediction_path = archive_paths(stage, case)
            record = read(record_path)
            if record['manifest_sha256'] != hashes[str(source)]:
                raise ValueError(f'Archived prediction uses another manifest: {record_path}')
            verify_file(prediction_path, record['prediction_sha256'])
            originals[stage][case_key(stage, case)] = {
                'record_sha256': sha256(record_path), 'prediction_sha256': record['prediction_sha256']}
    sources = [Path(__file__), Path(tabular.__file__), Path(recsys.__file__), Path(llm.__file__)]
    sources += [ROOT / 'scripts' / name for name in (
        'evaluate_two_expert_router.py', 'evaluate_chunked_uci.py', 'evaluate_dwip_missbench_uci.py',
        'evaluate_matched_llm.py', 'compare_recsys_checkpoint.py', 'dwip_recsys_experiment.py',
        'recsys_evaluation_common.py', 'audit_dwip_missbench_uci.py')]
    sources += sorted((ROOT / 'src/tabimpute').rglob('*.py'))
    manifest = {
        'registered_unix': time.time(), 'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': CHECKPOINT_SHA,
        'config_sha256': sha256(CHECKPOINT.parent / 'config.json'), 'archive': str(ARCHIVE),
        'protocols': protocols, 'original_predictions': originals, 'input_sha256': hashes,
        'source_sha256': {str(path): sha256(path) for path in sources},
        'policy': {
            'training': False, 'paper_changes': False, 'orientation_selection': 'none; report all three',
            'preprocessing': 'Freeze the exact original-axis observed-only normalization; transpose normalized inputs without refitting statistics',
            'tabular': 'Preserve original column statistics including the existing UCI outer normalization and interface epsilon/constant rules',
            'recsys': 'Keep original observed-only two-way biases and user scales; do not reinterpret user scaling as item scaling',
            'llm': 'Keep original context and 200-row test batches, masks, selections, scoring means/scales; transpose each normalized full context/test batch',
            'windows': 'MissBench/LLM full matrices; UCI/ratings 2000 rows x 1000 columns in each orientation; transpose can change window membership',
            'averaging': 'Arithmetic mean of aligned predictions; ratings clip each orientation before averaging, LLM clips only in the unchanged scorer',
            'standard': 'Hash-verified frozen predictions; representative numerical parity controls are rerun',
            'precision': 'FP32 weights, BF16 autocast, predictive mean; original SDPA settings per suite',
        },
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'numpy': np.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)},
    }
    for source in sources:
        destination = output / 'source_snapshot' / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    write_json(manifest_path, manifest)
    write_json(output / 'status.json', {'state': 'registered', 'updated_unix': time.time()})
    return manifest


def load_manifest(output):
    manifest = read(output / 'manifest.json')
    verify_file(CHECKPOINT, manifest['checkpoint_sha256'])
    verify_file(CHECKPOINT.parent / 'config.json', manifest['config_sha256'])
    for path, expected in {**manifest['source_sha256'], **manifest['input_sha256']}.items():
        verify_file(path, expected)
    return manifest


def original(stage, case, manifest):
    record_path, prediction_path = archive_paths(stage, case)
    hashes = manifest['original_predictions'][stage][case_key(stage, case)]
    verify_file(record_path, hashes['record_sha256'])
    verify_file(prediction_path, hashes['prediction_sha256'])
    with np.load(prediction_path, allow_pickle=False) as saved:
        arrays = dict(saved)
    return read(record_path), arrays


def interface_normalization(matrix):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        means, scales = np.nanmean(matrix, axis=0), np.nanstd(matrix, axis=0)
    means = np.where(np.isnan(means), 0, means)
    scales = np.where(np.isnan(scales), 1, scales) + 1e-16
    return (matrix - means) / scales, means, scales


def aligned_completion(model, normalized, means, scales, transpose=False, windowed=False):
    source = normalized.T if transpose else normalized
    if windowed:
        configure_windows(model, source.shape, 2000, 1000)
    else:
        model.max_num_rows = model.max_num_cols = None
    completed, _ = model.get_imputation(source.copy(order='K'))
    completed = completed.T if transpose else completed
    result = completed.astype(normalized.dtype) * scales + means
    if result.shape != normalized.shape or not np.isfinite(result).all():
        raise ValueError('Invalid aligned completion')
    return result


def predict_tabular(model, observed, suite, transpose):
    if suite == 'uci':
        source, outer_mean, outer_scale, _ = standardize_uci(observed)
    else:
        source, outer_mean, outer_scale = observed, 0., 1.
    normalized, means, scales = interface_normalization(source.copy())
    completed = aligned_completion(model, normalized, means, scales, transpose, suite == 'uci')
    completed = completed * outer_scale + outer_mean
    completed[~np.isnan(observed)] = observed[~np.isnan(observed)]
    return completed


def predict_llm(model, context, visible, mask, mean, scale, transpose):
    predictions = np.empty(visible.shape, dtype=np.float64)
    for start in range(0, len(visible), llm.BATCH_ROWS):
        stop = min(start + llm.BATCH_ROWS, len(visible))
        batch = np.vstack([context, visible[start:stop]])
        normalized, means, scales = interface_normalization(batch)
        completed = aligned_completion(model, normalized, means, scales, transpose)
        predictions[start:stop] = completed[len(context):]
    return ((predictions - mean) / scale)[mask]


def combine(standard, transposed):
    if standard.shape != transposed.shape or not np.isfinite(standard).all() or not np.isfinite(transposed).all():
        raise ValueError('Invalid orientation predictions')
    return {'standard': standard, 'transpose': transposed, 'average': (standard + transposed) / 2}


def cache(output, stage, key, fingerprint):
    path = output / stage / 'predictions' / (key + '.npz')
    record_path = path.with_suffix('.json')
    if not record_path.exists():
        return path, None
    record = read(record_path)
    if record['manifest_sha256'] != fingerprint:
        raise ValueError('Changed cache identity')
    verify_file(path, record['prediction_sha256'])
    return path, record


def save_case(path, record, arrays, fingerprint):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(path, **arrays)
    write_json(path.with_suffix('.json'), {**record, 'manifest_sha256': fingerprint,
                                         'prediction_sha256': sha256(path)})


def parity_record(key, actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    return {'case_id': key, 'entries': int(actual.size), 'max_abs_difference': float(np.max(np.abs(actual - expected))),
            'bit_identical': bool(np.array_equal(actual, expected))}


def progress(output, stage, completed, expected, **details):
    write_json(output / stage / 'status.json', {'state': 'evaluating', 'completed': completed,
               'expected': expected, 'pid': os.getpid(), 'updated_unix': time.time(), **details})


def run_tabular(output, manifest, model):
    cases = manifest['protocols']['tabular']['cases']
    fingerprint = sha256(output / 'manifest.json')
    scorers = {suite: tabular.load_original(path, 'compute_normalized_rmse_columnwise')
               for suite, path in tabular.SCORERS.items()}
    parity_cases = {}
    for case in cases:
        parity_cases.setdefault((case['suite'], case['pattern']), case['case_id'])
    for suite in ('missbench', 'uci'):
        selected = max((case for case in cases if case['suite'] == suite), key=lambda case: np.prod(case['shape']))
        parity_cases[suite, 'largest'] = selected['case_id']
    parity = []
    for index, case in enumerate(cases):
        tabular.check_reserve(output)
        observed, truth = tabular.load_case(case)
        old, arrays = original('tabular', case, manifest)
        mask = np.isnan(observed)
        standard = arrays['prediction']
        if tuple(arrays['shape']) != observed.shape or standard.shape != (int(mask.sum()),):
            raise ValueError('Archived tabular geometry mismatch')
        if case['case_id'] in parity_cases.values():
            actual = predict_tabular(model, observed, case['suite'], False)[mask]
            parity.append(parity_record(case['case_id'], actual, standard))
            write_json(output / 'tabular/parity.json', {'checks': parity})
        path, existing = cache(output, 'tabular', case['case_id'], fingerprint)
        if existing is None:
            start = time.monotonic()
            transposed = predict_tabular(model, observed, case['suite'], True)[mask]
            predictions = combine(standard, transposed)
            metrics = {}
            for variant, values in predictions.items():
                completed = observed.astype(values.dtype)
                completed[mask] = values
                score, _ = tabular.checked_score(truth, completed, observed, case['suite'], scorers[case['suite']])
                metrics[variant] = {'nrmse': score}
            np.testing.assert_allclose(metrics['standard']['nrmse'], old['new_nrmse'], rtol=1e-7, atol=1e-7)
            record = {key: case[key] for key in ('case_id', 'suite', 'dataset', 'pattern', 'missingness', 'repeat')}
            save_case(path, {**record, 'metrics': metrics, 'transpose_seconds': time.monotonic() - start},
                      {**predictions, 'shape': np.array(observed.shape)}, fingerprint)
        progress(output, 'tabular', index + 1, len(cases), case_id=case['case_id'])
        if (index + 1) % 100 == 0:
            print(f'Tabular {index + 1}/{len(cases)}', flush=True)


def run_llm(output, manifest, model):
    cases = {case['case_id']: case for case in manifest['protocols']['llm']['cases']}
    fingerprint = sha256(output / 'manifest.json')
    parity = []
    seen = set()
    for specification, frozen, inputs in llm.frozen_settings():
        key = specification['case_id']
        if key in seen or key not in cases:
            raise ValueError('Unexpected/duplicate LLM case')
        seen.add(key)
        case = cases[key]
        llm.verify_hashes(case['reference_sha256'])
        context, visible, mask, truth = inputs
        old, arrays = original('llm', case, manifest)
        np.testing.assert_array_equal(truth, arrays['target_z'])
        query_rows, query_columns = np.where(mask)
        np.testing.assert_array_equal(query_rows, arrays['query_rows'])
        np.testing.assert_array_equal(query_columns, arrays['query_columns'])
        standard = arrays['student_z']
        if specification['fold'] == 0 and (specification['method'], specification['budget']) in (('entropy', 1), ('mi', 8), ('random', 15)):
            actual = predict_llm(model, context, visible, mask, frozen['mean'], frozen['scale'], False)
            parity.append(parity_record(key, actual, standard))
            write_json(output / 'llm/parity.json', {'checks': parity})
        path, existing = cache(output, 'llm', key, fingerprint)
        if existing is None:
            start = time.monotonic()
            transposed = predict_llm(model, context, visible, mask, frozen['mean'], frozen['scale'], True)
            predictions = combine(standard, transposed)
            metrics = {variant: llm.checked_metrics(truth, prediction) for variant, prediction in predictions.items()}
            for metric in ('r2', 'rmse'):
                np.testing.assert_allclose(metrics['standard'][metric], old['student'][metric], rtol=1e-12, atol=1e-12)
            save_case(path, {**specification, 'metrics': metrics, 'transpose_seconds': time.monotonic() - start},
                      {**predictions, 'target_z': truth, 'query_rows': query_rows, 'query_columns': query_columns}, fingerprint)
        progress(output, 'llm', len(seen), len(cases), case_id=key)
        if len(seen) % 90 == 0:
            print(f'LLM {len(seen)}/{len(cases)}', flush=True)
    if seen != set(cases):
        raise ValueError('Missing LLM cases')


def rating_queries(model, observed, users, items, normalizer, rating_range, transpose, output=None, fingerprint=None):
    oriented_shape = observed.shape[::-1] if transpose else observed.shape
    oriented_rows, oriented_columns = (items, users) if transpose else (users, items)
    row_spans = overlapping_spans(oriented_shape[0], 2000)
    column_spans = overlapping_spans(oriented_shape[1], 1000)
    windows = [(row_start, row_stop, column_start, column_stop)
               for row_start, row_stop in row_spans for column_start, column_stop in column_spans]
    sums = np.zeros(len(users), dtype=np.float64)
    counts = np.zeros(len(users), dtype=np.uint8)
    first_window = 0
    if output is not None and (output / 'progress.json').exists():
        record = read(output / 'progress.json')
        if record['manifest_sha256'] != fingerprint:
            raise ValueError('Changed rating window identity')
        verify_file(output / 'progress.npz', record['sha256'])
        with np.load(output / 'progress.npz') as saved:
            sums, counts = saved['sums'], saved['counts']
        first_window = record['next_window']
    model.max_num_rows, model.max_num_cols = 2000, 1000
    for window_index, (row_start, row_stop, column_start, column_stop) in enumerate(windows):
        if window_index < first_window:
            continue
        selected = np.flatnonzero((oriented_rows >= row_start) & (oriented_rows < row_stop) &
                                  (oriented_columns >= column_start) & (oriented_columns < column_stop))
        if selected.size:
            user_slice = slice(column_start, column_stop) if transpose else slice(row_start, row_stop)
            item_slice = slice(row_start, row_stop) if transpose else slice(column_start, column_stop)
            block = observed[user_slice, item_slice].copy()
            block = ((block - normalizer['offsets'][user_slice, None] - normalizer['means'][item_slice]) /
                     normalizer['scales'][item_slice]) / normalizer['row_scales'][user_slice, None]
            source = block.T.copy() if transpose else block
            _, raw = model.impute(source, means=np.zeros(source.shape[1]), stds=np.ones(source.shape[1]), return_full=True)
            query = raw[oriented_rows[selected] - row_start, oriented_columns[selected] - column_start]
            prediction = (query * normalizer['row_scales'][users[selected]] * normalizer['scales'][items[selected]] +
                          normalizer['means'][items[selected]] + normalizer['offsets'][users[selected]])
            if not np.isfinite(prediction).all():
                raise ValueError('Nonfinite rating prediction')
            sums[selected] += prediction
            counts[selected] += 1
        if output is not None and ((window_index + 1) % 10 == 0 or window_index + 1 == len(windows)):
            output.mkdir(parents=True, exist_ok=True)
            atomic_npz(output / 'progress.npz', sums=sums, counts=counts)
            write_json(output / 'progress.json', {'next_window': window_index + 1, 'windows': len(windows),
                       'sha256': sha256(output / 'progress.npz'), 'manifest_sha256': fingerprint})
            print(f'{output.name}: {window_index + 1}/{len(windows)} rating windows', flush=True)
    np.testing.assert_array_equal(counts, recsys.expected_coverage(oriented_shape, oriented_rows, oriented_columns))
    return np.clip(sums / counts, *rating_range), counts


def run_recsys(output, manifest, model):
    cases = manifest['protocols']['recsys']['cases']
    fingerprint = sha256(output / 'manifest.json')
    for index, case in enumerate(cases):
        tabular.check_reserve(output)
        recsys.verify_hashes(case['reference_sha256'])
        verify_file(case['normalization'], case['normalization_sha256'])
        split = recsys.load_split(case['split'])
        observed = recsys.observed_matrix(split)
        with np.load(case['normalization']) as saved:
            normalizer = dict(saved)
        old, arrays = original('recsys', case, manifest)
        users, items = split['test_users'], split['test_items']
        standard = arrays['prediction']
        np.testing.assert_array_equal(arrays['counts'], recsys.expected_coverage(observed.shape, users, items))
        if case['dataset'] == '100K':
            actual, _ = rating_queries(model, observed, users, items, normalizer, case['rating_range'], False)
            write_json(output / 'recsys/parity.json', {'checks': [parity_record('100K', actual, standard)]})
        path, existing = cache(output, 'recsys', case['dataset'], fingerprint)
        if existing is None:
            progress(output, 'recsys', index, len(cases), dataset=case['dataset'])
            start = time.monotonic()
            transposed, counts = rating_queries(model, observed, users, items, normalizer, case['rating_range'], True,
                                                output / 'recsys/windows' / case['dataset'], fingerprint)
            predictions = combine(standard, transposed)
            metrics = {variant: recsys.audit_scores(split, prediction) for variant, prediction in predictions.items()}
            for metric in ('RMSE', 'MAE'):
                np.testing.assert_allclose(metrics['standard'][metric], old['metrics'][metric], rtol=1e-12, atol=1e-12)
            save_case(path, {'dataset': case['dataset'], 'label': case['label'], 'metrics': metrics,
                            'transpose_seconds': time.monotonic() - start},
                      {**predictions, 'counts': counts, 'test_users': users, 'test_items': items,
                       'target': split['test_ratings']}, fingerprint)
            print(f"{case['label']}: {metrics}", flush=True)
        del observed, split, normalizer
        gc.collect()
        torch.cuda.empty_cache()
        progress(output, 'recsys', index + 1, len(cases))


def worker(output, stage):
    manifest = load_manifest(output)
    folder = output / stage
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        torch.manual_seed(42)
        np.random.seed(42)
        torch.backends.cuda.enable_cudnn_sdp(stage == 'llm')
        torch.cuda.set_per_process_memory_fraction(.9)
        model = tabular.make_model(CHECKPOINT)
        {'tabular': run_tabular, 'llm': run_llm, 'recsys': run_recsys}[stage](output, manifest, model)
        write_json(folder / 'status.json', {'state': 'complete', 'updated_unix': time.time(),
                                          'cases': len(manifest['protocols'][stage]['cases'])})


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(output)
        for stage in STAGES:
            write_json(output / 'status.json', {'state': 'running', 'stage': stage, 'pid': os.getpid(), 'updated_unix': time.time()})
            with (output / f'{stage}.log').open('a') as stream:
                try:
                    subprocess.run([sys.executable, '-u', str(Path(__file__)), '--stage', stage, '--output', str(output)],
                                   stdout=stream, stderr=subprocess.STDOUT, check=True)
                except subprocess.CalledProcessError:
                    write_json(output / 'status.json', {'state': 'failed', 'stage': stage, 'updated_unix': time.time()})
                    raise
        write_json(output / 'status.json', {'state': 'inference_complete', 'updated_unix': time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--stage', choices=('prepare', 'run', *STAGES), default='run')
    arguments = parser.parse_args()
    if arguments.stage == 'prepare':
        prepare(arguments.output)
    elif arguments.stage == 'run':
        run(arguments.output)
    else:
        worker(arguments.output, arguments.stage)


if __name__ == '__main__':
    main()
