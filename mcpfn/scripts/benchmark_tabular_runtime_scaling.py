"""Extend archived GPU latency curves with complete CPU fit-and-impute calls."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from benchmark_tabimpute_runtime_scaling import make_input


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / '69d7e55fd826e0325c3115d0'
DEFAULT_OUTPUT = Path('/mnt/matrixformer_scratch/runtime_baselines_20261124_v2')
METHODS = dict(knn='K-NN', softimpute='SoftImpute', missforest='MissForest',
               hyperimpute='HyperImpute', mice='MICE')
THREAD_ENV = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                   'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_sources(manifest):
    for filename, expected in manifest['source_sha256'].items():
        if digest(filename) != expected:
            raise ValueError(f'Source changed during timing: {filename}')


def validate_prediction(matrix, prediction):
    if prediction.shape != matrix.shape or not np.isfinite(prediction).all():
        raise ValueError('Require a finite prediction for the whole matrix')
    observed = np.isfinite(matrix)
    if not np.array_equal(prediction[observed], matrix[observed]):
        raise ValueError('Observed entries changed')


def timed_call(imputer, matrix, method, seed):
    sample = matrix.copy()
    started = time.perf_counter()
    prediction = imputer(method, sample, seed)
    elapsed = time.perf_counter() - started
    validate_prediction(matrix, prediction)
    return elapsed, prediction


def register(output, archive):
    if output.exists():
        raise FileExistsError('Use --resume for an existing campaign')
    archived = read(archive / 'manifest.json')
    source_run = Path(archived.get('source_run', archive))
    if 'source_run' not in archived and read(source_run / 'status.json')['state'] != 'complete':
        raise ValueError('GPU extension must finish before CPU timing starts')
    sources = [Path(__file__).resolve(), ROOT / 'scripts/benchmark_tabimpute_runtime_scaling.py',
               ROOT / 'scripts/repeated_imputation_methods.py', archive / 'manifest.json', archive / 'timings.csv']
    for folder in ('cases', 'logs', 'inputs', 'predictions', 'source_snapshot'):
        (output / folder).mkdir(parents=True, exist_ok=True)
    source_hashes = {str(path): digest(path) for path in sources}
    for path in sources:
        shutil.copyfile(path, output / 'source_snapshot' / path.name)
    input_hashes = {}
    for rows in archived['rows']:
        original = source_run / 'inputs' / f'rows_{rows}.npy'
        matrix = np.load(original, allow_pickle=False)
        np.testing.assert_array_equal(matrix, make_input(rows, archived['columns'],
                                                       archived['missing_fraction'], archived['seed']))
        for model in ('matrixformer', 'legacy'):
            record = read(source_run / 'cases' / f'{model}_{rows}.json')
            if record['input_sha256'] != digest(original) or record['status'] != 'ok':
                raise ValueError('Archived GPU call did not use this exact input')
        shutil.copyfile(original, output / 'inputs' / original.name)
        input_hashes[str(rows)] = digest(original)
    packages = ['numpy', 'scipy', 'pandas', 'scikit-learn', 'hyperimpute', 'miceforest',
                'lightgbm', 'xgboost', 'torch', 'joblib', 'threadpoolctl']
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), methods=METHODS,
                    rows=archived['rows'], columns=archived['columns'], seed=archived['seed'],
                    missing_fraction=archived['missing_fraction'], warmup=archived['warmup'],
                    repeats=archived['repeats'], archived_gpu=str(archive.resolve()),
                    source_sha256=source_hashes, input_sha256=input_hashes,
                    versions={package: importlib.metadata.version(package) for package in packages},
                    cpu=subprocess.check_output(['lscpu'], text=True), platform=platform.platform(),
                    thread_environment=THREAD_ENV, torch_threads=1, threadpool_limit=1,
                    estimator_parallelism='Unmodified evaluation defaults; random-forest plugins use half the logical CPUs; MICE uses num_threads=1.',
                    execution='One configuration at a time; CPU only; no GPU or concurrent benchmark workers.',
                    timing='Wall-clock entire adapter call: observed-value scaling, fresh estimator construction, all fitting/model selection, full-matrix imputation, inverse scaling, observed-entry restoration and NumPy output. No division by rows, columns, entries or imputations.',
                    exclusions='Input generation/copy, process/import startup, two warmups, external validation and file writes. No prefit baseline estimators are reused.',
                    mice='Same MICE-Forest adapter as accuracy evaluation: five datasets, five iterations, deterministic predictions (mean_match_candidates=0), averaged completions; all five included in each timing.',
                    softimpute='Default shrinkage cross-validation and every iterative solve included.',
                    hyperimpute='Default adaptive per-column/per-iteration model selection and fitting included.',
                    gpu_note='GPU source measurements retained unchanged; measured separately on the same H200 host with four PyTorch CPU threads. CPU/GPU ratios compare deployed implementations, not device-normalized algorithmic complexity.')
    write(output / 'manifest.json', manifest)
    return manifest


def worker(output, method, rows):
    import torch
    from threadpoolctl import threadpool_limits, threadpool_info
    from repeated_imputation_methods import impute

    manifest = read(output / 'manifest.json')
    verify_sources(manifest)
    torch.set_num_threads(1)
    path = output / 'inputs' / f'rows_{rows}.npy'
    if digest(path) != manifest['input_sha256'][str(rows)]:
        raise ValueError('Input changed')
    matrix = np.load(path, allow_pickle=False)
    record_path = output / 'cases' / f'{method}_{rows}.json'
    record = read(record_path) if record_path.exists() else dict(
        model=METHODS[method], model_key=method, rows=rows, columns=manifest['columns'],
        input_sha256=digest(path), seconds=[], warmup_seconds=[], status='running')
    if record.get('status') == 'ok':
        return
    if record['warmup_seconds']:
        record.setdefault('previous_warmup_seconds', []).append(record['warmup_seconds'])
        record['warmup_seconds'] = []
    try:
        for repeat in range(manifest['warmup'] + manifest['repeats']):
            phase = 'warmup_seconds' if repeat < manifest['warmup'] else 'seconds'
            index = repeat if phase == 'warmup_seconds' else repeat - manifest['warmup']
            if index < len(record[phase]):
                continue
            with threadpool_limits(limits=1):
                elapsed, prediction = timed_call(impute, matrix, method, manifest['seed'])
                pools = threadpool_info()
                if phase == 'seconds' and any(pool['num_threads'] != 1 for pool in pools):
                    raise ValueError('A native thread pool escaped the timing limit')
                record['threadpools'] = pools
            record[phase].append(elapsed)
            if phase == 'seconds':
                prediction_path = output / 'predictions' / f'{method}_{rows}_{index + 1}.npy'
                np.save(prediction_path, prediction, allow_pickle=False)
            write(record_path, record)
            print(f'{method} rows={rows} {phase} call={index + 1}: {elapsed:.6f}s', flush=True)
        record.update(status='ok', updated_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        record.update(status='error', error=repr(error))
        write(record_path, record)
        raise
    write(record_path, record)


def collect(output):
    manifest = read(output / 'manifest.json')
    records = []
    for method in METHODS:
        for rows in manifest['rows']:
            path = output / 'cases' / f'{method}_{rows}.json'
            if not path.exists():
                raise ValueError(f'Missing timings: {method}/{rows}')
            case = read(path)
            seconds = case['seconds']
            if case['status'] != 'ok' or len(seconds) != manifest['repeats']:
                raise ValueError(f'Incomplete timings: {method}/{rows}')
            if not np.isfinite(seconds).all() or min(seconds) <= 0:
                raise ValueError('Invalid timing')
            records.extend(dict(model=METHODS[method], model_key=method, rows=rows,
                                columns=manifest['columns'], repeat=index + 1, seconds=elapsed)
                           for index, elapsed in enumerate(seconds))
    frame = pd.DataFrame(records)
    frame.to_csv(output / 'timings.csv', index=False)
    return frame


def run(output, archive, resume=False):
    manifest = read(output / 'manifest.json') if resume else register(output, archive)
    verify_sources(manifest)
    environment = dict(os.environ, **THREAD_ENV, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    completed = 0
    try:
        for method in METHODS:
            for rows in manifest['rows']:
                write(output / 'status.json', dict(state='running', method=method, rows=rows,
                                                  completed=completed, expected=len(METHODS) * len(manifest['rows'])))
                with (output / 'logs' / f'{method}_{rows}.log').open('a') as stream:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--output', str(output),
                                    '--worker-method', method, '--rows', str(rows)], env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=7200)
                completed += 1
                print(f'Completed {completed}: {method}, {rows} rows', flush=True)
        frame = collect(output)
        verify_sources(manifest)
        write(output / 'status.json', dict(state='complete', configurations=completed, timed_calls=len(frame)))
    except Exception as error:
        write(output / 'status.json', dict(state='failed', completed=completed, error=repr(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--archive', type=Path, default=PAPER / 'results/runtime_scaling')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker-method', choices=METHODS)
    parser.add_argument('--rows', type=int)
    arguments = parser.parse_args()
    if arguments.worker_method:
        worker(arguments.output.resolve(), arguments.worker_method, arguments.rows)
    else:
        run(arguments.output.resolve(), arguments.archive.resolve(), arguments.resume)


if __name__ == '__main__':
    main()
