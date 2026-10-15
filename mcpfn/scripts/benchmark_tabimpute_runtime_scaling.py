"""Measure fixed-width, full-matrix imputation latency with synchronized CUDA calls."""

import argparse
import csv
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / 'runs/streamed_training/large_matrices_98k_20260930/best_validation_checkpoint.pth'
DEFAULT_OUTPUT = ROOT / 'runs/benchmark_evaluations/runtime_scaling_legacy_matrixformer_20261001'
MODELS = {'legacy': 'TabImpute (legacy)', 'matrixformer': 'MatrixFormer'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def make_input(rows, columns, missing_fraction, seed):
    generator = np.random.default_rng(np.random.SeedSequence([seed, rows, columns]))
    rank = min(4, rows, columns)
    matrix = generator.standard_normal((rows, rank)) @ generator.standard_normal((rank, columns)) / np.sqrt(rank)
    matrix += 0.05 * generator.standard_normal((rows, columns))
    selected = generator.choice(matrix.size, round(matrix.size * missing_fraction), replace=False)
    matrix.flat[selected] = np.nan
    if np.isnan(matrix).all(axis=0).any():
        raise ValueError('Generated input contains an unobserved column')
    return matrix


def live_process(identity):
    try:
        process = psutil.Process(identity['pid'])
        return process if process.create_time() == identity['created'] else None
    except psutil.NoSuchProcess:
        return None


def resume_processes(identities):
    for identity in reversed(identities):
        process = live_process(identity)
        if process is not None:
            try:
                process.resume()
            except psutil.NoSuchProcess:
                pass


def gpu_pids():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return {int(value) for value in text.split()}


def watchdog(output):
    path = output / 'gpu_guard.json'
    while True:
        guard = json.loads(path.read_text())
        if guard['state'] != 'active':
            return
        if live_process(guard['owner']) is None or time.time() > guard['deadline']:
            owner = live_process(guard['owner'])
            if owner is not None:
                for child in owner.children(recursive=True):
                    if child.pid != os.getpid():
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                owner.terminate()
            resume_processes(guard['suspended'])
            guard.update(state='watchdog_resumed', released_unix=time.time())
            write_json(output / 'watchdog_action.json', guard)
            return
        time.sleep(2)


@contextmanager
def exclusive_compute(output, pause_pid):
    parent = None
    identities = []
    if pause_pid is not None:
        parent = psutil.Process(pause_pid)
        if not any(value.endswith('/compare_recsys_column_context.py') for value in parent.cmdline()) or '--mode' in parent.cmdline():
            raise ValueError('Only the identified recommender campaign supervisor may be suspended')
        processes = [parent, *parent.children(recursive=True)]
        if any(process.status() == psutil.STATUS_STOPPED for process in processes):
            raise ValueError('A process is already suspended; do not take ownership of its pause')
        identities = [dict(pid=process.pid, created=process.create_time()) for process in processes]
    guard = dict(state='active', owner=dict(pid=os.getpid(), created=psutil.Process().create_time()),
                 suspended=identities, deadline=time.time() + 900, started_unix=time.time())
    write_json(output / 'gpu_guard.json', guard)
    with (output / 'watchdog.log').open('a') as stream:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--output', str(output), '--watchdog'],
                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        if parent is not None:
            parent.suspend()
            descendants = parent.children(recursive=True)
            identities = [identities[0], *[dict(pid=process.pid, created=process.create_time()) for process in descendants]]
            guard['suspended'] = identities
            write_json(output / 'gpu_guard.json', guard)
            for process in descendants:
                process.suspend()
        allowed = {identity['pid'] for identity in identities}
        if gpu_pids() - allowed:
            raise RuntimeError('Unrelated GPU processes are active; refusing contended timing')
        idle_samples = 0
        for _ in range(60):
            utilization = int(subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip())
            idle_samples = idle_samples + 1 if utilization == 0 else 0
            if idle_samples == 3:
                break
            time.sleep(1)
        else:
            raise RuntimeError('GPU did not become idle after suspension')
        yield
    finally:
        resume_processes(identities)
        guard.update(state='released', released_unix=time.time())
        write_json(output / 'gpu_guard.json', guard)


def check_exclusive(output):
    guard = json.loads((output / 'gpu_guard.json').read_text())
    allowed = {os.getpid()}
    for identity in guard['suspended']:
        process = live_process(identity)
        if process is not None:
            if process.status() not in (psutil.STATUS_STOPPED, psutil.STATUS_ZOMBIE):
                raise RuntimeError('Suspended workload resumed during timing')
            allowed.add(process.pid)
    if gpu_pids() - allowed:
        raise RuntimeError('Concurrent GPU computation detected')


def worker(arguments):
    import torch
    from tabimpute.interface import ImputePFN

    output = arguments.output
    manifest = json.loads((output / 'manifest.json').read_text())
    model = arguments.worker_model
    spec = manifest['models'][model]
    matrix = np.load(output / 'inputs' / f'rows_{arguments.rows}.npy', allow_pickle=False)
    torch.set_num_threads(4)
    torch.manual_seed(manifest['seed'])
    np.random.seed(manifest['seed'])
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(0.25)
    record = dict(model=MODELS[model], model_key=model, rows=arguments.rows, columns=manifest['columns'],
                  missing_cells=int(np.isnan(matrix).sum()), input_sha256=digest(output / 'inputs' / f'rows_{arguments.rows}.npy'),
                  checkpoint_sha256=spec['checkpoint_sha256'], precision=spec['precision'],
                  prediction_stat=spec['prediction_stat'], status='running', seconds=[])
    target = output / 'cases' / f'{model}_{arguments.rows}.json'
    try:
        imputer = ImputePFN(device='cuda', checkpoint_path=spec['checkpoint'], entry_wise_features=model == 'legacy',
                            json_config=spec.get('config'), prediction_stat=spec['prediction_stat'],
                            precision_mode=spec['precision'], max_num_rows=None, max_num_cols=None,
                            max_num_chunks=1, preprocessors=None)
        imputer.model.eval()
        phase = 'warmup'
        for iteration in range(manifest['warmup'] + manifest['repeats']):
            check_exclusive(output)
            sample = matrix.copy()
            if iteration == manifest['warmup']:
                phase = 'timed'
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            prediction = imputer.impute(sample)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if prediction.shape != matrix.shape or not np.isfinite(prediction).all():
                raise ValueError('Invalid imputation')
            observed = np.isfinite(matrix)
            np.testing.assert_allclose(prediction[observed], matrix[observed], rtol=1e-6, atol=1e-6)
            if phase == 'timed':
                record['seconds'].append(elapsed)
            del prediction
            write_json(target, record)
        check_exclusive(output)
        record.update(status='ok', median_seconds=float(np.median(record['seconds'])),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    except torch.OutOfMemoryError as error:
        record.update(status='out_of_memory', error=str(error))
    except Exception as error:
        record.update(status='error', error=repr(error))
    write_json(target, record)
    print(json.dumps(record), flush=True)


def summarize(output):
    records = [json.loads(path.read_text()) for path in sorted((output / 'cases').glob('*.json'))]
    rows = [dict(model=record['model'], model_key=record['model_key'], rows=record['rows'], columns=record['columns'],
                 repeat=repeat, seconds=seconds) for record in records if record['status'] == 'ok'
            for repeat, seconds in enumerate(record['seconds'], start=1)]
    if rows:
        with (output / 'timings.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    return records


def run(arguments):
    from huggingface_hub import hf_hub_download

    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('inputs', 'cases', 'logs'):
        (output / folder).mkdir()
    legacy = Path(hf_hub_download('Tabimpute/TabImpute', 'tabimpute_001.ckpt', local_files_only=True))
    config = json.loads((arguments.checkpoint.parent / 'config.json').read_text())
    sources = [Path(__file__).resolve(), *sorted((ROOT / 'src/tabimpute').rglob('*.py')),
               ROOT / 'src/tabimpute/data/encoder.pth', ROOT / 'src/tabimpute/data/borders.pt',
               arguments.checkpoint.parent / 'config.json']
    manifest = dict(rows=arguments.row_sizes, columns=arguments.columns, missing_fraction=0.4,
                    seed=42, warmup=2, repeats=5, rank=4, noise_std=0.05,
                    timing='Synchronized end-to-end impute(); normalization, transfers, inference and NumPy output included; model loading, input copy, and warmup excluded',
                    context='Full matrices; no row/column chunking, no preprocessing/permutation ensemble, no fitting',
                    runtime_scope='Single-matrix synthetic workload; not timings of the archived benchmark ensemble/chunking configurations',
                    worker_memory_fraction=0.25, suspension='Recommender computation paused during timing; its GPU allocations remain resident; watchdog resumes it within 15 minutes',
                    models={'legacy': dict(checkpoint=str(legacy), checkpoint_sha256=digest(legacy),
                                          precision='full_bfloat16', prediction_stat='median'),
                            'matrixformer': dict(checkpoint=str(arguments.checkpoint), checkpoint_sha256=digest(arguments.checkpoint),
                                                 config={'config': config['model']}, precision='amp_bfloat16', prediction_stat='mean')},
                    source_sha256={str(path): digest(path) for path in sources},
                    nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True))
    write_json(output / 'manifest.json', manifest)
    for path in sources:
        snapshot = output / 'source_snapshot' / path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    for rows in arguments.row_sizes:
        np.save(output / 'inputs' / f'rows_{rows}.npy', make_input(rows, arguments.columns, 0.4, 42), allow_pickle=False)
    def interrupted(signum, frame):
        raise SystemExit(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with exclusive_compute(output, arguments.pause_pid):
            for index, rows in enumerate(arguments.row_sizes):
                order = ('matrixformer', 'legacy') if index % 2 == 0 else ('legacy', 'matrixformer')
                for model in order:
                    write_json(output / 'status.json', dict(state='running', rows=rows, model=MODELS[model]))
                    with (output / 'logs' / f'{model}_{rows}.log').open('w') as stream:
                        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--output', str(output),
                                        '--worker-model', model, '--rows', str(rows)], stdout=stream,
                                       stderr=subprocess.STDOUT, check=True, timeout=300)
                    records = summarize(output)
                    current = next(record for record in records if record['model_key'] == model and record['rows'] == rows)
                    print(f"{rows} x {arguments.columns}: {MODELS[model]} {current['status']} {current.get('median_seconds', '')}", flush=True)
                    if current['status'] != 'ok':
                        raise RuntimeError(f'Incomplete measurement: {model}/{rows}; see saved case diagnostics')
        for path, expected in manifest['source_sha256'].items():
            assert digest(path) == expected, path
        for spec in manifest['models'].values():
            assert digest(spec['checkpoint']) == spec['checkpoint_sha256']
        write_json(output / 'status.json', dict(state='complete', configurations=len(records), timed_calls=len(records) * manifest['repeats']))
    except BaseException as error:
        write_json(output / 'status.json', dict(state='failed', error=repr(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--columns', type=int, default=10)
    parser.add_argument('--row-sizes', nargs='+', type=int, default=[16, 32, 64, 128, 256, 512, 1024, 2048])
    parser.add_argument('--pause-pid', type=int)
    parser.add_argument('--worker-model', choices=tuple(MODELS))
    parser.add_argument('--rows', type=int)
    parser.add_argument('--watchdog', action='store_true')
    arguments = parser.parse_args()
    if arguments.watchdog:
        watchdog(arguments.output)
    elif arguments.worker_model:
        worker(arguments)
    else:
        run(arguments)


if __name__ == '__main__':
    main()
