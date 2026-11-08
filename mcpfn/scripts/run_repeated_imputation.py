"""Register, prepare, and resume a resource-bounded five-seed imputation study."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from recsys_evaluation_common import atomic_npz, sha256, write_json
from repeated_imputation_masks import VERSION, PATTERNS, DETERMINISTIC, NEURAL_TILE, seed_for, family_state, mask_at_rate
from repeated_imputation_methods import CPU_METHODS, GPU_METHODS, R_METHODS, MODEL_PATHS, LABELS, TABIMPUTE
from repeated_imputation_statistics import export


REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/mnt/matrixformer_scratch/repeated_imputation_20261108')
TRAINING = Path('/mnt/matrixformer_scratch/scientific_prior_training_20261108/million_v1')
DEFAULT_OUTPUT = ROOT / 'five_seed_v1'


def read(path):
    return json.loads(Path(path).read_text())


def verify_hashes(hashes):
    for filename, expected in hashes.items():
        if sha256(Path(filename)) != expected:
            raise ValueError(f'Frozen input/source changed: {filename}')


def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def stamp(path, **values):
    write_json(path, dict(pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat(), **values))


def affine_equivalent(first, second):
    from repeated_imputation_masks import standardize

    return first.shape == second.shape and np.allclose(standardize(first), standardize(second), rtol=2e-5, atol=2e-5)


def register(output, methods=None, datasets=None):
    if output.exists():
        raise FileExistsError('Use a new output directory; registered protocols are immutable')
    source = TRAINING / 'monitor_protocol/protocol.json'
    historical = read(source)['stages']['tabular']
    methods = list(methods or (*CPU_METHODS, *GPU_METHODS, *R_METHODS))
    if len(set(methods)) != len(methods) or not set(methods).issubset(LABELS):
        raise ValueError('Unknown or duplicate methods')
    sources = dict(read(TRAINING / 'config.json')['source_sha256'])
    local = [*Path(__file__).parent.glob('repeated_imputation_*.py'), Path(__file__),
             REPO / 'scripts/grzesiak_bridge.R', REPO / 'scripts/grzesiak_benchmark.py',
             REPO / 'benchmark/uci/download_uci_datasets.py', REPO / 'benchmark/openml/get_openml_errors.py',
             REPO / 'src/tabimpute/prepreocess.py',
             REPO / 'benchmark/cacti_wrapper.py', REPO / 'benchmark/remasker_wrapper.py',
             REPO / 'benchmark/diffputer_wrapper.py']
    for folder in ('benchmark/CACTI', 'benchmark/DiffPuter', 'benchmark/remasker'):
        local.extend((REPO / folder).rglob('*.py'))
    local.extend((REPO / 'runs/benchmark_evaluations/grzesiak_20260925/upstream/R').glob('*.R'))
    for path in local:
        sources[str(path.resolve())] = sha256(path)
    verify_hashes(sources)
    output.mkdir(parents=True)
    for filename in sources:
        destination = output / 'sources' / Path(filename).relative_to('/root/tabular')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(filename, destination)
    catalog = {}
    historical = sorted(historical, key=lambda case: (case['suite'], case['dataset'], case['pattern'] != 'MCAR', case['case_id']))
    for case in historical:
        key = case['suite'], case['dataset']
        if datasets and case['dataset'] not in datasets:
            continue
        if key in catalog:
            continue
        original = Path(case['source_directory']) / 'true.npy'
        truth = np.load(original, allow_pickle=False)
        if not np.isfinite(truth).all():
            raise ValueError(f'Incomplete truth: {key}')
        name = 'statlog_landsat_satellite' if key == ('uci', 'blood_transfusion') else case['dataset']
        destination = output / 'data' / case['suite'] / name / 'truth.npy'
        destination.parent.mkdir(parents=True)
        shutil.copyfile(original, destination)
        catalog[key] = dict(suite=case['suite'], dataset=name, historical_name=case['dataset'],
                            original_path=str(original), original_sha256=sha256(original),
                            truth_path=str(destination), truth_sha256=sha256(destination), shape=list(truth.shape))
    if not catalog:
        raise ValueError('No datasets selected')
    if datasets is None and {suite: sum(key[0] == suite for key in catalog) for suite in ('missbench', 'uci')} != dict(missbench=42, uci=12):
        raise ValueError('Unexpected dataset inventory')
    coordinate_audit = []
    for case in historical:
        key = case['suite'], case['dataset']
        if key in catalog:
            truth = np.load(Path(case['source_directory']) / 'true.npy', allow_pickle=False)
            reference = np.load(catalog[key]['truth_path'], allow_pickle=False)
            if not affine_equivalent(truth, reference):
                raise ValueError(f'Truth differs beyond positive column-affine scaling: {key}, {case["case_id"]}')
            coordinate_audit.append(dict(case_id=case['case_id'], exact_match=bool(np.array_equal(truth, reference)),
                                         affine_equivalent=True))
    aliases = {}
    unique = []
    for dataset in catalog.values():
        truth = np.load(dataset['truth_path'], allow_pickle=False)
        matches = [other for other in unique if other['suite'] == dataset['suite']
                   and affine_equivalent(truth, np.load(other['truth_path'], allow_pickle=False))]
        if matches:
            aliases[f"{dataset['suite']}/{dataset['dataset']}"] = f"{matches[0]['suite']}/{matches[0]['dataset']}"
        else:
            unique.append(dataset)
    cases = []
    for dataset in unique:
        for pattern, (label, family) in PATTERNS.items():
            repeats = range(1) if pattern in DETERMINISTIC else range(5)
            for repeat in repeats:
                seed = seed_for(VERSION, dataset['suite'], dataset['dataset'], pattern, repeat)
                for rate in ((.4,) if dataset['suite'] == 'missbench' else (.1, .2, .3, .4, .5)):
                    identifier = f"{dataset['suite']}/{dataset['dataset']}/{pattern}/rate_{rate:.1f}/seed_{repeat}"
                    cases.append(dict(case_id=identifier, suite=dataset['suite'], dataset=dataset['dataset'],
                                      pattern=pattern, pattern_label=label, family=family, repeat=repeat,
                                      missingness=rate, mask_seed=seed, deterministic=pattern in DETERMINISTIC,
                                      truth_path=dataset['truth_path'], truth_sha256=dataset['truth_sha256']))
    models = {}
    for name, path in MODEL_PATHS.items():
        if name in methods:
            config = path.parent / 'config.json'
            models[name] = dict(checkpoint=str(path), checkpoint_sha256=sha256(path),
                                config=str(config), config_sha256=sha256(config))
    if 'tabimpute' in methods:
        models['tabimpute'] = dict(checkpoint=str(TABIMPUTE), checkpoint_sha256=sha256(TABIMPUTE),
                                  recipe='Paper masters_mcar checkpoint; nhead=2, no EMA, median/full-BF16; two row-column, one row, one column permutations')
    package_names = ('numpy', 'scipy', 'torch', 'scikit-learn', 'hyperimpute', 'miceforest', 'ForestDiffusion')
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = dict(version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
                    historical_protocol=str(source), historical_protocol_sha256=sha256(source),
                    datasets=list(catalog.values()), dataset_aliases=aliases, cases=cases, methods=methods, models=models,
                    source_sha256=sources, packages=packages, inference_seed=42, mask_repeats=5,
                    deterministic_repeats=1, gpu_memory_gib=8., timeout_seconds=1800,
                    algorithm_randomness='Fixed seed 42 separately from mask seeds; no algorithm-seed repeats',
                    input_preprocessing='Observed-only column z-score for every method; scale 1 for constants, center 0 for empty columns',
                    scoring='Historical columnwise held-out-truth SD NRMSE primary; observed-SD NRMSE diagnostic',
                    context='Native orientation, predictive mean, 2000x1000 overlapping tail windows for MatrixFormer; no train/test row split',
                    masks='New rate-controlled protocol, not a reproduction of the historical single-mask numbers',
                    neural_mask_tiles=list(NEURAL_TILE),
                    neural_mask_definition='Original independent random per-cell MLPs within bounded local tiles; fixed architecture distribution and per-tile seeds; standardized complete truth used only for mask generation; clipped logit recalibration',
                    block_mask_definition='Balanced contiguous approximately 3x3 blocks; pooled values feed tiled original MLPs; one shared Bernoulli draw per block; area-weighted rate calibration',
                    conditional_mask_definition='Col/Phase-MAR protect floor(0.4*d), minimum one columns; Self-MNAR targets all columns; beta=2; calibration targets global expected missing fraction',
                    threshold_mask_definition='Pol central quantile interval is deterministic; Censor chooses a tail per column/seed; strict inequalities preserve ties, realized rates may differ',
                    uncertainty='Average seeds, then rates, then patterns within each dataset; 95% t intervals across dataset means; two-sided paired Wilcoxon overall with Holm adjustment within suite',
                    coverage='Only complete dataset-panels summarized; incomplete suite summaries provisional; no silent failure dropping',
                    training='Report-only development evaluation; never alter million-run sources, gradients, or checkpoint selection',
                    dataset_correction=dict(historical_name='blood_transfusion', corrected_name='statlog_landsat_satellite',
                                            uci_id=146, official_url='https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite',
                                            reason='Historical downloader requests ID 146; actual Blood Transfusion is ID 176. Same 6435x36 matrix retained, not replaced.'))
    manifest['truth_coordinates'] = 'Freeze the first historical MCAR truth matrix per dataset; historical MissBench copies use different column-affine scales. All copies verified equivalent up to positive column-affine scaling; no historic masks reused. Fresh observed-only preprocessing is fitted after each new mask.'
    manifest['dataset_independence'] = 'Positive column-affine duplicate matrices with identical row/column order are aliases, not independent datasets; evaluate the first name once.'
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'historical_coordinate_audit.json', coordinate_audit)
    stamp(output / 'status.json', state='registered', cases=len(cases), methods=len(methods))
    return manifest


def load_manifest(output):
    manifest = read(output / 'manifest.json')
    verify_hashes(manifest['source_sha256'])
    verify_hashes({item['truth_path']: item['truth_sha256'] for item in manifest['datasets']})
    return manifest


def case_paths(output, case):
    return output / 'masks' / (case['case_id'] + '.npz'), output / 'masks' / (case['case_id'] + '.json')


def prepare(output, manifest):
    handle = lock(output / 'prepare.lock')
    import torch

    torch.set_num_threads(1)
    groups = defaultdict(list)
    for case in manifest['cases']:
        groups[case['suite'], case['dataset'], case['pattern'], case['repeat']].append(case)
    completed = 0
    for key, cases in groups.items():
        if all(case_paths(output, case)[1].exists() for case in cases):
            for case in cases:
                mask_path, record_path = case_paths(output, case)
                if sha256(mask_path) != read(record_path)['mask_sha256']:
                    raise ValueError(f'Mask changed: {case["case_id"]}')
            completed += len(cases)
            continue
        stamp(output / 'preparation_status.json', state='generating', completed=completed,
              expected=len(manifest['cases']), current=list(key))
        truth = np.load(cases[0]['truth_path'], allow_pickle=False)
        started = time.monotonic()
        state = family_state(truth, cases[0]['pattern'], cases[0]['mask_seed'])
        for case in cases:
            mask, diagnostics = mask_at_rate(state, case['missingness'])
            if not mask.any() or mask.all():
                raise ValueError(f'Degenerate mask, no retries permitted: {case["case_id"]}')
            mask_path, record_path = case_paths(output, case)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if record_path.exists():
                if not np.array_equal(load_mask(mask_path), mask):
                    raise ValueError('Regenerated mask differs from cached mask')
            else:
                atomic_npz(mask_path, packed=np.packbits(mask.reshape(-1)), shape=np.asarray(mask.shape))
                write_json(record_path, dict(case | diagnostics, mask_sha256=sha256(mask_path),
                                             mask_identity=hashlib.sha256(mask.tobytes()).hexdigest()))
            completed += 1
        print(f'prepared {completed}/{len(manifest["cases"])} {key} {time.monotonic() - started:.1f}s', flush=True)
    audit_masks(output, manifest)
    stamp(output / 'preparation_status.json', state='complete', completed=completed, expected=len(manifest['cases']))
    handle.close()


def load_mask(path):
    with np.load(path, allow_pickle=False) as stored:
        shape = tuple(stored['shape'])
        return np.unpackbits(stored['packed'], count=int(np.prod(shape))).reshape(shape).astype(bool)


def audit_masks(output, manifest):
    import pandas as pd

    records = []
    for case in manifest['cases']:
        mask_path, record_path = case_paths(output, case)
        record = read(record_path)
        if any(record[key] != value for key, value in case.items()) or sha256(mask_path) != record['mask_sha256']:
            raise ValueError('Mask registration mismatch')
        mask = load_mask(mask_path)
        if float(mask.mean()) != record['realized_rate'] or hashlib.sha256(mask.tobytes()).hexdigest() != record['mask_identity']:
            raise ValueError('Mask diagnostics mismatch')
        records.append(record)
    frame = pd.DataFrame(records)
    frame.to_csv(output / 'mask_audit.csv', index=False)
    unique = frame.groupby(['suite', 'dataset', 'pattern', 'missingness']).agg(
        seeds=('repeat', 'count'), unique_masks=('mask_identity', 'nunique'),
        realized_mean=('realized_rate', 'mean'), realized_min=('realized_rate', 'min'), realized_max=('realized_rate', 'max'))
    unique.to_csv(output / 'mask_uniqueness.csv')
    write_json(output / 'mask_audit.json', dict(passed=True, cases=len(records),
               distinct_cases=int(unique.unique_masks.sum()), independent_unit='dataset',
               repeated_identical_masks=int((unique.seeds - unique.unique_masks).sum())))


def setup_worker(output, method, manifest):
    import torch
    from sklearn import set_config
    from repeated_imputation_methods import build_model

    torch.set_num_threads(1)
    set_config(working_memory=256)
    if method in GPU_METHODS:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(manifest['gpu_memory_gib'] * 1024**3 / total)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.mha.set_fastpath_enabled(False)
    if method in manifest['models']:
        specification = manifest['models'][method]
        hashes = {specification['checkpoint']: specification['checkpoint_sha256']}
        if 'config' in specification:
            hashes[specification['config']] = specification['config_sha256']
        verify_hashes(hashes)
        return build_model(method, specification)
    return None


def worker(output, method, limit=None):
    from repeated_imputation_methods import impute, metrics
    from threadpoolctl import threadpool_limits

    manifest = load_manifest(output)
    if method not in manifest['methods']:
        raise ValueError('Unregistered method')
    destination = output / 'results' / method
    destination.mkdir(parents=True, exist_ok=True)
    handle = lock(destination / 'worker.lock')
    status = destination / 'status.json'
    stamp(status, state='initializing', current=None, started_unix=time.time())
    model = setup_worker(output, method, manifest)
    cases = [case for case in manifest['cases'] if method != 'tabimpute' or case['suite'] == 'missbench']
    if limit:
        cases = cases[:limit]
    completed = 0
    for case in cases:
        target = destination / 'cases' / (case['case_id'] + '.json')
        prediction_path = destination / 'predictions' / (case['case_id'] + '.npz')
        mask_path, mask_record_path = case_paths(output, case)
        if not mask_record_path.exists():
            raise FileNotFoundError('Prepare all masks before evaluating methods')
        mask_record = read(mask_record_path)
        if sha256(mask_path) != mask_record['mask_sha256']:
            raise ValueError('Mask changed')
        if target.exists():
            saved = read(target)
            if saved['manifest_sha256'] != sha256(output / 'manifest.json') or saved['mask_sha256'] != mask_record['mask_sha256']:
                raise ValueError('Cached result protocol mismatch')
            if saved['status'] == 'ok' and sha256(prediction_path) != saved['prediction_sha256']:
                raise ValueError('Cached prediction changed')
            completed += 1
            continue
        if shutil.disk_usage(output).free < 20 * 1024**3:
            raise RuntimeError('20 GiB scratch reserve reached')
        if method in GPU_METHODS:
            while True:
                free_mib = float(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                  '--format=csv,noheader,nounits'], text=True).splitlines()[0])
                if free_mib >= 10 * 1024:
                    break
                stamp(status, state='waiting_gpu_memory', current=case['case_id'], free_mib=free_mib,
                      completed=completed, expected=len(cases))
                time.sleep(30)
        stamp(status, state='running', current=case['case_id'], completed=completed,
              expected=len(cases), started_unix=time.time())
        record = dict(**case, method=method, mask_sha256=mask_record['mask_sha256'],
                      manifest_sha256=sha256(output / 'manifest.json'))
        started = time.monotonic()
        try:
            truth = np.load(case['truth_path'], allow_pickle=False).astype(np.float64)
            mask = load_mask(mask_path)
            missing = truth.copy()
            missing[mask] = np.nan
            with threadpool_limits(limits=1):
                prediction = impute(method, missing, manifest['inference_seed'], model, manifest['timeout_seconds'])
            record.update(metrics(truth, missing, prediction, case['suite']))
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_npz(prediction_path, missing_values=prediction[mask], shape=np.asarray(mask.shape))
            record.update(status='ok', prediction_sha256=sha256(prediction_path))
        except (ImportError, ModuleNotFoundError):
            stamp(status, state='blocked_dependency', error=traceback.format_exc(), current=case['case_id'])
            raise
        except Exception as error:
            record.update(status='error', error=str(error), traceback=traceback.format_exc())
        record['seconds'] = time.monotonic() - started
        write_json(target, record)
        completed += 1
        print(f'{method} {completed}/{len(cases)} {case["case_id"]} {record["status"]} {record["seconds"]:.2f}s', flush=True)
        if method in GPU_METHODS:
            import torch

            torch.cuda.empty_cache()
    stamp(status, state='finished', completed=completed, expected=len(cases))
    handle.close()


def timeout_record(output, method, manifest, status):
    case = next(case for case in manifest['cases'] if case['case_id'] == status['current'])
    target = output / 'results' / method / 'cases' / (case['case_id'] + '.json')
    if target.exists():
        return
    mask_record = read(case_paths(output, case)[1])
    write_json(target, dict(**case, method=method, status='timeout', nrmse=None,
                           error=f"Exceeded {manifest['timeout_seconds']} second per-case budget",
                           mask_sha256=mask_record['mask_sha256'], manifest_sha256=sha256(output / 'manifest.json')))


def run_method(output, method, manifest):
    log_path = output / 'logs' / f'{method}.log'
    log_path.parent.mkdir(exist_ok=True)
    temporary = output / 'temporary'
    temporary.mkdir(exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       NUMEXPR_NUM_THREADS='1', LOKY_MAX_CPU_COUNT='1', PYTHONDONTWRITEBYTECODE='1',
                       RCPP_PARALLEL_NUM_THREADS='1', TMPDIR=str(temporary),
                       PYTHONPATH=f'{REPO / "src"}:{REPO / "scripts"}')
    if method not in GPU_METHODS:
        environment['CUDA_VISIBLE_DEVICES'] = ''
    while True:
        with log_path.open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker',
                                        '--output', str(output), '--method', method],
                                       stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        timed_out = False
        try:
            while process.poll() is None:
                time.sleep(5)
                status_path = output / 'results' / method / 'status.json'
                if status_path.exists():
                    status = read(status_path)
                    if status.get('pid') == process.pid and status['state'] == 'running' and time.time() - status['started_unix'] > manifest['timeout_seconds'] + 30:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        timeout_record(output, method, manifest, status)
                        timed_out = True
                        break
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
        if not timed_out:
            if process.returncode != 0:
                stamp(output / 'results' / method / 'supervisor_status.json', state='worker_failed',
                      returncode=process.returncode, log=str(log_path))
            return


def pipeline(output):
    handle = lock(output / 'pipeline.lock')
    manifest = load_manifest(output)
    stamp(output / 'status.json', state='preparing', cases=len(manifest['cases']), methods=len(manifest['methods']))
    prepare(output, manifest)
    stamp(output / 'status.json', state='evaluating', cases=len(manifest['cases']), methods=len(manifest['methods']))

    def lane(methods):
        for method in methods:
            run_method(output, method, manifest)

    cpu = [method for method in manifest['methods'] if method not in GPU_METHODS]
    gpu = [method for method in manifest['methods'] if method in GPU_METHODS]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(lane, methods) for methods in (cpu, gpu)]
        while not all(future.done() for future in futures):
            export(output, manifest)
            time.sleep(60)
        for future in futures:
            future.result()
    export(output, manifest)
    import pandas as pd

    coverage = pd.read_csv(output / 'summary/coverage.csv')
    complete = bool(coverage.complete.all())
    stamp(output / 'status.json', state='complete' if complete else 'finished_with_gaps',
          successful=int(coverage.successful.sum()), failed=int(coverage.failed.sum()), pending=int(coverage.pending.sum()))
    handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('register', 'prepare', 'run', 'worker', 'summarize', 'audit'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--method', choices=tuple(LABELS))
    parser.add_argument('--methods', nargs='+', choices=tuple(LABELS))
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--limit', type=int)
    arguments = parser.parse_args()
    if hasattr(os, 'nice'):
        os.nice(10)
    output = arguments.output.resolve()
    if arguments.command == 'register':
        register(output, arguments.methods, arguments.datasets)
    elif arguments.command == 'worker':
        worker(output, arguments.method, arguments.limit)
    elif arguments.command == 'run':
        pipeline(output)
    else:
        manifest = load_manifest(output)
        if arguments.command == 'prepare':
            prepare(output, manifest)
        elif arguments.command == 'audit':
            audit_masks(output, manifest)
        else:
            export(output, manifest)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        destination = Path(sys.argv[sys.argv.index('--output') + 1]) if '--output' in sys.argv else DEFAULT_OUTPUT
        if destination.exists() and 'worker' not in sys.argv and 'register' not in sys.argv:
            stamp(destination / 'fatal_error.json', state='failed', traceback=traceback.format_exc())
            stamp(destination / 'status.json', state='failed', error_file='fatal_error.json')
        raise
