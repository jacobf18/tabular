"""Shard frozen CPU evaluations without modifying their protocol or adapters."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback

import numpy as np


CPU_METHODS = {'forestdiffusion', 'mice', 'softimpute', 'hyperimpute', 'missforest'}
THREAD_ENV = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                   'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                                   'LOKY_MAX_CPU_COUNT', 'RCPP_PARALLEL_NUM_THREADS')}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stamp(path, **values):
    write(path, dict(pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat(), **values))


def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return handle


def runtime(root):
    scripts = Path(root) / 'runtime/mcpfn/scripts'
    sys.path[:0] = [str(scripts), str(scripts.parent / 'src')]
    module = importlib.import_module('run_repeated_imputation')
    if Path(module.__file__).resolve() != (scripts / 'run_repeated_imputation.py').resolve():
        raise ValueError('Expected the frozen runtime, not the mutable working checkout')
    return module


def partition(cases, workers, costs):
    if workers < 1:
        raise ValueError('At least one worker is required')
    if len({case['case_id'] for case in cases}) != len(cases):
        raise ValueError('Duplicate case IDs')
    assignments, loads = [[] for unused in range(workers)], [0] * workers
    order = {case['case_id']: index for index, case in enumerate(cases)}
    for case in sorted(cases, key=lambda item: (-costs[item['case_id']], item['case_id'])):
        selected = min(range(workers), key=lambda index: (loads[index], index))
        assignments[selected].append(case['case_id'])
        loads[selected] += costs[case['case_id']]
    return [sorted(assigned, key=order.get) for assigned in assignments]


def select_cases(manifest, assignments, shard):
    if not 0 <= shard < len(assignments):
        raise ValueError('Invalid shard index')
    by_id = {case['case_id']: case for case in manifest['cases']}
    flat = [case_id for assigned in assignments for case_id in assigned]
    if len(flat) != len(set(flat)) or not set(flat).issubset(by_id):
        raise ValueError('Assignments overlap or contain unknown cases')
    return [by_id[case_id] for case_id in assignments[shard]]


def shard_hosts(plan):
    hosts = plan.get('shard_hosts', [plan['host']] * plan['workers'])
    if len(hosts) != plan['workers'] or any(not isinstance(host, str) or not host for host in hosts):
        raise ValueError('Every shard must have exactly one execution host')
    if plan['host'] not in hosts:
        raise ValueError('Primary host must own a shard')
    return hosts


def host_shards(plan, host=None):
    hosts = shard_hosts(plan)
    if host is None:
        if len(set(hosts)) != 1:
            raise ValueError('Multi-host plans require an explicit execution host')
        host = hosts[0]
    selected = [index for index, owner in enumerate(hosts) if owner == host]
    if not selected:
        raise ValueError('Execution host has no assigned shards')
    return selected


def prepare(root, folder, method, host, workers, hosts=None, replace_plan=None):
    root, folder = Path(root).resolve(), Path(folder).resolve()
    if method not in CPU_METHODS or folder.exists() or not 1 <= workers <= 64:
        raise ValueError('Use a supported CPU method, a new folder and 1–64 workers')
    placement = list(hosts) if hosts is not None else [host] * workers
    shard_hosts(dict(host=host, workers=workers, shard_hosts=placement))
    frozen = runtime(root)
    manifest = frozen.load_manifest(root)
    if method not in manifest['methods'] or not manifest.get('frozen_masks'):
        raise ValueError('Method must be registered with frozen masks')
    with lock(root / 'results' / method / 'worker.lock'):
        ownership = root / 'results' / method / 'delegation.json'
        previous = read(ownership) if ownership.exists() else None
        if replace_plan is not None:
            replace_plan = Path(replace_plan).resolve()
            if (previous is None or previous.get('plan') != str(replace_plan)
                    or previous.get('plan_sha256') != digest(replace_plan)):
                raise ValueError('Replacement must match the current immutable ownership claim')
            old = read(replace_plan)
            if old['root'] != str(root) or old['method'] != method or old['manifest_sha256'] != digest(root / 'manifest.json'):
                raise ValueError('Replacement must retain the same method and protocol')
        elif previous is not None:
            raise ValueError(f'Method already delegated: {ownership}')
        pending = [case for case in manifest['cases']
                   if not (root / 'results' / method / 'cases' / (case['case_id'] + '.json')).exists()]
        if not pending:
            raise ValueError('No unfinished cases to delegate')
        shapes = {item['truth_path']: np.load(item['truth_path'], mmap_mode='r').shape
                  for item in manifest['datasets']}
        costs = {case['case_id']: int(np.prod(shapes[case['truth_path']])) for case in pending}
        packages = ['numpy', 'scipy', 'pandas', 'scikit-learn', 'torch', 'joblib', 'threadpoolctl']
        packages += (['ForestDiffusion', 'xgboost', 'lightgbm', 'catboost'] if method == 'forestdiffusion'
                     else ['miceforest', 'lightgbm'] if method == 'mice'
                     else ['hyperimpute', 'xgboost', 'lightgbm', 'catboost'])
        folder.mkdir(parents=True)
        driver = folder / Path(__file__).name
        shutil.copyfile(Path(__file__).resolve(), driver)
        plan = dict(root=str(root), method=method, host=host, workers=workers,
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    manifest_sha256=digest(root / 'manifest.json'), driver_sha256=digest(driver),
                    shard_hosts=placement,
                    assignments=partition(pending, workers, costs),
                    expected_packages={name: importlib.metadata.version(name).split('+')[0] for name in packages},
                    policy='Disjoint pending cases, greedily balanced by matrix size. Original adapters, masks, '
                           'seeds, timeout, scoring and prediction files are unchanged. Local execution of this '
                           'method must stop before preparing the plan. No model-error-based selection.')
        write(folder / 'plan.json', plan)
        if previous is not None:
            write(folder / 'previous_ownership.json', previous)
        write(ownership, dict(plan=str(folder / 'plan.json'), host=host,
                              hosts=list(dict.fromkeys(placement)),
                              plan_sha256=digest(folder / 'plan.json')))
        print(json.dumps(dict(plan=str(folder / 'plan.json'), method=method, pending=len(pending),
                              shards=list(map(len, plan['assignments'])))), flush=True)


def load_plan(path):
    path = Path(path).resolve()
    plan = read(path)
    if plan['method'] not in CPU_METHODS:
        raise ValueError('Only CPU methods may be offloaded')
    if digest(Path(__file__)) != plan['driver_sha256']:
        raise ValueError('Execution driver changed; use the registered snapshot')
    root = Path(plan['root'])
    if digest(root / 'manifest.json') != plan['manifest_sha256']:
        raise ValueError('Parent protocol changed')
    manifest = read(root / 'manifest.json')
    if plan['method'] not in manifest['methods']:
        raise ValueError('Unregistered method')
    if len(plan['assignments']) != plan['workers']:
        raise ValueError('Worker count must match assignment count')
    shard_hosts(plan)
    select_cases(manifest, plan['assignments'], 0)
    return path, plan, root, manifest


def environment(plan):
    packages = {name: importlib.metadata.version(name) for name in plan['expected_packages']}
    mismatched = {name: version for name, version in packages.items()
                  if version.split('+')[0] != plan['expected_packages'][name]}
    if mismatched:
        raise ValueError(f'Package mismatch: {mismatched}')
    return dict(hostname=socket.gethostname(), python=sys.version, packages=packages,
                thread_environment={name: os.environ.get(name) for name in THREAD_ENV})


def shard_output(folder, shard):
    return Path(folder) / 'shards' / f'shard_{shard:02d}'


def worker(path, shard):
    path, plan, root, manifest = load_plan(path)
    environment(plan)
    frozen = runtime(root)
    original_load = frozen.load_manifest
    assigned = select_cases(manifest, plan['assignments'], shard)
    output = shard_output(path.parent, shard)
    output.mkdir(parents=True, exist_ok=True)
    for name in ('manifest.json', 'masks'):
        link = output / name
        if not link.exists():
            link.symlink_to(root / name, target_is_directory=name == 'masks')
        if link.resolve() != (root / name).resolve():
            raise ValueError('Shard input link points outside the frozen campaign')

    def assigned_manifest(destination):
        result = original_load(destination)
        return dict(result, cases=assigned)

    frozen.load_manifest = assigned_manifest
    frozen.worker(output, plan['method'])


def supervise_shard(path, plan, root, manifest, shard, stop):
    output = shard_output(path.parent, shard)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'results' / plan['method'] / 'status.json'
    while not stop.is_set():
        with (output / 'worker.log').open('ab', buffering=0) as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), 'worker',
                                        '--plan', str(path), '--shard', str(shard)],
                                       stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       start_new_session=True, env=dict(os.environ, **THREAD_ENV,
                                                                     CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
        timed_out = False
        try:
            while process.poll() is None:
                if stop.wait(5):
                    return
                status = read(status_path) if status_path.exists() else {}
                if (status.get('pid') == process.pid and status.get('state') == 'running'
                        and time.time() - status['started_unix'] > manifest['timeout_seconds'] + 30):
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    runtime(root).timeout_record(output, plan['method'], manifest, status)
                    timed_out = True
                    break
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        if timed_out:
            continue
        status = read(status_path) if status_path.exists() else {}
        if process.returncode or status.get('state') not in ('recycling', 'finished'):
            raise RuntimeError(f'Shard {shard} failed: exit={process.returncode}; status={status}')
        if status['state'] == 'finished':
            return


def serve(path):
    path, plan, root, manifest = load_plan(path)
    host_shards(plan)
    with lock(path.parent / 'serve.lock'), lock(root / 'distributed_locks' / (plan['method'] + '.lock')):
        runtime(root).load_manifest(root)
        write(path.parent / 'environment.json', environment(plan))
        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda unused_signal, unused_frame: stop.set())
        signal.signal(signal.SIGINT, lambda unused_signal, unused_frame: stop.set())
        with ThreadPoolExecutor(max_workers=plan['workers']) as executor:
            futures = [executor.submit(supervise_shard, path, plan, root, manifest, shard, stop)
                       for shard in range(plan['workers'])]
            try:
                while not all(future.done() for future in futures) and not stop.is_set():
                    for future in futures:
                        if future.done():
                            future.result()
                    stamp(path.parent / 'status.json', state='running', method=plan['method'],
                          workers=plan['workers'], finished_shards=sum(future.done() for future in futures))
                    stop.wait(10)
                if not stop.is_set():
                    for future in futures:
                        future.result()
                    stamp(path.parent / 'status.json', state='finished', method=plan['method'])
                else:
                    stamp(path.parent / 'status.json', state='stopped', method=plan['method'])
            except BaseException:
                stop.set()
                stamp(path.parent / 'status.json', state='failed', error=traceback.format_exc())
                raise


def validate_record(record, prediction_path, case, manifest_hash, mask_path, mask_record, scorer):
    if any(record.get(key) != value for key, value in case.items()):
        raise ValueError('Result case registration mismatch')
    if record.get('manifest_sha256') != manifest_hash:
        raise ValueError('Result manifest mismatch')
    if digest(mask_path) != mask_record['mask_sha256'] or record.get('mask_sha256') != mask_record['mask_sha256']:
        raise ValueError('Result mask mismatch')
    if record['status'] not in ('ok', 'error', 'timeout', 'unevaluable_mask'):
        raise ValueError('Unrecognized result status')
    if record['status'] != 'ok':
        return
    if digest(prediction_path) != record.get('prediction_sha256'):
        raise ValueError('Prediction checksum mismatch')
    if digest(case['truth_path']) != case['truth_sha256']:
        raise ValueError('Truth checksum mismatch')
    truth = np.load(case['truth_path'], allow_pickle=False).astype(np.float64)
    with np.load(mask_path, allow_pickle=False) as stored:
        mask = np.unpackbits(stored['packed'], count=int(np.prod(stored['shape']))).reshape(tuple(stored['shape'])).astype(bool)
    with np.load(prediction_path, allow_pickle=False) as stored:
        if tuple(stored['shape']) != truth.shape or stored['missing_values'].shape != (int(mask.sum()),):
            raise ValueError('Prediction dimensions mismatch')
        prediction = truth.copy()
        prediction[mask] = stored['missing_values']
    missing = truth.copy()
    missing[mask] = np.nan
    for key, value in scorer(truth, missing, prediction, case['suite']).items():
        saved = record.get(key)
        if value is None:
            agrees = saved is None
        else:
            agrees = saved is not None and math.isclose(saved, value, rel_tol=1e-10, abs_tol=1e-12)
        if not agrees:
            raise ValueError(f'Recomputed score mismatch: {key}')


def import_results(path, plan, root, manifest, shards=None, incoming_root=None):
    frozen = runtime(root)
    scorer = importlib.import_module('repeated_imputation_methods').metrics
    by_id = {case['case_id']: case for case in manifest['cases']}
    imported = 0
    plan_hash = digest(path)
    selected = set(range(plan['workers']) if shards is None else shards)
    incoming_root = Path(incoming_root) if incoming_root is not None else path.parent / 'incoming'
    for shard, assigned in enumerate(plan['assignments']):
        if shard not in selected:
            continue
        incoming = incoming_root / 'shards' / f'shard_{shard:02d}' / 'results' / plan['method']
        for case_id in assigned:
            source = incoming / 'cases' / (case_id + '.json')
            if not source.exists():
                continue
            record = read(source)
            if record.get('method') != plan['method'] or record.get('case_id') != case_id:
                raise ValueError('Imported result is outside its assigned work')
            target = root / 'results' / plan['method'] / 'cases' / (case_id + '.json')
            if target.exists():
                existing = read(target)
                if (existing.get('distributed_execution', {}).get('plan_sha256') != plan_hash
                        or existing.get('status') != record.get('status')
                        or existing.get('prediction_sha256') != record.get('prediction_sha256')):
                    raise ValueError(f'Refusing to overwrite a conflicting result: {case_id}')
                continue
            prediction = incoming / 'predictions' / (case_id + '.npz')
            if record['status'] == 'ok' and not prediction.exists():
                continue
            mask, receipt = frozen.case_paths(root, by_id[case_id])
            validate_record(record, prediction, by_id[case_id], plan['manifest_sha256'], mask, read(receipt), scorer)
            if record['status'] == 'ok':
                destination = root / 'results' / plan['method'] / 'predictions' / (case_id + '.npz')
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix('.importing')
                shutil.copyfile(prediction, temporary)
                temporary.replace(destination)
            record['distributed_execution'] = dict(host=shard_hosts(plan)[shard], shard=shard,
                                                    plan_sha256=plan_hash, driver_sha256=plan['driver_sha256'])
            write(target, record)
            imported += 1
    return imported


def transfer_command(path, host, shards, incoming):
    return ['rsync', '-rlt', '--delay-updates', '--timeout=120', '-e',
            'ssh -o BatchMode=yes -o ConnectTimeout=15', '--exclude=manifest.json',
            '--exclude=masks', '--exclude=*.lock', '--exclude=*.log',
            '--include=/status.json', '--include=/environment.json', '--include=/execution_resources.json',
            '--include=/shards/'] + [f'--include=/shards/shard_{shard:02d}/***' for shard in shards] + [
            '--exclude=*', f'{host}:{path.parent}/', str(incoming) + '/']


def export_when_idle(root, manifest, frozen):
    try:
        extension_lock = lock(root / 'extension.lock')
    except BlockingIOError:
        return False
    with extension_lock:
        try:
            pipeline_lock = lock(root / 'pipeline.lock')
        except BlockingIOError:
            return False
        with pipeline_lock:
            frozen.export(root, manifest)
    return True


def collect_shared(path, plan, root, manifest, interval):
    frozen = runtime(root)
    frozen.load_manifest(root)
    hosts = list(dict.fromkeys(shard_hosts(plan)))
    with lock(path.parent / 'collect.lock'), lock(root / 'results' / plan['method'] / 'worker.lock'):
        ownership = read(root / 'results' / plan['method'] / 'delegation.json')
        if ownership.get('plan_sha256') != digest(path):
            raise ValueError('Method is owned by another execution plan')
        stamp(path.parent / 'collector_status.json', state='waiting_for_remote', owns_method_lock=True)
        while True:
            remote_by_host, node_states, errors = {}, {}, {}
            imported = 0
            for index, host in enumerate(hosts):
                incoming = path.parent / 'incoming' / f'host_{index:02d}'
                incoming.mkdir(parents=True, exist_ok=True)
                shards = host_shards(plan, host)
                try:
                    subprocess.run(transfer_command(path, host, shards, incoming), capture_output=True,
                                   text=True, timeout=300, check=True)
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                    node_states[host] = 'connection_retry'
                    errors[host] = str(error)[-2000:]
                    continue
                remote = read(incoming / 'status.json') if (incoming / 'status.json').exists() else {}
                remote_by_host[host] = remote
                try:
                    imported += import_results(path, plan, root, manifest, shards, incoming)
                except Exception:
                    node_states[host] = 'import_failed'
                    errors[host] = traceback.format_exc()
                    continue
                complete = all((root / 'results' / plan['method'] / 'cases' / (case_id + '.json')).exists()
                               for shard in shards for case_id in plan['assignments'][shard])
                node_states[host] = ('finished' if complete and remote.get('state') == 'finished'
                                     else 'remote_failed' if remote.get('state') in ('failed', 'stopped', 'finished')
                                     else 'distributed_running')
            statuses = [read(record)['status'] for record in (root / 'results' / plan['method'] / 'cases').rglob('*.json')]
            finished = len(statuses) == len(manifest['cases']) and all(state == 'finished' for state in node_states.values())
            values = dict(state='finished' if finished else 'distributed_running', owns_method_lock=True,
                          hosts=hosts, node_states=node_states, errors=errors, remote_by_host=remote_by_host,
                          completed=len(statuses), expected=len(manifest['cases']), successful=statuses.count('ok'),
                          imported_this_poll=imported)
            stamp(path.parent / 'collector_status.json', **values)
            stamp(root / 'results' / plan['method'] / 'status.json', **values)
            export_when_idle(root, manifest, frozen)
            if finished:
                return
            time.sleep(interval)


def collect(path, interval):
    path, plan, root, manifest = load_plan(path)
    if len(set(shard_hosts(plan))) > 1:
        return collect_shared(path, plan, root, manifest, interval)
    incoming = path.parent / 'incoming'
    incoming.mkdir(exist_ok=True)
    frozen = runtime(root)
    frozen.load_manifest(root)
    with lock(path.parent / 'collect.lock'), lock(root / 'results' / plan['method'] / 'worker.lock'):
        ownership = read(root / 'results' / plan['method'] / 'delegation.json')
        if ownership.get('plan_sha256') != digest(path):
            raise ValueError('Method is owned by another execution plan')
        stamp(path.parent / 'collector_status.json', state='waiting_for_remote', owns_method_lock=True)
        while True:
            command = ['rsync', '-rlt', '--delay-updates', '--timeout=120', '-e',
                       'ssh -o BatchMode=yes -o ConnectTimeout=15', '--exclude=manifest.json',
                       '--exclude=masks', '--exclude=*.lock', '--exclude=*.log',
                       '--include=/status.json', '--include=/environment.json',
                       '--include=/shards/', '--include=/shards/***', '--exclude=*',
                       f"{plan['host']}:{path.parent}/", str(incoming) + '/']
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:
                stamp(path.parent / 'collector_status.json', state='connection_retry', owns_method_lock=True,
                      error='Result transfer timed out; local method remains locked')
                time.sleep(interval)
                continue
            if result.returncode:
                stamp(path.parent / 'collector_status.json', state='connection_retry', owns_method_lock=True,
                      error=result.stderr[-2000:])
                time.sleep(interval)
                continue
            try:
                imported = import_results(path, plan, root, manifest)
            except Exception:
                stamp(path.parent / 'collector_status.json', state='import_failed', owns_method_lock=True,
                      error=traceback.format_exc())
                time.sleep(interval)
                continue
            statuses = [read(record)['status'] for record in (root / 'results' / plan['method'] / 'cases').rglob('*.json')]
            remote = read(incoming / 'status.json') if (incoming / 'status.json').exists() else {}
            complete = len(statuses) == len(manifest['cases'])
            state = 'finished' if complete else 'remote_failed' if remote.get('state') == 'failed' else 'distributed_running'
            values = dict(state=state, owns_method_lock=True, host=plan['host'], completed=len(statuses), expected=len(manifest['cases']),
                          successful=statuses.count('ok'), imported_this_poll=imported, remote=remote)
            stamp(path.parent / 'collector_status.json', **values)
            stamp(root / 'results' / plan['method'] / 'status.json', **values)
            export_when_idle(root, manifest, frozen)
            if complete and remote.get('state') == 'finished':
                return
            if remote.get('state') in ('failed', 'stopped', 'finished'):
                stamp(path.parent / 'collector_status.json', state='remote_failed', owns_method_lock=True,
                      remote=remote, error='Assignments incomplete; local method remains locked')
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'worker', 'serve', 'collect'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--method', choices=sorted(CPU_METHODS))
    parser.add_argument('--host')
    parser.add_argument('--shard-hosts', nargs='+')
    parser.add_argument('--replace-plan', type=Path,
                        help='Replace this claim only after stopping its services and importing final results')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--shard', type=int)
    parser.add_argument('--interval', type=int, default=120)
    args = parser.parse_args()
    if args.command == 'prepare':
        if not all((args.root, args.folder, args.method, args.host)):
            parser.error('prepare requires --root, --folder, --method and --host')
        prepare(args.root, args.folder, args.method, args.host, args.workers, args.shard_hosts, args.replace_plan)
    else:
        if args.plan is None or args.interval < 1:
            parser.error('A plan and a positive polling interval are required')
        if args.command == 'worker':
            if args.shard is None:
                parser.error('worker requires --shard')
            worker(args.plan, args.shard)
        elif args.command == 'serve':
            serve(args.plan)
        else:
            collect(args.plan, args.interval)


if __name__ == '__main__':
    main()
