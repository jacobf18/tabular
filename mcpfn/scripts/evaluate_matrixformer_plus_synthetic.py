"""Plan and run frozen MatrixFormer+ synthetic matrix-completion evaluations."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from recsys_evaluation_common import atomic_npz, sha256, write_json
from synthetic_low_rank_benchmark import (
    BASELINES, METHODS, NEURAL_METHODS, array_digest, cases_for_profile, evaluate_baseline,
    generate_case, predict_experts, score_prediction, support_diagnostics,
)


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / 'evaluation_protocols/matrixformer_plus_synthetic.json'
CHECKPOINTS = {
    'general': REPO / 'runs/streamed_training/large_matrices_98k_20260930/best_validation_checkpoint.pth',
    'specialist': REPO / 'runs/recsys_goal_20260920/synthetic_user_scaled_parent_risk_400/best_validation_checkpoint.pth',
}
CHECKPOINT_SHA256 = {
    'general': 'c0ddf75274c8ff92ca5fd22c1263369ffa7e55eb7cb85fe80961b0afa101f7a9',
    'specialist': '1fcadc0c5846f9c35fbbae4e3b519b9244b608272cb1eda68c0850132321f770',
}


def read(path):
    return json.loads(Path(path).read_text())


def source_paths():
    scripts = [Path(__file__).resolve(), REPO / 'scripts/synthetic_low_rank_benchmark.py',
               REPO / 'scripts/summarize_matrixformer_plus_synthetic.py', REPO / 'scripts/recsys_evaluation_common.py']
    package = [path for path in (REPO / 'src/tabimpute').rglob('*.py') if 'workdir' not in path.parts]
    return sorted(set([*scripts, *package, REPO / 'src/tabimpute/data/borders.pt']))


def check_storage(output, reserve):
    if shutil.disk_usage(output).free < reserve * 1024 ** 3:
        raise RuntimeError(f'Preserve the registered {reserve:g} GiB free-space reserve')


def plan(args):
    config_path = args.config.resolve()
    config = read(config_path)
    if config.get('version') != 1 or config['methods'] != list(METHODS):
        raise ValueError('Unsupported protocol or method inventory')
    cases = cases_for_profile(config['profiles'][args.profile])
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Planning requires an empty output directory; run an existing plan to resume')
    device = args.device or 'cuda'
    precision = args.precision or ('float32' if device == 'cpu' else 'amp_bfloat16')
    if device == 'cpu' and precision != 'float32':
        raise ValueError('CPU diagnostic runs use explicit float32 inference')
    checkpoint_data = {}
    for name, path in CHECKPOINTS.items():
        checkpoint_data[name] = {'path': str(path), 'sha256': sha256(path),
            'config': str(path.parent / 'config.json'), 'config_sha256': sha256(path.parent / 'config.json')}
        if checkpoint_data[name]['sha256'] != CHECKPOINT_SHA256[name]:
            raise ValueError(f'Checkpoint differs from the established MatrixFormer+ expert: {name}')
    sources = {str(path): sha256(path) for path in source_paths()}
    packages = {name: importlib.metadata.version(name) for name in
                ('numpy', 'scipy', 'scikit-learn', 'fancyimpute', 'torch', 'matplotlib')}
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'evaluation_only': True,
        'training_permitted': False, 'config': str(config_path), 'config_sha256': sha256(config_path),
        'profile': args.profile, 'protocol': config, 'cases': cases, 'checkpoints': checkpoint_data,
        'source_sha256': sources, 'packages': packages, 'device': device, 'precision': precision,
        'backend': args.backend or 'dense', 'threads': args.threads, 'reserve_gib': args.reserve_gib,
        'inference': 'Frozen predictive means; full matrix context; no rating clipping; observed_blend with observed-column scaling',
        'primary_metric': 'Cell-pooled RMSE over every hidden entry within each matrix; mean and uncertainty across seed replicates',
        'secondary_metric': 'Hidden-entry RMSE against the latent noise-free signal, labeled separately',
        'storage_precision': 'Float32 data and predictions; scores accumulated in float64 from these archived values',
        'limitations': ['Observed blending uses in-sample reconstruction, not held-out routing evidence.',
            'MCAR synthetic stress tests are not claims about arbitrary MNAR matrices.',
            'Observations/DOF is a necessary-count diagnostic, not a recovery guarantee.',
            'Gaussian factors are related to pretraining; fresh seeds do not make this a wholly unseen distribution.',
            'CPU and smoke-profile results are infrastructure checks, not the main paper experiment.']}
    output.mkdir(parents=True, exist_ok=True)
    check_storage(output, args.reserve_gib)
    for source in sources:
        path = Path(source)
        destination = output / 'source_snapshot' / path.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    shutil.copyfile(config_path, output / 'protocol.json')
    write_json(output / 'manifest.json', manifest)
    (output / 'manifest.sha256').write_text(sha256(output / 'manifest.json') + '\n')
    write_json(output / 'status.json', {'state': 'planned', 'expected_cases': len(cases),
        'expected_method_results': len(cases) * len(METHODS), 'neural_updates': 0})
    dense_bytes = sum(case['rows'] * case['cols'] * (13 + 4 * len(METHODS) * case['missing_rate']) for case in cases)
    print(json.dumps({'output': str(output), 'cases': len(cases), 'method_results': len(cases) * len(METHODS),
        'uncompressed_array_storage_estimate_gib': dense_bytes / 1024 ** 3,
        'device': device, 'precision': precision, 'training_permitted': False}), flush=True)


def verify_manifest(output):
    if sha256(output / 'manifest.json') != (output / 'manifest.sha256').read_text().strip():
        raise ValueError('Manifest changed; create a new output directory')
    manifest = read(output / 'manifest.json')
    if not manifest['evaluation_only'] or manifest['training_permitted']:
        raise ValueError('Evaluation-only manifest required')
    if (sha256(Path(manifest['config'])) != manifest['config_sha256'] or
            read(output / 'protocol.json') != manifest['protocol'] or
            read(manifest['config']) != manifest['protocol']):
        raise ValueError('Protocol changed; create a new output directory')
    if manifest['cases'] != cases_for_profile(manifest['protocol']['profiles'][manifest['profile']]):
        raise ValueError('Registered cases differ from the protocol')
    for source, digest in manifest['source_sha256'].items():
        if sha256(Path(source)) != digest or sha256(output / 'source_snapshot' / Path(source).relative_to(REPO)) != digest:
            raise ValueError(f'Source changed; do not mix runs: {source}')
    for checkpoint in manifest['checkpoints'].values():
        for path_key, hash_key in (('path', 'sha256'), ('config', 'config_sha256')):
            if sha256(Path(checkpoint[path_key])) != checkpoint[hash_key]:
                raise ValueError(f'Frozen checkpoint/config changed: {checkpoint[path_key]}')
    for name, version in manifest['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Dependency changed: {name}')
    return manifest


def require_idle_gpu():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                      '--format=csv,noheader'], text=True)
    if output.strip():
        raise RuntimeError(f'GPU is in use; do not overlap another evaluation:\n{output}')


class FrozenExperts:
    def __init__(self, manifest):
        import torch
        from tabimpute.interface import ImputePFN
        from tabimpute.streamed_inference import StreamedImputePFN

        torch.set_num_threads(manifest['threads'])
        torch.manual_seed(manifest['protocol']['master_seed'])
        torch.backends.cuda.enable_cudnn_sdp(False)
        factory = StreamedImputePFN if manifest['backend'] == 'streamed' else ImputePFN
        self.models = {}
        for name, checkpoint in manifest['checkpoints'].items():
            model = factory(device=manifest['device'], checkpoint_path=checkpoint['path'],
                json_config={'config': read(checkpoint['config'])['model']}, model_family='new_stable',
                precision_mode=manifest['precision'], prediction_stat='mean', entry_wise_features=False,
                max_num_rows=None, max_num_cols=None, max_num_chunks=1, verbose=False)
            model.model.eval().requires_grad_(False)
            self.models[name] = model
        self.versions = self.state_versions()

    def state_versions(self):
        return {name: {key: tensor._version for key, tensor in model.model.state_dict().items()}
                for name, model in self.models.items()}

    def check_frozen(self):
        if self.state_versions() != self.versions:
            raise RuntimeError('Neural model state mutated during evaluation')
        for model in self.models.values():
            if model.model.training or any(parameter.requires_grad or parameter.grad is not None
                                           for parameter in model.model.parameters()):
                raise RuntimeError('Neural training/gradients are prohibited')

    def predict(self, observed):
        import torch

        self.check_frozen()
        with torch.inference_mode():
            result = predict_experts(observed, self.models['general'], self.models['specialist'])
        self.check_frozen()
        return result


def record_result(directory, case, method, prediction, details, truth, signal, observed, input_sha256):
    scores, saved = score_prediction(truth, signal, observed, prediction)
    path = directory / f'{method}.npz'
    atomic_npz(path, prediction=saved)
    record = {**case, 'method': method, 'status': 'ok', **scores, **details,
              'inputs_sha256': input_sha256, 'predictions_sha256': sha256(path), 'neural_updates': 0}
    write_json(directory / f'{method}.json', record)
    return record


def verify_result(directory, method, case, input_sha256):
    path = directory / f'{method}.json'
    if not path.exists():
        return None
    record = read(path)
    if record['case_id'] != case['case_id'] or record['method'] != method or record['inputs_sha256'] != input_sha256:
        raise ValueError('Existing result belongs to different inputs')
    if record['status'] == 'ok' and sha256(directory / f'{method}.npz') != record['predictions_sha256']:
        raise ValueError('Archived prediction changed')
    return record


@contextmanager
def run_lock(output):
    with (output / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        except BaseException as error:
            status = read(output / 'status.json')
            if status['state'] == 'running':
                write_json(output / 'status.json', {**status, 'state': 'interrupted',
                    'error': f'{type(error).__name__}: {error}'})
            raise


def run(args):
    output = args.output.resolve()
    with run_lock(output):
        manifest = verify_manifest(output)
        for name in ('device', 'precision', 'backend'):
            if getattr(args, name) is not None and getattr(args, name) != manifest[name]:
                raise ValueError('Runtime inference settings must match the plan')
        requested = set(args.methods or METHODS)
        requested = (requested - {'baselines', 'neural'}) | (set(BASELINES) if 'baselines' in requested else set()) | (
            set(NEURAL_METHODS) if 'neural' in requested else set())
        if not requested.issubset(METHODS):
            raise ValueError('Unknown methods')
        if requested.intersection(NEURAL_METHODS):
            requested.update(NEURAL_METHODS)
        experts = None
        processed = 0
        settings = manifest['protocol']['baseline_settings']
        for case in manifest['cases']:
            if args.case_limit is not None and processed >= args.case_limit:
                break
            directory = output / 'cases' / case['case_id']
            directory.mkdir(parents=True, exist_ok=True)
            check_storage(output, manifest['reserve_gib'])
            signal, truth, observed, missing = generate_case(case, manifest['protocol']['master_seed'])
            inputs = directory / 'inputs.npz'
            if inputs.exists():
                with np.load(inputs, allow_pickle=False) as archived:
                    if array_digest(archived['signal'], archived['truth'], archived['observed'], archived['missing']) != array_digest(signal, truth, observed, missing):
                        raise ValueError('Stored synthetic inputs do not regenerate exactly')
            else:
                atomic_npz(inputs, signal=signal, truth=truth, observed=observed, missing=missing)
            input_sha256 = sha256(inputs)
            pending = []
            for method in METHODS:
                if method not in requested:
                    continue
                record = verify_result(directory, method, case, input_sha256)
                if record is not None and record['status'] != 'ok' and args.retry_failed:
                    previous = directory / f'{method}.failed_attempt_{time.time_ns()}.json'
                    (directory / f'{method}.json').rename(previous)
                    record = None
                if record is None:
                    pending.append(method)
            if not pending:
                continue
            processed += 1
            write_json(output / 'status.json', {'state': 'running', 'case_id': case['case_id'],
                'methods': sorted(requested), 'pid': os.getpid(), 'neural_updates': 0})
            diagnostics = support_diagnostics(observed, case['rank'])
            original = array_digest(observed)
            for method in pending:
                if method not in BASELINES:
                    continue
                try:
                    prediction, details = evaluate_baseline(observed.copy(), method, settings, case['seed'])
                    record_result(directory, case, method, prediction, {**details, 'support': diagnostics},
                                  truth, signal, observed, input_sha256)
                except Exception as error:
                    write_json(directory / f'{method}.json', {**case, 'method': method, 'status': 'failed',
                        'error': f'{type(error).__name__}: {error}', 'inputs_sha256': input_sha256,
                        'support': diagnostics, 'neural_updates': 0})
            neural_pending = set(pending).intersection(NEURAL_METHODS)
            if neural_pending:
                if experts is None:
                    if manifest['device'].startswith('cuda'):
                        require_idle_gpu()
                    experts = FrozenExperts(manifest)
                started = time.perf_counter()
                try:
                    predictions, details = experts.predict(observed.copy())
                    elapsed = time.perf_counter() - started
                    for method in sorted(neural_pending):
                        seconds = (sum(details['expert_seconds'].values()) if method == 'matrixformer_plus' else
                                   details['expert_seconds']['general' if method == 'matrixformer' else 'specialist'])
                        record_result(directory, case, method, predictions[method],
                            {**details, 'seconds': seconds, 'shared_bundle_wall_seconds': elapsed, 'support': diagnostics},
                            truth, signal, observed, input_sha256)
                except Exception as error:
                    experts.check_frozen()
                    for method in sorted(neural_pending):
                        write_json(directory / f'{method}.json', {**case, 'method': method, 'status': 'failed',
                            'error': f'{type(error).__name__}: {error}', 'inputs_sha256': input_sha256,
                            'support': diagnostics, 'neural_updates': 0})
            if array_digest(observed) != original:
                raise RuntimeError('Evaluator mutated the supplied observations')
            print(json.dumps({'case_id': case['case_id'], 'methods_attempted': pending, 'neural_updates': 0}), flush=True)
        if experts is not None:
            experts.check_frozen()
        verify_manifest(output)
        counts = {'ok': 0, 'failed': 0, 'pending': 0}
        for case in manifest['cases']:
            for method in METHODS:
                path = output / 'cases' / case['case_id'] / f'{method}.json'
                counts[read(path)['status'] if path.exists() else 'pending'] += 1
        write_json(output / 'status.json', {'state': 'partial' if counts['pending'] else (
            'complete_with_failures' if counts['failed'] else 'complete'), **counts,
            'expected_method_results': len(manifest['cases']) * len(METHODS),
            'checkpoints_unchanged': True, 'neural_updates': 0})
        print(json.dumps(read(output / 'status.json')), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--profile', choices=('smoke', 'main', 'stress'), default='main')
    parser.add_argument('--device', choices=('cuda', 'cpu'))
    parser.add_argument('--precision', choices=('amp_bfloat16', 'float32'))
    parser.add_argument('--backend', choices=('dense', 'streamed'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--reserve-gib', type=float, default=20.)
    parser.add_argument('--methods', nargs='+', choices=(*METHODS, 'baselines', 'neural'))
    parser.add_argument('--case-limit', type=int)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if args.threads <= 0 or args.reserve_gib < 0 or (args.case_limit is not None and args.case_limit <= 0):
        parser.error('Threads and case limit must be positive; reserve must be nonnegative')
    from threadpoolctl import threadpool_limits

    threads = args.threads if args.action == 'plan' else read(args.output / 'manifest.json')['threads']
    with threadpool_limits(limits=threads):
        plan(args) if args.action == 'plan' else run(args)


if __name__ == '__main__':
    main()
