"""Finish V53's untouched qualification decision after storage-only recovery."""

import argparse
import fcntl
from pathlib import Path
import time

import run_single_model_balanced_structured_pipeline as original
import relocate_single_model_v53_root_archive as preservation
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


OUTPUT = original.ROOT / 'balanced_structured_components_v53_capacity_recovery'
FAILURE = "OSError('Preserve 6442450944 bytes on root before confirm: free=6448852992, budget=16777216')"
TRAINING_AUDIT_SHA = '32519e3d1144c1cfc5cb69963c45b122820d684956726aa13ef2949cbbab8919'
ENDPOINT_AUDIT_SHA = '72fbbd56c21a7d2316f9cf91c5e8390ecacf61355f88577e2bf621082a1546f7'
PRESERVATION_SHA = 'fc4fcf99ac506d260a7186552e44011fd54133656e5054b4eacf0a7b899a2370'
REASON = 'No trained boundary retains every original guard and strictly improves minimax'
POLICY = (
    'Orchestration only: preserve original V53 failed coordinator/config/status/log bytes. '
    'Original64-attempt/59-update training,1792-task audit,56-grid endpoint replay and CPU audit '
    'already completed; never repeat any of them. The original coordinator failed its unchanged '
    'root capacity check before launching confirm. Bind audited step0/no eligible selection '
    'and the untouched confirmation reservation. After independently verified storage-only '
    'preservation restores all original reserves, run only the unchanged original confirm CLI '
    'once, through its original capacity admission and qualification logic. This must produce '
    'the original negative gate without generating any of284 fresh confirmation cases. '
    'Keep both original source sets, seeds, all22 guards,32 confirmation comparisons and every '
    'frozen real benchmark unchanged. No manual selection, altered tolerance, new scientific '
    'trial, inference, benchmark execution or goal-success claim. Publish only additive '
    'recovery orchestration metadata/logs and the original previously unstarted gate; '
    'leave the original failed coordinator failed forever.'
)


def read(path):
    return original.read(path)


def status(state, **fields):
    original.status(OUTPUT, state, **fields)


def validate_original(failed, trained, audit, endpoint, replay):
    if (failed.get('state') != 'failed' or failed.get('error') != FAILURE
            or trained.get('state') != 'training_complete' or trained.get('step') != 64
            or trained.get('accepted_updates') != 59 or trained.get('best', {}).get('step') != 0
            or audit.get('complete') is not True or audit.get('accepted_updates') != 59
            or audit.get('training_episodes') != 1792 or audit.get('best', {}).get('step') != 0
            or endpoint.get('complete') is not True or endpoint.get('endpoint_grids') != 56
            or endpoint.get('selected_grids') != 0 or endpoint.get('cuda_initialized') is not False
            or replay.get('state') != 'complete' or replay.get('selected_validation_passed') is not False):
        raise ValueError('Require the exact audited, unqualified V53 and its pre-confirm capacity failure')


def require_original():
    original.verify(original.OUTPUT)
    failed = read(original.OUTPUT / 'status.json')
    trained = read(original.training.OUTPUT / 'status.json')
    audit = read(original.AUDIT / 'audit.json')
    endpoint = read(original.REPLAY / 'audit.json')
    replay = read(original.REPLAY / 'status.json')
    validate_original(failed, trained, audit, endpoint, replay)
    for current in (failed, trained, replay):
        if original.process_identity(current['pid']) is not None:
            raise ValueError('Never recover over a live original scientific stage')
    if (sha256(original.AUDIT / 'audit.json') != TRAINING_AUDIT_SHA
            or sha256(original.REPLAY / 'audit.json') != ENDPOINT_AUDIT_SHA
            or not read(original.REPLAY / 'gate.json')['endpoint_reproduced']):
        raise ValueError('Bind the original complete independent audits, never replacement runs')
    receipt = preservation.OUTPUT / 'independent_recheck.json'
    checked = read(receipt)
    if (sha256(receipt) != PRESERVATION_SHA or checked.get('complete') is not True
            or checked.get('all_original_bytes_and_paths_preserved') is not True
            or checked.get('original_failed_pipeline_unchanged') is not True
            or checked.get('protected_v53_sources_checked') != 555
            or checked['config_sha256'] != sha256(preservation.OUTPUT / 'config.json')
            or read(preservation.OUTPUT / 'status.json')['state'] != 'complete'):
        raise ValueError('Require the exact independently rechecked storage-only recovery')
    preservation.scratch.require_mount()
    preservation.mounts.check_mount(read(preservation.OUTPUT / 'config.json')['record'], preservation.DESTINATION)
    preservation.capacity(restored=True)


def require_unused():
    if (read(original.CONFIRMATION / 'status.json')['state'] != 'registered'
            or {path.name for path in original.CONFIRMATION.iterdir()} != {'config.json', 'registration.json', 'status.json'}):
        raise ValueError('Only the original completely unused confirmation reservation may run')
    if (original.CAMPAIGN.exists() or original.REVIEW.exists()
            or (original.training.OUTPUT / 'selected_checkpoint.pth').exists()
            or (original.training.OUTPUT / 'best_validation_checkpoint.pth').exists()
            or (original.OUTPUT / 'result.json').exists()):
        raise ValueError('Never recover over a selected model, real campaign or fabricated original closure')


def validate_negative(gate, names):
    expected = {'passed': False, 'selected_step': 0, 'confirmation_evaluated': False,
        'reason': REASON, 'real_benchmark_success': False}
    if gate != expected or set(names) != {'config.json', 'registration.json', 'status.json', 'gate.json'}:
        raise ValueError('Require the exact original negative decision, without consuming confirmation')


def register():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError('Never repeat this one-shot orchestration recovery')
    require_original()
    require_unused()
    original.prerequisite(original.OUTPUT)
    original.require_stage_capacity('confirm')
    paths = [original.OUTPUT / name for name in ('config.json', 'registration.json', 'status.json',
        'run.log', 'replay.log', 'audit-replay.log')]
    paths.extend(original.training.OUTPUT / name for name in ('config.json', 'registration.json',
        'status.json', 'training.jsonl', 'validation.jsonl'))
    paths.extend([original.AUDIT / 'audit.json', original.REPLAY / 'audit.json', original.REPLAY / 'gate.json',
        original.REPLAY / 'result.json', original.REPLAY / 'status.json'])
    paths.extend(original.CONFIRMATION / name for name in ('config.json', 'registration.json'))
    paths.extend(preservation.OUTPUT / name for name in ('config.json', 'status.json', 'preservation.json',
        'durable_copy.json', 'independent_recheck.json'))
    sources = [Path(__file__).resolve(), Path(original.__file__).resolve(), Path(preservation.__file__).resolve(),
        REPO / 'tests/test_single_model_v53_capacity_recovery.py', REPO / 'tests/test_single_model_v53_root_archive.py']
    config = {'policy': POLICY, 'original_pipeline': str(original.OUTPUT), 'stages': ['confirm'],
        'input_sha256': {str(path): sha256(path) for path in paths},
        'source_sha256': {str(path): sha256(path) for path in sources}}
    OUTPUT.mkdir()
    write_json(OUTPUT / 'config.json', config)
    write_json(OUTPUT / 'registration.json', {'config_sha256': sha256(OUTPUT / 'config.json'),
        'registered_unix': time.time(), 'scientific_stage_repeated': False})
    status('registered')


def verify():
    config = read(OUTPUT / 'config.json')
    if (read(OUTPUT / 'registration.json')['config_sha256'] != sha256(OUTPUT / 'config.json')
            or config['policy'] != POLICY or config['original_pipeline'] != str(original.OUTPUT)
            or config['stages'] != ['confirm']):
        raise ValueError('Preserve the one-shot registered recovery scope')
    verify_hashes(config['source_sha256'])
    verify_hashes(config['input_sha256'])
    require_original()


def run():
    verify()
    require_unused()
    original.run_child(OUTPUT, 'confirm', Path(original.__file__).resolve(),
        ['--stage', 'confirm', '--output', str(original.OUTPUT)], cpu=False)
    gate = read(original.CONFIRMATION / 'gate.json')
    validate_negative(gate, [path.name for path in original.CONFIRMATION.iterdir()])
    if read(original.CONFIRMATION / 'status.json')['state'] != 'complete':
        raise ValueError('Require the original completed negative gate')
    verify()
    write_json(OUTPUT / 'result.json', {'complete': True, 'gate': gate,
        'training_audit_sha256': TRAINING_AUDIT_SHA, 'endpoint_audit_sha256': ENDPOINT_AUDIT_SHA,
        'original_failed_pipeline': str(original.OUTPUT),
        'original_failure_sha256': sha256(original.OUTPUT / 'status.json'),
        'campaign': None, 'review': None, 'scientific_stages_repeated': False,
        'confirmation_cases_generated': 0, 'no_inference_ensemble': True, 'goal_success_not_assessed': True})
    status('complete', confirmed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('register', 'run'), required=True)
    arguments = parser.parse_args()
    if arguments.stage == 'register':
        register()
        return
    with (OUTPUT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(OUTPUT / 'status.json')['state'] != 'registered':
            raise ValueError('Never restart the recovery')
        try:
            run()
        except BaseException as error:
            status('failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
