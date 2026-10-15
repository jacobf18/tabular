"""Audit V53's completed negative gate without repeating any scientific stage."""

import fcntl
import os
from pathlib import Path
import time

import recover_single_model_v53_capacity as recovery
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


OUTPUT = recovery.original.ROOT / 'balanced_structured_components_v53_closure_audit'
FAILURE = "ValueError('Require the exact original negative decision, without consuming confirmation')"
RECOVERY_CONFIG_SHA = '805625c695f103acfeaaf228317281feefb9995bb03b3093bd74f7024cf0c15b'
RECOVERY_FAILURE_SHA = '6831cae0f074892a37c18cf7480279a921e7a2d701a0d2f7a505d15a22e3f0c5'
GATE_SHA = '3016c6c32e38311fecae4a0c250e409167d7eb92827c715d1efecbc80ca71a94'
POLICY = (
    'Read-only scientific closure: original V53 confirm already exited successfully and wrote '
    'its negative gate without consuming confirmation. The additive capacity recovery then '
    'rejected the normal zero-byte confirmation run.lock because its inventory expected only '
    'four files. Preserve that failed recovery and the original capacity failure unchanged. '
    'Validate exactly the original four metadata files plus one regular empty single-link '
    'run.lock, acquire it exclusively and apply the identical original negative-gate validator '
    'to the four metadata names. Recheck original training/endpoint audits, both original '
    'source inventories, preservation receipt and recovery registration. Bind all failure, '
    'gate and audit hashes. Never run register/run/confirm/replay, generate data or predictions, '
    'change any source already registered, select a model, retry a scientific stage, change '
    'capacity reserves or claim real benchmark success. Write only this separate CPU closure '
    'audit. A completed negative trial is not achievement of the single-model research goal.'
)


def layout(directory, gate):
    directory = Path(directory)
    names = {path.name for path in directory.iterdir()}
    lock = directory / 'run.lock'
    if (names != {'config.json', 'registration.json', 'status.json', 'gate.json', 'run.lock'}
            or lock.is_symlink() or not lock.is_file() or lock.stat().st_size != 0
            or lock.stat().st_nlink != 1):
        raise ValueError('Require only original metadata and its exact empty regular lock, never confirmation data')
    recovery.validate_negative(gate, names - {'run.lock'})


def original_closure():
    original = recovery.original
    recovery.preservation.require_cpu()
    recovery.verify()
    failed = recovery.read(recovery.OUTPUT / 'status.json')
    confirmation = recovery.read(original.CONFIRMATION / 'status.json')
    if (failed.get('state') != 'failed' or failed.get('error') != FAILURE
            or sha256(recovery.OUTPUT / 'config.json') != RECOVERY_CONFIG_SHA
            or sha256(recovery.OUTPUT / 'status.json') != RECOVERY_FAILURE_SHA
            or confirmation.get('state') != 'complete' or confirmation.get('confirmation_evaluated') is not False
            or original.process_identity(failed['pid']) is not None
            or original.process_identity(confirmation['pid']) is not None):
        raise ValueError('Require the stopped bookkeeping failure after the original successful negative gate')
    gate = recovery.read(original.CONFIRMATION / 'gate.json')
    if sha256(original.CONFIRMATION / 'gate.json') != GATE_SHA:
        raise ValueError('Never replace or recompute the original gate')
    layout(original.CONFIRMATION, gate)
    if original.CAMPAIGN.exists() or original.REVIEW.exists():
        raise ValueError('No real campaign belongs to this unqualified trial')
    return gate


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError('Never overwrite or repeat this closure audit')
    gate = original_closure()
    original = recovery.original
    with (original.CONFIRMATION / 'run.lock').open('r') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        layout(original.CONFIRMATION, gate)
        paths = [recovery.OUTPUT / name for name in ('config.json', 'registration.json', 'status.json', 'run.log', 'confirm.log')]
        paths.extend(original.CONFIRMATION.iterdir())
        paths.extend([original.OUTPUT / 'status.json', original.AUDIT / 'audit.json', original.REPLAY / 'audit.json',
            recovery.preservation.OUTPUT / 'independent_recheck.json'])
        sources = [Path(__file__).resolve(), REPO / 'tests/test_single_model_v53_closure_audit.py']
        config = {'policy': POLICY, 'input_sha256': {str(path): sha256(path) for path in paths},
            'source_sha256': {str(path): sha256(path) for path in sources}, 'scientific_stages_to_run': []}
        OUTPUT.mkdir()
        write_json(OUTPUT / 'config.json', config)
        verify_hashes(config['input_sha256'])
        verify_hashes(config['source_sha256'])
        write_json(OUTPUT / 'result.json', {'complete': True, 'unqualified': True, 'gate': gate,
            'config_sha256': sha256(OUTPUT / 'config.json'), 'original_negative_gate_sha256': GATE_SHA,
            'training_audit_sha256': recovery.TRAINING_AUDIT_SHA, 'endpoint_audit_sha256': recovery.ENDPOINT_AUDIT_SHA,
            'original_failed_pipeline': str(original.OUTPUT), 'failed_capacity_recovery': str(recovery.OUTPUT),
            'normal_empty_lock_verified': True, 'confirmation_cases_generated': 0,
            'scientific_stages_repeated': False, 'campaign': None, 'review': None,
            'no_inference_ensemble': True, 'goal_success_not_assessed': True})
        write_json(OUTPUT / 'status.json', {'state': 'complete', 'pid': os.getpid(),
            'updated_unix': time.time(), 'confirmed': False})


if __name__ == '__main__':
    main()
