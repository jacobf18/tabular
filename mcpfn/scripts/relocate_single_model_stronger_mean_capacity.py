"""Preserve five closed, unselected artifacts before a stronger-mean full trial."""

import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch

import preflight_single_model_stronger_mean_distillation as probe
import recover_single_model_stronger_mean_audit as recovery
import relocate_single_model_idle_diagnostics_to_root as readers
import relocate_single_model_idle_diagnostics_to_scratch as copying
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from relocate_single_model_gradient_archive import sync_directory


ROOT = probe.ROOT
OUTPUT = ROOT / 'stronger_mean_capacity_relocation_v1'
DESTINATION = copying.SCRATCH / 'single_model_goal_20261005_owned/stronger_mean_capacity_v1'
TRAINING_ADMISSION_BYTES = 18 * 1024**3 + 2432 * 1024**2
SELECTED = ROOT / 'all_imputation_blend_v45/selected_checkpoint.pth'
SELECTED_SHA256 = probe.SELECTED_SHA256
TARGETS = {
    'all_imputation_blend_v45/latest_checkpoint.pth': 'tail',
    'stronger_mean_distillation_preflight_v1/candidate_parameters.pt': 'vector',
    'family_projection_preflight_v1/projected_step.pth': 'family_projected',
    'family_projection_preflight_v1/unprojected_step.pth': 'family_unprojected',
    'maintenance_20261007/completed_pytest_fixtures.tar.gz': 'test_archive',
}
CLOSURE_HASHES = {
    str(recovery.OUTPUT / 'recovery_result.json'): '4e3d75aa81aef0d488eee631f4fedff72d98658ae407f5e66a316ccb86303df3',
    str(ROOT / 'family_projection_preflight_v1_audit_recovery1/audit.json'):
        'cd2a2040c1ea64553bcb1bd67553870d145d0ceab1b67b094e4204f3adaf2434',
    str(ROOT / 'maintenance_20261007/pytest_archive_audit.json'):
        '733f86d91d64d3a49c3c176c5b5dba27bae6b0648abe5c7efd808b41efc76d5a',
}
POLICY = (
    'Storage only: preserve exact bytes of the distinct unselected V45 step64 optimizer tail, '
    'the nonpromotable stronger-mean probe vector, both independently audited family-projection '
    'feasibility checkpoints and the already verified completed-pytest-fixture tar.gz. Preserve '
    'both probes original failed statuses and their completed additive audit recoveries. The '
    'failed, unaudited V28 optimizer tail is explicitly excluded. Keep selected checkpoints, '
    'teachers, initializer and all scientific artifacts unchanged. Require stopped producers and '
    'no open readers; copy/fsync/hash and typed tensor or full gzip-CRC verification before atomic '
    'canonical symlinks. Independently recheck every source/destination and original proof. Restore '
    '18 GiB plus2432 MiB admission, keeping18/6/50 GiB reserves. UUID-bound scratch is machine-local '
    'storage, not external/provider-persistent backup. No deletion of evidence, training, seed, '
    'selection, qualification retry, inference change or benchmark-success claim.'
)


def read(path):
    return json.loads(Path(path).read_text())


def status(state, **fields):
    write_json(OUTPUT / 'status.json', {'state': state, 'pid': os.getpid(), 'updated_unix': time.time(), **fields})


def require_stopped():
    states = {
        'all_imputation_blend_v45': 'training_complete',
        'all_imputation_blend_v45_pipeline': 'complete',
        'all_imputation_blend_v45_endpoint_replay': 'complete',
        'all_imputation_blend_v45_confirmation': 'complete',
        'all_imputation_blend_v45_benchmarks': 'complete',
        'all_imputation_blend_v45_review': 'complete',
        'stronger_mean_distillation_preflight_v1': 'failed',
        'stronger_mean_distillation_preflight_v1_audit_recovery_v1': 'complete',
        'family_projection_preflight_v1': 'failed',
        'family_projection_preflight_v1_audit_recovery1': 'complete',
    }
    for name, expected in states.items():
        current = read(ROOT / name / 'status.json')
        if current['state'] != expected or probe.diagnosis.original.process_identity(current['pid']) is not None:
            raise ValueError('Require stopped owners and their unchanged complete or preserved-failure states')


def validate_payload(payload, kind, *, full_parameters=77998496, model_tensors=254, selected_model=None):
    if kind == 'vector':
        if (not isinstance(payload, torch.Tensor) or payload.shape != (full_parameters,)
                or payload.dtype != torch.float32 or not torch.isfinite(payload).all()):
            raise ValueError('Preserve every finite coordinate of the original full probe vector')
        return
    if kind not in {'tail', 'family_projected', 'family_unprojected'}:
        raise ValueError('Only explicitly declared nonselected tensor artifacts may move')
    if (not isinstance(payload, dict) or not isinstance(payload.get('model'), dict)
            or len(payload['model']) != model_tensors
            or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in payload['model'].values())):
        raise ValueError('Preserve complete finite model tensors without dropping buffers')
    if kind == 'tail':
        if (payload.get('step') != 64 or payload.get('best_step') != 50
                or payload.get('gradient_updates_performed') != 56 or payload.get('attempted_batches') != 64
                or payload.get('independent_confirmation_passed') is not False
                or payload.get('preflight_only') is not False or payload.get('full_network_training') is not True
                or not payload.get('optimizer') or selected_model is None
                or set(selected_model) != set(payload['model'])
                or all(torch.equal(value, selected_model[name]) for name, value in payload['model'].items())):
            raise ValueError('Move only the distinct audited unselected V45 terminal optimizer state')
    elif (payload.get('step') != 1 or payload.get('epoch') != 1 or payload.get('gradient_updates_performed') != 1
            or payload.get('preflight_only') is not True or payload.get('selection_qualified') is not False
            or payload.get('role') != kind.removeprefix('family_') or 'optimizer' in payload):
        raise ValueError('Preserve the original nonpromotable family-projection feasibility roles')


def require_selected():
    if SELECTED.is_symlink() or SELECTED.stat().st_dev != ROOT.stat().st_dev or sha256(SELECTED) != SELECTED_SHA256:
        raise ValueError('Keep the distinct selected model physically on checkpoint storage')


def validate_file(path, kind):
    if kind == 'test_archive':
        with gzip.open(path, 'rb') as source:
            for chunk in iter(lambda: source.read(8 * 1024**2), b''):
                pass
        return
    selected = None
    if kind == 'tail':
        require_selected()
        selected = torch.load(SELECTED, map_location='cpu', weights_only=True, mmap=True)['model']
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    validate_payload(payload, kind, selected_model=selected)


def evidence():
    identities = probe.require_completed_v45()
    verify_hashes(CLOSURE_HASHES)
    identities.update(CLOSURE_HASHES)
    probe.verify(probe.OUTPUT)
    recovered = read(recovery.OUTPUT / 'recovery_result.json')
    if (recovered['complete'] is not True or recovered['accepted'] is not True
            or recovered['original_failure_preserved'] is not True or recovered['original_inventory_unchanged'] is not True
            or any(recovered[key] is not False for key in ('training_repeated', 'model_or_reference_inference_repeated',
                'new_seeds_consumed', 'selection_or_benchmarks_evaluated', 'cuda_initialized', 'gate_changed'))):
        raise ValueError('Bind completed additive recovery, never relabel the failed original audit')
    identities.update({str(recovery.OUTPUT / 'recovery_config.json'): recovered['recovery_config_sha256'],
        str(recovery.OUTPUT / 'retention_audit.json'): recovered['retention_audit_sha256'],
        str(probe.OUTPUT / 'audit.json'): recovered['base_audit_sha256'],
        str(probe.OUTPUT / 'gate.json'): recovered['base_gate_sha256']})
    verify_hashes(identities)
    recovery_config = read(recovery.OUTPUT / 'recovery_config.json')
    verify_hashes(recovery_config['source_sha256'])
    verify_hashes(recovery_config['input_sha256'])
    retention = read(recovery.OUTPUT / 'retention_audit.json')
    if retention['reference_grids'] != 26 or retention['reference_role_counts'] != {'v8': 26}:
        raise ValueError('Retain every recovered V8 reference check')
    family = read(ROOT / 'family_projection_preflight_v1_audit_recovery1/audit.json')
    if (not family['complete'] or not family['recovery_complete'] or not family['original_artifacts_unchanged']
            or not family['original_failure_preserved'] or not family['preflight_only']
            or family['tasks'] != 14 or family['gradient_vectors_remeasured'] != 14
            or family['checkpoint_selection_performed'] is not False or family['training_restarted'] is not False
            or family['selection_tolerance_changed'] is not False or family['cuda_initialized'] is not False):
        raise ValueError('Keep the complete original nonpromotable family-projection audit recovery')
    verify_hashes(family['artifact_sha256'])
    verify_hashes(family['recovery_input_sha256'])
    archive = read(ROOT / 'maintenance_20261007/pytest_archive_audit.json')
    if (not archive['complete'] or not archive['gzip_integrity_checked'] or not archive['tar_compared_to_all_original_files']
            or archive['scientific_evidence_or_baselines_removed'] is not False or archive['registered_input_references'] != []):
        raise ValueError('Move only the previously verified complete test-fixture archive')
    tail = read(ROOT / 'all_imputation_blend_v45_audit/audit.json')
    hashes = {**tail['artifact_sha256'], **recovery_config['input_sha256'], **family['artifact_sha256'],
              archive['archive']: archive['archive_sha256']}
    paths = [SELECTED, ROOT / 'all_imputation_blend_v45/status.json', probe.OUTPUT / 'status.json',
             recovery.OUTPUT / 'status.json', ROOT / 'family_projection_preflight_v1/status.json',
             ROOT / 'family_projection_preflight_v1_audit_recovery1/status.json']
    identities.update({str(path): sha256(path) for path in paths})
    return hashes, identities


def inventory():
    copying.require_mount()
    require_stopped()
    require_selected()
    hashes, inputs = evidence()
    records = []
    for name, kind in TARGETS.items():
        source = ROOT / name
        if (source.is_symlink() or not source.is_file() or source.stat().st_nlink != 1
                or source.stat().st_dev != ROOT.stat().st_dev or source.samefile(SELECTED)
                or str(source) not in hashes or sha256(source) != hashes[str(source)]):
            raise ValueError('Move only the exact audited single-link nonselected artifacts')
        validate_file(source, kind)
        records.append({'source': str(source), 'destination': str(DESTINATION / name), 'kind': kind,
                        'bytes': source.stat().st_size, 'sha256': hashes[str(source)]})
    readers.require_no_artifact_readers([row['source'] for row in records])
    return records, inputs


def copy_and_verify(row):
    source, destination = Path(row['source']), Path(row['destination'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as reader, destination.open('xb') as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024**2)
        writer.flush()
        os.fsync(writer.fileno())
    sync_directory(destination.parent)
    if destination.stat().st_size != row['bytes'] or sha256(destination) != row['sha256']:
        raise ValueError('Never publish an inexact preservation copy')
    validate_file(destination, row['kind'])


def recheck():
    config = read(OUTPUT / 'config.json')
    if (read(OUTPUT / 'registration.json')['config_sha256'] != sha256(OUTPUT / 'config.json')
            or config['policy'] != POLICY or config['scratch_uuid'] != copying.UUID):
        raise ValueError('Preserve the registered storage-only operation')
    copying.require_mount()
    require_stopped()
    require_selected()
    verify_hashes(config['source_sha256'])
    verify_hashes(config['input_sha256'])
    hashes = {}
    for row in config['records']:
        source, destination = Path(row['source']), Path(row['destination'])
        if not source.is_symlink() or source.resolve() != destination or destination.is_symlink():
            raise ValueError('Preserve every canonical path to its original exact bytes')
        validate_file(source, row['kind'])
        hashes.update({str(source): row['sha256'], str(destination): row['sha256']})
    verify_hashes(hashes)
    if (shutil.disk_usage(ROOT).free < TRAINING_ADMISSION_BYTES or shutil.disk_usage('/').free < 6 * 1024**3
            or shutil.disk_usage(copying.SCRATCH).free < 50 * 1024**3):
        raise OSError('Restore full training admission without lowering reserves')
    return {'complete': True, 'canonical_artifacts_checked': 5, 'artifact_sha256': hashes,
        'all_original_serialized_bytes_and_paths_preserved': True, 'selected_checkpoints_unchanged': True,
        'bytes_relocated': sum(row['bytes'] for row in config['records']),
        'checkpoint_available_bytes': shutil.disk_usage(ROOT).free, 'provider_persistent_backup': False,
        'config_sha256': sha256(OUTPUT / 'config.json'), 'training_admission_bytes': TRAINING_ADMISSION_BYTES}


def preserve():
    if OUTPUT.exists() or DESTINATION.exists():
        raise FileExistsError('Never restart or overwrite a begun preservation')
    records, inputs = inventory()
    total = sum(row['bytes'] for row in records)
    if (shutil.disk_usage(ROOT).free < 18 * 1024**3 + 16 * 1024**2
            or shutil.disk_usage(ROOT).free + total < TRAINING_ADMISSION_BYTES + 16 * 1024**2
            or shutil.disk_usage('/').free < 6 * 1024**3 + 16 * 1024**2
            or shutil.disk_usage(copying.SCRATCH).free - total < 50 * 1024**3):
        raise OSError('Keep all reserves and enough verified recovery for training')
    sources = {str(Path(module.__file__).resolve()): sha256(module.__file__) for module in tuple(sys.modules.values())
        if getattr(module, '__file__', None) and str(module.__file__).endswith('.py')
        and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file()}
    test_path = REPO / 'tests/test_single_model_stronger_mean_capacity.py'
    sources[str(test_path)] = sha256(test_path)
    OUTPUT.mkdir()
    DESTINATION.mkdir(parents=True)
    write_json(OUTPUT / 'config.json', {'policy': POLICY, 'records': records, 'input_sha256': inputs,
        'source_sha256': sources, 'scratch_uuid': copying.UUID})
    write_json(OUTPUT / 'registration.json', {'config_sha256': sha256(OUTPUT / 'config.json'), 'registered_unix': time.time()})
    write_json(DESTINATION / 'OWNERSHIP.json', {'registered_at': str(OUTPUT), 'config_sha256': sha256(OUTPUT / 'config.json')})
    status('copying_closed_nonselected_artifacts')
    try:
        for row in records:
            copy_and_verify(row)
        require_stopped()
        readers.require_no_artifact_readers([row['source'] for row in records])
        for row in records:
            copying.publish(row)
        write_json(OUTPUT / 'independent_recheck.json', recheck())
        status('complete', bytes_relocated=total)
    except BaseException as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preserve', 'recheck'), required=True)
    arguments = parser.parse_args()
    with probe.diagnosis.original.arithmetic_recovery.previous.original_arithmetic():
        if arguments.stage == 'preserve':
            preserve()
        else:
            receipt = OUTPUT / 'external_recheck.json'
            if receipt.exists():
                raise FileExistsError('Never overwrite an independent preservation receipt')
            write_json(receipt, recheck())
