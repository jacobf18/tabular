"""Preserve all twelve rejected displacement-grid checkpoints before a new full trial."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types

import torch

import preflight_single_model_balanced_native_coverage as probe
import relocate_single_model_all_imputation_column_mean_capacity as blueprint
from relocate_single_model_all_imputation_column_mean_capacity import copying, readers, sync_directory
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


ROOT = probe.ROOT
OUTPUT = ROOT / 'balanced_native_capacity_relocation_v1'
DESTINATION = copying.SCRATCH / 'single_model_goal_20261005_owned/balanced_native_capacity_v1'
TRAINING_ADMISSION_BYTES = blueprint.TRAINING_ADMISSION_BYTES
SELECTED, SELECTED_SHA256 = probe.PARENT, probe.PARENT_SHA256
GRID = [{'id': f'candidate_{index:02d}', 'alpha': alpha, 'beta': beta}
    for index, (alpha, beta) in enumerate((alpha, beta)
        for alpha in (.1, .25, .4) for beta in (0., .025, .05, .1))]
TARGETS = {f'displacement_grid_v1/{candidate["id"]}/checkpoint.pth': candidate['id'] for candidate in GRID}
STATES = {'displacement_grid_v1': 'rejected', 'displacement_grid_v1_qualification': 'complete',
    'balanced_native_coverage_preflight_v1': 'complete'}
PROBE_CONFIG_SHA256 = '1b3b5fae260b205702287b5fa858b199531b82d5fee138823f6b393e24e9416b'
GRID_CONFIG_SHA256 = '5699cfa321205611e8987aebe6100360487d13fd9cc2810ef24b352ef247a780'
GRID_HASHES = {
    str(ROOT / 'displacement_grid_v1/config.json'): GRID_CONFIG_SHA256,
    str(ROOT / 'displacement_grid_v1/build.json'): '3dd0ba8ea6a84702c939db16536addfe13c78eed03fc0e3879901560aac358ee',
    str(ROOT / 'displacement_grid_v1_qualification/gate.json'): 'fe19ea8593d59343dd5967bc10875a5cb5f322ab77f2838aa36e001a48167652',
    str(ROOT / 'displacement_grid_v1_qualification/selection/audit.json'): '1fa8e0263e8d91df162ca311f97fa833d00eb0979be92323a7ff4b9aa56cc3c8',
    str(ROOT / 'displacement_grid_v1_qualification/selection/decision.json'): '39a5434e826ecd2dc0d511729730f2fd6522c2c73cfa9e0d4729f3c8870683fc',
}
POLICY = (
    'Storage only after the full-context native-coverage probe closes accepted with all original '
    'component/endpoint/mean audits and independent terminal review. Preserve all twelve original '
    'displacement-grid candidate checkpoints: every candidate failed the original development '
    'selection, none was selected or confirmed. Keep their derived step400/source-update semantics '
    'and zero new training updates. Preserve all3,744,936,228 serialized bytes, original canonical '
    'paths and metadata; never move actual V29, V54, any selected model or teacher. Require stopped '
    'owners, unlocked original locks and no open readers. Copy/fsync/hash/typed-load every complete '
    'checkpoint before publishing any canonical symlink, then independently recheck in a separate '
    'CPU process. Preserve18/6/50 GiB reserves and restore18 GiB plus2432 MiB full-trial admission. '
    'Scratch is UUID-bound machine-local storage, not external or provider-persistent backup. No '
    'evidence deletion, scientific inference, new seed, training, selection, relaxed guard, scientific '
    'retry or qualification claim. This operation does not authorize a full trial.'
)


def require_stopped():
    for name, expected in STATES.items():
        state = read(ROOT / name / 'status.json')
        if (state.get('state') != expected or type(state.get('pid')) is not int or state['pid'] <= 0
                or probe.closed.original.process_identity(state['pid']) is not None):
            raise ValueError('Preserve all stopped original owners, including the rejected grid')
        for filename in ('run.lock', 'build.lock', 'qualification.lock'):
            path = ROOT / name / filename
            if path.exists():
                with path.open('rb') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def validate_payload(payload, kind, *, model_tensors=254, model_elements=77998688):
    expected = next((candidate for candidate in GRID if candidate['id'] == kind), None)
    if (expected is None or not isinstance(payload, dict) or payload.get('candidate') != expected
            or payload.get('config_sha256') != GRID_CONFIG_SHA256 or payload.get('step') != 400
            or payload.get('epoch') != 400 or payload.get('derived_checkpoint') is not True
            or payload.get('gradient_updates_performed_here') != 0 or 'optimizer' in payload
            or not isinstance(payload.get('model'), dict) or len(payload['model']) != model_tensors
            or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                for value in payload['model'].values())
            or sum(value.numel() for value in payload['model'].values()) != model_elements):
        raise ValueError('Preserve only complete original derived-grid states with zero new training')


def validate_file(path, kind):
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    validate_payload(payload, kind)


def validate_grid(config, build, audit, decision, gate):
    identifiers = {candidate['id'] for candidate in GRID}
    if (config['grid'] != GRID or config['gradient_updates_performed_here'] != 0
            or build['complete'] is not True or set(build['candidates']) != identifiers
            or build['single_parameter_set_per_candidate'] is not True or build['cuda_initialized'] is not False
            or audit['complete'] is not True or audit['phase'] != 'selection' or audit['cases'] != 152
            or audit['prediction_grids_rescored'] != 2280 or audit['candidates_audited'] != 12
            or set(audit['arithmetic']) != identifiers or audit['research_selection_tolerance'] != 0
            or audit['cuda_initialized'] is not False or audit['goal_success_not_assessed'] is not True
            or audit['decision_sha256'] != GRID_HASHES[str(ROOT / 'displacement_grid_v1_qualification/selection/decision.json')]
            or decision['selected'] is not None or decision['eligible_candidates'] != []
            or decision['candidates_evaluated'] != 12 or decision['passed'] is not False
            or gate['passed'] is not False or gate['selected_candidate'] is not None
            or gate['checkpoint_sha256'] is not None or gate['selected_step'] != 0
            or gate['confirmation_evaluated'] is not False or gate['real_benchmark_success'] is not False
            or gate['gradient_updates_performed_here'] != 0 or gate['derived_checkpoint'] is not True):
        raise ValueError('Preserve the complete independently audited rejection of all twelve candidates')
    for candidate in GRID:
        identity = candidate['id']
        built, checked = build['candidates'][identity], audit['arithmetic'][identity]
        if (any(checked[key] != value for key, value in built.items() if key != 'checkpoint_sha256')
                or checked['tensors'] != 254 or checked['elements'] != 77998688
                or checked['changed_tensors'] != 238 or checked['arithmetic_independently_reconstructed'] is not True
                or checked['allowed_output_ulps'] != 1
                or audit['artifact_sha256'][str(ROOT / 'displacement_grid_v1' / identity / 'checkpoint.pth')]
                    != built['checkpoint_sha256']):
            raise ValueError('Bind every candidate to its original arithmetic audit and exact byte identity')


def validate_probe(gate, audit, retention, mean, review):
    if (gate['passed'] is not True or gate['preflight_only'] is not True or gate['real_benchmark_success'] is not False
            or audit['complete'] is not True or audit['training_tasks'] != 28 or audit['prediction_grids'] != 56
            or audit['teacher_replay']['component_grids'] != 52 or audit['cuda_initialized'] is not False
            or retention['complete'] is not True or retention['reference_grids'] != 26
            or retention['component_grids'] != 50 or retention['component_role_counts'] != {'v8': 24, 'v29': 26}
            or mean['complete'] is not True or mean['tasks'] != 28 or mean['imputation_reweighted_tasks'] != 26
            or mean['unchanged_rating_tasks'] != 2 or mean['cuda_initialized'] is not False
            or review['complete'] is not True or review['accepted'] is not True
            or review['config_sha256'] != PROBE_CONFIG_SHA256 or review['mean_tasks'] != 28
            or review['goal_achieved'] is not False or review['full_trial_authorized'] is not False):
        raise ValueError('Require the original accepted full-context probe and every completed independent audit')


def evidence():
    identities = {**GRID_HASHES, str(probe.OUTPUT / 'config.json'): PROBE_CONFIG_SHA256,
        str(SELECTED): SELECTED_SHA256}
    verify_hashes(identities)
    grid, build, audit, decision, gate = [read(ROOT / name) for name in (
        'displacement_grid_v1/config.json', 'displacement_grid_v1/build.json',
        'displacement_grid_v1_qualification/selection/audit.json',
        'displacement_grid_v1_qualification/selection/decision.json', 'displacement_grid_v1_qualification/gate.json')]
    validate_grid(grid, build, audit, decision, gate)
    review = read(probe.OUTPUT / 'external_recheck.json')
    mean_path = probe.OUTPUT.with_name(probe.OUTPUT.name + '_mean_audit') / 'audit.json'
    validate_probe(read(probe.OUTPUT / 'gate.json'), read(probe.OUTPUT / 'audit.json'),
        read(probe.OUTPUT / 'retention_audit.json'), read(mean_path), review)
    verify_hashes(review['input_report_sha256'])
    verify_hashes(review['review_source_sha256'])
    for mapping in (audit['artifact_sha256'], review['input_report_sha256'], review['review_source_sha256']):
        for path, digest in mapping.items():
            if path in identities and identities[path] != digest:
                raise ValueError('Never reconcile changed original grid or feasibility identities')
            identities[path] = digest
    paths = [ROOT / name / 'status.json' for name in STATES]
    paths.extend(ROOT / 'displacement_grid_v1' / filename for filename in ('registration.json', 'qualification.json'))
    paths.extend(probe.OUTPUT / filename for filename in ('registration.json', 'external_recheck.json', 'OUTCOME.md'))
    paths.extend((copying.UNIT, mean_path))
    identities.update({str(path): sha256(path) for path in paths})
    verify_hashes(identities)
    hashes = {str(ROOT / name): build['candidates'][kind]['checkpoint_sha256'] for name, kind in TARGETS.items()}
    return hashes, identities


for name in ('read', 'status', 'require_selected', 'inventory', 'copy_and_verify'):
    function = getattr(blueprint, name)
    if function.__closure__ is not None:
        raise ValueError('Storage helpers must bind the exact rejected-grid scope')
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
RECHECK_SOURCE = replace_once(blueprint.RECHECK_SOURCE, "'canonical_artifacts_checked': 2", "'canonical_artifacts_checked': 12")
exec(compile(RECHECK_SOURCE, __file__, 'exec'), globals())
PRESERVE_SOURCE = replace_once(blueprint.PRESERVE_SOURCE,
    'tests/test_single_model_all_imputation_column_mean_capacity.py', 'tests/test_single_model_balanced_native_capacity.py')
exec(compile(PRESERVE_SOURCE, __file__, 'exec'), globals())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preserve', 'recheck'), required=True)
    arguments = parser.parse_args()
    with probe.diagnosis.original.arithmetic_recovery.previous.original_arithmetic():
        if arguments.stage == 'preserve':
            preserve()
        else:
            receipt = OUTPUT / 'external_recheck.json'
            if receipt.exists() or receipt.is_symlink():
                raise FileExistsError('Never overwrite independent preservation evidence')
            write_json(receipt, recheck())
