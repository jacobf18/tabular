"""Preserve three closed nonselected artifacts before the balanced-prior trial."""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types

import torch

import preflight_single_model_balanced_structured_components as probe
import relocate_single_model_all_imputation_column_mean_capacity as blueprint
from relocate_single_model_all_imputation_column_mean_capacity import readers, copying, sync_directory
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


ROOT = probe.ROOT
OUTPUT = ROOT / 'balanced_structured_capacity_relocation_v1'
DESTINATION = copying.SCRATCH / 'single_model_goal_20261005_owned/balanced_structured_capacity_v1'
TRAINING_ADMISSION_BYTES = blueprint.TRAINING_ADMISSION_BYTES
SELECTED, SELECTED_SHA256 = probe.PARENT, probe.PARENT_SHA256
TAIL_CONFIG_SHA256 = '11e32073be40fb14e715caa7689c2546f89453490c730d2b210bbce5f94b9ead'
TARGETS = {
    'all_imputation_column_mean_v52/latest_checkpoint.pth': 'tail',
    'general_native_retention_v47_development_diagnosis_v1/columns.csv': 'columns',
    'balanced_structured_components_preflight_v1/candidate_parameters.pt': 'vector',
}
STATES = {
    'all_imputation_column_mean_v52': 'training_complete',
    'all_imputation_column_mean_v52_audit_task_replay': 'complete',
    'all_imputation_column_mean_v52_endpoint_replay': 'complete',
    'all_imputation_column_mean_v52_confirmation': 'complete',
    'all_imputation_column_mean_v52_pipeline': 'complete',
    'balanced_structured_components_preflight_v1': 'complete',
}
POLICY = (
    'Storage only: preserve exact serialized bytes of the audited unqualified V52 step64/57-update '
    'tail, the completed V47 development diagnosis columns and the completed balanced-prior '
    'probe vector. Preserve all135775 diagnostic column-boundary rows including21835 empty-query '
    'rows, every adverse outcome and all audit limitations. The V52 endpoint is already on '
    'scratch through its parent directory and cannot reclaim checkpoint capacity. Never move actual V29, original '
    'teachers, selected checkpoints or failed V28 evidence. Require stopped owners, unlocked '
    'producer locks and no open readers. Copy/fsync/hash/typed-load before atomic canonical '
    'symlink publication and separate independent recheck. Restore18 GiB plus2432 MiB admission '
    'without lowering18/6/50 GiB reserves. Scratch is machine-local, not external or provider-'
    'persistent backup. No training, inference, seed, selection, qualification or efficacy claim.'
)


def validate_payload(payload, kind, *, full_parameters=77998496, model_tensors=254, selected_model=None):
    if kind == 'vector':
        return blueprint.validate_payload(payload, kind, full_parameters=full_parameters)
    if kind == 'full_replay':
        return blueprint.blueprint.validate_payload(payload, 'decoder_replay', decoder_parameters=full_parameters)
    if kind != 'tail':
        raise ValueError('Move only the three declared nonselected artifact kinds')
    if (not isinstance(payload, dict) or payload.get('step') != 64 or payload.get('best_step') != 0
            or payload.get('gradient_updates_performed') != 57 or payload.get('attempted_batches') != 64
            or payload.get('config_sha256') != TAIL_CONFIG_SHA256
            or payload.get('parent_checkpoint_sha256') != SELECTED_SHA256
            or payload.get('independent_confirmation_passed') is not False
            or payload.get('preflight_only') is not False or payload.get('full_network_training') is not True
            or not payload.get('optimizer') or not isinstance(payload.get('model'), dict)
            or len(payload['model']) != model_tensors or selected_model is None
            or set(selected_model) != set(payload['model'])
            or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in payload['model'].values())
            or all(torch.equal(value, selected_model[name]) for name, value in payload['model'].items())):
        raise ValueError('Preserve only the distinct audited unqualified V52 tail, never actual V29')


def evidence():
    identities = probe.require_completed_v52()
    probe.verify(probe.OUTPUT)
    paths = [ROOT / 'all_imputation_column_mean_v52_audit/audit.json', probe.OUTPUT / 'audit.json',
        probe.OUTPUT.with_name(probe.OUTPUT.name + '_mean_audit') / 'audit.json',
        ROOT / 'all_imputation_column_mean_v52_endpoint_replay/audit.json']
    reports = [read(path) for path in paths]
    tail, feasibility, mean_audit, endpoint = reports
    retention, gate = read(probe.OUTPUT / 'retention_audit.json'), read(probe.OUTPUT / 'gate.json')
    if (tail['complete'] is not True or tail['through_step'] != 64 or tail['best']['step'] != 0
            or tail['accepted_updates'] != 57 or feasibility['complete'] is not True
            or feasibility['training_tasks'] != 28 or feasibility['prediction_grids'] != 56
            or gate['passed'] is not True or gate['preflight_only'] is not True
            or retention['complete'] is not True or retention['reference_grids'] != 26
            or retention['component_grids'] != 50 or retention['component_role_counts'] != {'v8': 24, 'v29': 26}
            or mean_audit['complete'] is not True or mean_audit['tasks'] != 28
            or mean_audit['imputation_reweighted_tasks'] != 26 or mean_audit['structured_reweighted_tasks'] != 24
            or mean_audit['native_reweighted_tasks'] != 2 or mean_audit['unchanged_rating_tasks'] != 2
            or endpoint['complete'] is not True or endpoint['endpoint_grids'] != 56 or endpoint['selected_grids'] != 0
            or endpoint['cuda_initialized'] is not False or endpoint['autograd_replayed'] is not False
            or endpoint['research_selection_tolerance'] != 0 or endpoint['goal_success_not_assessed'] is not True):
        raise ValueError('Require complete original V52 and balanced-prior probe audits without qualification')
    hashes = {path: digest for report in reports for path, digest in report['artifact_sha256'].items()}
    directory = ROOT / 'general_native_retention_v47_development_diagnosis_v1'
    diagnostic_identities = {
        str(directory / 'audit.json'): 'c17bd6606868cf1e3c31f70cfc3a314879ff220db8471e23ccf08b8da5c72fd7',
        str(directory / 'summary.json'): '69795c68753c7762a1575f0597d85df113f36ce3ed39fa186df8a8004897356c',
    }
    verify_hashes(diagnostic_identities)
    summary, audit, configuration, current = [read(directory / name) for name in
        ('summary.json', 'audit.json', 'config.json', 'status.json')]
    if (summary['complete'] is not True or audit['complete'] is not True or current['state'] != 'complete'
            or audit['summary_sha256'] != diagnostic_identities[str(directory / 'summary.json')]
            or current['summary_sha256'] != audit['summary_sha256']
            or summary['config_sha256'] != sha256(directory / 'config.json')
            or summary['imputation_columns'] != 135775 or summary['empty_query_columns'] != 21835
            or audit['column_boundaries'] != 135775 or audit['imputation_case_boundaries'] != 680
            or audit['csv_regrouping_recomputed'] is not True or audit['cuda_initialized'] is not False
            or audit['model_inference_performed'] is not False or audit['raw_grid_replay_performed'] is not False):
        raise ValueError('Preserve the complete original CPU-only V47 column diagnosis')
    verify_hashes(configuration['source_sha256'])
    verify_hashes(configuration['input_sha256'])
    verify_hashes(summary['artifact_sha256'])
    hashes.update(summary['artifact_sha256'])
    identities.update(diagnostic_identities)
    identities.update(configuration['input_sha256'])
    paths.extend(directory / name for name in ('config.json', 'status.json'))
    paths.extend(ROOT / name / 'status.json' for name in STATES)
    paths.extend(probe.OUTPUT / name for name in ('config.json', 'registration.json', 'result.json',
        'retention_audit.json', 'gate.json', 'external_recheck.json', 'OUTCOME.md'))
    paths.extend((SELECTED, ROOT / 'all_imputation_column_mean_v52/config.json'))
    identities.update({str(path): sha256(path) for path in paths})
    verify_hashes(identities)
    return hashes, identities


for name in ('read', 'status', 'require_selected', 'require_stopped', 'validate_file', 'inventory', 'copy_and_verify'):
    function = getattr(blueprint, name)
    if function.__closure__ is not None:
        raise ValueError('Preservation callbacks must bind the exact three-artifact scope')
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)


validate_tensor_file = validate_file


def validate_columns(path, *, expected_rows=135775, expected_cases=680, expected_empty=21835):
    count, empty = 0, 0
    cases = set()
    with Path(path).open() as stream:
        reader = csv.DictReader(stream)
        if not {'step', 'index', 'column', 'queries', 'candidate_common_contribution'}.issubset(reader.fieldnames or ()):
            raise ValueError('Preserve the original diagnostic CSV schema')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Preserve complete CSV rows')
            step, index, column, queries = (int(row[key]) for key in ('step', 'index', 'column', 'queries'))
            if step not in (0, 10, 25, 50, 64) or min(index, column, queries) < 0:
                raise ValueError('Preserve original boundaries and nonnegative column metadata')
            cases.add((step, index))
            count += 1
            empty += queries == 0
    if (count, len(cases), empty) != (expected_rows, expected_cases, expected_empty):
        raise ValueError('Preserve every column-boundary including empty-query rows')


def validate_file(path, kind):
    if kind == 'columns':
        return validate_columns(path)
    return validate_tensor_file(path, kind)


RECHECK_SOURCE = replace_once(blueprint.RECHECK_SOURCE, "'canonical_artifacts_checked': 2", "'canonical_artifacts_checked': 3")
exec(compile(RECHECK_SOURCE, __file__, 'exec'), globals())
PRESERVE_SOURCE = replace_once(blueprint.PRESERVE_SOURCE,
    'tests/test_single_model_all_imputation_column_mean_capacity.py', 'tests/test_single_model_balanced_structured_capacity.py')
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
            if receipt.exists():
                raise FileExistsError('Never overwrite independent preservation evidence')
            write_json(receipt, recheck())
