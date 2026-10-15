"""Preserve two closed nonselected checkpoints before standardized-Huber training."""

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

import preflight_single_model_standardized_teacher_huber as probe
import relocate_single_model_all_imputation_column_mean_capacity as blueprint
from relocate_single_model_all_imputation_column_mean_capacity import readers, copying, sync_directory
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


ROOT = probe.ROOT
OUTPUT = ROOT / 'standardized_huber_capacity_relocation_v1'
DESTINATION = copying.SCRATCH / 'single_model_goal_20261005_owned/standardized_huber_capacity_v1'
TRAINING_ADMISSION_BYTES = blueprint.TRAINING_ADMISSION_BYTES
SELECTED, SELECTED_SHA256 = probe.PARENT, probe.PARENT_SHA256
TAIL_CONFIG_SHA256 = 'f1e90786a292bb5054ab0eb8ab7916aad5539ddb7da2e4bd3ca453ddbf64b5da'
HISTORICAL_CONFIG_SHA256 = 'd3f31f15bb94f54309c69754ce81c70486c75901f0a32372538c465bd5dab7c7'
HISTORICAL_SNAPSHOT_SHA256 = '2e33202648284fa3cacdd6970212d747dce6eeebbaeb826c464832eca35b0ca9'
TARGETS = {'balanced_structured_components_v53/latest_checkpoint.pth': 'tail',
    'area_mixture_v7/diagnostic_snapshots/training_step_10.pth': 'historical_training'}
STATES = {'balanced_structured_components_v53': 'training_complete',
    'balanced_structured_components_v53_audit_task_replay': 'complete',
    'balanced_structured_components_v53_endpoint_replay': 'complete',
    'balanced_structured_components_v53_confirmation': 'complete',
    'balanced_structured_components_v53_pipeline': 'failed',
    'balanced_structured_components_v53_capacity_recovery': 'failed',
    'balanced_structured_components_v53_closure_audit': 'complete',
    'standardized_teacher_huber_preflight_v1': 'complete', 'area_mixture_v7': 'training_complete'}
POLICY = (
    'Storage only: preserve exact serialized bytes of the audited unqualified V53 step64/'
    '59-update optimizer tail and the original V7 step10 optimizer diagnostic, whose original '
    'progress audit names its exact hash. V7 selected step300, not this early training snapshot. '
    'Bind V53 additive negative closure while preserving its two original failed workflows. '
    'Require complete accepted standardized-Huber feasibility and its original full/supplementary '
    'audits and terminal saved-evidence review. Keep actual V29, selected models, teachers, '
    'feasibility vectors and every scientific outcome untouched. Require stopped owners, '
    'unlocked original locks and no open readers; copy/fsync/hash/typed-load before canonical '
    'symlink publication and independent recheck. Restore18 GiB plus2432 MiB admission; preserve '
    '18/6/50 GiB reserves. No mounts, deletion of evidence, training, inference, seeds, '
    'selection, qualification or scientific retry. Scratch is machine-local, not external backup.'
)


def validate_payload(payload, kind, *, model_tensors=254, selected_model=None):
    if kind not in {'tail', 'historical_training'}:
        raise ValueError('Preserve only the two declared nonselected optimizer checkpoints')
    if (not isinstance(payload, dict) or not isinstance(payload.get('model'), dict)
            or len(payload['model']) != model_tensors or not payload.get('optimizer')
            or selected_model is None or set(selected_model) != set(payload['model'])
            or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in payload['model'].values())
            or all(torch.equal(value, selected_model[name]) for name, value in payload['model'].items())):
        raise ValueError('Preserve the complete finite distinct nonselected optimizer snapshot')
    if kind == 'tail':
        if (payload.get('step') != 64 or payload.get('best_step') != 0
                or payload.get('gradient_updates_performed') != 59 or payload.get('attempted_batches') != 64
                or payload.get('config_sha256') != TAIL_CONFIG_SHA256
                or payload.get('parent_checkpoint_sha256') != SELECTED_SHA256
                or payload.get('independent_confirmation_passed') is not False
                or payload.get('preflight_only') is not False or payload.get('full_network_training') is not True):
            raise ValueError('Preserve only the audited unqualified V53 tail')
    elif (payload.get('step') != 10 or payload.get('epoch') != 10 or payload.get('best_step') != 0
            or payload.get('config_sha256') != HISTORICAL_CONFIG_SHA256
            or payload.get('bank_sha256') != '6fafd887113c99861921bd51fe72e941e4fc260c8932ef26153553667b76e0c8'):
        raise ValueError('Preserve only the original V7 step10 diagnostic, never its selected model')


def validate_file(path, kind):
    require_selected()
    selected_path = SELECTED if kind == 'tail' else ROOT / 'area_mixture_v7/best_validation_checkpoint.pth'
    selected_model = torch.load(selected_path, map_location='cpu', weights_only=False, mmap=True)['model']
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    validate_payload(payload, kind, selected_model=selected_model)


def evidence():
    identities = probe.require_completed_v53()
    probe.verify(probe.OUTPUT)
    review = read(probe.OUTPUT / 'external_recheck.json')
    if (review.get('complete') is not True or review.get('accepted') is not True
            or review.get('config_sha256') != '7bf3acc2e92274eda8e5254546913f49353f1700091292d78b4c0ed93a0641d8'
            or review.get('mean_tasks') != 28 or review.get('goal_achieved') is not False
            or review.get('full_trial_authorized') is not False):
        raise ValueError('Require the complete accepted original feasibility review without efficacy claims')
    verify_hashes(review['input_report_sha256'])
    verify_hashes(review['review_source_sha256'])
    identities.update(review['input_report_sha256'])
    identities.update(review['review_source_sha256'])
    old = ROOT / 'area_mixture_v7'
    progress = read(old / 'progress_audit_10.json')
    if (progress.get('complete') is not True or progress.get('checkpoint_step') != 10
            or progress.get('checkpoint_sha256') != HISTORICAL_SNAPSHOT_SHA256
            or progress.get('config_sha256') != HISTORICAL_CONFIG_SHA256
            or progress.get('finite_weights_and_optimizer') is not True
            or read(old / 'status.json').get('best_step') != 300):
        raise ValueError('Bind the original nonselected V7 diagnostic and its progress audit')
    hashes = dict(read(ROOT / 'balanced_structured_components_v53_audit/audit.json')['artifact_sha256'])
    hashes[str(old / 'diagnostic_snapshots/training_step_10.pth')] = HISTORICAL_SNAPSHOT_SHA256
    paths = [ROOT / name / 'status.json' for name in STATES]
    paths.extend(probe.OUTPUT / name for name in ('config.json', 'registration.json', 'result.json', 'audit.json',
        'retention_audit.json', 'gate.json', 'external_recheck.json', 'OUTCOME.md'))
    paths.extend((SELECTED, old / 'config.json', old / 'progress_audit_10.json', old / 'best_validation_checkpoint.pth',
        ROOT / 'balanced_structured_components_v53/config.json'))
    identities.update({str(path): sha256(path) for path in paths})
    verify_hashes(identities)
    return hashes, identities


for name in ('read', 'status', 'require_selected', 'require_stopped', 'inventory', 'copy_and_verify'):
    function = getattr(blueprint, name)
    if function.__closure__ is not None:
        raise ValueError('Preservation callbacks must bind the exact new two-artifact scope')
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
RECHECK_SOURCE = blueprint.RECHECK_SOURCE
exec(compile(RECHECK_SOURCE, __file__, 'exec'), globals())
PRESERVE_SOURCE = replace_once(blueprint.PRESERVE_SOURCE,
    'tests/test_single_model_all_imputation_column_mean_capacity.py',
    'tests/test_single_model_standardized_huber_capacity.py')
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
