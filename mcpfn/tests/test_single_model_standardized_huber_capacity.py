from copy import deepcopy

import pytest
import torch

import relocate_single_model_standardized_huber_capacity as capacity
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def payload(kind):
    result = {'model': {'weight': torch.ones(3)}, 'optimizer': {'state': {}}, 'best_step': 0}
    if kind == 'tail':
        result.update(step=64, gradient_updates_performed=59, attempted_batches=64,
            config_sha256=capacity.TAIL_CONFIG_SHA256, parent_checkpoint_sha256=capacity.SELECTED_SHA256,
            independent_confirmation_passed=False, preflight_only=False, full_network_training=True)
    else:
        result.update(step=10, epoch=10, config_sha256=capacity.HISTORICAL_CONFIG_SHA256,
            bank_sha256='6fafd887113c99861921bd51fe72e941e4fc260c8932ef26153553667b76e0c8')
    return result


@pytest.mark.parametrize('kind', ['tail', 'historical_training'])
@pytest.mark.parametrize('mutation', [None, 'step', 'best_step', 'config', 'optimizer', 'selected', 'nonfinite', 'unknown'])
def test_preserve_only_original_distinct_optimizer_snapshots(kind, mutation):
    value = payload(kind)
    selected = {'weight': torch.zeros(3)}
    if mutation == 'step':
        value['step'] += 1
    elif mutation == 'best_step':
        value['best_step'] = value['step']
    elif mutation == 'config':
        value['config_sha256'] = 'changed'
    elif mutation == 'optimizer':
        value.pop('optimizer')
    elif mutation == 'selected':
        selected = deepcopy(value['model'])
    elif mutation == 'nonfinite':
        value['model']['weight'][0] = float('nan')
    elif mutation == 'unknown':
        kind = 'selected'
    if mutation:
        with pytest.raises(ValueError):
            capacity.validate_payload(value, kind, model_tensors=1, selected_model=selected)
    else:
        capacity.validate_payload(value, kind, model_tensors=1, selected_model=selected)


@pytest.mark.parametrize('key,value', [('gradient_updates_performed', 58), ('attempted_batches', 63),
    ('parent_checkpoint_sha256', 'changed'), ('independent_confirmation_passed', True),
    ('preflight_only', True), ('full_network_training', False)])
def test_tail_preserves_v53_negative_outcome(key, value):
    current = payload('tail')
    current[key] = value
    with pytest.raises(ValueError):
        capacity.validate_payload(current, 'tail', model_tensors=1, selected_model={'weight': torch.zeros(3)})


def test_original_failures_and_selected_model_stay_unchanged():
    assert len(capacity.TARGETS) == 2
    assert capacity.STATES['balanced_structured_components_v53_pipeline'] == 'failed'
    assert capacity.STATES['balanced_structured_components_v53_capacity_recovery'] == 'failed'
    assert capacity.STATES['balanced_structured_components_v53_closure_audit'] == 'complete'
    assert capacity.SELECTED_SHA256 == '9747c69624220de33356145c5df9baa5256bc625719868d4f0e3740bce475c32'
    assert "'canonical_artifacts_checked': 2" in capacity.RECHECK_SOURCE
    assert capacity.TRAINING_ADMISSION_BYTES == 18 * 1024**3 + 2432 * 1024**2


def test_cloned_callbacks_are_fully_bound():
    for function in vars(capacity).values():
        if callable(function) and getattr(function, '__globals__', None) is vars(capacity):
            assert unresolved_globals(function) == []


def test_exact_copy_refuses_overwrite(tmp_path, monkeypatch):
    source, destination = tmp_path / 'original.pt', tmp_path / 'copy.pt'
    source.write_bytes(b'exact original serialized bytes')
    row = {'source': str(source), 'destination': str(destination), 'bytes': source.stat().st_size,
        'sha256': capacity.sha256(source), 'kind': 'tail'}
    monkeypatch.setattr(capacity, 'validate_file', lambda path, kind: None)
    capacity.copy_and_verify(row)
    assert destination.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        capacity.copy_and_verify(row)
