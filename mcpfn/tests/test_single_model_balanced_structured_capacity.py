from copy import deepcopy
import inspect

import pytest
import torch

import relocate_single_model_balanced_structured_capacity as capacity
import test_single_model_parent_structured_capacity as predecessor_tests
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


def test_exact_closed_three_artifact_scope_and_reserves():
    assert len(capacity.TARGETS) == 3
    assert set(capacity.TARGETS.values()) == {'tail', 'vector', 'columns'}
    assert capacity.SELECTED == capacity.probe.PARENT
    assert capacity.TRAINING_ADMISSION_BYTES == 18 * 1024**3 + 2432 * 1024**2
    assert not any('v28' in name or 'selected_checkpoint' in name for name in capacity.TARGETS)
    for function in vars(capacity).values():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(capacity):
            assert unresolved_globals(function) == []


@pytest.mark.parametrize('mutation', [None, 'step', 'best_step', 'gradient_updates_performed', 'config_sha256',
    'parent_checkpoint_sha256', 'independent_confirmation_passed', 'full_network_training', 'selected', 'kind'])
def test_only_distinct_unqualified_v52_tail_is_eligible(mutation):
    payload = {'step': 64, 'best_step': 0, 'gradient_updates_performed': 57, 'attempted_batches': 64,
        'config_sha256': capacity.TAIL_CONFIG_SHA256, 'parent_checkpoint_sha256': capacity.SELECTED_SHA256,
        'independent_confirmation_passed': False, 'preflight_only': False, 'full_network_training': True,
        'optimizer': {'state': {}}, 'model': {'weight': torch.ones(3)}}
    selected, kind = {'weight': torch.zeros(3)}, 'tail'
    if mutation == 'kind':
        kind = 'selected'
    elif mutation == 'selected':
        selected = deepcopy(payload['model'])
    elif mutation:
        payload[mutation] = {'step': 50, 'best_step': 64, 'gradient_updates_performed': 58,
            'config_sha256': 'changed', 'parent_checkpoint_sha256': 'changed',
            'independent_confirmation_passed': True, 'full_network_training': False}[mutation]
    if mutation:
        with pytest.raises(ValueError):
            capacity.validate_payload(payload, kind, model_tensors=1, selected_model=selected)
    else:
        capacity.validate_payload(payload, kind, model_tensors=1, selected_model=selected)


@pytest.mark.parametrize('mutation', [None, 'missing', 'dtype', 'shape', 'nonfinite', 'products', 'product_nan', 'norm'])
def test_full_reconstruction_keeps_all_coordinates_products_and_norm(mutation):
    payload = {'parameters': torch.zeros(3), 'products': [0.] * 84, 'norm': .1}
    if mutation == 'missing':
        payload.pop('norm')
    elif mutation in {'dtype', 'shape', 'nonfinite'}:
        payload['parameters'] = {'dtype': torch.zeros(3).double(), 'shape': torch.zeros(2),
            'nonfinite': torch.full((3,), float('nan'))}[mutation]
    elif mutation == 'products':
        payload['products'].pop()
    elif mutation == 'product_nan':
        payload['products'][0] = float('nan')
    elif mutation == 'norm':
        payload['norm'] = -1.
    if mutation:
        with pytest.raises(ValueError):
            capacity.validate_payload(payload, 'full_replay', full_parameters=3)
    else:
        capacity.validate_payload(payload, 'full_replay', full_parameters=3)


def test_exact_copy_and_vector_contracts_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(predecessor_tests, 'capacity', capacity)
    predecessor_tests.test_vectors_keep_full_float32_finite_coordinates()
    predecessor_tests.test_preservation_copy_is_exact_and_refuses_overwrite(tmp_path, monkeypatch)


@pytest.mark.parametrize('mutation', [None, 'schema', 'short_row', 'step', 'negative', 'missing', 'empty'])
def test_diagnostic_csv_preserves_complete_and_empty_columns(tmp_path, mutation):
    path = tmp_path / 'columns.csv'
    rows = ['step,index,column,queries,candidate_common_contribution', '0,0,0,3,0.1', '10,0,0,0,0.0']
    if mutation == 'schema':
        rows[0] = rows[0].replace('queries', 'changed')
    elif mutation == 'short_row':
        rows[1] = '0,0,0,3'
    elif mutation == 'step':
        rows[1] = '1,0,0,3,0.1'
    elif mutation == 'negative':
        rows[1] = '0,0,-1,3,0.1'
    elif mutation == 'missing':
        rows.pop()
    elif mutation == 'empty':
        rows[2] = '10,0,0,1,0.0'
    path.write_text('\n'.join(rows) + '\n')
    if mutation:
        with pytest.raises(ValueError):
            capacity.validate_columns(path, expected_rows=2, expected_cases=2, expected_empty=1)
    else:
        capacity.validate_columns(path, expected_rows=2, expected_cases=2, expected_empty=1)


def test_inventory_rejects_files_already_on_another_device(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pathlib import Path

    source, selected = tmp_path / 'artifact.pt', tmp_path / 'selected.pt'
    source.write_bytes(b'exact artifact')
    selected.write_bytes(b'selected')
    monkeypatch.setattr(capacity, 'ROOT', tmp_path)
    monkeypatch.setattr(capacity, 'SELECTED', selected)
    monkeypatch.setattr(capacity, 'TARGETS', {'artifact.pt': 'vector'})
    for name in ('require_stopped', 'require_selected'):
        monkeypatch.setattr(capacity, name, lambda: None)
    monkeypatch.setattr(capacity.copying, 'require_mount', lambda: None)
    monkeypatch.setattr(capacity, 'evidence', lambda: ({str(source): capacity.sha256(source)}, {}))
    original = Path.stat

    def stat(path, *arguments, **keywords):
        result = original(path, *arguments, **keywords)
        if path == source:
            return SimpleNamespace(st_dev=result.st_dev + 1, st_nlink=1, st_mode=result.st_mode)
        return result

    monkeypatch.setattr(Path, 'stat', stat)
    with pytest.raises(ValueError, match='single-link nonselected'):
        capacity.inventory()
