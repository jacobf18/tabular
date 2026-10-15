from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import relocate_single_model_legacy_root_banks as preservation


def bank_fixture():
    bank = [{'source': torch.tensor([[[1., float('nan')], [2., 3.]]]),
        'target': torch.ones(1, 2, 2), 'mask': torch.tensor([[[False, True], [False, False]]]),
        'metadata': {'seed': 123, 'gradient_data': 'synthetic', 'domain': 'imputation'}}]
    reference = {'cases': [{'index': 0, 'seed': 123, 'gradient_data': 'synthetic',
        'domain': 'imputation', 'rows': 2, 'columns': 2}]}
    return bank, reference


@pytest.mark.parametrize('fault', [None, 'count', 'index', 'seed', 'domain', 'real', 'target_nan',
    'source_inf', 'mask_dtype', 'shape', 'empty_mask'])
def test_complete_typed_bank_and_original_case_identity(fault):
    bank, reference = bank_fixture()
    if fault == 'count':
        bank.clear()
    elif fault == 'index':
        reference['cases'][0]['index'] = 1
    elif fault in ('seed', 'domain', 'real'):
        key, value = {'seed': ('seed', 124), 'domain': ('domain', 'recsys'), 'real': ('gradient_data', 'real')}[fault]
        bank[0]['metadata'][key] = value
    elif fault == 'target_nan':
        bank[0]['target'][0, 0, 0] = float('nan')
    elif fault == 'source_inf':
        bank[0]['source'][0, 0, 0] = float('inf')
    elif fault == 'mask_dtype':
        bank[0]['mask'] = bank[0]['mask'].float()
    elif fault == 'shape':
        bank[0]['target'] = torch.ones(1, 3, 2)
    elif fault == 'empty_mask':
        bank[0]['mask'].zero_()
    if fault:
        with pytest.raises(ValueError):
            preservation.validate_bank(bank, reference, 1)
    else:
        before = deepcopy(bank)
        preservation.validate_bank(bank, reference, 1)
        torch.testing.assert_close(bank[0]['source'], before[0]['source'], equal_nan=True)


def test_copy_checks_before_publishing_and_keeps_source_bytes(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.write_bytes(b'original serialized bytes')
    destination = tmp_path / 'preserved' / 'bank.pt'
    row = {'source': str(source), 'destination': str(destination), 'bytes': source.stat().st_size, 'name': 'owner'}
    checked = []
    monkeypatch.setattr(preservation, 'check_file', lambda path, name: checked.append((path.read_bytes(), name)))
    preservation.copy_and_verify(row)
    assert checked == [(b'original serialized bytes', 'owner')]
    assert source.read_bytes() == destination.read_bytes() and not source.is_symlink()
    with pytest.raises(FileExistsError):
        preservation.copy_and_verify(row)


def test_wrong_copy_size_cannot_reach_typed_validation(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.write_bytes(b'original')
    monkeypatch.setattr(preservation, 'check_file', lambda *arguments: pytest.fail('Inexact bytes reached validation'))
    with pytest.raises(ValueError, match='incomplete'):
        preservation.copy_and_verify({'source': str(source), 'destination': str(tmp_path / 'copy' / 'bank.pt'),
            'bytes': 1000, 'name': 'owner'})
    assert source.read_bytes() == b'original'


@pytest.mark.parametrize('short_volume', [None, 'root', 'checkpoint', 'scratch'])
def test_all_original_reserves_and_root_headroom_are_mandatory(monkeypatch, short_volume):
    amounts = {'root': 6 * 1024**3 + 16 * 1024**2, 'checkpoint': 18 * 1024**3,
        'scratch': 50 * 1024**3 + preservation.BYTES}
    if short_volume:
        amounts[short_volume] -= 1

    def usage(path):
        label = 'root' if str(path) == '/' else 'checkpoint' if path == preservation.ROOT else 'scratch'
        return SimpleNamespace(free=amounts[label])

    monkeypatch.setattr(preservation.shutil, 'disk_usage', usage)
    if short_volume:
        with pytest.raises(OSError):
            preservation.require_reserves(preservation.BYTES)
    else:
        preservation.require_reserves(preservation.BYTES)


def test_exact_scope_excludes_every_model_checkpoint():
    assert len(preservation.OWNERS) == 4
    assert sum(value[1] for value in preservation.OWNERS.values()) == 148
    assert preservation.BYTES == 327541419 + 327536299 + 182325847 + 182301719
    assert all(len(value[2]) == 64 for value in preservation.OWNERS.values())
