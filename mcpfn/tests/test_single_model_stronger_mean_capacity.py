import gzip

import pytest
import torch

import relocate_single_model_stronger_mean_capacity as capacity


def family_payload(role):
    return {'model': {'matrix': torch.ones(3)}, 'step': 1, 'epoch': 1, 'gradient_updates_performed': 1,
            'preflight_only': True, 'selection_qualified': False, 'role': role}


@pytest.mark.parametrize('role', ['projected', 'unprojected'])
@pytest.mark.parametrize('field,value', [(None, None), ('preflight_only', False), ('selection_qualified', True),
    ('step', 2), ('role', 'selected'), ('optimizer', {'state': {}})])
def test_only_original_nonpromotable_family_states_are_admitted(role, field, value):
    payload = family_payload(role)
    if field:
        payload[field] = value
        with pytest.raises(ValueError):
            capacity.validate_payload(payload, f'family_{role}', model_tensors=1)
    else:
        capacity.validate_payload(payload, f'family_{role}', model_tensors=1)


def test_full_vector_requires_every_finite_fp32_coordinate():
    capacity.validate_payload(torch.zeros(3), 'vector', full_parameters=3)
    for value in [torch.zeros(2), torch.zeros(3, dtype=torch.float64), torch.tensor([0., 1., float('nan')])]:
        with pytest.raises(ValueError):
            capacity.validate_payload(value, 'vector', full_parameters=3)


def test_terminal_tail_must_differ_from_the_preserved_selection():
    selected = {'matrix': torch.zeros(3)}
    payload = {'model': {'matrix': torch.ones(3)}, 'step': 64, 'best_step': 50,
        'gradient_updates_performed': 56, 'attempted_batches': 64, 'independent_confirmation_passed': False,
        'preflight_only': False, 'full_network_training': True, 'optimizer': {'state': {}}}
    capacity.validate_payload(payload, 'tail', model_tensors=1, selected_model=selected)
    with pytest.raises(ValueError, match='distinct'):
        capacity.validate_payload(payload, 'tail', model_tensors=1, selected_model=payload['model'])
    payload['gradient_updates_performed'] = 57
    with pytest.raises(ValueError):
        capacity.validate_payload(payload, 'tail', model_tensors=1, selected_model=selected)


def test_archive_crc_is_checked_without_extracting_or_removing_files(tmp_path):
    archive = tmp_path / 'fixtures.tar.gz'
    archive.write_bytes(gzip.compress(b'original completed fixture bytes'))
    capacity.validate_file(archive, 'test_archive')
    corrupted = bytearray(archive.read_bytes())
    corrupted[-8] ^= 1
    archive.write_bytes(corrupted)
    with pytest.raises(gzip.BadGzipFile):
        capacity.validate_file(archive, 'test_archive')


def test_exact_scope_excludes_unaudited_tail_and_all_selected_models():
    assert len(capacity.TARGETS) == 5
    assert not any('anchor_rehearsal_v28' in name or 'selected_checkpoint' in name or 'best_validation' in name for name in capacity.TARGETS)
    assert capacity.TRAINING_ADMISSION_BYTES == 21877489664
    assert capacity.SELECTED.parent.name == 'all_imputation_blend_v45'


def test_copy_refuses_existing_destination_and_never_publishes_bad_bytes(tmp_path, monkeypatch):
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.write_bytes(b'complete evidence')
    row = {'source': str(source), 'destination': str(destination), 'bytes': source.stat().st_size,
           'sha256': capacity.sha256(source), 'kind': 'vector'}
    checked = []
    monkeypatch.setattr(capacity, 'validate_file', lambda path, kind: checked.append((path, kind)))
    capacity.copy_and_verify(row)
    assert destination.read_bytes() == source.read_bytes()
    assert checked == [(destination, 'vector')]
    with pytest.raises(FileExistsError):
        capacity.copy_and_verify(row)
