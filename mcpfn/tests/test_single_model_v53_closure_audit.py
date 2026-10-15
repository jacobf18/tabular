import pytest

import audit_single_model_v53_capacity_recovery as audit


def fixture_directory(tmp_path):
    for name in ('config.json', 'registration.json', 'status.json', 'gate.json', 'run.lock'):
        (tmp_path / name).write_text('' if name == 'run.lock' else '{}')
    return {'passed': False, 'selected_step': 0, 'confirmation_evaluated': False,
        'reason': audit.recovery.REASON, 'real_benchmark_success': False}


def test_exact_original_metadata_with_empty_lock(tmp_path):
    gate = fixture_directory(tmp_path)
    audit.layout(tmp_path, gate)


@pytest.mark.parametrize('extra', ['synthetic_bank.pt', 'manifest.json', 'predictions.npz', 'unrelated.json'])
def test_any_additional_data_rejected(tmp_path, extra):
    gate = fixture_directory(tmp_path)
    (tmp_path / extra).write_text('preserve')
    with pytest.raises(ValueError, match='never confirmation data'):
        audit.layout(tmp_path, gate)


def test_nonempty_lock_rejected(tmp_path):
    gate = fixture_directory(tmp_path)
    (tmp_path / 'run.lock').write_text('unexpected bytes')
    with pytest.raises(ValueError, match='empty regular lock'):
        audit.layout(tmp_path, gate)


def test_symlink_lock_rejected(tmp_path):
    gate = fixture_directory(tmp_path)
    (tmp_path / 'run.lock').unlink()
    (tmp_path / 'run.lock').symlink_to(tmp_path / 'gate.json')
    with pytest.raises(ValueError, match='empty regular lock'):
        audit.layout(tmp_path, gate)


def test_missing_lock_rejected(tmp_path):
    gate = fixture_directory(tmp_path)
    (tmp_path / 'run.lock').unlink()
    with pytest.raises(ValueError, match='empty regular lock'):
        audit.layout(tmp_path, gate)


@pytest.mark.parametrize('field,value', [('passed', True), ('selected_step', 50),
    ('confirmation_evaluated', True), ('reason', 'different'), ('real_benchmark_success', True)])
def test_original_negative_gate_must_remain_exact(tmp_path, field, value):
    gate = fixture_directory(tmp_path)
    gate[field] = value
    with pytest.raises(ValueError, match='exact original negative'):
        audit.layout(tmp_path, gate)


def test_existing_closure_cannot_be_overwritten(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, 'OUTPUT', tmp_path)
    with pytest.raises(FileExistsError, match='Never overwrite or repeat'):
        audit.main()
