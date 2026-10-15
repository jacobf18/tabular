import pytest

import recover_single_model_v53_capacity as recovery


def original_states():
    return [
        {'state': 'failed', 'error': recovery.FAILURE},
        {'state': 'training_complete', 'step': 64, 'accepted_updates': 59, 'best': {'step': 0}},
        {'complete': True, 'accepted_updates': 59, 'training_episodes': 1792, 'best': {'step': 0}},
        {'complete': True, 'endpoint_grids': 56, 'selected_grids': 0, 'cuda_initialized': False},
        {'state': 'complete', 'selected_validation_passed': False},
    ]


def test_original_unqualified_state_accepted():
    recovery.validate_original(*original_states())


@pytest.mark.parametrize('index,field,value', [
    (0, 'state', 'complete'), (0, 'error', 'unrelated failure'),
    (1, 'state', 'running'), (1, 'step', 63), (1, 'accepted_updates', 58), (1, 'best', {'step': 50}),
    (2, 'complete', False), (2, 'accepted_updates', 58), (2, 'training_episodes', 1791), (2, 'best', {'step': 10}),
    (3, 'complete', False), (3, 'endpoint_grids', 55), (3, 'selected_grids', 152), (3, 'cuda_initialized', True),
    (4, 'state', 'awaiting_independent_audit'), (4, 'selected_validation_passed', True),
])
def test_changed_scope_rejected(index, field, value):
    states = original_states()
    states[index][field] = value
    with pytest.raises(ValueError, match='exact audited, unqualified'):
        recovery.validate_original(*states)


def negative_gate():
    return {'passed': False, 'selected_step': 0, 'confirmation_evaluated': False,
        'reason': recovery.REASON, 'real_benchmark_success': False}


def test_exact_original_negative_gate_accepted():
    recovery.validate_negative(negative_gate(), ['config.json', 'registration.json', 'status.json', 'gate.json'])


@pytest.mark.parametrize('field,value', [('passed', True), ('selected_step', 64),
    ('confirmation_evaluated', True), ('reason', 'manually rejected'), ('real_benchmark_success', True)])
def test_changed_gate_rejected(field, value):
    gate = negative_gate()
    gate[field] = value
    with pytest.raises(ValueError, match='exact original negative'):
        recovery.validate_negative(gate, ['config.json', 'registration.json', 'status.json', 'gate.json'])


@pytest.mark.parametrize('extra', ['synthetic_bank.pt', 'manifest.json', 'student_predictions', 'unrelated.json'])
def test_confirmation_consumption_rejected(extra):
    with pytest.raises(ValueError, match='without consuming confirmation'):
        recovery.validate_negative(negative_gate(), ['config.json', 'registration.json', 'status.json', 'gate.json', extra])


def test_existing_recovery_not_restarted(monkeypatch, tmp_path):
    monkeypatch.setattr(recovery, 'OUTPUT', tmp_path)
    with pytest.raises(FileExistsError, match='Never repeat'):
        recovery.register()


def test_only_original_confirm_cli_is_launched(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(recovery, 'verify', lambda: None)
    monkeypatch.setattr(recovery, 'require_unused', lambda: None)
    monkeypatch.setattr(recovery.original, 'CONFIRMATION', tmp_path)
    monkeypatch.setattr(recovery.original, 'run_child', lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(recovery, 'read', lambda path: negative_gate() if path.name == 'gate.json' else {'state': 'complete'})
    monkeypatch.setattr(recovery, 'validate_negative', lambda *args: None)
    monkeypatch.setattr(recovery, 'write_json', lambda *args: None)
    monkeypatch.setattr(recovery, 'sha256', lambda path: 'original-failure')
    monkeypatch.setattr(recovery, 'status', lambda *args, **kwargs: None)
    recovery.run()
    assert len(calls) == 1
    arguments, options = calls[0]
    assert arguments[1] == 'confirm'
    assert arguments[2].resolve() == recovery.original.Path(recovery.original.__file__).resolve()
    assert arguments[3] == ['--stage', 'confirm', '--output', str(recovery.original.OUTPUT)]
    assert options == {'cpu': False}
