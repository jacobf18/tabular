import os

import pytest

import recover_single_model_stronger_mean_audit as recovery


def evidence():
    failed = {'state': 'failed', 'failed_stage': 'audit', 'error': 'NameError("name \'Counter\' is not defined")'}
    result = {'data_admission_only': False, 'preflight_only': True, 'trainable_parameters': 77998496,
              'record': {'accepted': True, 'accepted_updates': 1, 'reason': 'accepted'}}
    base = {'complete': True, 'training_tasks': 28, 'prediction_grids': 56, 'cuda_initialized': False,
            'autograd_replayed': False, 'raw_endpoint': {'gradient_vectors': 84, 'accepted_updates': 1},
            'frozen': {'frozen_tensors': 16}, 'teacher_replay': {'complete': True, 'full_grids': 28,
                'blend_tasks': 26, 'component_grids': 52, 'observed_weights_refitted': True}}
    replay = {'complete': True, 'records': [{} for index in range(26)]}
    gate = {'passed': True, 'preflight_only': True, 'reason': 'accepted', 'real_benchmark_success': False,
            'audit_sha256': recovery.BASE_AUDIT_SHA256}
    return failed, result, base, replay, gate, "NameError: name 'Counter' is not defined"


@pytest.mark.parametrize('entry,key,value', [(0, 'error', 'other error'), (0, 'failed_stage', 'run'),
    (1, 'data_admission_only', True), (1, 'record', {'accepted': False, 'accepted_updates': 0, 'reason': 'rejected'}),
    (2, 'complete', False), (2, 'prediction_grids', 55), (2, 'cuda_initialized', True),
    (2, 'teacher_replay', {'complete': True, 'full_grids': 28, 'blend_tasks': 25,
                         'component_grids': 50, 'observed_weights_refitted': True}),
    (3, 'records', [{} for index in range(25)]), (4, 'passed', False),
    (4, 'preflight_only', False), (4, 'audit_sha256', 'changed')])
def test_only_exact_audit_tail_failure_and_accepted_proposal_are_admitted(entry, key, value):
    reports = list(evidence())
    recovery.validate_failure(*reports)
    reports[entry][key] = value
    with pytest.raises(ValueError):
        recovery.validate_failure(*reports)


def test_view_never_redirects_original_inputs_or_allows_escape(tmp_path):
    original, destination = tmp_path / 'original', tmp_path / 'recovery'
    view = recovery.AuditView(original, destination)
    assert os.fspath(view) == str(original)
    for name in recovery.AUDIT_FILES:
        assert view / name == destination / name
    for name in ['config.json', 'result.json', 'raw_update.pt', 'retention_replay/reference_1.npz']:
        assert view / name == original / name
    for name in ['../escape', '/absolute']:
        with pytest.raises(ValueError):
            view / name


def test_counter_is_supplied_only_in_isolated_namespace_and_code_is_unchanged(tmp_path):
    original = dict(vars(recovery.probe))
    assert recovery.missing_globals(recovery.probe.audit, original) == {'Counter'}
    current = recovery.namespace(tmp_path)
    assert recovery.missing_globals(current['audit'], current) == set()
    assert current['audit'].__code__ is original['audit'].__code__
    assert current['audit'].__globals__ is current
    assert recovery.probe.audit is original['audit']
    assert 'Counter' not in vars(recovery.probe)
    for name in ['verify', 'retention_records', 'status']:
        assert current[name].__code__ is original[name].__code__
    with pytest.raises(ValueError, match='only the new'):
        current['write_json'](tmp_path / 'audit.json', {'complete': True})
    current['write_json'](tmp_path / 'retention_audit.json', {'complete': True})
    with pytest.raises(FileExistsError):
        current['write_json'](tmp_path / 'retention_audit.json', {'complete': True})


def test_base_audit_reuse_requires_exact_copied_bytes_and_bound_inputs(tmp_path, monkeypatch):
    original, destination = tmp_path / 'original', tmp_path / 'recovery'
    original.mkdir()
    destination.mkdir()
    payload = b'{"artifact_sha256": {}, "teacher_replay": {"artifact_sha256": {}}}'
    for directory in [original, destination]:
        (directory / 'audit.json').write_bytes(payload)
        (directory / 'gate.json').write_bytes(b'{"passed": true}')
    monkeypatch.setattr(recovery, 'ORIGINAL', original)
    monkeypatch.setattr(recovery, 'BASE_AUDIT_SHA256', recovery.sha256(original / 'audit.json'))
    monkeypatch.setattr(recovery, 'BASE_GATE_SHA256', recovery.sha256(original / 'gate.json'))
    current = recovery.namespace(destination)
    current['audit_original'](recovery.AuditView(original, destination))
    (destination / 'audit.json').write_bytes(payload + b' ')
    with pytest.raises(ValueError):
        current['audit_original'](recovery.AuditView(original, destination))
    with pytest.raises(ValueError, match='exact completed'):
        current['audit_original'](original)


def test_cpu_visibility_and_initialization_are_both_required(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError):
        recovery.require_cpu()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(recovery.torch.cuda, 'is_initialized', lambda: False)
    recovery.require_cpu()
    monkeypatch.setattr(recovery.torch.cuda, 'is_initialized', lambda: True)
    with pytest.raises(ValueError):
        recovery.require_cpu()
