from pathlib import Path

import test_single_model_balanced_native_capacity as tests_blueprint
from single_model_registered_adaptation import replace_once


SOURCE = Path(tests_blueprint.__file__).read_text()
SOURCE = replace_once(SOURCE, 'import relocate_single_model_balanced_native_capacity as capacity',
    'import relocate_single_model_balanced_native_quarter_capacity as capacity')
SOURCE = replace_once(SOURCE, "'teacher_replay': {'component_grids': 52}, 'cuda_initialized': False}",
    "'teacher_replay': {'component_grids': 52}, 'cuda_initialized': False,\n"
    "        'raw_endpoint': {'fixed_quarter_step': {'fraction': .25, 'parameters': 77998496,\n"
    "            'actual_saved_displacements_checked': True}, 'accepted': True,\n"
    "            'actual_final_parameters_checked': True},\n"
    "        'frozen': {'changed_trainable_tensors': list(range(238)), 'frozen_tensors': 16,\n"
    "            'unselected_parameters_and_all_buffers_bitwise_unchanged': True}}")
exec(compile(SOURCE, __file__, 'exec'), globals())


@pytest.mark.parametrize('path,value', [
    ('raw_endpoint.fixed_quarter_step.fraction', .5),
    ('raw_endpoint.fixed_quarter_step.parameters', 1),
    ('raw_endpoint.fixed_quarter_step.actual_saved_displacements_checked', False),
    ('raw_endpoint.accepted', False), ('raw_endpoint.actual_final_parameters_checked', False),
    ('frozen.changed_trainable_tensors', []), ('frozen.frozen_tensors', 15),
    ('frozen.unselected_parameters_and_all_buffers_bitwise_unchanged', False)])
def test_preservation_requires_a_complete_actual_quarter_update(path, value):
    reports = complete_probe()
    destination = reports[1]
    fields = path.split('.')
    for field in fields[:-1]:
        destination = destination[field]
    destination[fields[-1]] = value
    with pytest.raises(ValueError, match='accepted full-context probe'):
        capacity.validate_probe(*reports)


def test_exact_quarter_registration_and_both_original_owners_remain_bound():
    assert capacity.PROBE_CONFIG_SHA256 == '9b8418c5e0ee88fb64b9a163efa31594927fccc386419986cb0fd0e6f0c859e4'
    assert capacity.probe.OUTPUT.name == 'balanced_native_quarter_preflight_v1'
    assert capacity.STATES['balanced_native_coverage_preflight_v1'] == 'complete'
    assert capacity.STATES['balanced_native_quarter_preflight_v1'] == 'complete'
    assert capacity.OUTPUT.name == 'balanced_native_quarter_capacity_relocation_v1'
    assert capacity.DESTINATION.name == 'balanced_native_quarter_capacity_v1'
    assert capacity.native_blueprint.PROBE_CONFIG_SHA256 != capacity.PROBE_CONFIG_SHA256
    assert capacity.TARGETS == capacity.native_blueprint.TARGETS


@pytest.mark.parametrize('pid,live', [(1234, False), (1234, True), (None, False), (True, False),
    (0, False), (-1, False), ('1234', False)])
def test_stopped_owner_check_uses_actual_quarter_probe_namespace(monkeypatch, tmp_path, pid, live):
    monkeypatch.setattr(capacity, 'ROOT', tmp_path)
    monkeypatch.setattr(capacity, 'read', lambda path: {'state': capacity.STATES[path.parent.name], 'pid': pid})
    checked = []
    def process_identity(value):
        checked.append(value)
        return {'alive': True} if live else None
    monkeypatch.setattr(capacity.probe.probe_blueprint.closed.original, 'process_identity', process_identity)
    if type(pid) is int and pid > 0 and not live:
        capacity.require_stopped()
        assert checked == [pid] * len(capacity.STATES)
    else:
        with pytest.raises(ValueError, match='stopped original owners'):
            capacity.require_stopped()
