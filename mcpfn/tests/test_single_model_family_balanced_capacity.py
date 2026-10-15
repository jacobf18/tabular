import builtins
import inspect

import pytest

import relocate_single_model_family_balanced_capacity as relocation
import test_single_model_full_network_capacity as inherited
import test_single_model_cohort_capacity as cohort_tests
from recover_single_model_legacy_strata_replay import code_globals


@pytest.mark.parametrize('mutation', [None, 'live', 'updates', 'attempts', 'config', 'qualified',
    'preflight', 'scope', 'optimizer', 'selection'])
def test_unselected_v42_tail_keeps_exact_identity(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(inherited, 'relocation', relocation)
    inherited.test_each_tail_requires_its_exact_original_terminal_identity(tmp_path, monkeypatch, 'unselected_tail', mutation)


@pytest.mark.parametrize('count', [4, 6])
def test_vector_copy_keeps_bytes_and_canonical_path(tmp_path, monkeypatch, count):
    monkeypatch.setattr(inherited.inherited, 'relocation', relocation)
    inherited.inherited.test_vector_copy_and_publication_keep_exact_bytes_and_inventory(tmp_path, monkeypatch, count)


@pytest.mark.parametrize('entry,path,value', [
    (0, ('record', 'accepted'), False), (1, ('teacher_replay', 'full_grids'), 27),
    (1, ('teacher_replay', 'component_grids'), 3), (2, ('reference_grids',), 25),
    (2, ('reference_role_counts',), {'v8': 3, 'v29': 23}), (2, ('endpoint_penalties',), []),
    (3, ('passed',), False), (4, ('state',), 'running')])
def test_requires_complete_accepted_family_balanced_probe(monkeypatch, entry, path, value):
    monkeypatch.setattr(relocation, 'sha256', lambda path: 'digest')
    records = cohort_tests.probe_records()
    records[0]['record']['accepted'] = True
    relocation.validate_probe(*records)
    target = records[entry]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        relocation.validate_probe(*records)


def test_scope_reserves_and_all_generated_bindings():
    assert set(relocation.OWNERS) == {'cohort_anchor_retention_v42', 'cohort_anchor_retention_preflight_v1',
        'family_balanced_retention_preflight_v1'}
    assert relocation.OWNERS['cohort_anchor_retention_v42']['selected'] == 0
    assert relocation.OWNERS['cohort_anchor_retention_v42']['updates'] == 58
    assert relocation.TRAINING_ADMISSION_BYTES == 21877489664
    assert '"canonical_artifacts_checked": 3' in relocation.MAIN_SOURCE
    assert relocation.PROBE_CONFIG_SHA256 == '25dbd1d2782ce45af86648695205ec3655b76c3ad57e8140dbfd21fb6c07c816'
    for name in ('inventory', 'inventory_original', 'checkpoint_identity', 'copy_and_verify', 'require_idle',
            'original_probe_evidence', 'main'):
        function = inspect.unwrap(getattr(relocation, name))
        assert function.__globals__ is vars(relocation) and function.__closure__ is None
        assert not (code_globals(function.__code__) - set(vars(relocation)) - set(vars(builtins)))
