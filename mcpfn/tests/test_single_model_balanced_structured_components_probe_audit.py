import inspect

import pytest
import torch

import audit_single_model_balanced_structured_components_probe as audit
import audit_single_model_native_column_mean_probe as original_math
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals
from test_single_model_anchor_retention import metadata
from test_single_model_column_anchor_retention import column_example


@pytest.mark.parametrize('column_weighted', [False, True])
@pytest.mark.parametrize('adverse', [False, True])
def test_saved_mean_math_preserves_every_column_and_adverse_direction(column_weighted, adverse):
    target, mask = column_example()
    teacher = torch.nan_to_num(target)
    predictions = {'before': (teacher + (.1 if adverse else 1.)).numpy(), 'candidate': (teacher + .7).numpy()}
    rows, values = audit.describe(target, mask, teacher, predictions, native=column_weighted)
    assert (rows, values) == original_math.describe(target, mask, teacher, predictions, native=column_weighted)
    assert len(rows) == target.shape[-1]
    assert rows[-1]['queries'] == rows[-1]['applied_coefficient'] == 0
    assert rows[2]['constant_multi_query'] and rows[3]['single_query']
    assert (values['applied_change'] > 0) == adverse


@pytest.mark.parametrize('family', audit.probe.components.FAMILIES)
@pytest.mark.parametrize('component', audit.probe.components.COMPONENTS)
def test_every_structured_component_and_family_remains_column_weighted(family, component):
    assert audit.imputation_task({'cohort': family}, metadata(component, structured=True, family=family))


@pytest.mark.parametrize('invalid', [None, '', 'a' * 63, 'a' * 65, 'g' * 64, 'A' * 64, 123])
def test_configuration_identity_is_required_before_any_audit(invalid):
    with pytest.raises(ValueError, match='exact original'):
        audit.audit(invalid)


def test_explicit_identity_is_local_and_does_not_mutate_module(monkeypatch):
    def implementation():
        return globals()['CONFIG_SHA256']

    monkeypatch.setattr(audit, '_audit', implementation)
    assert audit.audit('a' * 64) == 'a' * 64
    assert audit.CONFIG_SHA256 is None


def test_complete_scope_and_callbacks_are_preserved():
    assert audit.OUTPUT.name == 'balanced_structured_components_preflight_v1_mean_audit'
    for text in ('{True: 26, False: 2}', '"component_grids"] != 50', '"reference_grids"] != 26',
            '"component_role_counts"] != {"v8": 24, "v29": 26}', 'config = probe.verify(probe.OUTPUT)',
            'process_identity', 'test_single_model_balanced_structured_components_probe_audit.py'):
        assert text in audit.AUDIT_SOURCE
    assert audit.describe.__code__ is original_math.describe.__code__
    for function in vars(audit).values():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(audit):
            assert not unresolved_globals(function), function.__name__
