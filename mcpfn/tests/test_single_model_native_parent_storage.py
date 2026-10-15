from copy import deepcopy
import types

import pytest

import test_single_model_dual_parent_storage as previous_tests
import train_single_matrixformer_native_parent as training


SOURCE = previous_tests.SOURCE.replace("import single_model_dual_parent_storage as diagnostics",
    "import single_model_native_parent_storage as diagnostics")
exec(compile(SOURCE, __file__, "exec"), globals())

for name in ("test_complete_component_accounting_includes_original_and_replay_grids",
        "test_component_budget_rejects_incomplete_scope_and_invalid_shapes"):
    function = getattr(previous_tests, name)
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__dict__.update(function.__dict__)
    globals()[name] = cloned


def test_native_parent_retains_every_allowance_and_binds_local_callbacks():
    assert diagnostics.WRITE_BUDGETS == previous_tests.diagnostics.WRITE_BUDGETS
    assert diagnostics.RESERVES == previous_tests.diagnostics.RESERVES
    for name in ("component_storage", "prepare", "verify", "require_capacity"):
        assert getattr(diagnostics, name).__globals__ is vars(diagnostics)


def test_only_new_training_allowances_expand_and_previous_module_is_unchanged(planned):
    assert diagnostics.RESERVES == diagnostics.template.RESERVES
    assert diagnostics.RESERVES is not diagnostics.template.RESERVES
    assert diagnostics.WRITE_BUDGETS == diagnostics.template.WRITE_BUDGETS
    assert diagnostics.WRITE_BUDGETS is not diagnostics.template.WRITE_BUDGETS
    assert diagnostics.WRITE_BUDGETS["training"] == {
        "checkpoint": 2432 * 1024**2, "root": 16 * 1024**2, "scratch": 48 * 1024**3}
    diagnostics.require_plan(planned, "training")
    changed = deepcopy(planned)
    changed["stage_write_budget_bytes"]["training"]["scratch"] = 40 * 1024**3
    with pytest.raises(ValueError):
        diagnostics.require_plan(changed, "training")
