from copy import deepcopy
import types

import pytest

import test_single_model_balanced_structured_storage as previous_tests
import train_single_matrixformer_standardized_huber as training


SOURCE = previous_tests.SOURCE.replace('import single_model_balanced_structured_storage as diagnostics',
    'import single_model_standardized_huber_storage as diagnostics')
exec(compile(SOURCE, __file__, 'exec'), globals())
for name in ('test_complete_component_accounting_includes_original_and_replay_grids',
        'test_component_budget_rejects_incomplete_scope_and_invalid_shapes',
        'test_only_new_training_allowances_expand_and_previous_module_is_unchanged'):
    function = getattr(previous_tests, name)
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__dict__.update(function.__dict__)
    globals()[name] = cloned


def test_standardized_huber_preserves_complete_budgets_and_local_callbacks():
    assert diagnostics.WRITE_BUDGETS == previous_tests.diagnostics.WRITE_BUDGETS
    assert diagnostics.RESERVES == previous_tests.diagnostics.RESERVES
    assert diagnostics.component_bytes(training.batches()) == diagnostics.component_bytes(training.unvalidated_batches())
    for name in ('component_storage', 'prepare', 'verify', 'require_capacity'):
        assert getattr(diagnostics, name).__globals__ is vars(diagnostics)
