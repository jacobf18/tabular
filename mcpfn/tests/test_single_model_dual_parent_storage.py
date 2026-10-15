from copy import deepcopy

import pytest

import test_single_model_all_imputation_storage as previous_tests
import train_single_matrixformer_dual_parent as training


SOURCE = previous_tests.SOURCE
SOURCE = SOURCE.replace("import single_model_all_imputation_storage as diagnostics",
    "import single_model_dual_parent_storage as diagnostics")
SOURCE = SOURCE.replace('"scratch": 40 * 1024**3}', '"scratch": 48 * 1024**3}')
SOURCE = SOURCE.replace('"checkpoint": 2304 * 1024**2, "root": 16 * 1024**2, "scratch": 32768 * 1024**2}',
    '"checkpoint": 2432 * 1024**2, "root": 16 * 1024**2, "scratch": 40 * 1024**3}')
exec(compile(SOURCE, __file__, "exec"), globals())


def test_complete_component_accounting_includes_original_and_replay_grids():
    accounting = diagnostics.component_storage(training.batches())
    assert accounting["tasks"] == 1664
    assert accounting["components"] == 3200
    assert accounting["archived_component_bytes"] == accounting["replay_component_bytes"] == 4 * accounting["component_cells"]
    assert diagnostics.component_bytes(training.batches()) == sum(value for key, value in accounting.items()
        if key not in {"tasks", "components", "component_cells"})


@pytest.mark.parametrize("mutation", ["batch", "task", "zero", "boolean", "dimension"])
def test_component_budget_rejects_incomplete_scope_and_invalid_shapes(mutation):
    batches = deepcopy(training.batches())
    if mutation == "batch":
        batches.pop()
    elif mutation == "task":
        batches[0]["tasks"].pop(0)
    else:
        specification = next(task["specification"] for task in batches[0]["tasks"]
            if task["specification"]["domain"] == "imputation")
        specification = specification.get("structured", specification)
        specification["shape"] = {"zero": [0, 8], "boolean": [True, 8], "dimension": [8]}[mutation]
    with pytest.raises(ValueError):
        diagnostics.component_storage(batches)
