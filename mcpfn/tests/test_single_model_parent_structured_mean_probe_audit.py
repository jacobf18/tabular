import inspect

import numpy as np
import pytest
import torch

import audit_single_model_parent_structured_mean_probe as audit
import audit_single_model_native_column_mean_probe as predecessor
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals
from test_single_model_column_anchor_retention import column_example


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("adverse", [False, True])
def test_unchanged_independent_math_covers_all_columns_endpoints_and_adverse_changes(native, adverse):
    target, mask = column_example()
    teacher = torch.nan_to_num(target)
    predictions = {"before": (teacher + .1 if adverse else teacher + 1.).numpy(),
                   "candidate": (teacher + .7).numpy()}
    rows, values = audit.describe(target, mask, teacher, predictions, native=native)
    assert (rows, values) == predecessor.describe(target, mask, teacher, predictions, native=native)
    assert len(rows) == target.shape[-1]
    assert rows[-1]["queries"] == rows[-1]["applied_coefficient"] == 0
    assert rows[2]["constant_multi_query"] and rows[3]["single_query"]
    assert (values["applied_change"] > 0) == adverse


@pytest.mark.parametrize("mutation", ["missing_endpoint", "nonfinite", "shape", "mask", "empty"])
def test_invalid_and_incomplete_saved_grids_fail_closed(mutation):
    target, mask = column_example()
    teacher = torch.nan_to_num(target)
    predictions = {"before": teacher.numpy().copy(), "candidate": teacher.numpy().copy()}
    if mutation == "missing_endpoint":
        predictions.pop("candidate")
    elif mutation == "nonfinite":
        predictions["candidate"][0, 0, 0] = np.nan
    elif mutation == "shape":
        predictions["before"] = predictions["before"][:, :1]
    elif mutation == "mask":
        mask = mask.float()
    else:
        mask.zero_()
    with pytest.raises(ValueError, match="complete"):
        audit.describe(target, mask, teacher, predictions, native=True)


def test_new_binding_retains_original_formula_without_changing_registered_sources():
    assert audit.describe.__code__ is predecessor.describe.__code__
    assert audit.describe.__globals__ is vars(audit)
    assert audit.probe.retention.__name__ == "single_model_parent_structured_retention"
    assert audit.OUTPUT.name == "parent_structured_retention_preflight_v1_mean_audit"
    assert audit.CONFIG_SHA256 != predecessor.CONFIG_SHA256
    assert 'config = probe.verify(probe.OUTPUT)' in audit.AUDIT_SOURCE
    assert 'process_identity' in audit.AUDIT_SOURCE
    assert 'Path(blueprint.__file__).resolve()' in audit.AUDIT_SOURCE
    assert 'test_single_model_parent_structured_mean_probe_audit.py' in audit.AUDIT_SOURCE
    for value in vars(audit).values():
        if inspect.isfunction(value) and inspect.unwrap(value).__globals__ is vars(audit):
            assert unresolved_globals(value) == []
