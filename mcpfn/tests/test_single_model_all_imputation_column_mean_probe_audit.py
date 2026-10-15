import inspect

import numpy as np
import pytest
import torch

import audit_single_model_all_imputation_column_mean_probe as audit
import audit_single_model_native_column_mean_probe as original_math
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals
from test_single_model_anchor_retention import metadata
from test_single_model_column_anchor_retention import column_example


@pytest.mark.parametrize("column_weighted", [False, True])
@pytest.mark.parametrize("adverse", [False, True])
def test_independent_arithmetic_preserves_all_columns_and_adverse_changes(column_weighted, adverse):
    target, mask = column_example()
    teacher = torch.nan_to_num(target)
    predictions = {"before": (teacher + .1 if adverse else teacher + 1.).numpy(), "candidate": (teacher + .7).numpy()}
    rows, values = audit.describe(target, mask, teacher, predictions, native=column_weighted)
    assert (rows, values) == original_math.describe(target, mask, teacher, predictions, native=column_weighted)
    assert len(rows) == target.shape[-1]
    assert rows[-1]["queries"] == rows[-1]["applied_coefficient"] == 0
    assert rows[2]["constant_multi_query"] and rows[3]["single_query"]
    assert (values["applied_change"] > 0) == adverse


@pytest.mark.parametrize("component", range(8))
def test_all_native_components_remain_weighted(component):
    fields = metadata(component, prior="anchor_latent_profile_mixed" if component == 7 else "anchor_latent_mcar")
    assert audit.imputation_task({"cohort": "legacy"}, fields)
    with pytest.raises(ValueError):
        audit.imputation_task({"cohort": "mar"}, fields)


@pytest.mark.parametrize("family", audit.probe.retention.FAMILIES)
@pytest.mark.parametrize("component", ("low_rank_mcar", "slow_tail_mcar", "steep_tail_mcar", "anchor_latent_mcar"))
def test_every_structured_component_is_weighted_but_ratings_are_not(family, component):
    fields = metadata(component, structured=True, family=family)
    assert audit.imputation_task({"cohort": family}, fields)
    assert not audit.imputation_task({"cohort": "recsys"}, metadata(domain="recsys"))
    with pytest.raises(ValueError):
        audit.imputation_task({"cohort": "legacy"}, fields)


@pytest.mark.parametrize("mutation", ["missing_endpoint", "nonfinite", "shape", "mask", "empty"])
def test_invalid_or_incomplete_grids_are_rejected(mutation):
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


def test_complete_registered_scope_and_original_gate_are_bound():
    classifications = []
    for task in audit.probe.batch()["tasks"]:
        cohort = task["cohort"]
        fields = metadata() if cohort == "legacy" else metadata(domain="recsys") if cohort == "recsys" else metadata(
            "low_rank_mcar", structured=True, family=cohort)
        classifications.append(audit.imputation_task(task, fields))
    assert len(classifications) == 28 and sum(classifications) == 26
    assert audit.CONFIG_SHA256 == "669d4b633a362e5e577771ed48d7838253a3ced0ea7e496d76087bc7c93f6d11"
    assert audit.OUTPUT.name == "all_imputation_column_mean_preflight_v1_mean_audit"
    assert '{True: 26, False: 2}' in audit.AUDIT_SOURCE
    assert '"component_role_counts"] != {"v8": 24, "v29": 26}' in audit.AUDIT_SOURCE
    assert '"component_grids"] != 50' in audit.AUDIT_SOURCE
    assert '"reference_grids"] != 26' in audit.AUDIT_SOURCE
    assert 'imputation_teacher_mean_tasks' in audit.AUDIT_SOURCE
    assert 'test_single_model_all_imputation_column_mean_probe_audit.py' in audit.AUDIT_SOURCE
    assert 'config = probe.verify(probe.OUTPUT)' in audit.AUDIT_SOURCE
    assert 'process_identity' in audit.AUDIT_SOURCE
    assert audit.describe.__code__ is original_math.describe.__code__
    for function in vars(audit).values():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(audit):
            assert not unresolved_globals(function), function.__name__
