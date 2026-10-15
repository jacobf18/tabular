import numpy as np
import pytest
import torch

import audit_single_model_native_column_mean_probe as audit
from test_single_model_column_anchor_retention import column_example


@pytest.mark.parametrize("native", [False, True])
def test_all_columns_and_both_endpoints_match_independent_applied_terms(native):
    target, mask = column_example()
    teacher = torch.nan_to_num(target) - .3
    before = torch.nan_to_num(target) + 1.2
    candidate = torch.nan_to_num(target) + .2
    predictions = {"before": before.numpy(), "candidate": candidate.numpy()}
    rows, values = audit.describe(target, mask, teacher, predictions, native=native)
    assert len(rows) == target.shape[-1]
    assert rows[-1]["queries"] == 0
    assert rows[-1]["applied_coefficient"] == 0
    assert rows[2]["constant_multi_query"] and rows[3]["single_query"]
    assert values["applied_change"] < 0
    if not native:
        assert values["before_applied_mean20"] == values["before_uniform_mean20"]
        assert values["candidate_applied_mean20"] == values["candidate_uniform_mean20"]
    for role, predictions in (("before", before), ("candidate", candidate)):
        unweighted = 20. * torch.nn.functional.huber_loss(predictions[mask], teacher[mask], delta=1., reduction="mean")
        assert values[role + "_uniform_mean20"] == pytest.approx(float(unweighted), rel=1e-6)
        assert sum(row[role + "_applied_mean20_contribution"] for row in rows) == pytest.approx(values[role + "_applied_mean20"])


def test_adverse_teacher_changes_are_not_filtered():
    target, mask = column_example()
    teacher = torch.nan_to_num(target)
    predictions = {"before": teacher.numpy(), "candidate": (teacher + .7).numpy()}
    rows, values = audit.describe(target, mask, teacher, predictions, native=True)
    assert values["applied_change"] > 0
    assert sum(row["applied_change"] > 0 for row in rows) == 4


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
