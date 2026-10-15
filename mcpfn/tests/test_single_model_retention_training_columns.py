from copy import deepcopy
import json

import numpy as np
import pytest
import torch

import analyze_single_model_retention_training_columns as analysis


def test_empty_single_constant_and_nonconstant_columns_are_all_retained():
    target = torch.tensor([[[0., 3., 5., 0.], [1., 8., 5., 2.], [2., 9., 5., 4.]]])
    mask = torch.tensor([[[False, True, True, True], [False, False, True, True], [False, False, True, True]]])
    source = target.masked_fill(mask, float("nan"))
    rows = analysis.column_rows(source, target, mask)
    assert len(rows) == 4
    assert rows[0]["queries"] == 0 and rows[0]["variance"] is None and rows[0]["unit_drift_penalty"] == 0.
    assert rows[1]["single_query"] and rows[1]["variance_floor_active"]
    assert rows[2]["constant_multi_query"] and rows[2]["variance_floor_active"]
    assert not rows[3]["variance_floor_active"]
    assert rows[3]["variance"] == pytest.approx(8. / 3.)
    assert sum(row["queries"] for row in rows) == int(mask.sum())
    actual = analysis.retention.column_coefficients(target, mask)
    assert float(analysis.retention.SPECIFICATION["weight"] * actual[mask].mean()) == pytest.approx(
        sum(row["unit_drift_penalty"] for row in rows), rel=1e-6)


def test_nonquery_nonfinite_labels_do_not_change_coefficients():
    target = torch.tensor([[[float("nan"), 1.], [3., 2.], [4., 4.]]])
    mask = torch.isfinite(target)
    source = torch.full_like(target, float("nan"))
    rows = analysis.column_rows(source, target, mask)
    assert len(rows) == 2 and all(np.isfinite(row["unit_drift_penalty"]) for row in rows)
    target[0, 1, 0] = float("inf")
    with pytest.raises(ValueError, match="finite-query"):
        analysis.column_rows(source, target, mask)


def test_independent_check_detects_wrong_streamed_coefficients(monkeypatch):
    target = torch.tensor([[[1.], [2.], [4.]]])
    mask = torch.ones_like(target, dtype=torch.bool)
    monkeypatch.setattr(analysis.retention, "column_coefficients", lambda values, selected: torch.zeros_like(values))
    with pytest.raises(ValueError, match="Independent column coefficient"):
        analysis.column_rows(target, target, mask)


@pytest.mark.parametrize("specification,expected", [
    ({"domain": "imputation", "seed": 1, "component": 6}, (1, True)),
    ({"domain": "imputation", "seed": 2, "component": 7}, (2, True)),
    ({"domain": "imputation", "seed": 3, "component": 0}, (3, False)),
    ({"domain": "recsys", "seed": 4}, (4, False)),
    ({"domain": "imputation", "structured": {"seed": 5, "component": "anchor_latent_mcar"}}, (5, True)),
    ({"domain": "imputation", "structured": {"seed": 6, "component": "slow_tail_mcar"}}, (6, False)),
])
def test_eligibility_uses_only_original_domain_and_generator(specification, expected):
    assert analysis.specification_identity({"specification": specification}) == expected


def test_complete_prefix_allows_later_appends_but_not_missing_or_reordered_attempts(tmp_path):
    path = tmp_path / "training.jsonl"
    initial = "".join(json.dumps({"step": step}) + "\n" for step in range(1, 4))
    path.write_text(initial + '{"step":')
    assert analysis.prefix(path, 3)[0] == initial
    with pytest.raises(ValueError):
        analysis.prefix(path, 4)
    path.write_text(json.dumps({"step": 2}) + "\n")
    with pytest.raises(ValueError, match="contiguous"):
        analysis.prefix(path, 1)


def test_aggregation_preserves_all_columns_and_original_task_weights():
    target = torch.tensor([[[1., 3.], [2., 3.], [4., 3.]]])
    mask = torch.ones_like(target, dtype=torch.bool)
    base = analysis.column_rows(target, target, mask)
    rows = []
    for step, weight in ((1, .25), (2, .01)):
        rows.extend({**deepcopy(row), "step": step, "task_index": 0, "cohort": "legacy",
            "family": "legacy", "task_weight": weight} for row in base)
    summary = analysis.aggregate(rows)
    overall = next(row for row in summary if row["grouping"] == "all")
    parts = [row for row in summary if row["grouping"] == "query_class"]
    assert overall["columns"] == 4 and overall["tasks"] == 2
    assert overall["unit_drift_penalty"] == pytest.approx(2 * sum(row["unit_drift_penalty"] for row in base))
    assert overall["task_weighted_unit_drift_penalty"] == pytest.approx(.26 * sum(row["unit_drift_penalty"] for row in base))
    assert sum(row["columns"] for row in parts) == overall["columns"]
    assert sum(row["task_weighted_unit_drift_penalty"] for row in parts) == pytest.approx(overall["task_weighted_unit_drift_penalty"])
