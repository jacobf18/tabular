from copy import deepcopy

import numpy as np
import pytest

import report_single_model_displacement_grid as report
from test_single_model_fixed_contraction import improved_metrics


def records():
    baseline, candidate = improved_metrics()
    base = {"retention_metrics": report.grid.mixture.column.RETENTION_METRICS}
    decision = report.grid.fixed.selection_decision(candidate, baseline, baseline, base)
    decisions = {spec["id"]: deepcopy(decision) for spec in report.grid.GRID}
    selection = report.grid.choose(decisions)
    gate = {"selected_candidate": selection["selected"], "confirmation_evaluated": True, "passed": False}
    return decisions, selection, gate


def test_full264_guard_report_does_not_promote_failed_confirmation():
    decisions, selection, gate = records()
    candidates, retention = report.summarize(decisions, selection, gate)
    assert len(candidates) == 12 and len(retention) == 264
    assert sum(row["selected"] for row in candidates) == 1
    assert all(row["eligible"] for row in candidates)
    assert not any(row["synthetic_confirmed"] for row in candidates)
    assert all(row["relative_percent"] < 0 for row in retention)
    assert {(row["group"], row["metric"]) for row in retention if row["candidate_id"] == "candidate_00"} == {
        (row["group"], row["metric"]) for row in decisions["candidate_00"]["comparisons"]}


@pytest.mark.parametrize("kind", ["omitted", "reordered", "missing_guard", "renamed_guard", "duplicate_guard",
                                  "waived_regression", "eligibility", "confirmation", "selection"])
def test_incomplete_or_modified_research_decisions_rejected(kind):
    decisions, selection, gate = records()
    current = decisions["candidate_00"]
    if kind == "omitted":
        decisions.pop("candidate_11")
    elif kind == "reordered":
        decisions = dict(reversed(list(decisions.items())))
    elif kind == "missing_guard":
        current["comparisons"].pop()
    elif kind == "renamed_guard":
        current["comparisons"][0]["group"] = "not_registered"
    elif kind == "duplicate_guard":
        current["comparisons"][-1] = deepcopy(current["comparisons"][0])
    elif kind == "waived_regression":
        current["comparisons"][0]["candidate"] = current["comparisons"][0]["initialization"] * 1.000000001
    elif kind == "eligibility":
        current["retained"] = False
    elif kind == "confirmation":
        gate["confirmation_evaluated"] = False
    else:
        gate["selected_candidate"] = report.grid.GRID[1]
    with pytest.raises(ValueError):
        report.summarize(decisions, selection, gate)


def test_zero_baseline_does_not_invent_a_percentage():
    decisions, selection, gate = records()
    row = decisions["candidate_00"]["comparisons"][0]
    row["initialization"], row["candidate"] = 0., 0.
    candidates, retention = report.summarize(decisions, selection, gate)
    assert retention[0]["relative_percent"] is None and retention[0]["retained"]


def prediction_records(tmp_path, difference=0.):
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    np.savez(first, student=np.ones((1, 2, 3), dtype=np.float32))
    values = np.ones((1, 2, 3), dtype=np.float32)
    values[0, 0, 0] += difference
    np.savez(second, student=values)
    reports = []
    for path in (first, second):
        cases = [{"index": index, "seed": 100 + index, "domain": "imputation", "student": {"rmse": 1.},
                  "prediction_file": str(path), "prediction_sha256": report.sha256(path)} for index in range(152)]
        reports.append({"cases": cases, "student": {"rmse": 1.}})
    return reports


@pytest.mark.parametrize("difference", [0., .25])
def test_full_control_values_compared_without_claiming_unequal_arrays_match(tmp_path, difference):
    previous, current = prediction_records(tmp_path, difference)
    summary, rows, hashes = report.compare_predictions(previous, current)
    assert summary["cases"] == len(rows) == 152
    assert summary["elements"] == 912
    assert summary["different_elements"] == (152 if difference else 0)
    assert summary["all_prediction_values_equal"] == (difference == 0.)
    assert summary["max_absolute_difference"] == difference
    assert len(hashes) == 2


@pytest.mark.parametrize("kind", ["missing", "seed", "domain", "index", "hash", "shape", "nan"])
def test_invalid_control_provenance_or_arrays_rejected(tmp_path, kind):
    previous, current = prediction_records(tmp_path)
    if kind == "missing":
        current["cases"].pop()
    elif kind in {"seed", "domain", "index"}:
        current["cases"][0][kind] = "different"
    elif kind == "hash":
        current["cases"][0]["prediction_sha256"] = "0" * 64
    else:
        path = tmp_path / "second.npz"
        values = np.ones((1, 2, 4) if kind == "shape" else (1, 2, 3), dtype=np.float32)
        if kind == "nan":
            values[0, 0, 0] = np.nan
        np.savez(path, student=values)
        for row in current["cases"]:
            row["prediction_sha256"] = report.sha256(path)
    with pytest.raises((ValueError, RuntimeError)):
        report.compare_predictions(previous, current)


def test_markdown_distinguishes_synthetic_metrics_from_real_results():
    decisions, selection, gate = records()
    candidates, retention = report.summarize(decisions, selection, gate)
    summary = {"confirmation_evaluated": True, "synthetic_gate_passed": False,
               "known_control": {"state_values_identical": True, "all_prediction_values_equal": False,
                                 "max_absolute_difference": .125, "aggregate_metrics_identical": False}}
    text = report.markdown(summary, candidates, retention)
    assert "not MissBench, UCI, recommender, or LLM benchmark measurements" in text
    assert "Final synthetic gate passed: **False**" in text
    assert "All152 complete prediction grids identical in value: **False**" in text
    assert "not a blinded candidate" in text
