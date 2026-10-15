from copy import deepcopy

import pytest

from check_single_model_point_finite_steps import REPEATS, ROLES, signed_change, summarize_state


def test_signed_changes_preserve_zero_baselines_and_strict_signs():
    assert signed_change(0., .1)["relative_difference"] is None
    assert signed_change(1., 1. + 1e-14)["direction"] == "worse"
    assert signed_change(1., .9)["direction"] == "better"
    assert signed_change(1., 1.)["direction"] == "equal"
    with pytest.raises(ValueError):
        signed_change(1., float("nan"))


def panel():
    tasks, entries, rows = [], [], []
    for index in range(14):
        task = f"task_{index}"
        tasks.append({"task": task, "metrics": {"rmse": {"value": 1.}, "common_nrmse": None}})
        entries.append({"task": task, "kind": "rmse", "index": index})
        for role in ROLES:
            for repeat in range(REPEATS):
                value = 1. if role == "before" else .9
                rows.append({"task": task, "role": role, "repeat": repeat, "products": [.1] * 14,
                             "metrics": {"rmse": {"value": value}, "common_nrmse": None}})
    return {"tasks": tasks, "entries": entries}, rows


def test_all_signed_metrics_and_undefined_slots_are_retained():
    result, rows = panel()
    summary = summarize_state(result, rows)
    for role in ROLES:
        counts = summary["roles"][role]
        assert counts["undefined"] == 14 and counts["available"]
        assert counts["equal" if role == "before" else "better"] == 14
    assert summary["maximum_repeat_metric_spread"] == 0
    assert summary["maximum_capture_replay_metric_difference"] == 0
    assert len(summary["changes"]) == 56


def test_missing_repeat_or_changed_coverage_is_rejected():
    result, rows = panel()
    with pytest.raises(ValueError, match="mandatory"):
        summarize_state(result, rows[:-1])
    changed = deepcopy(rows)
    changed[-1]["metrics"]["common_nrmse"] = {"value": 1.}
    with pytest.raises(ValueError, match="coverage"):
        summarize_state(result, changed)


def test_unavailable_variant_is_explicit_not_zero():
    result, rows = panel()
    summary = summarize_state(result, [row for row in rows if row["role"] != "composite_and_points"])
    assert summary["roles"]["composite_and_points"] == {"available": False}
