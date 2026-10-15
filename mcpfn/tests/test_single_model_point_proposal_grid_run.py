import pytest

from check_single_model_point_proposal_grid import panel


def records():
    cases = [{"name": "legacy"}, {"name": "recsys"}]
    rows = [{"role": "before", "task": case["name"], "repeat": repeat,
             "metrics": {"rmse": {"value": 1.}, "common": None}}
            for case in cases for repeat in range(2)]
    return cases, rows


def test_panel_preserves_tasks_repetitions_and_undefined_metrics():
    cases, rows = records()
    assert panel(rows, cases, "before") == {
        "legacy/rmse": [1., 1.], "legacy/common": [None, None],
        "recsys/rmse": [1., 1.], "recsys/common": [None, None]}


def test_panel_rejects_missing_or_duplicate_repeats():
    cases, rows = records()
    with pytest.raises(ValueError, match="Every original"):
        panel(rows[:-1], cases, "before")
    rows[-1]["repeat"] = 0
    with pytest.raises(ValueError, match="identity"):
        panel(rows, cases, "before")


def test_panel_rejects_changed_metric_keys():
    cases, rows = records()
    rows[-1]["metrics"].pop("common")
    with pytest.raises(ValueError, match="names"):
        panel(rows, cases, "before")
