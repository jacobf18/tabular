from copy import deepcopy

import pytest

from report_single_model_decoder_progress import paired_progress


def fixture():
    baseline = {"cases": [{"index": index, "seed": 7 + index, "rows": 20, "columns": 5, "domain": "imputation",
                           "structured": True, "family": "mar", "component": "low_rank",
                           "student": {"common_nrmse": 1.0}}
                          for index in range(2)]}
    parent, current = deepcopy(baseline), deepcopy(baseline)
    for report, value in ((parent, 1.02), (current, 1.01)):
        for row in report["cases"]:
            row["student"]["common_nrmse"] = value
        report["decision"] = {"comparisons": [{"group": "mar", "metric": "common_nrmse", "initialization": 1.,
                                                "candidate": value, "retained": value <= 1.}]}
    return baseline, parent, current


def test_inherited_regression_is_distinct_from_partial_decoder_repair():
    result = paired_progress(*fixture())
    guard = result["guards"][0]
    assert guard["inherited_gap"] == pytest.approx(.02)
    assert guard["remaining_gap"] == pytest.approx(.01)
    assert guard["decoder_change"] == pytest.approx(-.01)
    assert guard["inherited_gap_repaired_percent"] == pytest.approx(50.)
    assert not guard["retained"]
    assert sum(row["guard_delta_contribution"] for row in result["decoder_cases"]) == pytest.approx(-.01)
    assert sum(row["guard_delta_contribution"] for row in result["retention_cases"]) == pytest.approx(.01)


def test_decoder_can_improve_or_worsen_without_relabeling_retention():
    baseline, parent, current = fixture()
    for row in current["cases"]:
        row["student"]["common_nrmse"] = 1.03
    current["decision"]["comparisons"][0]["candidate"] = 1.03
    guard = paired_progress(baseline, parent, current)["guards"][0]
    assert guard["inherited_gap_repaired_percent"] == pytest.approx(-50.)
    assert not guard["retained"]
    for row in current["cases"]:
        row["student"]["common_nrmse"] = .99
    current["decision"]["comparisons"][0].update(candidate=.99, retained=True)
    guard = paired_progress(baseline, parent, current)["guards"][0]
    assert guard["inherited_gap_repaired_percent"] == pytest.approx(150.)
    assert guard["retained"]


def test_no_inherited_regression_has_no_fabricated_repair_percentage():
    baseline, parent, current = fixture()
    for row in parent["cases"]:
        row["student"]["common_nrmse"] = 1.
    parent["decision"]["comparisons"][0].update(candidate=1., retained=True)
    guard = paired_progress(baseline, parent, current)["guards"][0]
    assert guard["inherited_gap_repaired_percent"] is None
    assert guard["decoder_change"] == pytest.approx(.01)


@pytest.mark.parametrize("mutation", ["parent_score", "current_score", "baseline", "metadata", "drop", "swap", "decision"])
def test_paired_progress_rejects_coverage_or_score_drift(mutation):
    baseline, parent, current = fixture()
    if mutation == "parent_score":
        parent["cases"][0]["student"]["common_nrmse"] = 1.
    elif mutation == "current_score":
        current["cases"][0]["student"]["common_nrmse"] = 1.
    elif mutation == "baseline":
        baseline["cases"][0]["student"]["common_nrmse"] = .9
    elif mutation == "metadata":
        parent["cases"][0]["component"] = "anchor"
    elif mutation == "drop":
        current["cases"].pop()
    elif mutation == "swap":
        current["cases"].reverse()
    else:
        current["decision"]["comparisons"][0]["retained"] = True
    with pytest.raises(ValueError):
        paired_progress(baseline, parent, current)
