from copy import deepcopy

import pytest

import report_single_model_cached_continuation_progress as report


def rows(domain="imputation", structured=False, metric="rmse", values=(1.0, 2.0, 3.0, 4.0, .5)):
    baseline_value, parent_value, general_value, teacher_value, candidate_value = values
    meta = {"index": 0, "seed": 123, "domain": domain, "structured": structured, "family": "mar"}
    original = {**meta, "general": {metric: general_value}, "teacher": {metric: teacher_value}}
    initial = {**meta, "student": {metric: baseline_value}}
    parent = {**meta, "student": {metric: parent_value}}
    current = {**meta, "student": {metric: candidate_value}}
    group = "structured_overall" if structured else domain
    guard = {"group": group, "metric": metric, "initialization": baseline_value, "candidate": candidate_value,
             "retained": candidate_value <= baseline_value}
    return [original], [initial], [parent], [current], [guard]


@pytest.mark.parametrize("domain,structured,expected", [("imputation", False, 1.0),
    ("imputation", True, 3.0), ("recsys", False, 4.0)])
def test_original_domain_teacher_routing_is_not_reference_blend(domain, structured, expected):
    result = report.summarize(*rows(domain, structured))
    guard = result["guards"][0]
    assert guard["training_teacher"] == expected
    assert guard["warm_start"] == 2.0
    assert guard["candidate_change_vs_warm_start_percent"] == -75.0
    assert guard["candidate_minus_original_general"] == -2.5
    assert result["case_identities"] == [(0, 123)]


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_rejects_case_identity_or_metadata_changes(position):
    arguments = rows()
    arguments[position][0]["seed"] = 999
    with pytest.raises(ValueError, match="case identity"):
        report.summarize(*arguments)


def test_rejects_duplicate_guard_and_wrong_audited_score():
    arguments = rows()
    arguments[-1].append(deepcopy(arguments[-1][0]))
    with pytest.raises(ValueError, match="duplicate"):
        report.summarize(*arguments)
    arguments = rows()
    arguments[-1][0]["candidate"] += 1.0
    with pytest.raises(ValueError, match="actual audited guard"):
        report.summarize(*arguments)


def test_truth_defined_nulls_are_retained_in_case_output():
    arguments = rows(structured=True, metric="common_nrmse")
    for group in arguments[:4]:
        undefined = deepcopy(group[0])
        undefined.update(index=1, seed=456)
        for field in ("student", "general", "teacher"):
            if field in undefined:
                undefined[field]["common_nrmse"] = None
        group.append(undefined)
    result = report.summarize(*arguments)
    assert result["guards"][0]["undefined_cases"] == 1
    assert len(result["cases"]) == 2
    assert result["cases"][1]["training_teacher"] is None
    arguments[2][1]["student"]["common_nrmse"] = 0.0
    with pytest.raises(ValueError, match="null coverage"):
        report.summarize(*arguments)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_rejects_nonfinite_or_negative_scores(value):
    arguments = rows()
    arguments[2][0]["student"]["rmse"] = value
    with pytest.raises(ValueError, match="finite nonnegative"):
        report.summarize(*arguments)


def test_zero_reference_keeps_absolute_delta_without_fabricated_percentage():
    result = report.summarize(*rows(values=(1.0, 0.0, 3.0, 4.0, .5)))
    assert result["guards"][0]["candidate_change_vs_warm_start_percent"] is None
    assert result["guards"][0]["candidate_minus_warm_start"] == .5
