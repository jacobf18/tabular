from copy import deepcopy

import pytest
import torch

from single_model_distillation import Episode
import single_model_truth_defined_columns as corrected


def episode(singletons):
    target = torch.tensor([[[1., 3.], [2., 5.], [4., 7.]]])
    mask = torch.ones_like(target, dtype=torch.bool)
    if singletons:
        mask[0, 1:] = False
    source = target.masked_fill(mask, torch.nan)
    return Episode(source, target, mask, {"domain": "imputation", "gradient_data": "synthetic"})


@pytest.mark.parametrize("offset", [0., 1., 10.])
def test_singletons_stay_in_canonical_metrics_with_explicit_undefined_guard(offset):
    case = episode(True)
    result = corrected.episode_metrics(case.target + offset, case)
    assert result["common_nrmse"] is None and result["column_coverage"]["common_nonconstant_columns"] == 0
    assert result["rmse"] == offset and result["missbench_nrmse"] == result["uci_nrmse"] == 0.
    with pytest.raises(ValueError, match="common nonconstant"):
        corrected.column.episode_metrics(case.target + offset, case)


def test_fully_defined_metrics_are_identical_to_the_original():
    case = episode(False)
    prediction = case.target + .25
    assert corrected.episode_metrics(prediction, case) == corrected.column.episode_metrics(prediction, case)


def mixed_cases():
    singleton, defined = episode(True), episode(False)
    return [{"domain": "imputation", "student": corrected.episode_metrics(singleton.target + 10., singleton)},
            {"domain": "imputation", "student": corrected.episode_metrics(defined.target + 1., defined)},
            {"domain": "recsys", "student": {"rmse": .8, "mae": .7}}]


def test_guard_uses_defined_cases_but_canonical_metrics_still_count_the_bad_case():
    cases = mixed_cases()
    result = corrected.aggregate(cases, "student")
    assert result["imputation"]["rmse"] == 5.5
    assert result["imputation"]["missbench_nrmse"] == cases[1]["student"]["missbench_nrmse"] / 2
    assert result["imputation"]["common_nrmse"] == cases[1]["student"]["common_nrmse"]
    assert result["imputation"]["canonical_metric_cases"] == 2 and result["imputation"]["common_nrmse_cases"] == 1
    assert result["recsys"] == {"rmse": .8, "mae": .7}


@pytest.mark.parametrize("change", ["undefined_value", "omitted_defined", "nonfinite", "all_undefined", "canonical_nan"])
def test_no_model_dependent_exclusions_or_waived_guard(change):
    cases = mixed_cases()
    if change == "undefined_value":
        cases[0]["student"]["common_nrmse"] = 0.
    elif change == "omitted_defined":
        cases[1]["student"]["common_nrmse"] = None
    elif change == "nonfinite":
        cases[1]["student"]["common_nrmse"] = float("nan")
    elif change == "all_undefined":
        cases.pop(1)
    else:
        cases[0]["student"]["missbench_nrmse"] = float("nan")
    with pytest.raises(ValueError):
        corrected.aggregate(cases, "student")


@pytest.mark.parametrize("offset", [.5, 2.])
def test_same_truth_defined_coverage_for_every_prediction(offset):
    cases = mixed_cases()
    modified = deepcopy(cases)
    modified[1]["student"]["common_nrmse"] += offset
    first, second = corrected.aggregate(cases, "student"), corrected.aggregate(modified, "student")
    assert first["imputation"]["common_nrmse_cases"] == second["imputation"]["common_nrmse_cases"] == 1
    assert second["imputation"]["common_nrmse"] == first["imputation"]["common_nrmse"] + offset


def test_all_six_retention_checks_are_still_required():
    baseline = corrected.aggregate(mixed_cases(), "student")
    student = deepcopy(baseline)
    for domain, name in corrected.column.RETENTION_METRICS:
        student[domain][name] *= .9
    assert corrected.column.confirmation_gate(student, baseline, baseline, baseline)["passed"]
    student["imputation"]["common_nrmse"] = baseline["imputation"]["common_nrmse"] + .01
    assert not corrected.column.confirmation_gate(student, baseline, baseline, baseline)["passed"]


def test_callbacks_restore_after_failure():
    previous = corrected.column.episode_metrics, corrected.column.aggregate
    with pytest.raises(RuntimeError):
        with corrected.confirmation_metrics():
            assert corrected.column.episode_metrics is corrected.episode_metrics
            raise RuntimeError("deliberate")
    assert (corrected.column.episode_metrics, corrected.column.aggregate) == previous
