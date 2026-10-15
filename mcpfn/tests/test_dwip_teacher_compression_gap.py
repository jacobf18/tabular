from copy import deepcopy
import json
import math

import pytest
import torch

from diagnose_dwip_teacher_compression_gap import (
    BUCKETS, PANELS, compare_student, support_buckets,
)


def fixture():
    cases = [{"id": "case", "group": "group", "monitor_only": False,
              "source": torch.full((1, 2, 2), float("nan")),
              "target": torch.zeros(1, 2, 2), "mask": torch.ones(1, 2, 2, dtype=torch.bool)}]
    payload = {"metrics": {"macro_rmse": 1., "groups": {"group": {"n": 4, "monitor_only": False, "rmse": 1.}}},
               "predictions": {"case": {"mean": torch.ones(4), "target": torch.zeros(4), "flat_indices": torch.arange(4)}}}
    return cases, payload, deepcopy(payload), deepcopy(payload)


def set_mean(payload, value):
    payload["predictions"]["case"]["mean"].fill_(value)
    payload["metrics"]["macro_rmse"] = abs(value)
    payload["metrics"]["groups"]["group"]["rmse"] = abs(value)


@pytest.mark.parametrize("panel", PANELS)
def test_exact_teacher_matching_has_zero_compression(panel):
    cases, first, second, student = fixture()
    set_mean(second, -1.)
    set_mean(student, 0.)
    result = compare_student(panel, cases, first, second, student)
    assert result["cached_macro_rmse_reproduced"] == {
        "rank1": 1., "rank2": 1., "equal_prediction_mean": 0., "student": 0.}
    group = result["groups"]["group"]
    assert group["compression_mse"] == group["teacher_residual_cross_term"] == 0.
    assert group["identity_absolute_error"] == 0.
    assert group["student_minus_rank1_mse"] == -1.
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("student_mean,cross,gap", ((2., 2., 3.), (0., -2., -1.)))
def test_compression_error_can_hurt_or_correct_teacher_error(student_mean, cross, gap):
    cases, first, second, student = fixture()
    set_mean(student, student_mean)
    group = compare_student("fixed", cases, first, second, student)["groups"]["group"]
    assert group["compression_mse"] == 1.
    assert group["teacher_residual_cross_term"] == cross
    assert group["student_minus_teacher_mse"] == gap
    assert group["excess_compression_mse_vs_rank1"] + group["cross_term_change_vs_rank1"] == group["student_minus_rank1_mse"]


def test_zero_control_error_does_not_divide_by_zero():
    cases, first, second, student = fixture()
    set_mean(first, 0.)
    result = compare_student("fixed", cases, first, second, student)
    assert result["summary"]["normalization_groups"] == 0
    assert all(value is None for value in result["summary"]["rank1_mse_normalized_group_mean"].values())
    json.dumps(result, allow_nan=False)


def test_support_buckets_partition_scored_cells_using_observations_only():
    source = torch.full((1, 10, 10), float("nan"))
    for index, count in enumerate((0, 1, 2, 5)):
        source[0, index, 4:4 + count] = 1.
        source[0, 4:4 + count, index] = 1.
    mask = torch.zeros_like(source, dtype=torch.bool)
    for index in range(4):
        mask[0, index, index] = True
    case = {"source": source, "mask": mask}
    buckets = support_buckets(case)
    assert list(buckets) == list(BUCKETS)
    assert [int(subset.sum()) for subset in buckets.values()] == [1, 1, 1, 1]
    assert torch.stack(list(buckets.values())).sum(0).eq(1).all()
    source[torch.isfinite(source)] = 999.
    assert all(torch.equal(value, support_buckets(case)[name]) for name, value in buckets.items())


@pytest.mark.parametrize("change", ("dimensions", "observed", "infinity", "mask_dtype"))
def test_invalid_support_inputs_fail_closed(change):
    case = fixture()[0][0]
    if change == "dimensions":
        case["source"] = case["source"].flatten()
    elif change == "observed":
        case["source"][0, 0, 0] = 1.
    elif change == "infinity":
        case["source"][0, 0, 0] = float("inf")
    else:
        case["mask"] = case["mask"].float()
    with pytest.raises(ValueError):
        support_buckets(case)


def test_monitor_groups_and_unequal_cell_counts_do_not_change_macro_weighting():
    cases, first, second, student = fixture()
    for name, count, error, monitor in (("large", 8, 3., False), ("monitor", 12, 100., True)):
        case = {"id": name, "group": name, "monitor_only": monitor,
                "source": torch.full((1, 1, count), float("nan")),
                "target": torch.zeros(1, 1, count), "mask": torch.ones(1, 1, count, dtype=torch.bool)}
        cases.append(case)
        for payload in (first, second, student):
            payload["predictions"][name] = {"mean": torch.full((count,), error),
                                            "target": torch.zeros(count), "flat_indices": torch.arange(count)}
            payload["metrics"]["groups"][name] = {"n": count, "monitor_only": monitor, "rmse": error}
            payload["metrics"]["macro_rmse"] = 2.
    result = compare_student("independent_spectral", cases, first, second, student)
    assert result["summary"]["macro_rmse_fp64_residual"]["student"] == 2.
    assert result["summary"]["cells"] == 12
    assert result["summary"]["non_monitor_groups"] == 2
    assert result["minimum_axis_observed_support"]["0"]["summary"] == result["summary"]
    assert result["minimum_axis_observed_support"]["5_plus"]["summary"] is None
    assert result["groups"]["monitor"]["monitor_only"]


@pytest.mark.parametrize("label", (1, 2, 3))
@pytest.mark.parametrize("change", ("target", "indices", "dtype", "metrics", "nonfinite", "case_id"))
def test_all_three_prediction_caches_are_validated(label, change):
    values = fixture()
    payload = values[label]
    record = payload["predictions"]["case"]
    if change == "target":
        record["target"] += 1
    elif change == "indices":
        record["flat_indices"] += 1
    elif change == "dtype":
        record["mean"] = record["mean"].double()
    elif change == "metrics":
        payload["metrics"]["macro_rmse"] = .5
    elif change == "nonfinite":
        record["mean"][0] = float("nan")
    else:
        payload["predictions"]["other"] = payload["predictions"].pop("case")
    with pytest.raises(ValueError):
        compare_student("fixed", *values)


def test_no_public_panel_or_new_population_is_accepted():
    with pytest.raises(ValueError, match="registered synthetic"):
        compare_student("mteb", *fixture())


def test_fp64_identity_is_separate_from_cached_fp32_residual_arithmetic():
    cases, first, second, student = fixture()
    cases[0]["target"].fill_(.1)
    for payload, value in ((first, .3), (second, .7), (student, .4)):
        payload["predictions"]["case"]["target"].fill_(.1)
        payload["predictions"]["case"]["mean"].fill_(value)
        error = payload["predictions"]["case"]["mean"] - payload["predictions"]["case"]["target"]
        score = math.sqrt(float(error.double().square().mean()))
        payload["metrics"]["macro_rmse"] = score
        payload["metrics"]["groups"]["group"]["rmse"] = score
    result = compare_student("fixed", cases, first, second, student)
    assert result["groups"]["group"]["identity_absolute_error"] < 1e-12
    assert result["cached_macro_rmse_reproduced"]["student"] == student["metrics"]["macro_rmse"]
