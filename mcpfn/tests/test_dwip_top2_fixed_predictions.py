from copy import deepcopy

import pytest
import torch

from diagnose_dwip_top2_fixed_predictions import compare_fixed_predictions, top_two


def entry(queue, score):
    return {"queue": queue, "kind": "fixed_selected_training_checkpoint", "state": "complete",
            "case_count": 72, "case_digest": "same", "fixed_macro_rmse": score, "mteb": 999.}


def test_top_two_uses_only_fixed_scores_and_deterministic_ties():
    candidates = [entry("third", 1.1), entry("second", 1.), entry("first", 1.)]
    candidates[0]["mteb"] = 10000.
    candidates.append({"kind": "saved_average_not_promoted"})
    assert [item["queue"] for item in top_two(candidates)] == ["first", "second"]


@pytest.mark.parametrize("change", ("duplicate", "state", "cases", "nonfinite"))
def test_top_two_rejects_ambiguous_or_changed_pool(change):
    candidates = [entry("first", 1.), entry("second", 1.1)]
    if change == "duplicate":
        candidates[1]["queue"] = "first"
    elif change == "state":
        candidates[1]["state"] = "training"
    elif change == "cases":
        candidates[1]["case_digest"] = "different"
    else:
        candidates[1]["fixed_macro_rmse"] = float("nan")
    with pytest.raises(ValueError):
        top_two(candidates)


def fixture():
    cases = [{"id": "case", "group": "group", "monitor_only": False,
              "target": torch.zeros(2), "mask": torch.ones(2, dtype=torch.bool)}]
    before = {"metrics": {"macro_rmse": 1., "groups": {"group": {"n": 2, "monitor_only": False, "rmse": 1.}}},
              "predictions": {"case": {"mean": torch.ones(2), "target": torch.zeros(2), "flat_indices": torch.arange(2)}}}
    after = deepcopy(before)
    after["predictions"]["case"]["mean"] *= -1
    return cases, before, after


def test_fixed_prediction_mean_is_only_a_diagnostic_and_detects_complementarity():
    cases, before, after = fixture()
    result = compare_fixed_predictions(cases, before, after)
    assert result["macro_rmse"] == {"rank1": 1., "rank2": 1., "equal_prediction_mean": 0.}
    assert result["equal_prediction_mean_beats_both_raw"]
    assert result["groups"]["group"]["residual_cosine_uncentered"] == -1.


def test_fixed_prediction_diagnostic_rejects_overflow_from_finite_inputs():
    cases, before, after = fixture()
    cases[0]["target"].fill_(-3e38)
    for payload in (before, after):
        payload["predictions"]["case"]["target"].fill_(-3e38)
        payload["predictions"]["case"]["mean"].fill_(3e38)
    with pytest.raises(ValueError, match="Nonfinite fixed residual"):
        compare_fixed_predictions(cases, before, after)


@pytest.mark.parametrize("change", ("id", "target", "indices", "shape", "nonfinite", "metrics"))
def test_fixed_prediction_diagnostic_rejects_population_or_score_drift(change):
    cases, before, after = fixture()
    record = after["predictions"]["case"]
    if change == "id":
        after["predictions"]["other"] = after["predictions"].pop("case")
    elif change == "target":
        record["target"] += 1
    elif change == "indices":
        record["flat_indices"] += 1
    elif change == "shape":
        record["mean"] = torch.ones(3)
    elif change == "nonfinite":
        record["mean"][0] = float("nan")
    else:
        after["metrics"]["macro_rmse"] = .5
    with pytest.raises(ValueError):
        compare_fixed_predictions(cases, before, after)
