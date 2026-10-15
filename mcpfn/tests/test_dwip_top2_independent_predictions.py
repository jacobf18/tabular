from copy import deepcopy

import pytest
import torch

from diagnose_dwip_top2_independent_predictions import (
    PANELS, evidence_decision, independent_comparison, unchanged_pair,
)


def fixture():
    cases = [{"id": "case", "group": "group", "monitor_only": False,
              "target": torch.zeros(2), "mask": torch.ones(2, dtype=torch.bool)}]
    before = {"metrics": {"macro_rmse": 1., "groups": {"group": {"n": 2, "monitor_only": False, "rmse": 1.}}},
              "predictions": {"case": {"mean": torch.ones(2), "target": torch.zeros(2), "flat_indices": torch.arange(2)}}}
    after = deepcopy(before)
    after["predictions"]["case"]["mean"] *= -1
    return cases, before, after


def candidates():
    return [{"queue": name, "kind": "fixed_selected_training_checkpoint", "state": "complete",
             "case_count": 72, "case_digest": "fixed", "fixed_macro_rmse": score, "sha256": name}
            for name, score in (("first", 1.), ("second", 1.1), ("third", 1.2))]


def test_same_fixed_pair_ignores_benchmarks_and_rejected_averages():
    pool = candidates()
    registered = deepcopy(pool[:2])
    pool[2]["mteb"] = 999.
    pool.append({"kind": "saved_average_not_promoted", "fixed_macro_rmse": .1})
    assert unchanged_pair(pool, registered) == registered


@pytest.mark.parametrize("field", ("queue", "sha256", "fixed_macro_rmse", "case_digest"))
def test_pair_or_provenance_change_is_rejected(field):
    pool = candidates()
    registered = deepcopy(pool[:2])
    registered[0][field] = "different"
    with pytest.raises(ValueError, match="exact previously registered"):
        unchanged_pair(pool, registered)


def test_newly_better_model_cannot_change_the_frozen_pair():
    pool = candidates()
    registered = deepcopy(pool[:2])
    pool[2]["fixed_macro_rmse"] = .5
    with pytest.raises(ValueError, match="exact previously registered"):
        unchanged_pair(pool, registered)


@pytest.mark.parametrize("panel", PANELS)
def test_independent_result_is_not_mislabeled_as_fixed_selection(panel):
    result = independent_comparison(panel, *fixture())
    assert result["macro_rmse"] == {"rank1": 1., "rank2": 1., "equal_prediction_mean": 0.}
    assert result["non_monitor_group_count"] == result["non_monitor_groups_mean_beats_both_raw"] == 1
    assert not any("selector" in key for key in result)
    assert "not a selector" in result["scope"]
    assert result["case_count"] == 1


@pytest.mark.parametrize("panel", ("fixed", "mteb", "retention", "third_panel"))
def test_no_other_population_is_permitted(panel):
    with pytest.raises(ValueError, match="Only the two registered"):
        independent_comparison(panel, *fixture())


def test_cached_raw_reproduction_check_remains_required():
    cases, before, after = fixture()
    after["metrics"]["macro_rmse"] = .5
    with pytest.raises(ValueError, match="does not reproduce"):
        independent_comparison(PANELS[0], cases, before, after)


@pytest.mark.parametrize("first,second,expected", ((True, True, True), (True, False, False), (False, True, False), (False, False, False)))
def test_both_panels_required_for_compression_evidence_but_never_promotion(first, second, expected):
    panels = {name: {"equal_prediction_mean_beats_both_raw": improves}
              for name, improves in zip(PANELS, (first, second))}
    result = evidence_decision(panels)
    assert result["supports_studying_single_model_compression"] is expected
    assert not result["checkpoint_created_or_promoted"]
    assert not result["ensemble_deployment_proposed"]


def test_missing_panel_is_not_positive_evidence():
    with pytest.raises(ValueError, match="both complete"):
        evidence_decision({PANELS[0]: {"equal_prediction_mean_beats_both_raw": True}})


def test_tied_raw_predictions_do_not_support_compression():
    cases, before, after = fixture()
    after = deepcopy(before)
    result = independent_comparison(PANELS[0], cases, before, after)
    assert not result["equal_prediction_mean_beats_both_raw"]
