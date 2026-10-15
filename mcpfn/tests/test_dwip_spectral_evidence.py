from copy import deepcopy
import json

import pytest
import torch

import analyze_dwip_spectral_evidence as evidence
from analyze_dwip_spectral_evidence import compare_predictions, compare_scores


def score_fixture():
    return {"groups": {"small": {"rmse": 2., "nll": 1., "n": 10, "monitor_only": False},
                       "large": {"rmse": 10., "nll": 5., "n": 20, "monitor_only": True}}}


def prediction_fixture():
    return {"predictions": {"small_mixed_0": {"mean": torch.tensor([2., 4.]),
            "target": torch.tensor([1., 2.]), "flat_indices": torch.tensor([3, 7])}}}


def test_score_comparison_preserves_selector_and_reports_monitor_regression():
    baseline = score_fixture()
    candidate = deepcopy(baseline)
    candidate["groups"]["small"]["rmse"] = 1.
    candidate["groups"]["large"]["rmse"] = 20.
    result = compare_scores(baseline, candidate)
    assert result["baseline_macro_rmse"] == 2.
    assert result["candidate_macro_rmse"] == 1.
    assert result["mean_relative_group_rmse"] == .5
    assert result["groups_improved"] == result["group_count"] == 1
    assert result["groups"]["large"]["rmse_delta"] == 10.


@pytest.mark.parametrize("key,value", [("n", 9), ("monitor_only", True), ("rmse", float("nan"))])
def test_score_comparison_rejects_invalid_population_or_metrics(key, value):
    baseline = score_fixture()
    candidate = deepcopy(baseline)
    candidate["groups"]["small"][key] = value
    with pytest.raises(ValueError):
        compare_scores(baseline, candidate)


def test_score_comparison_rejects_different_groups():
    baseline = score_fixture()
    candidate = deepcopy(baseline)
    del candidate["groups"]["large"]
    with pytest.raises(ValueError, match="groups differ"):
        compare_scores(baseline, candidate)


def test_score_comparison_zero_baseline_has_no_relative_ratio():
    baseline = score_fixture()
    baseline["groups"]["small"]["rmse"] = 0.
    result = compare_scores(baseline, baseline)
    assert result["mean_relative_group_rmse"] is None


def test_paired_predictions_compare_identical_targets_and_queries():
    baseline = prediction_fixture()
    candidate = deepcopy(baseline)
    candidate["predictions"]["small_mixed_0"]["mean"] = torch.tensor([1., 2.])
    result = compare_predictions(baseline, candidate)
    assert result["case_count"] == result["cases_improved"] == 1
    assert result["cases"]["small_mixed_0"]["candidate_rmse"] == 0.
    assert result["cases"]["small_mixed_0"]["rmse_delta"] == pytest.approx(-(2.5 ** .5))


@pytest.mark.parametrize("key,value", [
    ("target", torch.tensor([1., 3.])), ("flat_indices", torch.tensor([2, 7])),
    ("mean", torch.tensor([1.])), ("mean", torch.tensor([float("nan"), 2.])),
])
def test_paired_predictions_reject_mismatches_and_nonfinite_values(key, value):
    baseline = prediction_fixture()
    candidate = deepcopy(baseline)
    candidate["predictions"]["small_mixed_0"][key] = value
    with pytest.raises(ValueError):
        compare_predictions(baseline, candidate)


def test_paired_predictions_reject_different_case_ids():
    baseline = prediction_fixture()
    candidate = deepcopy(baseline)
    candidate["predictions"]["other_case"] = candidate["predictions"].pop("small_mixed_0")
    with pytest.raises(ValueError, match="identities differ"):
        compare_predictions(baseline, candidate)


def test_collector_keeps_mteb_report_only_and_distinguishes_stale_status(tmp_path, monkeypatch):
    root, work = tmp_path / "queue", tmp_path / "work"
    root.mkdir()
    work.mkdir()
    (root / "status.json").write_text(json.dumps({"state": "training"}))
    (root / "baseline_mteb.json").write_text(json.dumps({"mi_r2_5": .69}))
    (root / "dwip1e5_mteb.json").write_text(json.dumps({"mi_r2_5": .99}))
    training = {"step": 2000, "loss_missing_ema": 1., "lr": 1e-5,
                "loss_missing": 1., "nll_missing": .9, "grad_norm": 2., "grad_was_clipped": 1.}
    (work / "metrics_stream.jsonl").write_text(json.dumps(training) + "\n")
    initial = {**score_fixture(), "step": 0, "macro_rmse": 2.}
    latest = deepcopy(initial)
    latest.update(step=2000, macro_rmse=3.)
    latest["groups"]["small"]["rmse"] = 3.
    (work / "validation_stream.jsonl").write_text("\n".join(json.dumps(row) for row in (initial, latest)))
    monkeypatch.setattr(evidence.psutil, "process_iter", lambda attrs: [])
    monkeypatch.setattr(evidence, "analyze", lambda work, out: {"recommendation": "in progress"})
    report = evidence.collect(root, work)
    assert report["queue_state"]["state"] == "training"
    assert report["live_trainers"] == []
    assert report["holdouts"] == {}
    assert report["validation"]["best_step"] == 0
    assert report["training"]["nonfinite_sampled_steps"] == []
    assert report["mteb_report_only"]["mi_r2_5_gap_to_reference"] == pytest.approx(.228)
