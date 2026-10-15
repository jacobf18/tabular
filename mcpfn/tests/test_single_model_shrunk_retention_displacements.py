from copy import deepcopy

import numpy as np
import pytest
import torch

import probe_single_model_shrunk_retention_displacements as panel


def point_values():
    values = {}
    for cohort in ("legacy", "ratings", *(f"structured_{index}" for index in range(12))):
        for draw in range(2):
            for metric in ("rmse", "common_nrmse"):
                values[f"{cohort}_draw_{draw}/{metric}"] = 1.
    return values


@pytest.mark.parametrize("fraction", panel.FRACTIONS)
def test_exact_controls_and_independent_arithmetic(fraction):
    before = torch.tensor([1., -2., 0., 1e-8, 42.])
    projected = torch.tensor([1.2, -2.1, 1e-6, 2e-8, 41.])
    candidate = panel.contract(before, projected, fraction)
    report = panel.audit_arithmetic(before, projected, candidate, fraction)
    assert report["parameters"] == 5
    assert report["non_bitwise_elements"] == 0
    if fraction in (0., 1.):
        assert torch.equal(candidate, before if fraction == 0. else projected)
    candidate[0] += .01
    with pytest.raises(ValueError):
        panel.audit_arithmetic(before, projected, candidate, fraction)


def test_rejects_unregistered_fractions():
    with pytest.raises(ValueError):
        panel.contract(torch.ones(2), torch.zeros(2), .01)


def test_rollback_after_is_not_the_projected_proposal():
    before, projected = torch.zeros(5), torch.ones(5)
    raw = {"snapshot": {"before": before, "after": before.clone()},
        "record": {"accepted": False, "reason": "finite_metrics_rejected"}}
    assert torch.equal(panel.vectors(raw, projected)[1], projected)
    with pytest.raises(ValueError):
        panel.vectors(raw, raw["snapshot"]["after"])
    raw["snapshot"]["after"] = projected
    with pytest.raises(ValueError):
        panel.vectors(raw, projected)


def test_both_repeats_preserve_all30_point_guards_and_individual_regressions():
    initial = point_values()
    before = {str(repeat): dict(initial) for repeat in panel.REPEATS}
    candidate = deepcopy(before)
    for values in candidate.values():
        values["structured_0_draw_0/rmse"] -= .1
    assert panel.finite(before, candidate)["finite_feasible"]
    candidate["1"]["legacy_draw_0/rmse"] += 1e-12
    decision = panel.finite(before, candidate)
    assert not decision["finite_feasible"]
    assert decision["repetitions"]["1"]["defined_slots"] == 30
    assert decision["repetitions"]["1"]["individual_draw_diagnostics"]["changes"]["legacy_draw_0/rmse"] > 0
    candidate["1"]["legacy_draw_0/rmse"] = 1.
    candidate["1"]["structured_0_draw_1/rmse"] += .05
    decision = panel.finite(before, candidate)
    assert decision["finite_feasible"]
    assert not decision["repetitions"]["1"]["individual_draw_diagnostics"]["all_retained"]
    assert not panel.finite(before, before)["finite_feasible"]
    with pytest.raises(ValueError):
        panel.finite(before, {"0": candidate["0"]})


def test_prediction_checks_the_entire_grid_not_only_metrics():
    prediction = np.zeros((3, 4), dtype=np.float32)
    panel.check_prediction(prediction, prediction.copy())
    different = prediction.copy()
    different[0, 0] = np.nextafter(np.float32(0), np.float32(1))
    with pytest.raises(ValueError):
        panel.check_prediction(different, prediction)
    with pytest.raises(ValueError):
        panel.check_prediction(prediction.astype(np.float64), prediction)


def test_cpu_audit_requires_both_guards(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    with pytest.raises(ValueError):
        panel.require_cpu()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    panel.require_cpu()
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ValueError):
        panel.require_cpu()


@pytest.mark.parametrize("field,value", [("accepted", True), ("gate_changed", True), ("complete", False)])
def test_only_completed_negative_predecessor_is_admitted(field, value):
    result = {"complete": True, "accepted": False, "original_rejection_preserved": True, "gate_changed": False}
    audit = {"complete": True, "training_tasks": 28, "prediction_grids": 56, "cuda_initialized": False}
    gate = {"passed": False, "reason": "finite_metrics_rejected", "preflight_only": True, "real_benchmark_success": False}
    panel.validate_predecessor(result, audit, gate)
    result[field] = value
    with pytest.raises(ValueError):
        panel.validate_predecessor(result, audit, gate)


def test_capacity_retains_all_reserves(monkeypatch):
    class Usage:
        free = 0
    monkeypatch.setattr(panel.shutil, "disk_usage", lambda root: Usage())
    with pytest.raises(OSError):
        panel.capacity()
