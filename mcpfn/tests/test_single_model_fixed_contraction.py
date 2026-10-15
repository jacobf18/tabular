from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import single_model_fixed_contraction as contraction
from test_single_model_structured_mixture import metrics


def test_one_fixed_parameter_set_preserves_inventory_and_sources():
    parent = {"weight": torch.tensor([1., -2., 0.]), "counter": torch.tensor(3)}
    tail = {"weight": torch.tensor([5., 2., 4.]), "counter": torch.tensor(3)}
    original_parent, original_tail = deepcopy(parent), deepcopy(tail)
    candidate = contraction.interpolate(parent, tail)
    assert torch.equal(candidate["weight"], torch.tensor([2., -1., 1.]))
    assert candidate["weight"].dtype == parent["weight"].dtype
    report = contraction.check_arithmetic(parent, tail, candidate)
    assert report["changed_tensors"] == 1
    assert report["elements"] == 4
    assert report["non_bitwise_elements"] == 0
    assert all(torch.equal(parent[key], value) for key, value in original_parent.items())
    assert all(torch.equal(tail[key], value) for key, value in original_tail.items())


@pytest.mark.parametrize("kind", ["alpha", "inventory", "shape", "dtype", "nonfinite", "integer"])
def test_invalid_weight_constructions_are_rejected(kind):
    parent = {"weight": torch.tensor([1., 2.]), "counter": torch.tensor(3)}
    tail = deepcopy(parent)
    alpha = contraction.ALPHA
    if kind == "alpha":
        alpha = .5
    elif kind == "inventory":
        del tail["counter"]
    elif kind == "shape":
        tail["weight"] = torch.ones(3)
    elif kind == "dtype":
        tail["weight"] = tail["weight"].double()
    elif kind == "nonfinite":
        tail["weight"][0] = torch.nan
    else:
        tail["counter"] += 1
    with pytest.raises(ValueError):
        contraction.interpolate(parent, tail, alpha)


def test_independent_arithmetic_detects_wrong_candidate():
    parent = {"weight": torch.tensor([1., 2.])}
    tail = {"weight": torch.tensor([2., 3.])}
    candidate = contraction.interpolate(parent, tail)
    candidate["weight"][0] += .01
    with pytest.raises(ValueError, match="does not reproduce"):
        contraction.check_arithmetic(parent, tail, candidate)


def improved_metrics():
    baseline = metrics()
    candidate = deepcopy(baseline)
    for domain, name in contraction.mixture.column.RETENTION_METRICS:
        candidate[domain][name] *= .9
    for name in contraction.mixture.STRUCTURED_METRICS:
        candidate["structured"]["overall"][name] *= .9
    for family in contraction.mixture.FAMILIES:
        candidate["structured"]["families"][family]["common_nrmse"] *= .9
    return baseline, candidate


def test_selection_requires_all22_guards_and_strict_minimax():
    baseline, candidate = improved_metrics()
    base = {"retention_metrics": contraction.mixture.column.RETENTION_METRICS}
    result = contraction.selection_decision(candidate, baseline, baseline, base)
    assert result["passed"] and len(result["comparisons"]) == 22
    assert len(result["ratios"]) == 16
    assert not contraction.selection_decision(baseline, baseline, baseline, base)["passed"]
    for group, metric in contraction.mixture.column.RETENTION_METRICS:
        changed = deepcopy(candidate)
        changed[group][metric] = baseline[group][metric] * 1.000001
        assert not contraction.selection_decision(changed, baseline, baseline, base)["passed"]
    for family in contraction.mixture.FAMILIES:
        changed = deepcopy(candidate)
        changed["structured"]["families"][family]["common_nrmse"] = baseline["structured"]["families"][family]["common_nrmse"] * 1.000001
        assert not contraction.selection_decision(changed, baseline, baseline, base)["passed"]
    for metric in contraction.mixture.STRUCTURED_METRICS:
        changed = deepcopy(candidate)
        changed["structured"]["overall"][metric] = baseline["structured"]["overall"][metric] * 1.000001
        assert not contraction.selection_decision(changed, baseline, baseline, base)["passed"]


def test_fresh_confirmation_has_every_family_generator_and_both_regimes():
    cases = contraction.confirmation_cases()
    assert len(cases) == 284
    assert sum("structured" in case for case in cases) == 192
    assert len(contraction.seeds_in(cases)) == 284
    assert not contraction.seeds_in(cases).intersection(contraction.seeds_in(
        contraction.mixture.structured_bank(contraction.mixture.SEEDS["structured_confirmation_seed"], confirmation=True)))
    for family in contraction.mixture.FAMILIES:
        selected = [case["structured"] for case in cases if case.get("structured", {}).get("family") == family]
        assert len(selected) == 16
        assert {case["component"] for case in selected} == set(contraction.mixture.COMPONENTS)
        assert {case["regime"] for case in selected} == {"legacy", "area_scaled"}


def test_reference_floor_keeps_the_original_two_controls():
    general, teacher = improved_metrics()
    reference = contraction.reference_values(general, teacher)
    assert reference["imputation"] == {name: teacher["imputation"][name] for name in contraction.mixture.STRUCTURED_METRICS}
    assert reference["recsys"] is teacher["recsys"]
    assert set(reference["structured"]["families"]) == set(contraction.mixture.FAMILIES)


def test_candidate_must_declare_its_derived_source_step():
    state = {"step": 400, "epoch": 400, "config_sha256": "fixed", "derived_checkpoint": True,
             "gradient_updates_performed_here": 0, "alpha": .25, "step_semantics": contraction.SEMANTICS}
    contraction.check_candidate_metadata(state, "fixed")
    for key, value in (("step", 250), ("alpha", .5), ("derived_checkpoint", False),
                       ("gradient_updates_performed_here", 400), ("config_sha256", "changed")):
        changed = {**state, key: value}
        with pytest.raises(ValueError):
            contraction.check_candidate_metadata(changed, "fixed")


@pytest.mark.parametrize("confirmed", [False, True])
def test_qualification_report_does_not_hide_failed_general_gate(tmp_path, confirmed):
    baseline, candidate = improved_metrics()
    base = {"retention_metrics": contraction.mixture.column.RETENTION_METRICS}
    selection = contraction.selection_decision(candidate, baseline, baseline, base)
    (tmp_path / "selection").mkdir()
    contraction.write_json(tmp_path / "selection/decision.json", selection)
    gate = {"passed": False, "confirmation_evaluated": confirmed,
            "legacy_general_retained": False, "structured_general_retained": False}
    if confirmed:
        (tmp_path / "confirmation").mkdir()
        contraction.write_json(tmp_path / "confirmation/decision.json", selection)
        contraction.write_json(tmp_path / "confirmation/initialization.json", {"student": baseline})
        contraction.write_json(tmp_path / "confirmation/student.json", {"student": candidate})
    contraction.report_qualification(tmp_path, gate, base)
    text = (tmp_path / "RESULTS.md").read_text()
    assert "Final synthetic gate passed: **False**" in text
    assert text.count("all22 initialization retention guards") == 1 + int(confirmed)
    assert "Real-benchmark success is not assessed" in text
    if confirmed:
        assert "legacy general-control retention: False" in text


def test_raw_audit_binds_the_complete_prediction_grid_and_file_hash(tmp_path, monkeypatch):
    episode = SimpleNamespace(metadata={"seed": 123}, target=torch.zeros(1, 2, 2))
    monkeypatch.setattr(contraction.mixture.backend, "restore_episode", lambda cached, device: episode)
    monkeypatch.setattr(contraction.mixture.defined, "episode_metrics", lambda prediction, restored: {"rmse": float(prediction.square().mean().sqrt())})
    path = tmp_path / "prediction.npz"
    np.savez(path, student=np.zeros((1, 2, 2), dtype=np.float32))
    report = {"cases": [{"index": 0, "seed": 123, "prediction_file": str(path),
                         "prediction_sha256": contraction.sha256(path), "student": {"rmse": 0.}}]}
    count, hashes = contraction.raw_audit([{}], [(report, ("student",))])
    assert count == 1 and hashes == {str(path): contraction.sha256(path)}
    np.savez(path, student=np.ones((1, 2, 2), dtype=np.float32))
    with pytest.raises((ValueError, RuntimeError)):
        contraction.raw_audit([{}], [(report, ("student",))])
