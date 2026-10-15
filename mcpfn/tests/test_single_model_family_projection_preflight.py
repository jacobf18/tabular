from copy import deepcopy

import numpy as np
import pytest
import torch

import preflight_single_model_family_projection as preflight


def test_registration_has_every_family_fresh_seeds_and_explicit_maximum_shapes():
    cases = preflight.specifications()
    assert len(cases) == 14 and [case["index"] for case in cases] == list(range(14))
    assert [case["name"] for case in cases] == ["legacy", *preflight.mixture.FAMILIES, "recsys"]
    assert sum(case["weight"] for case in cases) == pytest.approx(1.)
    assert cases[0]["weight"] == .25 and cases[-1]["weight"] == .5
    assert sum(case["weight"] for case in cases[1:-1]) == pytest.approx(.25)
    assert len(preflight.fixed.seeds_in(cases)) == 28
    assert cases[0]["specification"]["shape"] == cases[-1]["specification"]["shape"] == [2000, 1000]
    for case in cases[1:-1]:
        record = case["specification"]["structured"]
        expected = preflight.mixture.COMPACT_SHAPES if case["name"] in preflight.mixture.NEURAL_FAMILIES else preflight.mixture.SPECTRAL_SHAPES
        assert record["shape"] == list(expected[-1]) and record["requested_missingness"] == .6


def test_flatten_and_assign_preserve_exact_original_coordinates():
    model = torch.nn.Linear(3, 2)
    original = preflight.flatten(model)
    proposed = original + .125
    preflight.assign(model, proposed)
    assert torch.equal(preflight.flatten(model), proposed)
    preflight.assign(model, torch.ones_like(original), gradients=True)
    assert torch.equal(preflight.flatten(model, gradients=True), torch.ones_like(original))
    assert torch.equal(preflight.flatten(model), proposed)


@pytest.mark.parametrize("kind", ["missing_gradient", "nonfinite_gradient", "nonfinite_parameter", "inventory"])
def test_partial_or_nonfinite_parameter_coordinates_are_rejected(kind):
    model = torch.nn.Linear(3, 2)
    if kind == "missing_gradient":
        with pytest.raises(ValueError):
            preflight.flatten(model, gradients=True)
    elif kind == "nonfinite_gradient":
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, torch.nan)
        with pytest.raises(ValueError):
            preflight.flatten(model, gradients=True)
    elif kind == "nonfinite_parameter":
        with torch.no_grad():
            model.weight[0, 0] = torch.inf
        with pytest.raises(ValueError):
            preflight.flatten(model)
    else:
        with pytest.raises(ValueError):
            preflight.assign(model, torch.ones(3))


def test_saved_checkpoint_is_explicitly_unselected_preflight_and_never_overwritten(tmp_path):
    preflight.write_json(tmp_path / "config.json", {"fixture": True})
    model = torch.nn.Linear(3, 2)
    record = preflight.save_checkpoint(model, tmp_path, "projected")
    state = torch.load(record["checkpoint"], map_location="cpu", weights_only=True)
    assert state["preflight_only"] and not state["selection_qualified"]
    assert state["step"] == state["epoch"] == state["gradient_updates_performed"] == 1
    assert state["role"] == "projected"
    assert state["config_sha256"] == preflight.sha256(tmp_path / "config.json")
    assert not (tmp_path / "best_validation_checkpoint.pth").exists()
    with pytest.raises(FileExistsError):
        preflight.save_checkpoint(model, tmp_path, "projected")
    with pytest.raises(ValueError):
        preflight.save_checkpoint(model, tmp_path, "selected")


def certificate():
    tasks = torch.tensor([[1., 0.], [-10., 1.]])
    direction = torch.tensor([2., 2.])
    gram, products, norm = preflight.projection.gram_and_products(tasks, direction)
    result = preflight.projection.solve_projection(gram, products, norm)
    return gram, products, norm, result


def test_cpu_certificate_reconstructs_coefficients_and_all_kkt_conditions():
    gram, products, norm, result = certificate()
    preflight.verify_certificate(gram, products, norm, result, preflight.projection.SPECIFICATION["relative_inward_margin"])


@pytest.mark.parametrize("kind", ["coefficient", "negative_dual", "zero_update", "loosened_margin", "products"])
def test_certificate_cannot_hide_changed_projection_or_failed_constraint(kind):
    gram, products, norm, result = certificate()
    result = deepcopy(result)
    margin = preflight.projection.SPECIFICATION["relative_inward_margin"]
    if kind == "coefficient":
        result["raw_gradient_coefficients"][1] += .1
    elif kind == "negative_dual":
        result["dual_multipliers"][1] = -1.
    elif kind == "zero_update":
        norm = 0.
    elif kind == "loosened_margin":
        margin = 1.
    else:
        products = np.asarray(products) - 1.
    with pytest.raises((ValueError, AssertionError)):
        preflight.verify_certificate(gram, products, norm, result, margin)
