import numpy as np
import pytest
import torch

from single_model_family_projection import gram_and_products
from single_model_point_proposal_grid import (
    direction_products, displacement_decision, finite_decision, scaled_parameters, specifications,
)


def test_nine_fixed_unique_proposals():
    cases = specifications()
    assert len(cases) == len({case["name"] for case in cases}) == 9
    assert cases[0] == {"name": "margin_0.001_scale_1", "margin": .001, "scale": 1.}
    assert cases[-1] == {"name": "margin_0.1_scale_0.25", "margin": .1, "scale": .25}


@pytest.mark.parametrize("scale", [1., .5, .25])
def test_contraction_uses_actual_fp32_endpoint(scale):
    before = torch.tensor([1., -.05, 1e-9])
    after = before - torch.tensor([1e-5, 1e-7, 1e-12])
    expected = (before.double() + scale * (after.double() - before.double())).float()
    torch.testing.assert_close(scaled_parameters(before, after, scale), expected, rtol=0, atol=0)


def test_contraction_rejects_unregistered_scale():
    with pytest.raises(ValueError, match="registered"):
        scaled_parameters(torch.ones(2), torch.zeros(2), .1)


def test_products_reproduce_full_gram_helper_without_recomputing_gram():
    torch.manual_seed(57)
    gradients, direction = torch.randn(7, 101), torch.randn(101).double()
    _, expected, norm = gram_and_products(gradients, direction, chunk_elements=11)
    actual, actual_norm = direction_products(gradients, direction, chunk_elements=11)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    assert actual_norm == norm


def test_roundoff_adversity_and_excessive_norm_are_not_accepted():
    assert not displacement_decision([1., -1e-20], 1., 1.)["admissible"]
    assert not displacement_decision([1., 0.], 2. + 1e-12, 1.)["admissible"]
    assert not displacement_decision([0., 0.], 1., 1.)["admissible"]
    assert displacement_decision([1., 0.], 2., 1.)["admissible"]


def test_all_repeats_not_only_the_mean_must_retain():
    before = {"rmse": [1., 1.]}
    after = {"rmse": [.9, 1.05]}
    result = finite_decision(before, after)
    assert result["improved_slots"] == 1
    assert not result["all_repeats_retained"] and not result["finite_feasible"]


def test_strict_sign_and_complete_coverage():
    before = {"rmse": [1., 1.], "common": [None, None]}
    result = finite_decision(before, {"rmse": [1. + 1e-14, 1. + 1e-14], "common": [None, None]})
    assert result["worsened_slots"] == 1 and not result["finite_feasible"]
    assert result["undefined_slots"] == 1
    with pytest.raises(ValueError, match="every task"):
        finite_decision(before, {"rmse": [.9, .9]})
    with pytest.raises(ValueError, match="coverage"):
        finite_decision(before, {"rmse": [.9, .9], "common": [1., 1.]})


def test_some_strict_improvement_is_required():
    before = {"rmse": [1., 1.]}
    assert not finite_decision(before, before)["finite_feasible"]
    assert finite_decision(before, {"rmse": [.9, .9]})["finite_feasible"]
