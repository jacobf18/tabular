import random

import pytest
import torch

from single_model_column_objective import column_majorizer
from single_model_query_budget import budget_for_seed, query_budget_majorizer, subsample_query_mask


def fixture_arrays():
    generator = torch.Generator().manual_seed(123)
    truth = torch.randn((1, 32, 5), generator=generator)
    mask = torch.rand(truth.shape, generator=generator) > .3
    mask[:, :, 0] = False
    mask[:, 0, 0] = True
    mask[:, :, 1] = False
    teacher = truth + .2 * torch.randn(truth.shape, generator=generator)
    return truth, mask, teacher


@pytest.mark.parametrize("budget", [2, 4, 8, 16])
def test_query_budget_keeps_exact_per_column_counts_and_original_mask(budget):
    _, mask, _ = fixture_arrays()
    previous = mask.clone()
    subset = subsample_query_mask(mask, seed=43, budget=budget)
    assert torch.equal(mask, previous)
    assert not (subset & ~mask).any()
    assert torch.equal(subset.sum(1), mask.sum(1).clamp_max(budget))
    assert torch.equal(subset, subsample_query_mask(mask, seed=43, budget=budget))


def test_private_sampling_preserves_global_rng_states():
    _, mask, _ = fixture_arrays()
    python_state, torch_state = random.getstate(), torch.random.get_rng_state().clone()
    budget = budget_for_seed(73)
    subsample_query_mask(mask, seed=73, budget=budget)
    assert random.getstate() == python_state
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_combined_weights_reproduce_two_separately_averaged_objectives_and_gradients():
    truth, mask, teacher = fixture_arrays()
    seed = 42
    subset = subsample_query_mask(mask, seed=seed, budget=budget_for_seed(seed))
    full_weights, full_offset, _ = column_majorizer(truth, mask, teacher)
    subset_weights, subset_offset, _ = column_majorizer(truth, subset, teacher)
    combined, offset, diagnostics = query_budget_majorizer(truth, mask, teacher, seed=seed)
    prediction = (teacher + .1).clone().requires_grad_()
    error = (prediction - truth).square()
    expected = .5 * ((full_weights * error)[mask].mean() + full_offset)
    expected += .5 * ((subset_weights * error)[subset].mean() + subset_offset)
    actual = (combined * error)[mask].mean() + offset
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual, prediction, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, prediction)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.isfinite(actual_gradient).all() and not actual_gradient[~mask].any()
    assert diagnostics["full_to_subsampled_query_ratio"] == int(mask.sum()) / int(subset.sum())


def test_large_budget_reduces_to_original_objective_without_extra_queries():
    truth, mask, teacher = fixture_arrays()
    truth, mask, teacher = truth[:, :2], mask[:, :2], teacher[:, :2]
    expected_weights, expected_offset, _ = column_majorizer(truth, mask, teacher)
    actual_weights, actual_offset, _ = query_budget_majorizer(truth, mask, teacher, seed=42)
    torch.testing.assert_close(actual_weights[mask], expected_weights[mask])
    torch.testing.assert_close(actual_offset, expected_offset)


def test_bad_masks_budgets_and_seeds_fail_closed():
    _, mask, _ = fixture_arrays()
    for budget in (0, 3, True):
        with pytest.raises(ValueError):
            subsample_query_mask(mask, seed=1, budget=budget)
    for seed in (-1, True, 1.5):
        with pytest.raises(ValueError):
            budget_for_seed(seed)
    with pytest.raises(ValueError):
        subsample_query_mask(torch.zeros_like(mask), seed=1, budget=2)
