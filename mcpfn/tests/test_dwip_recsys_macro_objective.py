import itertools

import pytest
import torch

from tabimpute.prior.recsys_rating_episode import (
    expected_macro_rmse, rating_objective, sample_rating_episode,
)


def test_singleton_users_use_exact_expected_absolute_error():
    prediction = torch.tensor([2.4, 3.8], requires_grad=True)
    probabilities = torch.tensor([[.1, .2, .3, .3, .1], [.0, .1, .1, .4, .4]])
    actual = expected_macro_rmse(prediction, probabilities, torch.tensor([0, 3]), 5, seed=41)
    expected = ((prediction[:, None] - torch.arange(1, 6)).abs() * probabilities).sum(1).mean()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert torch.isfinite(prediction.grad).all()


def test_expected_macro_rmse_matches_exact_finite_query_risk_and_gradient():
    prediction = torch.tensor([2.4, 3.8, 4.1], requires_grad=True)
    probabilities = torch.tensor([[.1, .2, .3, .3, .1], [.0, .1, .1, .4, .4], [.0, .2, .3, .4, .1]])
    exact_multi = prediction.sum() * 0
    for first, second in itertools.product(range(5), repeat=2):
        squared = (prediction[:2] - torch.tensor([first + 1, second + 1])).square().mean()
        exact_multi = exact_multi + probabilities[0, first] * probabilities[1, second] * (squared + 1e-6).sqrt()
    exact_single = ((prediction[2] - torch.arange(1, 6)).abs() * probabilities[2]).sum()
    exact = (exact_multi + exact_single) / 2
    estimate = expected_macro_rmse(prediction, probabilities, torch.tensor([0, 0, 1]), 2, seed=42, samples=32768)
    torch.testing.assert_close(estimate, exact, atol=.005, rtol=0)
    exact_gradient = torch.autograd.grad(exact, prediction, retain_graph=True)[0]
    estimated_gradient = torch.autograd.grad(estimate, prediction)[0]
    torch.testing.assert_close(estimated_gradient, exact_gradient, atol=.004, rtol=0)


def test_expected_macro_rmse_is_reproducible_without_global_rng_changes():
    prediction = torch.tensor([1.5, 4.5])
    probabilities = torch.full((2, 5), .2)
    users = torch.zeros(2, dtype=torch.long)
    state = torch.get_rng_state().clone()
    first = expected_macro_rmse(prediction, probabilities, users, 1, seed=43)
    second = expected_macro_rmse(prediction, probabilities, users, 1, seed=43)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(state, torch.get_rng_state())


def test_deterministic_targets_match_user_rmse_and_have_finite_perfect_gradient():
    prediction = torch.tensor([2., 4., 5.], requires_grad=True)
    probabilities = torch.nn.functional.one_hot(torch.tensor([1, 3, 4]), 5).float()
    actual = expected_macro_rmse(prediction, probabilities, torch.tensor([0, 0, 1]), 2, seed=44)
    assert actual.item() == pytest.approx(.0005)
    actual.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.equal(prediction.grad, torch.zeros(3))


def test_macro_objective_uses_synthetic_distribution_not_real_or_denoised_labels():
    class Criterion:
        def mean(self, logits):
            return logits[:, 0]

        def __call__(self, logits, target):
            return (logits[:, 0] - target).square()

    episode = sample_rating_episode(45, 24, 20, parent_rows=48, density_range=(.2, .2),
                                     ordinal_targets=True, point_target="expected_rating")
    logits = torch.zeros(int(episode.query.sum()), 1, requires_grad=True)
    loss, diagnostics = rating_objective(logits, episode, Criterion(), nll_weight=0, point_objective="expected_user_rmse")
    assert diagnostics["point_objective"] == "expected_user_rmse"
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    episode.point_target.add_(100)
    changed, _ = rating_objective(logits, episode, Criterion(), nll_weight=0, point_objective="expected_user_rmse")
    torch.testing.assert_close(loss, changed, rtol=0, atol=0)
    episode.metadata["gradient_data"] = "inner_training_ratings"
    with pytest.raises(ValueError, match="requires synthetic ordinal"):
        rating_objective(logits, episode, Criterion(), point_objective="expected_user_rmse")


@pytest.mark.parametrize("samples", [0, 1, 3])
def test_invalid_monte_carlo_sample_count_rejected(samples):
    with pytest.raises(ValueError, match="even sample count"):
        expected_macro_rmse(torch.zeros(2), torch.full((2, 5), .2), torch.zeros(2, dtype=torch.long), 1, 0, samples)


def test_parent_count_surrogate_matches_exact_repeated_profile_risk_and_gradient():
    prediction = torch.tensor([3.6], requires_grad=True)
    probabilities = torch.tensor([[.1, .1, .2, .4, .2]])
    exact = prediction.sum() * 0
    for first, second in itertools.product(range(5), repeat=2):
        error = ((prediction[0] - first - 1).square() + (prediction[0] - second - 1).square()) / 2
        exact = exact + probabilities[0, first] * probabilities[0, second] * (error + 1e-6).sqrt()
    actual = expected_macro_rmse(prediction, probabilities, torch.tensor([0]), 1, seed=46,
                                 samples=32768, effective_counts=torch.tensor([2]))
    torch.testing.assert_close(actual, exact, rtol=0, atol=.005)
    expected_gradient = torch.autograd.grad(exact, prediction, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, prediction)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=.005)


def test_parent_counts_equal_to_decoded_counts_preserve_risk_bitwise():
    prediction = torch.tensor([2.4, 3.8, 4.1])
    probabilities = torch.full((3, 5), .2)
    users = torch.tensor([0, 2, 2])
    original = expected_macro_rmse(prediction, probabilities, users, 4, seed=47)
    matched = expected_macro_rmse(prediction, probabilities, users, 4, seed=47, effective_counts=torch.tensor([1, 0, 2, 0]))
    torch.testing.assert_close(original, matched, rtol=0, atol=0)


def test_parent_count_replication_is_user_balanced_with_unsorted_ids_and_remainders():
    prediction = torch.tensor([2., 4., 3.], requires_grad=True)
    probabilities = torch.nn.functional.one_hot(torch.tensor([0, 4, 1]), 5).float()
    actual = expected_macro_rmse(prediction, probabilities, torch.tensor([2, 0, 2]), 4, seed=48,
                                 effective_counts=torch.tensor([3, 0, 5, 10]))
    torch.testing.assert_close(actual, torch.tensor(1.000001).sqrt())
    actual.backward()
    assert torch.isfinite(prediction.grad).all()
    assert actual.item() < 1.01


def test_parent_query_metadata_counts_the_full_parent_not_only_decoded_crop():
    episode = sample_rating_episode(49, 24, 20, parent_rows=96, parent_cols=80, query_limit=20,
                                     density_range=(.2, .2), exclude_unknown_queries=True)
    local = episode.rating_query[0].sum(1)
    assert torch.all(episode.global_query_counts >= local)
    assert torch.any(episode.global_query_counts[local > 0] > local[local > 0])


@pytest.mark.parametrize("counts", [torch.tensor([0]), torch.tensor([1.]), torch.tensor([1, 2])])
def test_invalid_parent_query_counts_rejected(counts):
    with pytest.raises(ValueError, match="query counts"):
        expected_macro_rmse(torch.tensor([3.]), torch.full((1, 5), .2), torch.tensor([0]), 1, 50, effective_counts=counts)


@pytest.mark.parametrize("user_noise_range", [(0., 0.), (.2, .6)])
def test_parent_population_objective_integrates_synthetic_mixture_and_finite_gradients(user_noise_range):
    class Criterion:
        def mean(self, logits):
            return logits[:, 0]

        def __call__(self, logits, target):
            return (logits[:, 0] - target).square()

    episode = sample_rating_episode(51, 24, 20, parent_rows=96, parent_cols=80, query_limit=48,
                                     density_range=(.2, .2), ordinal_targets=True, point_target="expected_rating",
                                     uniform_noise_range=(.1, .1), user_noise_log_std_range=user_noise_range)
    logits = torch.zeros(int(episode.query.sum()), 1, requires_grad=True)
    loss, diagnostics = rating_objective(logits, episode, Criterion(), point_objective="expected_user_rmse", macro_query_scope="parent")
    active = episode.rating_query[0].any(1)
    assert diagnostics["macro_population_queries"] == int(episode.global_query_counts[active].sum())
    assert diagnostics["macro_population_queries"] > int(episode.rating_query.sum())
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    episode.global_query_counts = None
    with pytest.raises(ValueError, match="synthetic global query counts"):
        rating_objective(logits, episode, Criterion(), point_objective="expected_user_rmse", macro_query_scope="parent")
