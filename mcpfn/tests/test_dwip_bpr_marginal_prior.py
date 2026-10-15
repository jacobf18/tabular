from dataclasses import fields

import pytest
import torch

from tabimpute.prior.recsys_rating_episode import item_marginal_positive_probability, ordinal_rating_probabilities, sample_rating_episode
from tabimpute.prior.recsys_ranking_objective import synthetic_ranking_objective


def test_item_marginal_matches_full_ordinal_latent_expectation():
    preference = torch.tensor([[2., 3., 4.], [3.5, 4., 5.], [4.5, 2.5, 3.]])
    exposure = torch.tensor([[.2, .4, .8], [.1, .3, .6], [.5, .7, .9]])
    noise = torch.tensor([[.5], [.8], [1.2]])
    actual = item_marginal_positive_probability(exposure, preference, noise, .15)
    positive = ordinal_rating_probabilities(preference, noise, .15)[..., 3:].sum(-1)
    torch.testing.assert_close(actual, (exposure * positive).mean(0))
    assert not actual.requires_grad and torch.all((actual >= 0) & (actual <= 1))


def test_item_marginal_matches_synthetic_monte_carlo():
    generator = torch.Generator().manual_seed(230)
    preference = torch.tensor([[2., 4.], [4.5, 3.5], [3.5, 2.5]])
    exposure = torch.tensor([[.2, .8], [.7, .1], [.4, .6]])
    noise = torch.tensor([[.5], [.8], [1.2]])
    shape = (100000, *preference.shape)
    ratings = (preference + noise * torch.randn(shape, generator=generator)).round().clamp(1, 5)
    ratings = torch.where(torch.rand(shape, generator=generator) < .1,
                          torch.randint(1, 6, shape, generator=generator).float(), ratings)
    exposed = torch.rand(shape, generator=generator) < exposure
    empirical = (exposed & (ratings >= 4)).float().mean((0, 1))
    expected = item_marginal_positive_probability(exposure, preference, noise, .1)
    torch.testing.assert_close(empirical, expected, atol=.003, rtol=0)


def test_uniform_rating_mixture_and_zero_exposure_boundaries():
    exposure = torch.tensor([[0., .4], [0., .8]])
    preference = torch.tensor([[2., 4.], [4., 2.]])
    target = item_marginal_positive_probability(exposure, preference, 1., 1.)
    torch.testing.assert_close(target, .4 * exposure.mean(0))


def test_marginal_target_preserves_every_other_episode_tensor():
    arguments = dict(seed=231, rows=16, cols=24, parent_rows=64, parent_cols=96, density_range=(.2, .2),
                     ranking_query_mode="user_catalog", relevance_targets=True, ranking_max_users=4, negatives_per_user=100,
                     query_limit=50, ordinal_targets=True, point_target="expected_rating", normalization="two_way_user_scale",
                     uniform_noise_range=(.1, .1), user_noise_log_std_range=(.3, .3))
    conditional = sample_rating_episode(**arguments)
    marginal = sample_rating_episode(**arguments, relevance_target="item_marginal_positive")
    for field in fields(conditional):
        original = getattr(conditional, field.name)
        if isinstance(original, torch.Tensor) and field.name != "relevance_probabilities":
            torch.testing.assert_close(original, getattr(marginal, field.name), rtol=0, atol=0, equal_nan=True)
    probability = marginal.relevance_probabilities
    torch.testing.assert_close(probability, probability[:, :1].expand_as(probability), rtol=0, atol=0)
    assert not torch.equal(probability, conditional.relevance_probabilities)
    assert marginal.metadata["relevance_target"] == "item_marginal_positive"
    assert marginal.metadata["gradient_data"] == "synthetic"


def test_marginal_target_requires_explicit_synthetic_relevance_supervision():
    with pytest.raises(ValueError, match="relevance"):
        sample_rating_episode(232, relevance_target="item_marginal_positive")
    with pytest.raises(ValueError, match="Unknown"):
        sample_rating_episode(232, relevance_targets=True, relevance_target="real_popularity")


def test_complete_marginal_objective_is_finite_and_rejects_real_gradients():
    class Criterion:
        def mean(self, logits):
            return logits[:, 0]

        def __call__(self, logits, target):
            return (logits[:, 0] - target).square()

    episode = sample_rating_episode(233, rows=16, cols=24, parent_rows=64, parent_cols=96,
        density_range=(.2, .2), ranking_query_mode="user_catalog", relevance_targets=True,
        relevance_target="item_marginal_positive", ranking_max_users=4, negatives_per_user=100,
        query_limit=50, ordinal_targets=True, point_target="expected_rating")
    logits = torch.zeros(int(episode.query.sum()), 1, requires_grad=True)
    loss, _ = synthetic_ranking_objective(logits, episode, Criterion(), ranking_objective="topk_expected_pairwise")
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    episode.metadata["gradient_data"] = "real"
    with pytest.raises(ValueError, match="synthetic"):
        synthetic_ranking_objective(logits, episode, Criterion(), ranking_objective="topk_expected_pairwise")
