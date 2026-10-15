import pytest
import torch

from tabimpute.prior.recsys_rating_episode import conditional_relevance_probability, sample_rating_episode
from tabimpute.prior.recsys_ranking_objective import expected_listwise_loss, synthetic_ranking_objective


def test_conditional_relevance_matches_exact_probability_and_boundaries():
    exposure = torch.tensor([0., .2, .8, 1.], dtype=torch.float64)
    positive = torch.tensor([.1, .3, .8, 1.], dtype=torch.float64)
    probability = conditional_relevance_probability(exposure, positive, .2)
    torch.testing.assert_close(probability, torch.tensor([0., .012 / .84, .128 / .36, 1.], dtype=torch.float64))
    assert conditional_relevance_probability(torch.ones(1), torch.ones(1), .27).item() <= 1
    with pytest.raises(ValueError):
        conditional_relevance_probability(exposure, positive, 0)
    with pytest.raises(ValueError):
        conditional_relevance_probability(torch.tensor([1.2]), torch.ones(1), .2)


def test_conditional_relevance_matches_synthetic_monte_carlo():
    generator = torch.Generator().manual_seed(201)
    exposure = torch.tensor([.05, .3, .8, 1.])
    positive = torch.tensor([.2, .5, .8, .9])
    exposed = torch.rand(200000, 4, generator=generator) < exposure
    held_out = exposed & (torch.rand(200000, 4, generator=generator) < .25)
    observed = exposed & ~held_out
    relevant = held_out & (torch.rand(200000, 4, generator=generator) < positive)
    empirical = relevant.sum(0) / (~observed).sum(0)
    expected = conditional_relevance_probability(exposure, positive, .25)
    torch.testing.assert_close(empirical, expected, atol=.004, rtol=0)


def test_relevance_targets_do_not_change_default_synthetic_context_or_queries():
    arguments = {"seed": 202, "rows": 24, "cols": 32, "parent_rows": 64, "parent_cols": 128,
                 "normalization": "two_way_user_scale", "ordinal_targets": True, "point_target": "expected_rating",
                 "ranking_query_mode": "user_matched", "context_ordering": "first_observation",
                 "density_range": (.1, .1), "uniform_noise_range": (.1, .1)}
    original = sample_rating_episode(**arguments)
    oracle = sample_rating_episode(**arguments, relevance_targets=True)
    for name in ("source", "target", "query", "rating_query", "means", "stds", "point_target", "row_scales", "ordinal_probabilities"):
        torch.testing.assert_close(getattr(original, name), getattr(oracle, name), atol=0, rtol=0, equal_nan=True)
    assert original.relevance_probabilities is None
    assert torch.isfinite(oracle.relevance_probabilities).all()
    assert torch.all((oracle.relevance_probabilities >= 0) & (oracle.relevance_probabilities <= 1))


def test_full_catalog_contains_all_known_unobserved_candidates_for_chosen_users():
    episode = sample_rating_episode(203, rows=24, cols=32, parent_rows=64, parent_cols=128,
                                     density_range=(.2, .2), ranking_query_mode="user_catalog", relevance_targets=True,
                                     ranking_max_users=4, negatives_per_user=100, query_limit=50)
    catalog_users = (episode.query[0] & ~episode.rating_query[0]).any(1)
    assert catalog_users.sum() == 4
    eligible = torch.isnan(episode.source[0]) & (episode.global_col_support > 0)[None]
    assert torch.equal(episode.query[0, catalog_users], eligible[catalog_users])
    with pytest.raises(ValueError, match="relevance"):
        sample_rating_episode(203, ranking_query_mode="user_catalog")


def test_listwise_optimum_matches_target_entropy_and_has_zero_gradient():
    distribution = torch.tensor([.6, .3, .1])
    prediction = distribution.log().requires_grad_()
    loss, diagnostics = expected_listwise_loss(prediction, distribution * .2, torch.zeros(3, dtype=torch.long),
                                                torch.arange(3), torch.tensor([True, False, False]), (1, 3))
    torch.testing.assert_close(loss, -(distribution * distribution.log()).sum())
    assert diagnostics["ranking_kl"] == pytest.approx(0, abs=1e-6)
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros(3), atol=1e-7, rtol=0)


def test_listwise_loss_macro_averages_users_and_ignores_rating_only_rows():
    prediction = torch.tensor([1., 3., 2., 4., 1., 2.], requires_grad=True)
    probabilities = torch.tensor([.1, .3, .2, .1, .4, .9])
    users = torch.tensor([0, 0, 1, 1, 1, 2])
    items = torch.tensor([0, 1, 0, 1, 2, 0])
    selected = torch.tensor([True, False, True, False, False, True])
    actual, diagnostics = expected_listwise_loss(prediction, probabilities, users, items, selected, (3, 3), temperature=.5)
    first = -(probabilities[:2] / probabilities[:2].sum() * (prediction[:2] / .5).log_softmax(0)).sum()
    second = -(probabilities[2:5] / probabilities[2:5].sum() * (prediction[2:5] / .5).log_softmax(0)).sum()
    torch.testing.assert_close(actual, (first + second) / 2)
    assert diagnostics["ranking_paired_users"] == 2
    actual.backward()
    assert prediction.grad[5] == 0 and torch.isfinite(prediction.grad).all()


def test_complete_expected_ranking_is_synthetic_only_and_has_finite_gradients():
    class Criterion:
        def mean(self, logits):
            return logits[:, 0]

        def __call__(self, logits, target):
            return (logits[:, 0] - target).square()

    episode = sample_rating_episode(204, rows=24, cols=32, parent_rows=64, parent_cols=128,
                                     density_range=(.2, .2), ranking_query_mode="user_catalog", relevance_targets=True,
                                     ranking_max_users=4, negatives_per_user=100, query_limit=50, ordinal_targets=True,
                                     point_target="expected_rating", normalization="two_way_user_scale")
    logits = torch.zeros(int(episode.query.sum()), 1, requires_grad=True)
    loss, diagnostics = synthetic_ranking_objective(logits, episode, Criterion(), ranking_objective="expected_listwise", temperature=.5)
    assert diagnostics["ranking_paired_users"] == 4
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="catalog positives"):
        synthetic_ranking_objective(logits, episode, Criterion())
    episode.metadata["gradient_data"] = "real"
    with pytest.raises(ValueError, match="synthetic"):
        synthetic_ranking_objective(logits, episode, Criterion(), ranking_objective="expected_listwise")
