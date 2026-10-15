from dataclasses import replace

import pytest
import torch

from tabimpute.prior.recsys_rating_episode import sample_rating_episode
from tabimpute.prior.recsys_ranking_objective import synthetic_ranking_objective, topk_expected_pairwise_loss


def test_top_one_loss_matches_hand_computed_expected_gain_weights():
    prediction = torch.tensor([3., 2., 1.], requires_grad=True)
    loss, diagnostics = topk_expected_pairwise_loss(prediction, torch.tensor([.1, .8, .2]),
        torch.zeros(3, dtype=torch.long), torch.arange(3), torch.tensor([True, False, False]), (1, 3), top_k=1)
    expected = (.7 * torch.nn.functional.softplus(prediction[0] - prediction[1])
                + .1 * torch.nn.functional.softplus(prediction[0] - prediction[2])) / .8
    torch.testing.assert_close(loss, expected)
    actual_gradient = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, prediction)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert actual_gradient[0] > 0 and actual_gradient[1] < 0 and actual_gradient[2] < 0
    assert diagnostics["ranking_paired_users"] == 1


def test_topk_loss_matches_explicit_unique_pair_enumeration():
    prediction = torch.tensor([1., 4., 3., 2.], requires_grad=True)
    probability = torch.tensor([.8, .1, .3, .6])
    loss, _ = topk_expected_pairwise_loss(prediction, probability, torch.zeros(4, dtype=torch.long),
        torch.arange(4), torch.tensor([True, False, False, False]), (1, 4), temperature=.3, top_k=2)
    numerator = prediction.sum() * 0
    denominator = 0
    ordering = prediction.argsort(descending=True)
    for first_position in range(2):
        for second_position in range(first_position + 1, 4):
            first_item, second_item = ordering[first_position], ordering[second_position]
            first_discount = 1 / torch.tensor(first_position + 2.).log2()
            second_discount = 1 / torch.tensor(second_position + 2.).log2() if second_position < 2 else 0
            difference = probability[first_item] - probability[second_item]
            weight = difference.abs() * (first_discount - second_discount)
            numerator = numerator + weight * torch.nn.functional.softplus(
                -difference.sign() * (prediction[first_item] - prediction[second_item]) / .3)
            denominator = denominator + weight
    torch.testing.assert_close(loss, numerator / denominator)


def test_user_macro_weighting_ignores_missing_items_and_rating_only_users():
    prediction = torch.tensor([3., 2., 2., 3., 9.], requires_grad=True)
    probabilities = torch.tensor([.1, .8, .2, .1, .9])
    users = torch.tensor([0, 0, 1, 1, 2])
    items = torch.tensor([0, 2, 1, 3, 0])
    selected = torch.tensor([True, False, True, False, True])
    loss, diagnostics = topk_expected_pairwise_loss(prediction, probabilities, users, items, selected, (3, 5), top_k=3)
    torch.testing.assert_close(loss, torch.nn.functional.softplus(torch.tensor(1.)))
    assert diagnostics["ranking_paired_users"] == 2
    loss.backward()
    assert prediction.grad[4] == 0 and torch.isfinite(prediction.grad).all()


def test_equal_probabilities_have_no_ranking_gradient():
    prediction = torch.tensor([3., 1.], requires_grad=True)
    loss, diagnostics = topk_expected_pairwise_loss(prediction, torch.tensor([.2, .2]),
        torch.zeros(2, dtype=torch.long), torch.arange(2), torch.tensor([True, False]), (1, 2))
    assert loss.item() == 0 and diagnostics["ranking_paired_users"] == 0
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros(2))


@pytest.mark.parametrize("temperature,top_k", [(0., 10), (float("nan"), 10), (1., 0)])
def test_invalid_topk_configuration_is_rejected(temperature, top_k):
    with pytest.raises(ValueError):
        topk_expected_pairwise_loss(None, None, None, None, None, (1, 2), temperature, top_k)


def test_low_temperature_leaves_correctly_ordered_margins_nearly_unchanged():
    prediction = torch.tensor([4., 3., 2.], requires_grad=True)
    loss, _ = topk_expected_pairwise_loss(prediction, torch.tensor([.8, .3, .1]),
        torch.zeros(3, dtype=torch.long), torch.arange(3), torch.tensor([True, False, False]), (1, 3), temperature=.1)
    loss.backward()
    assert loss.item() < .0001 and prediction.grad.abs().max() < .001


@pytest.mark.parametrize("probabilities", [[.1, .8, .2], [.8, .2, .1]])
def test_one_sided_ranking_only_lowers_the_less_relevant_side(probabilities):
    prediction = torch.tensor([3., 2., 1.], requires_grad=True)
    arguments = (prediction, torch.tensor(probabilities), torch.zeros(3, dtype=torch.long),
                 torch.arange(3), torch.tensor([True, False, False]), (1, 3))
    original, _ = topk_expected_pairwise_loss(*arguments, top_k=1)
    one_sided, _ = topk_expected_pairwise_loss(*arguments, top_k=1, negative_only=True)
    torch.testing.assert_close(original, one_sided, rtol=0, atol=0)
    one_sided.backward()
    assert torch.all(prediction.grad >= 0) and prediction.grad.sum() > 0
    assert prediction.grad[torch.tensor(probabilities).argmax()] == 0


def test_centered_one_sided_gradients_preserve_each_users_score_offset():
    prediction = torch.tensor([4., 3., 2., 4., 3., 9.], requires_grad=True)
    probabilities = torch.tensor([.1, .8, .2, .1, .7, .9])
    users = torch.tensor([0, 0, 0, 1, 1, 2])
    items = torch.tensor([0, 1, 2, 0, 1, 0])
    selected = torch.tensor([True, False, False, True, False, True])
    arguments = (prediction, probabilities, users, items, selected, (3, 4))
    ordinary, _ = topk_expected_pairwise_loss(*arguments, top_k=1, negative_only=True)
    centered, _ = topk_expected_pairwise_loss(*arguments, top_k=1, negative_only=True, center_negative_only=True)
    torch.testing.assert_close(ordinary, centered, rtol=0, atol=0)
    raw_gradient = torch.autograd.grad(ordinary, prediction, retain_graph=True)[0]
    actual = torch.autograd.grad(centered, prediction)[0]
    expected = raw_gradient.clone()
    for user in (0, 1):
        expected[users == user] -= raw_gradient[users == user].mean()
        torch.testing.assert_close(actual[users == user].sum(), torch.tensor(0.), atol=1e-7, rtol=0)
    torch.testing.assert_close(actual, expected)
    assert actual[0] > 0 and actual[3] > 0 and actual[5] == 0


def test_gradient_centering_requires_explicit_one_sided_mode():
    with pytest.raises(ValueError, match="one-sided"):
        topk_expected_pairwise_loss(None, None, None, None, None, (1, 2), center_negative_only=True)


@pytest.mark.parametrize("negative_only,center_negative_only", [(False, False), (True, False), (True, True)])
def test_complete_topk_objective_requires_synthetic_relevance_and_preserves_components(negative_only, center_negative_only):
    class Criterion:
        def mean(self, logits):
            return logits[:, 0]

        def __call__(self, logits, target):
            return (logits[:, 0] - target).square()

    episode = sample_rating_episode(219, rows=16, cols=24, parent_rows=64, parent_cols=96,
        density_range=(.2, .2), ranking_query_mode="user_catalog", relevance_targets=True, ranking_max_users=4,
        negatives_per_user=100, query_limit=50, ordinal_targets=True, point_target="expected_rating")
    logits = torch.zeros(int(episode.query.sum()), 1, requires_grad=True)
    loss, diagnostics, components = synthetic_ranking_objective(logits, episode, Criterion(),
        ranking_objective="topk_expected_pairwise", temperature=.1, return_components=True, ranking_boundary_weight=1,
        negative_only_ranking=negative_only, center_negative_only_ranking=center_negative_only)
    assert diagnostics["ranking_paired_users"] == 4
    torch.testing.assert_close(loss, components["protected"] + components["ranking"])
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="relevance"):
        synthetic_ranking_objective(logits, replace(episode, relevance_probabilities=None), Criterion(),
                                   ranking_objective="topk_expected_pairwise")
    with pytest.raises(ValueError, match="synthetic"):
        synthetic_ranking_objective(logits, replace(episode, metadata={**episode.metadata, "gradient_data": "real"}),
                                   Criterion(), ranking_objective="topk_expected_pairwise")
