from types import SimpleNamespace

import pytest
import torch

from tabimpute.prior.recsys_rating_episode import sample_rating_episode
from tabimpute.prior.recsys_ranking_objective import bounded_ranking_scores, expected_listwise_loss, synthetic_ranking_objective


def test_bounded_scores_use_frozen_inference_clipping():
    prediction = torch.tensor([.3, 3., 5.8], requires_grad=True)
    scores, penalty = bounded_ranking_scores(prediction, torch.zeros(3, dtype=torch.long),
                                             torch.tensor([True, False, False]), 1)
    torch.testing.assert_close(scores, torch.tensor([1., 3., 5.]))
    torch.testing.assert_close(penalty, torch.tensor((.75 ** 2 + .85 ** 2) / 3))


def test_boundary_loss_resolves_ties_with_finite_inward_gradients():
    prediction = torch.tensor([5.2, 5.8], requires_grad=True)
    users = torch.zeros(2, dtype=torch.long)
    selected = torch.tensor([True, False])
    scores, penalty = bounded_ranking_scores(prediction, users, selected, 1)
    ranking, _ = expected_listwise_loss(scores, torch.tensor([.8, .2]), users, torch.arange(2), selected, (1, 2))
    torch.testing.assert_close(ranking, torch.tensor(2.).log())
    ranking.backward(retain_graph=True)
    torch.testing.assert_close(prediction.grad, torch.zeros(2))
    prediction.grad = None
    penalty.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([.25, .85]))


def test_boundary_is_user_macro_and_excludes_rating_only_users():
    prediction = torch.tensor([6., 6., 0., 9.], requires_grad=True)
    users = torch.tensor([0, 0, 1, 2])
    selected = torch.tensor([True, False, False, True])
    _, penalty = bounded_ranking_scores(prediction, users, selected, 3)
    assert penalty.item() == pytest.approx(1.05 ** 2)
    penalty.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([.525, .525, -1.05, 0.]))


def test_interior_and_rating_only_cases_have_zero_boundary_loss():
    for values, selected in (([2., 4.], [True, False]), ([6., 7.], [True, True])):
        prediction = torch.tensor(values, requires_grad=True)
        _, penalty = bounded_ranking_scores(prediction, torch.zeros(2, dtype=torch.long),
                                            torch.tensor(selected), 1)
        assert penalty.item() == 0
        penalty.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros(2))


@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf")])
def test_invalid_boundary_weights_are_rejected(weight):
    episode = SimpleNamespace(metadata={"gradient_data": "synthetic"})
    with pytest.raises(ValueError, match="nonnegative"):
        synthetic_ranking_objective(None, episode, None, ranking_boundary_weight=weight)


@pytest.mark.parametrize("boundary_weight", [0., 1.])
def test_bounded_synthetic_objective_preserves_protected_decomposition(boundary_weight):
    class Criterion:
        def mean(self, logits):
            return logits[:, 0]

        def __call__(self, logits, target):
            return (logits[:, 0] - target).square()

    episode = sample_rating_episode(217, rows=16, cols=24, parent_rows=64, parent_cols=96,
                                     density_range=(.2, .2), ranking_query_mode="user_catalog", relevance_targets=True,
                                     ranking_max_users=4, negatives_per_user=100, query_limit=50, ordinal_targets=True,
                                     point_target="expected_rating", normalization="two_way_user_scale")
    logits = torch.full((int(episode.query.sum()), 1), 3., requires_grad=True)
    loss, diagnostics, components = synthetic_ranking_objective(
        logits, episode, Criterion(), ranking_objective="expected_listwise", ranking_weight=.7,
        ranking_boundary_weight=boundary_weight, return_components=True)
    torch.testing.assert_close(loss, components["protected"] + components["ranking"])
    assert components["ranking"].item() == pytest.approx(
        .7 * (diagnostics["ranking"] + boundary_weight * diagnostics["ranking_boundary"]))
    assert diagnostics["ranking_score_transform"] == ("frozen_clip_1_5" if boundary_weight else "raw")
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    episode.metadata["gradient_data"] = "real"
    with pytest.raises(ValueError, match="synthetic"):
        synthetic_ranking_objective(logits, episode, Criterion(), ranking_boundary_weight=boundary_weight)
