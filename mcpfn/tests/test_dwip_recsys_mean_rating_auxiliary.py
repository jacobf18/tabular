import pytest
import torch

from tabimpute.prior.recsys_rating_episode import rating_objective, sample_rating_episode


class IdentityCriterion:
    def mean(self, logits):
        return logits[:, 0]

    def __call__(self, logits, target):
        return (logits[:, 0] - target).square()


def episode_and_logits():
    episode = sample_rating_episode(107, 24, 20, parent_rows=64, parent_cols=48, query_limit=64,
                                     normalization="two_way_user_scale", density_range=(.2, .2),
                                     ordinal_targets=True, point_target="expected_rating",
                                     uniform_noise_range=(0., .15), user_noise_log_std_range=(.15, .6))
    logits = torch.zeros(int(episode.query.sum()), 1, requires_grad=True)
    return episode, logits


def test_zero_auxiliary_preserves_loss_diagnostics_and_gradients_exactly():
    episode, logits = episode_and_logits()
    arguments = dict(point_objective="expected_user_rmse", macro_query_scope="parent", mae_weight=.1,
                     ranking_weight=.05)
    original, original_diagnostics = rating_objective(logits, episode, IdentityCriterion(), **arguments)
    explicit, explicit_diagnostics = rating_objective(logits, episode, IdentityCriterion(), mean_rating_weight=0,
                                                      **arguments)
    assert torch.equal(original, explicit)
    assert original_diagnostics == explicit_diagnostics
    assert torch.equal(torch.autograd.grad(original, logits)[0], torch.autograd.grad(explicit, logits)[0])


def test_auxiliary_matches_raw_unit_conditional_mse_gradient_with_equal_user_weights():
    episode, logits = episode_and_logits()
    original, _ = rating_objective(logits, episode, IdentityCriterion(), point_objective="expected_user_rmse",
                                   macro_query_scope="parent", mae_weight=.1, ranking_weight=.05)
    combined, diagnostics = rating_objective(logits, episode, IdentityCriterion(), mean_rating_weight=.25,
                                             point_objective="expected_user_rmse", macro_query_scope="parent",
                                             mae_weight=.1, ranking_weight=.05)
    indices = episode.query.nonzero()
    rows, cols = indices[:, 1], indices[:, 2]
    prediction = logits[:, 0] * episode.row_scales[rows] * episode.stds[cols] + episode.means[cols] + episode.row_offsets[rows]
    selected = episode.rating_query[episode.query]
    counts = torch.bincount(rows[selected], minlength=episode.source.shape[1])
    weights = 1 / counts[rows[selected]].float() / (counts > 0).sum()
    probabilities = episode.ordinal_probabilities[episode.query][selected]
    rating_values = torch.arange(1, 6)
    means = (probabilities * rating_values).sum(-1)
    torch.testing.assert_close(means, episode.point_target[episode.query][selected])
    oracle_mse = (((prediction[selected, None] - rating_values).square() * probabilities).sum(-1) * weights).sum()
    centered_mse = ((prediction[selected] - means).square() * weights).sum()
    torch.testing.assert_close(combined, original + .25 * centered_mse)
    assert diagnostics["mse"] == pytest.approx(centered_mse.item())
    auxiliary_gradient = torch.autograd.grad(combined - original, logits, retain_graph=True)[0]
    oracle_gradient = torch.autograd.grad(.25 * oracle_mse, logits)[0]
    torch.testing.assert_close(auxiliary_gradient, oracle_gradient, rtol=2e-5, atol=2e-7)
    assert torch.isfinite(auxiliary_gradient).all()
    assert torch.equal(auxiliary_gradient[~selected], torch.zeros_like(auxiliary_gradient[~selected]))


@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf")])
def test_invalid_auxiliary_weight_rejected(weight):
    episode, logits = episode_and_logits()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        rating_objective(logits, episode, IdentityCriterion(), mean_rating_weight=weight)


@pytest.mark.parametrize("invalid", ["real", "observed", "missing"])
def test_auxiliary_rejects_real_observed_or_missing_targets(invalid):
    episode, logits = episode_and_logits()
    if invalid == "real":
        episode.metadata["gradient_data"] = "inner_training_ratings"
    elif invalid == "observed":
        episode.metadata["point_target"] = "observed"
    else:
        episode.point_target = None
    with pytest.raises(ValueError, match="requires synthetic expected-rating"):
        rating_objective(logits, episode, IdentityCriterion(), mean_rating_weight=.25)
