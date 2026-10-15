from contextlib import nullcontext

import numpy as np
import pytest
import torch

import dwip_recsys_experiment as experiment
from evaluate_dwip_recsys import user_rating_metrics
from evaluate_dwip_recsys_candidate import impute_candidate_ratings
from tabimpute.prior.recsys_rating_episode import (
    RatingEpisode, observed_user_scales, rating_objective, sample_rating_episode,
)


class IdentityCriterion:
    def mean(self, logits):
        return logits[:, 0]

    def __call__(self, logits, target):
        return (logits[:, 0] - target).square()


class ConstantImputer:
    def impute(self, source, *, means, stds):
        self.source = source.copy()
        self.means, self.stds = means.copy(), stds.copy()
        return np.where(np.isfinite(source), source, means + .25 * stds)


def test_user_scales_match_shrunk_unbiased_variance_and_cold_fallbacks():
    ratings = torch.tensor([[0., 2., 4.], [3., float("nan"), float("nan")], [float("nan")] * 3])
    scales = observed_user_scales(ratings, torch.isfinite(ratings), torch.zeros(3), torch.ones(3), torch.zeros(3))
    torch.testing.assert_close(scales, torch.tensor([(28 / 22) ** .5, 1., 1.]))


def test_user_scales_are_bounded_for_constant_and_extreme_profiles():
    ratings = torch.stack((torch.zeros(100), torch.tensor([-100., 100.]).repeat(50)))
    scales = observed_user_scales(ratings, torch.ones_like(ratings, dtype=torch.bool), torch.zeros(100), torch.ones(100), torch.zeros(2))
    torch.testing.assert_close(scales, torch.tensor([.5, 2.]))


def test_unobserved_values_cannot_change_user_scales():
    ratings = torch.tensor([[1., 2., float("nan")], [2., float("nan"), 5.]])
    observed = torch.isfinite(ratings)
    original = observed_user_scales(ratings, observed, torch.full((3,), 3.), torch.ones(3), torch.zeros(2))
    replaced = observed_user_scales(ratings.masked_fill(~observed, -1e9), observed, torch.full((3,), 3.), torch.ones(3), torch.zeros(2))
    torch.testing.assert_close(original, replaced, rtol=0, atol=0)


def test_user_scaled_episode_preserves_raw_targets_queries_and_rng():
    arguments = dict(parent_rows=64, parent_cols=48, query_limit=64, ordinal_targets=True,
                     point_target="expected_rating", density_range=(.2, .2),
                     user_noise_log_std_range=(.15, .6), uniform_noise_range=(0., .15))
    state = torch.get_rng_state().clone()
    original = sample_rating_episode(101, 24, 20, normalization="two_way", **arguments)
    scaled = sample_rating_episode(101, 24, 20, normalization="two_way_user_scale", **arguments)
    assert original.row_scales is None
    assert scaled.row_scales.shape == (24,)
    assert torch.all((scaled.row_scales >= .5) & (scaled.row_scales <= 2))
    torch.testing.assert_close(scaled.source * scaled.row_scales[None, :, None], original.source, equal_nan=True)
    torch.testing.assert_close(scaled.target * scaled.row_scales[None, :, None], original.target)
    for field in ("query", "rating_query", "point_target", "ordinal_probabilities", "global_query_counts"):
        torch.testing.assert_close(getattr(original, field), getattr(scaled, field), rtol=0, atol=0)
    assert scaled.metadata["gradient_data"] == "synthetic"
    assert torch.equal(state, torch.get_rng_state())


def test_rating_objective_restores_row_and_column_units_before_loss():
    target = torch.tensor([[[.1, -.1], [.1, -.1]]])
    query = torch.ones_like(target, dtype=torch.bool)
    episode = RatingEpisode(source=target, target=target, query=query, rating_query=query,
                            means=torch.tensor([3., 3.]), stds=torch.tensor([2., 4.]),
                            metadata={"gradient_data": "synthetic", "seed": 102},
                            row_offsets=torch.tensor([.1, -.1]), row_scales=torch.tensor([.5, 2.]),
                            point_target=torch.tensor([[[3.2, 2.9], [3.3, 2.1]]]))
    logits = torch.zeros(4, 1, requires_grad=True)
    loss, diagnostics = rating_objective(logits, episode, IdentityCriterion(), nll_weight=0)
    assert diagnostics["mse"] == pytest.approx(.2125)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    perfect, _ = rating_objective(target[query, None], episode, IdentityCriterion(), nll_weight=0)
    assert perfect.item() == pytest.approx(0., abs=1e-12)


def test_scaled_synthetic_episode_has_finite_parent_risk_gradients():
    episode = sample_rating_episode(106, 24, 20, parent_rows=64, parent_cols=48, query_limit=64,
                                     normalization="two_way_user_scale", density_range=(.2, .2),
                                     ordinal_targets=True, point_target="expected_rating",
                                     uniform_noise_range=(0., .15), user_noise_log_std_range=(.15, .6))
    restored = episode.target[0] * episode.row_scales[:, None] * episode.stds + episode.means + episode.row_offsets[:, None]
    torch.testing.assert_close(restored, restored.round(), rtol=0, atol=1e-6)
    assert torch.all((restored >= 1 - 1e-6) & (restored <= 5 + 1e-6))
    logits = torch.zeros(int(episode.query.sum()), 1, requires_grad=True)
    loss, diagnostics = rating_objective(logits, episode, IdentityCriterion(), .1, .05, .1,
                                         point_objective="expected_user_rmse", macro_query_scope="parent")
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert diagnostics["macro_population_queries"] >= int(episode.rating_query.sum())


@pytest.mark.parametrize("mode", ["two_way", "two_way_user_scale"])
def test_public_and_query_only_evaluation_use_the_same_units(monkeypatch, mode):
    original = (np.arange(8 * 21).reshape(8, 21) % 5 + 1).astype(np.float64)
    users = np.array([0, 1, 1, 4, 5, 5, 7])
    items = np.array([1, 2, 8, 4, 6, 12, 19])
    truth = original[users, items].copy()
    observed = original.copy()
    observed[users, items] = np.nan
    imputer = ConstantImputer()
    prediction = impute_candidate_ratings(imputer, observed, mode)
    known = np.isfinite(observed)
    np.testing.assert_allclose(prediction[known], observed[known], rtol=1e-12, atol=1e-12)
    if mode == "two_way_user_scale":
        np.testing.assert_array_equal(imputer.means, np.zeros(21))
        np.testing.assert_array_equal(imputer.stds, np.ones(21))
    else:
        means, stds, offsets = experiment.inference_components(observed, mode)
        np.testing.assert_allclose(imputer.source, observed - offsets[:, None], equal_nan=True)
        np.testing.assert_array_equal(imputer.means, means)
        np.testing.assert_array_equal(imputer.stds, stds)

    class ConstantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, source, decode_mask):
            return source.new_full((int(decode_mask.sum()), 1), .25) + self.anchor * 0

    monkeypatch.setattr(experiment, "DATASETS", {"toy": {"folder": "toy"}})
    monkeypatch.setattr(experiment, "load_split", lambda folder, partition: (observed, None, (users, items, truth)))
    monkeypatch.setattr(experiment, "compute_context", nullcontext)
    actual = experiment.rating_validation(ConstantModel(), IdentityCriterion(), normalization=mode)["toy"]
    expected = user_rating_metrics(users, truth, prediction[users, items])
    for metric in ("MAE", "RMSE"):
        assert actual[metric] == pytest.approx(expected[metric], abs=1e-12)


def test_existing_modes_do_not_add_a_row_scale_transform():
    observed = np.array([[1., 2.], [3., np.nan]])
    for mode in ("column", "global", "shrunk_column", "two_way"):
        components = experiment.inference_components(observed, mode)
        assert experiment.inference_row_scales(observed, mode, *components) is None
