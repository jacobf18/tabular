import pytest
import torch

from tabimpute.prior.recsys_rating_episode import (
    expected_ordinal_rating, ordinal_rating_probabilities, sample_rating_episode,
)


def test_user_noise_probabilities_broadcast_and_match_scalar_rows():
    preference = torch.linspace(1., 5., 21).reshape(3, 7)
    scales = torch.tensor([[.25], [.8], [1.6]])
    probabilities = ordinal_rating_probabilities(preference, scales, .1)
    mean = expected_ordinal_rating(preference, scales, .1)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(preference))
    torch.testing.assert_close((probabilities * torch.arange(1, 6)).sum(-1), mean)
    for row in range(3):
        expected = ordinal_rating_probabilities(preference[row], float(scales[row, 0]), .1)
        torch.testing.assert_close(probabilities[row], expected, rtol=0, atol=0)


def test_user_noise_empirical_ordinal_distribution_matches_probabilities():
    generator = torch.Generator().manual_seed(81)
    preference = torch.tensor([2.2, 3., 3.7, 4.4])
    scales = torch.tensor([.25, .7, 1.3, 2.])
    observations = (preference + scales * torch.randn(40000, 4, generator=generator)).round().clamp(1, 5)
    probabilities = torch.stack([(observations == rating).float().mean(0) for rating in range(1, 6)], dim=-1)
    torch.testing.assert_close(probabilities, ordinal_rating_probabilities(preference, scales), atol=.008, rtol=0)


def test_zero_user_noise_spread_preserves_episode_defaults_bitwise():
    arguments = dict(parent_rows=48, parent_cols=40, ordinal_targets=True, point_target="expected_rating", uniform_noise_range=(0., .15))
    implicit = sample_rating_episode(82, 24, 20, **arguments)
    explicit = sample_rating_episode(82, 24, 20, user_noise_log_std_range=(0., 0.), **arguments)
    for name in ("source", "target", "query", "rating_query", "point_target", "ordinal_probabilities", "global_query_counts"):
        torch.testing.assert_close(getattr(implicit, name), getattr(explicit, name), rtol=0, atol=0, equal_nan=True)
    assert implicit.metadata == explicit.metadata


def test_heterogeneous_episode_is_seeded_synthetic_and_has_exact_ordinal_mean():
    state = torch.get_rng_state().clone()
    arguments = dict(parent_rows=48, parent_cols=40, ordinal_targets=True, point_target="expected_rating",
                     user_noise_log_std_range=(.2, .6), uniform_noise_range=(0., .15))
    first = sample_rating_episode(83, 24, 20, **arguments)
    second = sample_rating_episode(83, 24, 20, **arguments)
    assert first.metadata["gradient_data"] == "synthetic"
    assert .2 <= first.metadata["user_noise_log_std"] <= .6
    torch.testing.assert_close(first.source, second.source, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(first.point_target, (first.ordinal_probabilities * torch.arange(1, 6)).sum(-1))
    assert torch.isfinite(first.ordinal_probabilities).all()
    assert torch.all(first.ordinal_probabilities >= 0)
    torch.testing.assert_close(first.ordinal_probabilities.sum(-1), torch.ones_like(first.point_target))
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("bounds", [(-.1, .3), (.4, .2), (.1, 1.1), (0., float("nan"))])
def test_invalid_user_noise_spread_rejected(bounds):
    with pytest.raises(ValueError, match="noise log-std"):
        sample_rating_episode(84, 24, 20, parent_rows=48, user_noise_log_std_range=bounds)


@pytest.mark.parametrize("scale", [0., float("inf"), float("nan"), torch.tensor([.5, 0.]), torch.tensor([.5, float("nan")])])
def test_nonpositive_or_nonfinite_noise_scale_rejected(scale):
    for function in (expected_ordinal_rating, ordinal_rating_probabilities):
        with pytest.raises(ValueError, match="finite positive"):
            function(torch.ones(2), scale)
