import pytest
import torch

from tabimpute.prior.recsys_rating_episode import (
    expected_ordinal_rating, ordinal_rating_probabilities, sample_rating_episode,
)


@pytest.mark.parametrize("probability", [0., .08, .5, 1.])
def test_mixture_probabilities_match_mean_and_remain_normalized(probability):
    preference = torch.linspace(-2, 8, 40)
    probabilities = ordinal_rating_probabilities(preference, .8, probability)
    mean = expected_ordinal_rating(preference, .8, probability)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(40))
    torch.testing.assert_close((probabilities * torch.arange(1, 6)).sum(-1), mean)
    assert torch.all(probabilities >= 0)
    original = expected_ordinal_rating(preference, .8)
    torch.testing.assert_close(mean, (1 - probability) * original + 3 * probability)


def test_gaussian_uniform_mixture_matches_empirical_ordinal_distribution():
    generator = torch.Generator().manual_seed(71)
    preference = torch.tensor([1.2, 2.5, 3.7, 4.8])
    observations = (preference + .8 * torch.randn(40000, 4, generator=generator)).round().clamp(1, 5)
    mask = torch.rand(observations.shape, generator=generator) < .12
    replacements = torch.randint(1, 6, observations.shape, generator=generator).float()
    observations = torch.where(mask, replacements, observations)
    frequencies = torch.stack([(observations == rating).float().mean(0) for rating in range(1, 6)], dim=-1)
    expected = ordinal_rating_probabilities(preference, .8, .12)
    torch.testing.assert_close(frequencies, expected, atol=.008, rtol=0)


def test_zero_noise_mixture_preserves_episode_defaults_bitwise():
    implicit = sample_rating_episode(72, 24, 20, parent_rows=48, ordinal_targets=True, point_target="expected_rating")
    explicit = sample_rating_episode(72, 24, 20, parent_rows=48, ordinal_targets=True, point_target="expected_rating", uniform_noise_range=(0., 0.))
    for name in ("source", "target", "query", "rating_query", "point_target", "ordinal_probabilities"):
        torch.testing.assert_close(getattr(implicit, name), getattr(explicit, name), rtol=0, atol=0, equal_nan=True)
    assert implicit.metadata == explicit.metadata


def test_uniform_only_episode_has_exact_uniform_targets_and_clean_provenance():
    episode = sample_rating_episode(73, 24, 20, parent_rows=48, density_range=(.2, .2),
                                     ordinal_targets=True, point_target="expected_rating", uniform_noise_range=(1., 1.))
    assert episode.metadata["gradient_data"] == "synthetic"
    assert episode.metadata["uniform_rating_probability"] == 1
    torch.testing.assert_close(episode.ordinal_probabilities, torch.full_like(episode.ordinal_probabilities, .2))
    torch.testing.assert_close(episode.point_target, torch.full_like(episode.point_target, 3.))
    raw = episode.target * episode.stds + episode.means
    assert torch.all((raw >= 1 - 1e-6) & (raw <= 5 + 1e-6))
    torch.testing.assert_close(raw, raw.round(), rtol=0, atol=1e-6)


def test_probability_changes_use_only_local_seeded_generator():
    state = torch.get_rng_state().clone()
    first = sample_rating_episode(74, 24, 20, parent_rows=48, uniform_noise_range=(.02, .12))
    repeated = sample_rating_episode(74, 24, 20, parent_rows=48, uniform_noise_range=(.02, .12))
    assert .02 <= first.metadata["uniform_rating_probability"] <= .12
    torch.testing.assert_close(first.source, repeated.source, equal_nan=True, rtol=0, atol=0)
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("probability", [-.1, 1.1, float("nan")])
def test_invalid_uniform_probability_rejected(probability):
    with pytest.raises(ValueError, match="probability"):
        expected_ordinal_rating(torch.ones(2), .8, probability)
    with pytest.raises(ValueError, match="probability"):
        ordinal_rating_probabilities(torch.ones(2), .8, probability)


@pytest.mark.parametrize("bounds", [(-.1, .1), (.2, .1), (0., 1.1), (0., float("nan"))])
def test_invalid_uniform_probability_range_rejected(bounds):
    with pytest.raises(ValueError, match="probability range"):
        sample_rating_episode(75, 24, 20, parent_rows=48, uniform_noise_range=bounds)
