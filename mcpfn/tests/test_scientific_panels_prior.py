import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from tabimpute.prior.scientific_panels import (
    MASK_PATTERNS, TRAIN_FAMILIES, TRANSFER_FAMILIES,
    observed_standardize, sample_scientific_episode,
)
from train_scientific_matrixformer import objective, selection_score, validation_specs
from tabimpute.model.bar_distribution import FullSupportBarDistribution


@pytest.mark.parametrize('family', TRAIN_FAMILIES)
@pytest.mark.parametrize('pattern', MASK_PATTERNS)
def test_all_families_and_masks_are_supported(family, pattern):
    episode = sample_scientific_episode(419, family=family, pattern=pattern, rows=64, columns=12, time_features=True)
    assert episode.source.shape == (64, 15)
    assert episode.query.any() and not episode.query[:, 12:].any()
    assert np.isfinite(episode.target).all()
    np.testing.assert_array_equal(np.isnan(episode.source), episode.query)
    np.testing.assert_array_equal(episode.source[~episode.query], episode.target[~episode.query])
    assert ((~episode.query).sum(axis=0) >= 4).all()
    assert episode.metadata['gradient_data'] == 'synthetic'


def test_generation_is_reproducible_without_mutating_global_numpy_rng():
    np.random.seed(11)
    before = np.random.get_state()
    first = sample_scientific_episode(921)
    second = sample_scientific_episode(921)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.testing.assert_array_equal(first.source, second.source)
    np.testing.assert_array_equal(first.target, second.target)
    assert first.metadata == second.metadata
    assert not np.array_equal(first.query, sample_scientific_episode(922).query)


def test_normalization_cannot_see_hidden_values():
    truth = np.arange(96, dtype=float).reshape(32, 3)
    missing = np.zeros_like(truth, dtype=bool)
    missing[5:10, :2] = True
    source, target, center, scale = observed_standardize(truth, missing)
    changed = truth.copy()
    changed[missing] += 100000.
    other_source, _, other_center, other_scale = observed_standardize(changed, missing)
    np.testing.assert_array_equal(source, other_source)
    np.testing.assert_array_equal(center, other_center)
    np.testing.assert_array_equal(scale, other_scale)
    np.testing.assert_allclose(target * scale + center, truth, atol=1e-5)
    np.testing.assert_allclose(np.nanmean(source, axis=0), 0., atol=1e-6)
    np.testing.assert_allclose(np.nanstd(source, axis=0), 1., atol=1e-6)


@pytest.mark.parametrize('family', TRANSFER_FAMILIES)
def test_transfer_families_are_excluded_from_training_and_selection(family):
    with pytest.raises(ValueError, match='Transfer-only'):
        sample_scientific_episode(11, family=family)
    episode = sample_scientific_episode(11, family=family, allow_transfer=True)
    assert np.isfinite(episode.target).all()
    assert family not in {spec.get('family') for spec in validation_specs()}


def test_replay_regression_blocks_checkpoint_promotion():
    baseline = {name: 1. for name in TRAIN_FAMILIES + ('imputation', 'recsys')}
    current = {name: .7 for name in baseline}
    current['recsys'] = 1.04
    score, eligible, ratios = selection_score(current, baseline)
    assert score < 1. and not eligible and ratios['recsys'] == 1.04
    current['recsys'] = 1.02
    assert selection_score(current, baseline)[1]


def test_distribution_objective_has_finite_gradients_and_detached_point_targets():
    distribution = FullSupportBarDistribution(torch.linspace(-4., 4., 17))
    logits = torch.randn(9, 16, requires_grad=True)
    truth = torch.linspace(-2., 2., 9)
    teacher = torch.randn(9, requires_grad=True)
    loss = objective(distribution, logits, truth, teacher)
    assert loss.shape == (9,) and torch.isfinite(loss).all()
    loss.mean().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert teacher.grad is None


@pytest.mark.parametrize('shape', [(16, 4), (384, 64)])
def test_supported_shape_boundaries(shape):
    for pattern in MASK_PATTERNS:
        episode = sample_scientific_episode(99, rows=shape[0], columns=shape[1], pattern=pattern)
        assert np.isfinite(episode.target).all()


def test_invalid_shape_and_unsupported_normalization_are_rejected():
    with pytest.raises(ValueError):
        sample_scientific_episode(11, rows=8)
    with pytest.raises(ValueError):
        observed_standardize(np.ones((16, 4)), np.ones((16, 4), dtype=bool))
