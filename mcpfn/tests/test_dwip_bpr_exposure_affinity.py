import argparse

import pytest
import torch

from dwip_bpr_experiment import episode_arguments
from tabimpute.prior.recsys_rating_episode import sample_rating_episode


def options(seed=71, context='first_observation'):
    return dict(seed=seed, rows=32, cols=64, parent_rows=128, parent_cols=256,
                device='cpu', query_limit=128, density_range=(.1, .15), normalization='two_way_user_scale',
                point_target='expected_rating', ordinal_targets=True, context_ordering=context,
                query_sampling='user_balanced', preference_family='mixed',
                uniform_noise_range=(.05, .15), user_noise_log_std_range=(.2, .6),
                ranking_query_mode='user_catalog', negatives_per_user=64, ranking_max_users=8,
                ranking_user_scope='parent_positive', relevance_targets=True,
                relevance_target='conditional', exclude_unknown_queries=True)


def assert_same_episode(actual, expected):
    assert vars(actual).keys() == vars(expected).keys()
    for name, value in vars(actual).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, getattr(expected, name), rtol=0, atol=0, equal_nan=True)
        else:
            assert value == getattr(expected, name)


@pytest.mark.parametrize('seed', [71, 72, 73])
@pytest.mark.parametrize('context', ['first_observation', 'first_observation_catalog'])
def test_default_exposure_affinity_is_opt_in_and_reproducible(seed, context):
    implicit = sample_rating_episode(**options(seed, context))
    explicit = sample_rating_episode(**options(seed, context), exposure_affinity_range=(0., 1.))
    assert_same_episode(implicit, explicit)
    assert 'exposure_affinity_range' not in implicit.metadata
    assert 'exposure_affinity_coefficient' not in implicit.metadata


@pytest.mark.parametrize('seed', [71, 72, 73])
@pytest.mark.parametrize('context', ['first_observation', 'first_observation_catalog'])
def test_stronger_affinity_changes_only_synthetic_exposure_prior(seed, context):
    arguments = options(seed, context)
    baseline = sample_rating_episode(**arguments)
    stronger = sample_rating_episode(**arguments, exposure_affinity_range=(1., 3.))
    assert_same_episode(stronger, sample_rating_episode(**arguments, exposure_affinity_range=(1., 3.)))
    assert stronger.metadata['gradient_data'] == 'synthetic'
    assert stronger.metadata['exposure_affinity_range'] == [1., 3.]
    assert 1 <= stronger.metadata['exposure_affinity_coefficient'] <= 3
    for name in ('density', 'rank', 'signal_scale', 'noise_scale', 'holdout_fraction',
                 'activity_sigma', 'popularity_sigma', 'uniform_rating_probability', 'user_noise_log_std'):
        assert stronger.metadata[name] == baseline.metadata[name]
    assert stronger.metadata['global_unknown_rating_queries'] == 0
    assert not torch.equal(torch.isfinite(stronger.source), torch.isfinite(baseline.source))
    assert torch.isnan(stronger.source[stronger.query]).all()
    assert torch.isfinite(stronger.point_target).all()
    assert torch.isfinite(stronger.relevance_probabilities).all()
    assert torch.all((stronger.relevance_probabilities >= 0) & (stronger.relevance_probabilities <= 1))
    assert not stronger.relevance_probabilities.requires_grad


@pytest.mark.parametrize('invalid', [(-1., 1.), (2., 1.), (0., float('inf')),
                                    (float('nan'), 1.), (0., float('nan')), (1.,), (0., 1., 2.)])
def test_invalid_exposure_affinity_range_fails_before_sampling(invalid):
    with pytest.raises(ValueError, match='Exposure-affinity'):
        sample_rating_episode(**options(), exposure_affinity_range=invalid)


@pytest.mark.parametrize('invalid', [(-1., 1.), (2., 1.), (0., float('inf')),
                                    (float('nan'), 1.), (0., float('nan'))])
def test_invalid_cli_range_fails_before_model_or_data_loading(invalid):
    arguments = argparse.Namespace(exposure_affinity_low=invalid[0], exposure_affinity_high=invalid[1])
    with pytest.raises(ValueError, match='Exposure-affinity'):
        episode_arguments(arguments)


def test_fixed_affinity_coefficient_preserves_other_random_draws():
    baseline = sample_rating_episode(**options())
    fixed = sample_rating_episode(**options(), exposure_affinity_range=(2., 2.))
    assert fixed.metadata['exposure_affinity_coefficient'] == 2
    for name in ('density', 'rank', 'signal_scale', 'noise_scale', 'holdout_fraction',
                 'activity_sigma', 'popularity_sigma', 'uniform_rating_probability', 'user_noise_log_std'):
        assert fixed.metadata[name] == baseline.metadata[name]
