import argparse
import sys

import pytest
import torch

import dwip_bpr_experiment as experiment
import dwip_recsys_experiment as trainer
from tabimpute.prior.recsys_rating_episode import sample_rating_episode


def forbidden(*args, **kwargs):
    raise AssertionError('Prior-range tests may not load real data or neural models')


@pytest.fixture(autouse=True)
def cpu_synthetic_only(monkeypatch):
    for name in ('load_split', 'cornac_split', 'ImputePFN'):
        monkeypatch.setattr(trainer, name, forbidden)
    for name in ('evaluate_public', 'make_imputer'):
        monkeypatch.setattr(experiment, name, forbidden)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('invalid', [(-1., 1.), (2., 1.), (0., float('inf')),
                                    (float('nan'), 1.), (0., float('nan'))])
def test_invalid_popularity_range_fails_before_model_or_data_load(invalid):
    with pytest.raises(ValueError, match='Item-popularity'):
        experiment.episode_arguments(argparse.Namespace(item_popularity_low=invalid[0],
                                                       item_popularity_high=invalid[1]))


def test_legacy_namespace_preserves_default_effective_arguments():
    implicit = experiment.episode_arguments(argparse.Namespace())
    explicit = experiment.episode_arguments(argparse.Namespace(item_popularity_low=1.4, item_popularity_high=2.8))
    assert vars(implicit) == vars(explicit)


def test_minimal_legacy_configuration_does_not_require_popularity_keys(monkeypatch, tmp_path):
    (tmp_path / 'config.json').write_text('{"arguments": {}}')
    monkeypatch.setattr(experiment, 'INITIAL', tmp_path / 'checkpoint.pth')
    selected = experiment.episode_arguments(argparse.Namespace())
    assert vars(selected) == {'selection_objective': 'rating_guarded_bpr'}


@pytest.mark.parametrize('supplied,expected', [([], (1.4, 2.8)),
                                             (['--item-popularity-low', '.7', '--item-popularity-high', '1.4'], (.7, 1.4)),
                                             (['--item-popularity-low', '0', '--item-popularity-high', '0'], (0., 0.))])
def test_cli_forwards_optional_prior_range(monkeypatch, tmp_path, supplied, expected):
    captured = []
    monkeypatch.setattr(experiment, 'train', lambda args: captured.append(experiment.episode_arguments(args)))
    monkeypatch.setattr(sys, 'argv', ['dwip_bpr_experiment.py', 'train', '--root', str(tmp_path), *supplied])
    experiment.main()
    assert len(captured) == 1
    assert (captured[0].item_popularity_low, captured[0].item_popularity_high) == expected


@pytest.mark.parametrize('fraction', [0., 1.])
@pytest.mark.parametrize('popularity', [(0., 0.), (.7, 1.4), (1.2, 1.2)])
def test_joint_window_sampler_receives_prior_range(monkeypatch, tmp_path, fraction, popularity):
    captured_arguments = []
    monkeypatch.setattr(experiment, 'train', lambda args: captured_arguments.append(args))
    monkeypatch.setattr(sys, 'argv', ['dwip_bpr_experiment.py', 'train', '--root', str(tmp_path)])
    experiment.main()
    arguments = captured_arguments[0]
    vars(arguments).update(seed=71, rows=32, cols=32, parent_rows=128, parent_cols=256, training_rows=32,
                           query_limit=128, negatives_per_user=64, ranking_max_users=8,
                           density_low=.1, density_high=.15, training_item_windows=2,
                           catalog_fraction=fraction, exposure_affinity_low=0., exposure_affinity_high=1.,
                           exposure_quality_low=1.5, exposure_quality_high=4.5,
                           item_popularity_low=popularity[0], item_popularity_high=popularity[1],
                           ranking_objective='topk_expected_pairwise', ranking_candidate_scope='topk_union',
                           ranking_query_mode='user_catalog', ranking_user_scope='parent_positive',
                           ranking_relevance_target='conditional')
    captured = []

    def cpu_sample(*args, **kwargs):
        captured.append(dict(kwargs))
        assert kwargs['device'] == 'cuda'
        return sample_rating_episode(*args, **{**kwargs, 'device': 'cpu'})

    monkeypatch.setattr(trainer, 'sample_rating_episode', cpu_sample)
    selected = experiment.episode_arguments(arguments)
    episode = experiment.ranking_episode(selected, 20)
    assert captured[-1]['item_popularity_sigma_range'] == popularity
    assert captured[-1]['exposure_affinity_range'] == (0., 1.)
    assert captured[-1]['exposure_quality_range'] == (1.5, 4.5)
    assert episode.metadata['gradient_data'] == 'synthetic'
    assert episode.metadata['relevance_target'] == 'conditional'
    assert episode.metadata['context_ordering'] == ('first_observation_catalog' if fraction else 'first_observation')
    assert popularity[0] <= episode.metadata['popularity_sigma'] <= popularity[1]
    assert 'exposure_affinity_range' not in episode.metadata
    assert 'exposure_affinity_coefficient' not in episode.metadata
    assert episode.metadata['exposure_quality_range'] == [1.5, 4.5]
    assert torch.isfinite(episode.relevance_probabilities).all()
    assert not episode.relevance_probabilities.requires_grad
