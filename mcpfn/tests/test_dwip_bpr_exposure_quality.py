import argparse
import sys

import pytest
import torch

import dwip_bpr_experiment as experiment
import dwip_recsys_experiment as trainer
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.prior.recsys_ranking_objective import raw_query_predictions
from tabimpute.prior.recsys_rating_episode import sample_rating_episode


def forbidden(*args, **kwargs):
    raise AssertionError('Quality-prior tests may not load real data or neural models')


@pytest.fixture(autouse=True)
def cpu_synthetic_only(monkeypatch):
    for name in ('load_split', 'cornac_split', 'ImputePFN'):
        monkeypatch.setattr(trainer, name, forbidden)
    for name in ('evaluate_public', 'make_imputer'):
        monkeypatch.setattr(experiment, name, forbidden)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def options(seed=71, context='first_observation', affinity_range=(1., 3.)):
    return dict(seed=seed, rows=32, cols=64, parent_rows=128, parent_cols=256,
                device='cpu', query_limit=128, density_range=(.1, .15), normalization='two_way_user_scale',
                point_target='expected_rating', ordinal_targets=True, context_ordering=context,
                query_sampling='user_balanced', preference_family='mixed',
                uniform_noise_range=(.05, .15), user_noise_log_std_range=(.2, .6),
                ranking_query_mode='user_catalog', negatives_per_user=64, ranking_max_users=8,
                ranking_user_scope='parent_positive', relevance_targets=True,
                relevance_target='conditional', exclude_unknown_queries=True,
                exposure_affinity_range=affinity_range)


def assert_same_episode(actual, expected):
    assert vars(actual).keys() == vars(expected).keys()
    for name, value in vars(actual).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, getattr(expected, name), rtol=0, atol=0, equal_nan=True)
        else:
            assert value == getattr(expected, name)


@pytest.mark.parametrize('seed', [71, 72, 73])
@pytest.mark.parametrize('context', ['first_observation', 'first_observation_catalog'])
@pytest.mark.parametrize('affinity_range', [(0., 1.), (1., 3.)])
def test_default_quality_prior_is_opt_in(seed, context, affinity_range):
    arguments = options(seed, context, affinity_range)
    implicit = sample_rating_episode(**arguments)
    explicit = sample_rating_episode(**arguments, exposure_quality_range=(0., 1.5))
    assert_same_episode(implicit, explicit)
    assert 'exposure_quality_range' not in implicit.metadata
    assert 'exposure_quality_coefficient' not in implicit.metadata


@pytest.mark.parametrize('seed', [71, 72, 73])
@pytest.mark.parametrize('context', ['first_observation', 'first_observation_catalog'])
def test_stronger_quality_is_finite_synthetic_and_reproducible(seed, context):
    arguments = options(seed, context)
    baseline = sample_rating_episode(**arguments)
    actual = sample_rating_episode(**arguments, exposure_quality_range=(1.5, 4.5))
    assert_same_episode(actual, sample_rating_episode(**arguments, exposure_quality_range=(1.5, 4.5)))
    assert actual.metadata['gradient_data'] == 'synthetic'
    assert actual.metadata['exposure_quality_range'] == [1.5, 4.5]
    assert 1.5 <= actual.metadata['exposure_quality_coefficient'] <= 4.5
    for name in ('density', 'rank', 'signal_scale', 'noise_scale', 'holdout_fraction',
                 'activity_sigma', 'popularity_sigma', 'uniform_rating_probability', 'user_noise_log_std',
                 'exposure_affinity_range', 'exposure_affinity_coefficient'):
        assert actual.metadata[name] == baseline.metadata[name]
    assert actual.metadata['global_unknown_rating_queries'] == 0
    assert not torch.equal(torch.isfinite(actual.source), torch.isfinite(baseline.source))
    assert torch.isnan(actual.source[actual.query]).all()
    assert torch.isfinite(actual.point_target).all()
    assert torch.isfinite(actual.relevance_probabilities).all()
    assert torch.all((actual.relevance_probabilities >= 0) & (actual.relevance_probabilities <= 1))
    assert not actual.relevance_probabilities.requires_grad


@pytest.mark.parametrize('invalid', [(-1., 1.), (2., 1.), (0., float('inf')),
                                    (float('nan'), 1.), (0., float('nan')), (1.,), (0., 1., 2.)])
def test_invalid_quality_range_fails_before_sampling(invalid):
    with pytest.raises(ValueError, match='Exposure-quality'):
        sample_rating_episode(**options(), exposure_quality_range=invalid)


@pytest.mark.parametrize('invalid', [(-1., 1.), (2., 1.), (0., float('inf')),
                                    (float('nan'), 1.), (0., float('nan'))])
def test_invalid_cli_range_fails_before_data_or_model_load(invalid):
    arguments = argparse.Namespace(exposure_quality_low=invalid[0], exposure_quality_high=invalid[1])
    with pytest.raises(ValueError, match='Exposure-quality'):
        experiment.episode_arguments(arguments)


def test_fixed_quality_preserves_affinity_and_other_random_draws():
    baseline = sample_rating_episode(**options())
    fixed = sample_rating_episode(**options(), exposure_quality_range=(3., 3.))
    assert fixed.metadata['exposure_quality_coefficient'] == 3
    assert fixed.metadata['exposure_affinity_coefficient'] == baseline.metadata['exposure_affinity_coefficient']
    assert fixed.metadata['holdout_fraction'] == baseline.metadata['holdout_fraction']


@pytest.mark.parametrize('supplied,expected', [([], (0., 1.5)),
                                             (['--exposure-quality-low', '1.5', '--exposure-quality-high', '4.5'], (1.5, 4.5))])
def test_cli_exposes_opt_in_prior(monkeypatch, tmp_path, supplied, expected):
    captured = []
    monkeypatch.setattr(experiment, 'train', lambda args: captured.append(experiment.episode_arguments(args)))
    monkeypatch.setattr(sys, 'argv', ['dwip_bpr_experiment.py', 'train', '--root', str(tmp_path), *supplied])
    experiment.main()
    assert len(captured) == 1
    assert (captured[0].exposure_quality_low, captured[0].exposure_quality_high) == expected


@pytest.mark.parametrize('context_fraction', [0., 1.])
@pytest.mark.parametrize('quality_range', [None, (0., 1.5), (1.5, 4.5), (3., 3.)])
def test_joint_window_forwarding_and_active_ranking_gradients(monkeypatch, tmp_path, context_fraction, quality_range):
    captured = []
    selected_arguments = []
    monkeypatch.setattr(experiment, 'train', lambda args: selected_arguments.append(args))
    monkeypatch.setattr(sys, 'argv', ['dwip_bpr_experiment.py', 'train', '--root', str(tmp_path)])
    experiment.main()
    arguments = selected_arguments[0]
    del arguments.exposure_quality_low, arguments.exposure_quality_high
    vars(arguments).update(seed=71, rows=32, cols=32, parent_rows=128, parent_cols=256, training_rows=32,
                           query_limit=128, negatives_per_user=64, ranking_max_users=8,
                           density_low=.1, density_high=.15, training_item_windows=2,
                           catalog_fraction=context_fraction, exposure_affinity_low=1., exposure_affinity_high=3.,
                           ranking_objective='topk_expected_pairwise', ranking_candidate_scope='topk_union',
                           ranking_query_mode='user_catalog', ranking_user_scope='parent_positive',
                           ranking_relevance_target='conditional', teacher_anchor_weight=50.)
    if quality_range is not None:
        arguments.exposure_quality_low, arguments.exposure_quality_high = quality_range

    def sample_cpu(*args, **kwargs):
        captured.append(dict(kwargs))
        assert kwargs['device'] == 'cuda'
        return sample_rating_episode(*args, **{**kwargs, 'device': 'cpu'})

    monkeypatch.setattr(trainer, 'sample_rating_episode', sample_cpu)
    selected = experiment.episode_arguments(arguments)
    episode = experiment.ranking_episode(selected, 20)
    assert captured[-1]['exposure_quality_range'] == (quality_range or (0., 1.5))
    assert captured[-1]['exposure_affinity_range'] == (1., 3.)
    expected = sample_rating_episode(selected.seed * 1000000 + 20, **{**captured[-1], 'device': 'cpu'})
    expected.metadata.update(training_item_windows=2, training_window_cols=selected.cols)
    assert_same_episode(episode, expected)
    assert episode.metadata['context_ordering'] == (
        'first_observation_catalog' if context_fraction else 'first_observation')
    logits = torch.randn((int(episode.query.sum()), 32),
                         generator=torch.Generator().manual_seed(71)).requires_grad_()
    criterion = FullSupportBarDistribution(torch.linspace(-3, 3, 33))
    teacher = raw_query_predictions(logits, episode, criterion)[0][episode.rating_query[episode.query]].detach() + .02
    loss, diagnostics = experiment.objective(logits, episode, criterion, selected, teacher)
    gradient = torch.autograd.grad(loss, logits)[0]
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    assert diagnostics['ranking'] > 0 and diagnostics['ranking_paired_users'] > 0
    assert diagnostics['ranking_objective'] == 'topk_expected_pairwise'
    assert diagnostics['ranking_candidate_scope'] == 'topk_union'
    assert diagnostics['teacher_anchor_weight'] == 50
    assert not teacher.requires_grad and teacher.grad is None
