import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from million_matrix_prior import (
    DOMAINS, SCIENCE_FAMILIES, MASK_FAMILIES, COMPONENTS, TRAIN_SEED, VALIDATION_SEED,
    WEIGHTS, shape_draw, teacher_weight, training_specs, validation_specs,
)
from million_matrix_monitor import summarize
from train_million_matrix_continuation import MILESTONES, learning_rate, objective
from tabimpute.model.bar_distribution import FullSupportBarDistribution


def test_counting_and_seed_separation():
    seeds = []
    for step in (1, 2, 6250, 25000, 62500, 125000, 250000):
        specs = training_specs(step)
        assert [spec['domain'] for spec in specs] == list(DOMAINS)
        for spec in specs:
            seeds += list(range(spec['seed'], spec['seed'] + 16))
    assert len(seeds) == len(set(seeds))
    assert max(seeds) < VALIDATION_SEED
    assert min(seeds) == TRAIN_SEED
    assert 250000 * len(DOMAINS) == 1000000
    assert all(matrices % len(DOMAINS) == 0 for matrices in MILESTONES)
    assert sum(WEIGHTS.values()) == pytest.approx(1.)


def test_all_prior_components_and_patterns_are_represented():
    specs = [spec for step in range(1, 121) for spec in training_specs(step)]
    assert {spec['family'] for spec in specs if spec['domain'] == 'science'} == set(SCIENCE_FAMILIES)
    native = [spec for spec in specs if spec.get('kind') == 'native']
    assert {spec['component'] for spec in native} == set(range(8))
    structured = [spec for spec in specs if spec.get('kind') == 'structured']
    assert {(spec['family'], spec['component']) for spec in structured} == {(family, component) for family in MASK_FAMILIES for component in COMPONENTS}
    assert {spec['pattern'] for spec in specs if spec['domain'] == 'science'} == {
        'isolated', 'bursts', 'network_outage', 'scheduled_assays', 'self_mnar', 'covariate_mar'}


def test_requested_shape_distribution_and_bounds():
    samples = [shape_draw(seed) for seed in range(10000)]
    frequency = {bucket: sum(size == bucket for _, size in samples) / len(samples) for bucket in ('small', 'medium', 'large')}
    assert .77 < frequency['small'] < .83
    assert .16 < frequency['medium'] < .20
    assert .01 < frequency['large'] < .03
    assert all(16 <= rows <= 2000 and 4 <= columns <= 1000 for (rows, columns), _ in samples)
    assert any(shape == [2000, 1000] for shape, _ in samples)


def test_validation_is_fresh_and_keeps_reserved_families_out():
    specs = validation_specs()
    assert len(specs) == 84
    assert len({spec['seed'] for spec in specs}) == len(specs)
    assert all(spec['seed'] >= VALIDATION_SEED for spec in specs)
    assert not {'damped_oscillators', 'spatial_field'} & {spec.get('family') for spec in specs}


def test_long_learning_rate_and_teacher_regularizer_schedule():
    config = dict(warmup_steps=1000, plateau_steps=50000, peak_lr=5e-6, steps=250000)
    assert learning_rate(1, config) == pytest.approx(5e-9)
    assert learning_rate(1000, config) == learning_rate(50000, config) == 5e-6
    assert learning_rate(250000, config) == pytest.approx(5e-7)
    assert teacher_weight('science', 0) == teacher_weight('science', 1000000) == 0.
    assert teacher_weight('imputation', 0) == 2.
    assert teacher_weight('imputation', 250000) == teacher_weight('imputation', 1000000) == .5
    assert teacher_weight('score_panel', 0) == 8.
    assert teacher_weight('recsys', 1000000) == 4.


@pytest.mark.parametrize('domain', DOMAINS)
def test_truth_objective_has_finite_gradients_and_detached_teacher(domain):
    distribution = FullSupportBarDistribution(torch.linspace(-4., 4., 17))
    logits = torch.randn(8, 16, requires_grad=True)
    truth = torch.linspace(-2., 2., 8)
    teacher = torch.randn(8, requires_grad=True)
    losses = objective(distribution, logits, truth, teacher, domain, 500000)
    assert losses.shape == truth.shape and torch.isfinite(losses).all()
    losses.mean().backward()
    assert torch.isfinite(logits.grad).all()
    assert teacher.grad is None


def monitor_rows():
    rows = [dict(stage='tabular', suite=suite, dataset=dataset, error=1., parent_error=1.)
            for suite in ('uci', 'missbench') for dataset in ('first', 'second')]
    rows += [dict(stage='llm', suite='llm', dataset='benchpress', error=.8, parent_error=.8, r2=.5, parent_r2=.5),
             dict(stage='recsys', suite='recsys', dataset='100K', error=.8, parent_error=.8, mae=.6, parent_mae=.6)]
    return rows


def test_missbench_individual_dataset_is_not_a_veto_but_llm_drift_is_reported():
    science = [dict(candidate=.8, parent=1.) for index in range(7)]
    rows = monitor_rows()
    warm = summarize(rows, science)
    rows[0]['error'] = .9
    rows[2]['error'], rows[3]['error'] = 1.25, .85
    result = summarize(rows, science, warm)
    assert result['joint_eligible'] and result['balanced_eligible']
    rows[-2]['r2'] = .45
    result = summarize(rows, science, warm)
    assert result['joint_eligible'] and not result['balanced_eligible']
    assert [check['name'] for check in result['failures']] == ['llm/benchpress/r2']


def test_joint_selection_requires_both_primary_benchmarks_to_retain_warm_start():
    rows = monitor_rows()
    science = [dict(candidate=.8, parent=1.) for index in range(7)]
    warm = summarize(rows, science)
    rows[0]['error'] = 1.1
    result = summarize(rows, [dict(candidate=.7, parent=1.) for index in range(7)], warm)
    assert not result['joint_eligible']


@pytest.mark.parametrize('step', [0, -1, 250001, 1.5])
def test_invalid_training_steps_are_rejected(step):
    with pytest.raises(ValueError):
        training_specs(step)
