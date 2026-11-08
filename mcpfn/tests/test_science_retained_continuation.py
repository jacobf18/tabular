import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from tabimpute.prior.scientific_panels import MASK_PATTERNS
from tabimpute.prior.scientific_processes import PROCESS_FAMILIES, sample_process_episode, sample_score_panel
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from science_continuation_monitor import science_score
from science_retention_benchmarks import gate_checks, selected_cases
from train_science_retained_continuation import objective, validation_specs, WEIGHTS


@pytest.mark.parametrize('family', PROCESS_FAMILIES)
@pytest.mark.parametrize('pattern', MASK_PATTERNS)
def test_process_families_and_masks(family, pattern):
    episode = sample_process_episode(913, family=family, rows=64, columns=16, pattern=pattern)
    assert episode.source.shape == episode.target.shape == episode.query.shape == (64, 16)
    assert np.isfinite(episode.target).all()
    np.testing.assert_array_equal(np.isnan(episode.source), episode.query)
    np.testing.assert_array_equal(episode.source[~episode.query], episode.target[~episode.query])
    assert ((~episode.query).sum(0) >= 4).all()
    other = sample_process_episode(913, family=family, rows=64, columns=16, pattern=pattern)
    np.testing.assert_array_equal(episode.source, other.source)


def test_score_panel_mask_has_context_and_sparse_test_columns():
    episode = sample_score_panel(14)
    boundary = episode.metadata['context_rows']
    observed = episode.metadata['observed_test_columns']
    assert not episode.query[boundary:, observed].any()
    assert episode.query[boundary:].sum() == (len(episode.source) - boundary) * (episode.source.shape[1] - len(observed))
    assert np.isnan(episode.source[episode.query]).all()
    np.testing.assert_allclose(np.nanmean(episode.source, axis=0), 0., atol=1e-6)


def test_all_seven_science_datasets_must_not_regress():
    names = ['wastewater', 'intel_temperature', 'potomac', 'susquehanna', 'ocean_alkalinity', 'air_quality', 'beijing']
    rows = [dict(dataset=name, candidate=.9, parent=1.) for name in names]
    assert science_score(rows)['all_not_worse']
    rows[0]['candidate'] = 1.01
    assert not science_score(rows)['all_not_worse']
    with pytest.raises(ValueError):
        science_score(rows[:-1])


def test_legacy_gate_catches_r2_and_error_drift():
    rows = [dict(stage='tabular', suite=suite, dataset='example', pattern='MCAR', error=1., parent_error=1.)
            for suite in ('missbench', 'uci')]
    rows += [dict(stage='llm', suite='llm', dataset='llm', error=.9, parent_error=.9, r2=.6, parent_r2=.6),
             dict(stage='recsys', suite='recsys', dataset='100K', error=.8, parent_error=.8, mae=.6, parent_mae=.6)]
    assert all(row['passed'] for row in gate_checks(rows))
    rows[2]['r2'] = .58
    rows[3]['error'] = .83
    failures = {row['name'] for row in gate_checks(rows) if not row['passed']}
    assert failures == {'llm/llm/r2', 'recsys/100K/rmse'}


def test_screen_selection_is_case_id_based_and_full_excludes_10m():
    cases = [dict(suite='missbench', case_id='mb/a'),
             dict(suite='uci', case_id='uci/a/2', missingness=.3, dataset='a', pattern='MAR'),
             dict(suite='uci', case_id='uci/a/1', missingness=.3, dataset='a', pattern='MAR')]
    assert [row['case_id'] for row in selected_cases(cases, 'tabular', False)] == ['mb/a', 'uci/a/1']
    assert selected_cases([dict(dataset='1M'), dict(dataset='10M')], 'recsys', True) == [dict(dataset='1M')]


@pytest.mark.parametrize('domain', list(WEIGHTS))
def test_retained_objective_gradient(domain):
    distribution = FullSupportBarDistribution(torch.linspace(-4., 4., 17))
    logits = torch.randn(8, 16, requires_grad=True)
    truth = torch.linspace(-2., 2., 8)
    target = torch.randn(8, requires_grad=True)
    losses = objective(distribution, logits, truth, target, domain)
    assert losses.shape == (8,) and torch.isfinite(losses).all()
    losses.mean().backward()
    assert torch.isfinite(logits.grad).all() and target.grad is None


def test_mixture_and_validation_inventory():
    assert sum(WEIGHTS.values()) == pytest.approx(1.)
    specs = validation_specs()
    assert len(specs) == 58
    assert len({record['seed'] for record in specs}) == 58
    assert {record['family'] for record in specs if record['domain'] == 'science'} >= set(PROCESS_FAMILIES)
