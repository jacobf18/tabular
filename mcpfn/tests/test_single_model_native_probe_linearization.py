from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import analyze_single_model_native_probe_linearization as analysis
from single_model_distillation import Episode
from test_single_model_point_metric_gradients import imputation_fixture, rating_fixture
from recover_single_model_stronger_mean_cuda_entry import unresolved_globals


@pytest.mark.parametrize('domain', ['imputation', 'recsys'])
def test_every_additive_term_and_metric_matches_original_definition(domain):
    before, episode = imputation_fixture() if domain == 'imputation' else rating_fixture()
    candidate = before + .1
    changes = {name: -.001 for name in analysis.metrics.point_metric_terms(before, episode)}
    rows = analysis.decompose(before, candidate, episode, changes)
    actual = analysis.metrics.point_metric_terms(candidate, episode)
    for row in rows:
        assert row['candidate'] == actual[row['metric']]['value']
        assert sum(row[key] for key in ('parameter_linear_change', 'parameter_to_output_residual', 'metric_remainder')) == pytest.approx(row['change'], abs=1e-14)
        if domain == 'imputation':
            assert row['metric_remainder'] >= -1e-12


def test_convex_output_curvature_can_reverse_a_favorable_linear_prediction():
    target = torch.tensor([[[0.], [1.]]], dtype=torch.float64)
    before = target + torch.tensor([[[1.], [0.]]], dtype=torch.float64)
    candidate = target + torch.tensor([[[.9], [1.]]], dtype=torch.float64)
    episode = Episode(target.clone(), target, torch.ones_like(target, dtype=torch.bool),
        {'domain': 'imputation', 'gradient_data': 'synthetic'})
    initial = analysis.metrics.point_metric_terms(before, episode)
    linear = {name: float((term['gradient'] * (candidate - before)).sum()) for name, term in initial.items()}
    rows = analysis.decompose(before, candidate, episode, linear)
    assert all(row['parameter_linear_change'] < 0 < row['change'] for row in rows)
    assert all(row['parameter_to_output_residual'] == 0 for row in rows)
    assert all(row['metric_remainder'] > 0 for row in rows)


def test_rating_clipping_can_have_a_negative_metric_remainder():
    target = torch.tensor([[[4.]]], dtype=torch.float64)
    scaling = SimpleNamespace(means=torch.zeros(1), stds=torch.ones(1), row_offsets=torch.zeros(1), row_scales=torch.ones(1))
    episode = Episode(target, target, torch.ones_like(target, dtype=torch.bool),
        {'domain': 'recsys', 'gradient_data': 'synthetic'}, scaling)
    rows = analysis.decompose(torch.tensor([[[4.5]]], dtype=torch.float64),
        torch.tensor([[[6.]]], dtype=torch.float64), episode, {'rmse': -.1, 'mae': -.1})
    assert all(row['metric_remainder'] == -1. for row in rows)


def test_all_undefined_columns_remain_null_not_zero():
    before, episode = imputation_fixture()
    episode.target.fill_(1.)
    rows = analysis.decompose(before, before + .1, episode, {'rmse': -.01, 'common_nrmse': None})
    common = next(row for row in rows if row['metric'] == 'common_nrmse')
    assert all(common[key] is None for key in ('before', 'candidate', 'change', 'parameter_linear_change',
        'output_linear_change', 'parameter_to_output_residual', 'metric_remainder'))
    with pytest.raises(ValueError, match='cannot acquire'):
        analysis.decompose(before, before + .1, episode, {'rmse': -.01, 'common_nrmse': 0.})


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'nonfinite', 'none', 'non_synthetic', 'nonfinite_prediction'])
def test_invalid_or_incomplete_inputs_are_rejected(mutation):
    before, episode = imputation_fixture()
    candidate = before + .1
    changes = {'rmse': -.01, 'common_nrmse': -.01}
    if mutation == 'missing':
        changes.pop('rmse')
    elif mutation == 'extra':
        changes['extra'] = -.1
    elif mutation == 'nonfinite':
        changes['rmse'] = float('nan')
    elif mutation == 'none':
        changes['rmse'] = None
    elif mutation == 'non_synthetic':
        episode.metadata['gradient_data'] = 'real'
    else:
        candidate[0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        analysis.decompose(before, candidate, episode, changes)


def group_fixture():
    rows = []
    for cohort in ('legacy', 'mar', 'recsys'):
        for replicate in range(2):
            for metric in ('rmse', 'mae' if cohort == 'recsys' else 'common_nrmse'):
                change = .1 if replicate else -.2
                rows.append({'task': f'{cohort}_draw_{replicate}', 'cohort': cohort, 'metric': metric,
                    'before': 1., 'candidate': 1. + change, 'change': change,
                    'parameter_linear_change': -.05, 'output_linear_change': -.04,
                    'parameter_to_output_residual': .01, 'metric_remainder': change + .04})
    changes = {f'legacy_draw_{replicate}/{metric}': .1 if replicate else -.2
        for replicate in range(2) for metric in ('rmse', 'common_nrmse')}
    changes.update({f'{cohort}/{metric}': -.05 for cohort in ('mar', 'recsys')
        for metric in ('rmse', 'mae' if cohort == 'recsys' else 'common_nrmse')})
    return rows, {'finite': {'changes': changes}}


def test_grouping_keeps_separate_native_guards_and_both_other_replicas():
    rows, record = group_fixture()
    saved = deepcopy(rows)
    groups = analysis.grouped(rows, record)
    assert rows == saved and len(groups) == 8
    assert len([row for row in groups if row['change'] > 0]) == 2
    assert all(row['members'] == (1 if row['comparison'].startswith('legacy_draw_') else 2) for row in groups)


@pytest.mark.parametrize('mutation', ['missing_group', 'missing_replica', 'changed_guard'])
def test_grouping_cannot_omit_a_replica_or_relax_the_recorded_change(mutation):
    rows, record = group_fixture()
    if mutation == 'missing_group':
        rows = [row for row in rows if row['cohort'] != 'mar']
    elif mutation == 'missing_replica':
        rows = [row for row in rows if row['task'] != 'mar_draw_1']
    else:
        record['finite']['changes']['mar/rmse'] = 0.
    with pytest.raises(ValueError):
        analysis.grouped(rows, record)


def closed_reports():
    status = {'state': 'complete', 'retention_audited': True}
    result = {'data_admission_only': False, 'record': {'accepted': False, 'accepted_updates': 0,
        'reason': 'finite_metrics_rejected'}}
    gate = {'passed': False, 'preflight_only': True, 'real_benchmark_success': False}
    audit = {'complete': True, 'training_tasks': 28, 'prediction_grids': 56,
        'raw_endpoint': {'gradient_vectors': 84, 'accepted': False, 'actual_final_parameters_checked': True},
        'teacher_replay': {'component_grids': 52}, 'cuda_initialized': False}
    retention = {'complete': True, 'component_grids': 50, 'reference_grids': 26}
    review = {'complete': True, 'accepted': False, 'config_sha256': analysis.CONFIG_SHA256,
        'full_trial_authorized': False, 'goal_achieved': False}
    return status, result, gate, audit, retention, review


@pytest.mark.parametrize('mutation', [None, (0, 'state', 'awaiting_independent_audit'),
    (0, 'retention_audited', False), (1, 'data_admission_only', True),
    (1, 'record', {'accepted': True}), (1, 'record', {'accepted': False, 'accepted_updates': 1}),
    (1, 'record', {'accepted': False, 'accepted_updates': 0, 'reason': 'exception'}),
    (2, 'passed', True), (2, 'preflight_only', False), (2, 'real_benchmark_success', True),
    (3, 'complete', False), (3, 'training_tasks', 27), (3, 'prediction_grids', 55),
    (3, 'raw_endpoint', {'gradient_vectors': 83}),
    (3, 'raw_endpoint', {'gradient_vectors': 84, 'accepted': True}),
    (3, 'raw_endpoint', {'gradient_vectors': 84, 'accepted': False, 'actual_final_parameters_checked': False}),
    (3, 'teacher_replay', {'component_grids': 51}), (3, 'cuda_initialized', True),
    (4, 'complete', False), (4, 'component_grids', 49), (4, 'reference_grids', 25),
    (5, 'complete', False), (5, 'accepted', True), (5, 'config_sha256', 'wrong'),
    (5, 'full_trial_authorized', True), (5, 'goal_achieved', True)])
def test_wait_for_exact_complete_negative_closure_without_relabeling(mutation):
    reports = closed_reports()
    if mutation is None:
        analysis.validate_reports(*reports)
    else:
        index, key, value = mutation
        reports[index][key] = value
        with pytest.raises(ValueError, match='fully audited rejection'):
            analysis.validate_reports(*reports)


def test_all_null_group_keeps_every_slot_and_no_manufactured_linearization():
    rows, record = group_fixture()
    for row in rows:
        if row['cohort'] == 'mar':
            for field in ('before', 'candidate', 'change', 'parameter_linear_change', 'output_linear_change',
                    'parameter_to_output_residual', 'metric_remainder'):
                row[field] = None
    record['finite']['changes'].update({'mar/rmse': None, 'mar/common_nrmse': None})
    groups = analysis.grouped(rows, record)
    assert len(groups) == 8
    assert all(row['defined_members'] == 0 and row['change'] is None for row in groups if row['comparison'].startswith('mar/'))


def test_diagnostic_callbacks_are_bound_without_inference_or_new_seed():
    assert 'not a causal attribution' in analysis.POLICY
    assert 'never evaluate another endpoint or fraction' in analysis.POLICY
    for function in vars(analysis).values():
        if callable(function) and getattr(function, '__globals__', None) is vars(analysis):
            assert unresolved_globals(function) == [], function.__name__
