from copy import deepcopy
import math

import pytest
import torch

import analyze_single_model_standardized_teacher_huber as analysis
import test_single_model_anchor_retention as fixtures
import test_single_model_column_anchor_retention as columns


def all_cases():
    return [{'role': role, 'task': f'{cohort}_draw_{replicate}'} for role in ('before', 'candidate')
        for replicate in (0, 1) for cohort in ('legacy', *analysis.objective.FAMILIES, 'recsys')]


def test_every_case_is_required():
    cases = all_cases()
    analysis.validate_case_inventory(cases)
    with pytest.raises(ValueError, match='all28'):
        analysis.validate_case_inventory(cases[:-1])
    with pytest.raises(ValueError, match='all28'):
        analysis.validate_case_inventory(cases + cases[:1])
    changed = deepcopy(cases)
    changed[0]['role'] = 'selected'
    with pytest.raises(ValueError, match='both original'):
        analysis.validate_case_inventory(changed)


@pytest.mark.parametrize('metadata', [fixtures.metadata(), fixtures.metadata('low_rank_mcar', structured=True, family='mar'),
    fixtures.metadata(domain='recsys')])
def test_complete_diagnostic_counts_and_unmodified_inputs(metadata):
    truth, mask = columns.column_example()
    before_truth, before_mask = truth.clone(), mask.clone()
    teacher = torch.nan_to_num(truth) - .25
    prediction = torch.nan_to_num(truth) + .3
    report = analysis.diagnostics(truth, mask, prediction, teacher, metadata, .125)
    assert report['queries'] == int(mask.sum())
    assert sum(report[name] for name in ('gradient_magnitude_decreased_cells',
        'gradient_magnitude_increased_cells', 'gradient_magnitude_equal_cells')) == int(mask.sum())
    assert report['teacher_huber_standardized'] >= 0
    if metadata['domain'] == 'recsys':
        assert report['teacher_huber_raw'] == report['teacher_huber_standardized']
        assert report['coordinate_gradient_norm_raw'] == report['coordinate_gradient_norm_standardized']
        assert report['rating_term_exactly_unchanged']
    torch.testing.assert_close(truth, before_truth, equal_nan=True, rtol=0, atol=0)
    torch.testing.assert_close(mask, before_mask, rtol=0, atol=0)


@pytest.mark.parametrize('weight', [0., -1., math.nan, math.inf])
def test_nonpositive_or_nonfinite_mass_rejected(weight):
    truth, mask = columns.column_example()
    values = torch.nan_to_num(truth)
    with pytest.raises(ValueError, match='positive task weight'):
        analysis.diagnostics(truth, mask, values, values, fixtures.metadata(), weight)


def test_zero_teacher_error_is_not_divided_by_zero():
    truth, mask = columns.column_example()
    values = torch.nan_to_num(truth)
    report = analysis.diagnostics(truth, mask, values, values, fixtures.metadata(), 1.)
    assert report['teacher_huber_raw'] == report['teacher_huber_standardized'] == 0.
    assert report['gradient_norm_ratio'] is None


def test_existing_analysis_cannot_be_overwritten(monkeypatch, tmp_path):
    monkeypatch.setattr(analysis, 'OUTPUT', tmp_path)
    with pytest.raises(FileExistsError, match='Never overwrite'):
        analysis.main()
