from copy import deepcopy
import math

import pytest

import recheck_single_model_standardized_teacher_huber_probe as review


def example():
    tasks = [{'name': f'{cohort}_draw_{draw}', 'cohort': cohort}
        for cohort in ('legacy', 'mar', 'recsys') for draw in range(2)]
    before = {f'{task["name"]}/{metric}': 1. for task in tasks
        for metric in ('rmse', 'mae' if task['cohort'] == 'recsys' else 'common_nrmse')}
    candidate = {name: .9 for name in before}
    candidate['mar_draw_0/rmse'] = 1.05
    before['mar_draw_0/common_nrmse'] = candidate['mar_draw_0/common_nrmse'] = None
    individual = {name: review.delta(value, candidate[name]) for name, value in before.items()}
    grouped = {name: value for name, value in individual.items() if name.startswith('legacy')}
    grouped.update({'mar/rmse': -.025, 'mar/common_nrmse': -.1, 'recsys/rmse': -.1, 'recsys/mae': -.1})
    record = {'accepted': False, 'point_before': before, 'point_after': candidate,
        'finite': {'individual_draw_diagnostics': {'changes': individual}, 'changes': grouped}}
    return record, tasks


def test_point_review_keeps_adverse_null_and_rejected_records():
    record, tasks = example()
    individual, grouped = review.point_tables(record, tasks)
    assert len(individual) == 12 and len(grouped) == 8
    assert sum(row['change'] is None for row in individual) == 1
    assert sum(row['change'] is not None and row['change'] > 0 for row in individual) == 1
    assert all(row['change'] < 0 for row in grouped)
    assert record['accepted'] is False


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'nan', 'changed_null', 'delta', 'group'])
def test_point_review_rejects_incomplete_or_incoherent_records(mutation):
    record, tasks = example()
    record = deepcopy(record)
    if mutation == 'missing':
        record['point_after'].pop('mar_draw_1/rmse')
    elif mutation == 'extra':
        record['point_before']['extra/rmse'] = 1.
    elif mutation == 'nan':
        record['point_after']['mar_draw_1/rmse'] = math.nan
    elif mutation == 'changed_null':
        record['point_after']['mar_draw_0/common_nrmse'] = .1
    elif mutation == 'delta':
        record['finite']['individual_draw_diagnostics']['changes']['mar_draw_1/rmse'] = 0.
    else:
        record['finite']['changes']['mar/rmse'] = 0.
    with pytest.raises(ValueError):
        review.point_tables(record, tasks)


def test_all_null_group_stays_null():
    record, tasks = example()
    record['point_before']['mar_draw_1/common_nrmse'] = None
    record['point_after']['mar_draw_1/common_nrmse'] = None
    record['finite']['individual_draw_diagnostics']['changes']['mar_draw_1/common_nrmse'] = None
    record['finite']['changes']['mar/common_nrmse'] = None
    _, grouped = review.point_tables(record, tasks)
    assert next(row for row in grouped if row['comparison'] == 'mar/common_nrmse')['change'] is None


@pytest.mark.parametrize('actual, expected', [(math.nan, 1.), (1., math.inf), (math.inf, math.inf), (1.01, 1.)])
def test_scalar_review_rejects_nonfinite_and_inexact_values(actual, expected):
    with pytest.raises(ValueError):
        review.close(actual, expected)


def test_hash_inventory_refuses_conflicting_evidence():
    inventory = {}
    review.merge_hashes({'source_sha256': {'path': 'original'}}, inventory)
    with pytest.raises(ValueError):
        review.merge_hashes({'nested': {'input_sha256': {'path': 'changed'}}}, inventory)


def test_csv_preserves_null_and_adverse_values(tmp_path):
    rows = [{'comparison': 'empty', 'change': None}, {'comparison': 'worse', 'change': .02}]
    path = tmp_path / 'rows.csv'
    path.write_text(review.csv_text(rows))
    result = review.read_csv(path)
    assert result[0]['change'] == '' and float(result[1]['change']) == .02


def test_never_overwrite_original_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'ROOT', tmp_path)
    path = tmp_path / 'saved.json'
    path.write_text('original')
    with pytest.raises(FileExistsError):
        review.write_new('saved.json', 'replacement')
    assert path.read_text() == 'original'
