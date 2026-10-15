from copy import deepcopy

import numpy as np
import pytest

import analyze_single_model_teacher_alignment as analysis


def test_count_shrinkage_keeps_constant_singleton_and_unequal_query_columns():
    truth = np.array([[2., 1.], [2., 3.], [2., 9.]])
    mask = np.array([[True, True], [True, False], [True, False]])
    prediction = truth + np.array([1., 2.])
    pooled, normalized = analysis.squared_distances(prediction, truth, truth, mask)
    assert pooled == pytest.approx(7. / 4.)
    assert normalized == pytest.approx((4. + 8.) / 2.)


def test_unqueried_columns_have_no_weight_but_all_queried_columns_remain():
    truth = np.array([[0., 100.], [2., 200.]])
    mask = np.array([[True, False], [True, False]])
    actual = analysis.squared_distances(truth + 1., truth, truth, mask)
    assert actual == pytest.approx((1., 1.))


@pytest.mark.parametrize('fault', ['empty', 'nonfinite', 'shape', 'mask_dtype'])
def test_invalid_evidence_is_rejected(fault):
    truth = np.ones((2, 3))
    prediction = truth.copy()
    mask = np.ones_like(truth, dtype=bool)
    if fault == 'empty':
        mask[:] = False
    elif fault == 'nonfinite':
        prediction[0, 0] = np.nan
    elif fault == 'shape':
        prediction = prediction[:1]
    else:
        mask = mask.astype(float)
    with pytest.raises(ValueError, match='complete finite'):
        analysis.squared_distances(prediction, truth, truth, mask)


def test_observed_queries_stay_in_both_masks_and_observed_targets_use_source():
    truth = np.zeros((2, 2))
    source = np.array([[3., np.nan], [np.nan, 5.]])
    query = np.array([[True, True], [False, False]])
    arrays = {role: truth.copy() for role in ['general', 'teacher', 'v8', 'student']}
    arrays['student'][:] = 2.
    rows = analysis.score_case(source, truth, query, arrays)
    student = {row['split']: row for row in rows if row['role'] == 'student'}
    assert len(rows) == 8
    assert student['query']['selected_cells'] == student['observed']['selected_cells'] == 2
    assert student['query']['target_mse'] == 4.
    assert student['observed']['target_mse'] == 5.
    assert student['query']['teacher_huber'] == 1.5


def test_role_specific_actual_initializer_is_not_required_on_confirmation():
    values = np.ones((2, 2))
    arrays = {role: values.copy() for role in ['general', 'teacher', 'v8', 'student']}
    assert len(analysis.score_case(values, values, values.astype(bool), arrays)) == 8
    arrays['actual_v29'] = values.copy()
    assert len(analysis.score_case(values, values, values.astype(bool), arrays)) == 10
    arrays['initialization'] = arrays.pop('v8')
    with pytest.raises(ValueError, match='explicit'):
        analysis.score_case(values, values, values.astype(bool), arrays)


@pytest.mark.parametrize('fault', ['missing', 'seed', 'domain', 'geometry', 'real', 'structured'])
def test_complete_pairing_rejects_omissions_or_identity_changes(fault):
    original = [{'index': 0, 'seed': 100, 'rows': 3, 'columns': 4,
                 'domain': 'imputation', 'gradient_data': 'synthetic',
                 'routing': {'target': 'same-input observed blend'}}]
    candidate = deepcopy(original)
    if fault == 'missing':
        candidate.clear()
    else:
        field, value = {'seed': ('seed', 101), 'domain': ('domain', 'recsys'),
                        'geometry': ('rows', 5), 'real': ('gradient_data', 'real'),
                        'structured': ('structured', True)}[fault]
        candidate[0][field] = value
    with pytest.raises(ValueError):
        analysis.paired_cases({'teacher': {'cases': original}, 'student': {'cases': candidate}}, {'legacy': 1})


def test_original_teacher_role_and_complete_case_inventory():
    original = [{'index': 0, 'seed': 100, 'rows': 3, 'columns': 4, 'domain': 'imputation',
                 'gradient_data': 'synthetic', 'routing': {'target': 'same-input observed blend'}}]
    reports = {'teacher': {'cases': original}, 'student': {'cases': deepcopy(original)}}
    assert analysis.paired_cases(reports, {'legacy': 1}) == original
    with pytest.raises(ValueError, match='coverage'):
        analysis.paired_cases(reports, {'legacy': 2})
    original[0]['routing']['target'] = 'general'
    with pytest.raises(ValueError, match='teacher target'):
        analysis.paired_cases(reports, {'legacy': 1})


def test_equal_case_summary_does_not_pool_large_matrices():
    rows = [{'bank': 'development', 'cohort': 'legacy', 'split': 'query', 'role': 'student',
             'family': 'anchor', 'index': index, 'selected_cells': count,
             **{metric: score for metric in analysis.METRICS}}
            for index, count, score in [(0, 1, 1.), (1, 999, 3.)]]
    summary = analysis.summarize(rows)
    assert all(row['teacher_mse'] == 2. and row['cases'] == 2 and row['cells'] == 1000 for row in summary)
    with pytest.raises(ValueError, match='duplicate'):
        analysis.summarize(rows + rows[:1])


def test_bound_evidence_identity_cannot_change(tmp_path):
    path = tmp_path / 'evidence'
    path.write_bytes(b'original')
    inputs = {}
    analysis.bind(path, inputs)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity changed'):
        analysis.bind(path, {}, next(iter(inputs.values())))
    with pytest.raises(ValueError, match='contradictory'):
        analysis.bind(path, inputs)
