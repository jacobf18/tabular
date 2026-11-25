import json

import numpy as np
import pandas as pd
import pytest

from evaluate_final_matrixformer_llm import (
    checked_metrics, fold_rows, kernel_ridge, summarize, verify_prediction,
)
from recsys_evaluation_common import atomic_npz, sha256, write_json


def test_ridge_clips_revealed_inputs_and_keeps_zero_support_rows():
    visible = np.array([[100., np.nan], [np.nan, np.nan]])
    frozen = dict(mean=np.zeros(2), scale=np.ones(2), correlation=np.array([[1., .5], [.5, 1.]]))
    query = np.array([[False, True], [True, True]])
    prediction = kernel_ridge(visible, frozen, query)
    np.testing.assert_allclose(prediction, [5. / 1.01, 0., 0.])


def test_fold_rows_keep_standardized_rmse_distinct_from_relative_rmse():
    specification = dict(dataset='merged', method='mi', budget=5, fold=0)
    scores = dict(matrixformer=checked_metrics(np.array([2., 2.]), np.zeros(2)))
    row = fold_rows(specification, scores)[0]
    assert row['rmse'] == 2.
    assert row['relative_rmse'] == 1.
    assert row['selection'] == 'mi' and row['method'] == 'matrixformer'


def test_prediction_resume_rechecks_arrays_and_hashes(tmp_path):
    path = tmp_path / 'prediction.npz'
    truth, prediction = np.array([2., 3.]), np.array([1., 4.])
    query = np.array([[True, False], [False, True]])
    specification = dict(dataset='merged', method='mi', budget=1, fold=0)
    atomic_npz(path, target_z=truth, student_z=prediction, kernel_ridge_z=np.zeros(2),
               query_rows=np.where(query)[0], query_columns=np.where(query)[1])
    scores = dict(matrixformer=checked_metrics(truth, prediction),
                  kernel_ridge=checked_metrics(truth, np.zeros(2)))
    record = dict(**specification, manifest_sha256='manifest', prediction_sha256=sha256(path), scores=scores)
    write_json(path.with_suffix('.json'), record)
    assert verify_prediction(path, specification, truth, query, 'manifest') == record
    with pytest.raises(ValueError, match='manifest'):
        verify_prediction(path, specification, truth, query, 'changed')
    with pytest.raises(AssertionError):
        verify_prediction(path, specification, truth[::-1], query, 'manifest')
    record['scores']['matrixformer']['rmse'] = 900.
    path.with_suffix('.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='rescore'):
        verify_prediction(path, specification, truth, query, 'manifest')


def test_summary_requires_complete_inventory_before_completion(tmp_path):
    records = []
    for fold, truth in enumerate([np.array([1.]), np.full(9, 3.)]):
        metric = checked_metrics(truth, np.zeros_like(truth))
        records.append(dict(dataset='merged', method='mi', budget=1, fold=fold,
                            scores=dict(matrixformer=metric, kernel_ridge=metric)))
    summarize(tmp_path, records)
    panels = pd.read_csv(tmp_path / 'panels.csv')
    np.testing.assert_array_equal(panels.rmse, [2., 2.])
    with pytest.raises(ValueError, match='Incomplete'):
        summarize(tmp_path, records, complete=True)
    with pytest.raises(ValueError, match='Duplicated'):
        summarize(tmp_path, records + records)
