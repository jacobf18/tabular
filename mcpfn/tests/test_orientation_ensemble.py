import numpy as np
import pytest

from evaluate_orientation_ensemble import BASE_VARIANTS, OrientationRouter, combine_predictions, summarize_tabular
from tabimpute.routing import TwoExpertRouter


class ColumnExpert:
    def __init__(self, shift):
        self.shift = shift
        self.inputs = []

    def impute(self, matrix, return_full=False):
        self.inputs.append(matrix.copy())
        raw = np.broadcast_to(np.nanmean(matrix, axis=0) + self.shift, matrix.shape).copy()
        return np.where(np.isnan(matrix), raw, matrix), raw


def test_transpose_maps_back_and_average_is_on_predictions_not_scores():
    matrix = np.array([[1., np.nan, 5.], [2., 3., np.nan]])
    experts = [ColumnExpert(0), ColumnExpert(.5)]
    actual = OrientationRouter(*experts).predict_all(matrix)
    original = TwoExpertRouter(ColumnExpert(0), ColumnExpert(.5)).predict_all(matrix, include_holdout=False)
    transposed = TwoExpertRouter(ColumnExpert(0), ColumnExpert(.5)).predict_all(matrix.T.copy(), include_holdout=False)
    for name in BASE_VARIANTS:
        np.testing.assert_array_equal(actual.predictions[name], original.predictions[name])
        np.testing.assert_array_equal(actual.predictions[name + '_transpose'], transposed.predictions[name].T)
        np.testing.assert_array_equal(actual.predictions[name + '_average'],
                                      (original.predictions[name] + transposed.predictions[name].T) / 2)
        for suffix in ('', '_transpose', '_average'):
            np.testing.assert_array_equal(actual.predictions[name + suffix][np.isfinite(matrix)], matrix[np.isfinite(matrix)])
    for expert in experts:
        np.testing.assert_array_equal(expert.inputs[0], matrix)
        np.testing.assert_array_equal(expert.inputs[1], matrix.T)
    assert actual.decisions['original']['evidence_cells'] == actual.decisions['transpose']['evidence_cells'] == 4


def test_combiner_rejects_wrong_shape_and_nonfinite_values():
    values = {name: np.array([1., 2.]) for name in BASE_VARIANTS}
    with pytest.raises(ValueError):
        combine_predictions(values, {**values, 'general': np.ones(3)})
    with pytest.raises(ValueError):
        combine_predictions(values, {**values, 'specialist': np.array([1., np.nan])})


def test_summary_keeps_dataset_balancing_and_all_variants(tmp_path):
    import csv
    from evaluate_orientation_ensemble import VARIANTS

    cases = [{'case_id': str(index), 'suite': 'uci', 'pattern': 'MCAR', 'dataset': 'first' if index < 2 else 'second',
              'missingness': .1, 'repeat': index} for index in range(3)]
    records = [{**case, 'nrmse': {name: 2. if index < 2 else 4. for name in VARIANTS}} for index, case in enumerate(cases)]
    summarize_tabular(tmp_path, records, cases)
    with (tmp_path / 'summary.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row['pattern'] for row in rows} == {'MCAR', 'overall'}
    assert all(float(row[name]) == 3. for row in rows for name in VARIANTS)


def test_empty_transposed_columns_use_neutral_statistics_without_filling():
    from evaluate_orientation_ensemble import standardize_orientation_uci

    matrix = np.array([[1., np.nan, 7.], [3., np.nan, 7.]], dtype=np.float32)
    normalized, means, scales = standardize_orientation_uci(matrix)
    np.testing.assert_array_equal(means, [2., 0., 7.])
    np.testing.assert_array_equal(scales, [1., 1., 1.])
    np.testing.assert_array_equal(np.isnan(normalized), np.isnan(matrix))
    np.testing.assert_array_equal(normalized * scales + means, matrix)


def test_nonempty_uci_orientation_normalization_retains_exact_original_values():
    from evaluate_orientation_ensemble import campaign, standardize_orientation_uci

    matrix = np.random.default_rng(42).normal(size=(307, 13)).astype(np.float32, order='F')
    matrix[::3, ::2] = np.nan
    expected = campaign.standardize_uci(matrix)[:3]
    actual = standardize_orientation_uci(matrix)
    for before, after in zip(expected, actual):
        np.testing.assert_array_equal(before, after)
