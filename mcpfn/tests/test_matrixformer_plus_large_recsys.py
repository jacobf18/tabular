import numpy as np

from evaluate_matrixformer_plus_large_recsys import collect_predictions, fit_observed_blend
from tabimpute.routing import fit_routing
from tabimpute.streamed_inference import overlapping_spans


class BlockExpert:
    max_num_rows = 3
    max_num_cols = 3

    def impute(self, matrix, means, stds, return_full):
        raw = np.full(matrix.shape, np.nanmean(matrix)) + np.arange(matrix.shape[1])
        return np.where(np.isfinite(matrix), matrix, raw), raw


def test_compact_collector_averages_raw_observed_and_query_predictions():
    matrix = np.arange(25, dtype=float).reshape(5, 5)
    matrix[::2, ::2] = np.nan
    users, items = np.indices(matrix.shape).reshape(2, -1)
    actual, counts = collect_predictions(BlockExpert(), matrix, users, items, np.zeros(5), np.ones(5))
    totals, coverage = np.zeros_like(matrix), np.zeros_like(matrix)
    for row_start, row_stop in overlapping_spans(5, 3):
        for col_start, col_stop in overlapping_spans(5, 3):
            selection = np.s_[row_start:row_stop, col_start:col_stop]
            _, raw = BlockExpert().impute(matrix[selection], None, None, True)
            totals[selection] += raw
            coverage[selection] += 1
    np.testing.assert_array_equal(actual, (totals / coverage).ravel())
    np.testing.assert_array_equal(counts, coverage.ravel())
    assert np.any(actual[np.isfinite(matrix.ravel())] != matrix.ravel()[np.isfinite(matrix.ravel())])


def test_compact_observed_fit_matches_full_matrix_and_ignores_queries():
    truth = np.array([[1., np.nan, 3.], [4., 5., np.nan]])
    general = np.array([[2., -999., 3.5], [3., 3., 999.]])
    specialist = np.array([[.5, 999., 4.], [5., 6., -999.]])
    evidence = np.isfinite(truth)
    expected = fit_routing(truth, general, specialist, evidence, np.full(3, np.nanstd(truth)))
    actual = fit_observed_blend(truth[evidence], general[evidence], specialist[evidence])
    assert actual == expected


def test_specialist_restores_global_preprocessing_before_overlap_average():
    matrix = np.arange(25, dtype=float).reshape(5, 5)
    matrix[::2, ::2] = np.nan
    means = np.linspace(1, 2, 5)
    scales = np.linspace(.5, 1, 5)
    offsets = np.linspace(-1, 1, 5)
    row_scales = np.linspace(.5, 2, 5)
    users, items = np.indices(matrix.shape).reshape(2, -1)
    actual, _ = collect_predictions(BlockExpert(), matrix, users, items, means, scales, offsets, row_scales)
    source = ((matrix - offsets[:, None] - means) / scales) / row_scales[:, None]
    totals, coverage = np.zeros_like(source), np.zeros_like(source)
    for row_start, row_stop in overlapping_spans(5, 3):
        for col_start, col_stop in overlapping_spans(5, 3):
            selection = np.s_[row_start:row_stop, col_start:col_stop]
            _, raw = BlockExpert().impute(source[selection], None, None, True)
            totals[selection] += raw
            coverage[selection] += 1
    expected = (totals / coverage) * row_scales[:, None] * scales + means + offsets[:, None]
    np.testing.assert_allclose(actual, expected.ravel(), rtol=0, atol=1e-14)
