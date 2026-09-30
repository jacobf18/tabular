import numpy as np
import pytest
from scipy.sparse import csr_matrix

from evaluate_large_recsys import predict_blocks, rating_statistics, score_independently
from tabimpute.streamed_inference import StreamedImputePFN, overlapping_spans


def test_overlapping_spans_cover_tail_without_short_context():
    assert overlapping_spans(5, 3) == [(0, 3), (2, 5)]
    assert overlapping_spans(6, 3) == [(0, 3), (3, 6)]
    assert overlapping_spans(2, 3) == [(0, 2)]
    with pytest.raises(ValueError):
        overlapping_spans(5, 0)


def test_sparse_statistics_match_dense_training_only_statistics():
    items = np.array([0, 1, 0, 2])
    ratings = np.array([1.0, 4.0, 5.0, 3.0])
    means, stds = rating_statistics(items, ratings, 3, "global")
    np.testing.assert_allclose(means, ratings.mean())
    np.testing.assert_allclose(stds, ratings.std())
    means, stds = rating_statistics(items, ratings, 3, "column")
    np.testing.assert_array_equal(means, [3.0, 4.0, 3.0])
    np.testing.assert_array_equal(stds, [2.0, 1.0, 1.0])


def test_sparse_query_aggregation_and_observation_preservation(tmp_path):
    class ConstantImputer:
        max_num_rows, max_num_cols = 3, 3

        def impute(self, matrix, **kwargs):
            assert np.isfinite(matrix).sum() > 0
            return np.where(np.isnan(matrix), 3.0, matrix)

    observed = np.eye(5)
    train = csr_matrix(observed)
    users, items = np.array([0, 2, 4]), np.array([1, 3, 2])
    prediction, counts = predict_blocks(ConstantImputer(), train, users, items, np.zeros(5), np.ones(5), tmp_path / "status.json", {})
    np.testing.assert_array_equal(prediction, [3.0, 3.0, 3.0])
    np.testing.assert_array_equal(counts, [1, 2, 2])
    with pytest.raises(ValueError, match="visible"):
        predict_blocks(ConstantImputer(), train, np.array([0]), np.array([0]), np.zeros(5), np.ones(5), tmp_path / "status.json", {})


def test_macro_rmse_is_mean_of_user_rmse_not_pooled():
    users = np.array([0, 1, 1, 1])
    result = score_independently(users, np.ones(4), np.array([4.0, 1.0, 1.0, 1.0]))
    assert result == {"MAE": 1.5, "RMSE": 1.5}


def test_standalone_adapter_normalizes_restores_and_keeps_observations():
    import torch

    imputer = StreamedImputePFN.__new__(StreamedImputePFN)
    imputer.device, imputer.direct_dtype = "cpu", torch.float32
    imputer.max_num_rows, imputer.max_num_cols = 3, 2
    imputer._stream_points = lambda values: torch.ones_like(values) * 3
    matrix = np.full((5, 3), np.nan)
    matrix[0, 0] = 1.0
    actual = imputer.impute(matrix, means=np.full(3, 10.0), stds=np.full(3, 2.0))
    assert actual[0, 0] == 1.0
    np.testing.assert_array_equal(actual[np.isnan(matrix)], 16.0)
    assert np.isnan(matrix[1:, :]).all()
    with pytest.raises(ValueError, match="columns"):
        imputer.impute(matrix, means=np.zeros(2), stds=np.ones(2))
