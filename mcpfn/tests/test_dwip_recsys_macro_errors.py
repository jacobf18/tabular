import numpy as np
import pytest

from diagnose_dwip_recsys_macro_errors import per_user_errors, load_predictions


def test_per_user_errors_preserve_finite_query_rmse_not_root_of_global_mse():
    users = np.array([0, 3, 3])
    counts, errors = per_user_errors(users, np.ones(3), np.array([2., 4., 5.]))
    assert np.array_equal(counts, [1, 0, 0, 2])
    assert errors["MSE"][[0, 3]].mean() == pytest.approx(6.75)
    assert errors["RMSE"][[0, 3]].mean() == pytest.approx((1 + 12.5 ** .5) / 2)
    assert errors["MAE"][[0, 3]].mean() == pytest.approx(2.25)


def test_prediction_loader_rejects_mismatched_query_values(tmp_path):
    path = tmp_path / "queries.npz"
    queries = (np.array([0]), np.array([1]), np.array([4.]))
    np.savez(path, users=queries[0], items=queries[1], truth=queries[2], prediction=np.array([3.5]))
    assert load_predictions(path, queries).tolist() == [3.5]
    with pytest.raises(ValueError, match="frozen inner"):
        load_predictions(path, (queries[0], queries[1], np.array([5.])))
