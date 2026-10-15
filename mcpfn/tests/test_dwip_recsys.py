import numpy as np
import pytest
from cornac.data import Dataset

from evaluate_dwip_recsys import FrozenRatings, normalize_training_ratings, user_rating_metrics


def test_user_metrics_weight_users_not_ratings():
    users = np.array([0, 0, 1])
    truth = np.array([1.0, 1.0, 1.0])
    prediction = np.array([1.0, 3.0, 4.0])
    result = user_rating_metrics(users, truth, prediction)
    assert result["users"] == 2
    assert result["MAE"] == pytest.approx(2.0)
    assert result["RMSE"] == pytest.approx((np.sqrt(2.0) + 3) / 2)


def test_normalization_uses_observed_training_entries_and_guards_constants():
    matrix = np.array([[1.0, 5.0], [3.0, np.nan], [np.nan, 5.0]])
    means, stds = normalize_training_ratings(matrix)
    assert np.array_equal(means, [2.0, 5.0])
    assert np.array_equal(stds, [1.0, 1.0])


def test_replay_adapter_does_not_run_imputation():
    train = Dataset.from_uir([("first", "item1", 1.0), ("second", "item2", 5.0)], seed=123)
    predictions = np.array([[1.0, 3.0], [2.0, 5.0]])
    model = FrozenRatings(prediction=predictions)
    model.fit(train)
    assert model.score(0, 1) == 3.0
    assert np.array_equal(model.score(1), [2.0, 5.0])
    with pytest.raises(Exception, match="Unknown user"):
        model.score(2, 0)


def test_replay_adapter_rejects_wrong_shape():
    train = Dataset.from_uir([("first", "item1", 1.0), ("second", "item2", 5.0)], seed=123)
    model = FrozenRatings(prediction=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="train mapping"):
        model.fit(train)
