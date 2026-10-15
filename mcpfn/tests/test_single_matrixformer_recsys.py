from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import evaluate_single_matrixformer_recsys as evaluation


def example_split():
    return {"shape": np.array([2, 3]), "train_users": np.array([0, 0, 1, 1]), "train_items": np.array([0, 1, 1, 2]),
            "train_ratings": np.array([.5, 2., 3., 5.]), "test_users": np.array([0, 1]), "test_items": np.array([2, 0]),
            "test_ratings": np.array([4., 1.])}


def test_observed_normalization_cannot_use_held_out_labels():
    split = example_split()
    original = evaluation.observed_matrix(split)
    changed = deepcopy(split)
    changed["test_ratings"] = np.array([999., -999.])
    actual = evaluation.observed_matrix(changed)
    np.testing.assert_array_equal(original, actual)
    assert not np.isfinite(original[split["test_users"], split["test_items"]]).any()
    first, second = evaluation.normalization(original), evaluation.normalization(actual)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


@pytest.mark.parametrize("problem", ("leakage", "duplicate", "bounds", "nonfinite"))
def test_invalid_rating_splits_fail_closed(problem):
    split = example_split()
    if problem == "leakage":
        split["test_items"][0] = 0
    elif problem == "duplicate":
        split["train_items"][1] = 0
    elif problem == "bounds":
        split["test_users"][0] = 2
    else:
        split["train_ratings"][0] = np.nan
    with pytest.raises(ValueError):
        evaluation.observed_matrix(split)


def test_overlapping_tail_query_coverage_is_independent_of_model():
    counts = evaluation.expected_coverage((3000, 1500), np.array([0, 1500, 2999]), np.array([0, 750, 1499]))
    np.testing.assert_array_equal(counts, [1, 4, 1])
    with pytest.raises(ValueError, match="Uncovered"):
        evaluation.expected_coverage((3000, 1500), np.array([3000]), np.array([0]))


def test_predictions_clip_after_collecting_all_overlaps_and_keep_half_star_scale(monkeypatch):
    model = SimpleNamespace()
    observed = np.full((2, 3), np.nan)
    users, items = np.array([0, 1, 1]), np.array([0, 1, 2])
    normalizer = {"means": np.zeros(3), "scales": np.ones(3), "offsets": np.zeros(2), "row_scales": np.ones(2)}
    monkeypatch.setattr(evaluation, "collect_predictions", lambda *args, **kwargs: (np.array([.3, 3., 7.]), np.ones(3, dtype=np.uint8)))
    prediction, counts = evaluation.predict_queries(model, observed, users, items, normalizer, [.5, 5.])
    np.testing.assert_array_equal(prediction, [.5, 3., 5.])
    assert (model.max_num_rows, model.max_num_cols) == (2000, 1000)
    np.testing.assert_array_equal(counts, np.ones(3))
    monkeypatch.setattr(evaluation, "collect_predictions", lambda *args, **kwargs: (np.array([.3, 3., 7.]), np.zeros(3, dtype=np.uint8)))
    with pytest.raises(AssertionError):
        evaluation.predict_queries(model, observed, users, items, normalizer, [.5, 5.])


def test_rating_metrics_are_user_macro_not_pooled():
    split = {"test_users": np.array([0, 0, 0, 1]), "test_ratings": np.array([1., 1., 1., 1.])}
    metrics = evaluation.audit_scores(split, np.array([2., 2., 2., 5.]))
    assert metrics["RMSE"] == 2.5
    assert metrics["MAE"] == 2.5
