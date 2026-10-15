import numpy as np
import pytest

from evaluate_single_matrixformer_llm import checked_metrics, panel_summaries, predict_setting, setting_inputs, setting_key


def test_setting_inputs_keep_zero_support_test_rows_and_exclude_query_labels():
    scores = np.arange(12, dtype=float).reshape(4, 3)
    observed = np.ones_like(scores, dtype=bool)
    observed[2, 0] = False
    mask = observed[2:].copy()
    mask[:, 0] = False
    frozen = {"context": np.array([0, 1]), "test": np.array([2, 3]), "order_entropy": np.array([0, 1, 2]),
              "mask_entropy_01": mask, "mean": np.zeros(3), "scale": np.ones(3)}
    context, visible, query, truth = setting_inputs(scores, observed, frozen, "entropy", 1)
    np.testing.assert_array_equal(context, scores[:2])
    assert np.isnan(visible[0]).all()
    assert query[0].sum() == 2
    assert not np.isfinite(visible[query]).any()
    np.testing.assert_array_equal(truth, [7., 8., 10., 11.])
    frozen["test"] = np.array([1, 3])
    with pytest.raises(ValueError, match="overlap"):
        setting_inputs(scores, observed, frozen, "entropy", 1)


def test_changed_frozen_query_mask_rejected():
    scores = np.arange(12, dtype=float).reshape(4, 3)
    observed = np.ones_like(scores, dtype=bool)
    frozen = {"context": np.array([0, 1]), "test": np.array([2, 3]), "order_entropy": np.array([0, 1, 2]),
              "mask_entropy_01": np.ones((2, 3), dtype=bool), "mean": np.zeros(3), "scale": np.ones(3)}
    with pytest.raises(AssertionError):
        setting_inputs(scores, observed, frozen, "entropy", 1)


def test_single_model_prediction_uses_identical_context_in_each_200_row_batch():
    class Model:
        def __init__(self):
            self.batches = []

        def impute(self, values, return_full=False):
            assert return_full
            self.batches.append(values.copy())
            completed = np.nan_to_num(values, nan=7.)
            return completed, completed.copy()

    model = Model()
    context = np.array([[1., 2.], [3., 4.]])
    visible = np.full((401, 2), np.nan)
    visible[:, 0] = 3.
    mask = np.isnan(visible)
    predictions = predict_setting(model, context, visible, mask, np.array([1., 1.]), np.array([2., 2.]))
    np.testing.assert_array_equal(predictions, np.full(401, 3.))
    assert [len(batch) for batch in model.batches] == [202, 202, 3]
    for batch in model.batches:
        np.testing.assert_array_equal(batch[:2], context)
        assert np.isnan(batch[2:, 1]).all()


def test_metric_uses_clipped_context_centered_r2_not_test_centered_r2():
    metrics = checked_metrics(np.array([20., -20., 2.]), np.array([0., -10., 1.]))
    assert metrics["ss_tot"] == 204.
    assert metrics["ss_res"] == 101.
    assert metrics["r2"] == pytest.approx(1 - 101 / 204)
    assert metrics["rmse"] == pytest.approx((101 / 3) ** .5)
    assert metrics["n_eval"] == 3
    empty = checked_metrics(np.array([]), np.array([]))
    assert empty["r2"] is None and empty["rmse"] is None and empty["n_eval"] == 0
    with pytest.raises(ValueError, match="finite"):
        checked_metrics(np.array([1.]), np.array([np.nan]))


def test_panel_means_are_equal_fold_not_query_weighted_and_require_all_folds():
    records = [{"dataset": "mteb", "method": "entropy", "budget": 1, "fold": fold,
                "student": {"r2": fold / 10, "rmse": 1., "n_eval": fold + 1},
                "blend": {"r2": fold / 10 - .1, "rmse": 2., "n_eval": fold + 1}}
               for fold in range(10)]
    summary = panel_summaries(records)[0]
    assert summary["complete"] and summary["student_r2"] == pytest.approx(.45)
    assert summary["delta_r2"] == pytest.approx(.1)
    assert summary["n_eval"] == 55
    assert panel_summaries(records[:-1])[0]["student_r2"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        panel_summaries(records + records[:1])
    assert setting_key("mteb", 3, "entropy", 1) == "mteb/fold_03/entropy_01"
