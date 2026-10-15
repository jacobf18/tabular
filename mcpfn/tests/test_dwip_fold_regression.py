import numpy as np
import pytest

from diagnose_dwip_fold_regression import CaptureImputer, decompose_scores, verify_reproduced_scores


def test_decomposition_preserves_official_clipping_and_excludes_selected_cells():
    target = np.array([[100., 20., 2.], [100., 1., 3.]])
    prediction = np.array([[-999., 0., 1.], [-999., 2., 1.]])
    observed = np.ones_like(target)
    result = decompose_scores(target, observed, prediction, [0], np.zeros(3), np.ones(3))
    assert result["metrics"]["n_eval"] == 4
    assert result["scored_sse"] == 106
    assert result["scored_sst"] == 114
    assert result["metrics"]["r2"] == pytest.approx(1 - 106 / 114)
    assert result["raw_rmse_report_only"] == pytest.approx((406 / 4) ** .5)
    assert result["columns"][0]["targets_clipped"] == 1
    assert not result["evaluation_mask"][:, 0].any()


def test_decomposition_rejects_nonfinite_predictions_and_empty_scoring_masks():
    target = np.ones((2, 2))
    with pytest.raises(ValueError, match="Nonfinite"):
        decompose_scores(target, target, np.full_like(target, np.nan), [], np.zeros(2), np.ones(2))
    with pytest.raises(ValueError, match="No evaluated"):
        decompose_scores(target, target, target, [0, 1], np.zeros(2), np.ones(2))


def test_capture_preserves_original_inputs_without_changing_returned_predictions():
    class MutatingImputer:
        def impute(self, source):
            source[0, 0] = 99.
            return source

    source = np.ones((2, 2))
    capture = CaptureImputer(MutatingImputer())
    result = capture.impute(source)
    assert result[0, 0] == 99.
    assert capture.calls[0]["source"][0, 0] == 1.
    result[0, 0] = 12.
    assert capture.calls[0]["prediction"][0, 0] == 99.


def test_score_reproduction_rejects_backend_sized_drift_and_count_mismatch():
    official = {"r2": .2988883819749727, "rmse": .6437514551807808, "n_eval": 985}
    cached = {f"random_{metric}": np.full((10, 15), official[metric]) for metric in ("r2", "rmse")}
    verify_reproduced_scores(official, cached, official, "original", 8, 10)
    drifted = {**official, "r2": .2974429532160293}
    with pytest.raises(RuntimeError, match="fresh=.*cached="):
        verify_reproduced_scores(drifted, cached, drifted, "original", 8, 10)
    with pytest.raises(RuntimeError, match="cell count"):
        verify_reproduced_scores(official, cached, {**official, "n_eval": 984}, "original", 8, 10)
