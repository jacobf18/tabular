import numpy as np
import pandas as pd
import pytest
import torch

from evaluate_tabarena_regression import checked_prediction, chunk_ranges, predict_in_chunks, summary_rows
from audit_tabarena_regression import regression_metrics
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


@pytest.mark.parametrize("count,batch_size", [(303, 8), (1914, 1), (501, 3), (15, 8), (256, 8)])
def test_chunks_preserve_inference_batch_layout(count, batch_size):
    chunks = chunk_ranges(count, batch_size)
    assert chunks[0][0] == 0 and chunks[-1][1] == count
    assert all(start % batch_size == 0 for start, stop in chunks)
    assert all(stop % batch_size == 0 for start, stop in chunks[:-1])
    rebuilt = [min(batch_size, stop - position) for start, stop in chunks
               for position in range(start, stop, batch_size)]
    original = [min(batch_size, count - position) for position in range(0, count, batch_size)]
    assert rebuilt == original


@pytest.mark.parametrize("values,count", [([1, np.nan], 2), ([1], 2), ([[1]], 1)])
def test_reject_incomplete_or_nonfinite_predictions(values, count):
    with pytest.raises(ValueError):
        checked_prediction(values, count)


def test_summary_ratios_are_unitless():
    records = []
    for dataset, scale in [("small_units", 1), ("large_units", 1000)]:
        records.append({"dataset": dataset, "task_id": scale,
                        "metrics": {method: {"rmse": error * scale, "mae": error * scale, "r2": 0.1}
                                    for method, error in [("MatrixFormer", 2), ("RandomForest", 3), ("Dummy", 4)]},
                        "train_rows": 2000, "test_rows": 1000, "context_rows": 1023,
                        "preprocessing": {"encoded_features": 5},
                        "timings": {method: {"fit_seconds": 1, "predict_seconds": 2}
                                    for method in ["MatrixFormer", "RandomForest", "Dummy"]}})
    summary = summary_rows(records)
    np.testing.assert_array_equal(summary[summary.method == "MatrixFormer"].rmse_over_dummy, [0.5, 0.5])
    np.testing.assert_array_equal(summary[summary.method == "RandomForest"].rmse_over_dummy, [0.75, 0.75])


def test_chunk_parameters_must_be_positive():
    with pytest.raises(ValueError):
        chunk_ranges(1, 0)


@pytest.mark.parametrize("truth,prediction", [([1, 4, 2], [2, 3, 2]), ([1, 1], [1, 1]), ([1, 1], [2, 2])])
def test_independent_audit_matches_sklearn(truth, prediction):
    values = regression_metrics(truth, prediction)
    assert values["rmse"] == pytest.approx(root_mean_squared_error(truth, prediction))
    assert values["mae"] == pytest.approx(mean_absolute_error(truth, prediction))
    assert values["r2"] == pytest.approx(r2_score(truth, prediction))


def test_interrupted_chunks_resume_without_repredicting_and_reject_corruption(tmp_path, monkeypatch):
    for function in ["reset_peak_memory_stats", "synchronize"]:
        monkeypatch.setattr(torch.cuda, function, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)

    class Predictor:
        batch_size_ = 2

        def __init__(self, fail_after_first):
            self.calls = []
            self.fail_after_first = fail_after_first

        def predict(self, features):
            self.calls.append(len(features))
            if self.fail_after_first and len(self.calls) > 1:
                raise RuntimeError("Simulated interruption")
            return features["value"].to_numpy(dtype=np.float64) * 2

    features = pd.DataFrame({"value": np.arange(70)})
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        predict_in_chunks(Predictor(True), features, tmp_path)
    resumed = Predictor(False)
    prediction, records = predict_in_chunks(resumed, features, tmp_path)
    np.testing.assert_array_equal(prediction, features.value * 2)
    assert resumed.calls == [32, 6]
    assert [(record["start"], record["stop"]) for record in records] == [(0, 32), (32, 64), (64, 70)]
    (tmp_path / "chunks/0000000_0000032.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Corrupt prediction chunk"):
        predict_in_chunks(resumed, features, tmp_path)
