import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_recsys_specialist_transfer import compare_llm_panels, decode_prediction, export_tabular, save_prediction


def panel(value):
    return {"benchmark_names": np.array(["a", "b"]), **{
        f"{method}_{metric}": np.full((10, 15), value)
        for method in ("entropy", "mi", "random") for metric in ("r2", "rmse")}}


def test_llm_full_coverage_and_delta():
    rows, folds = compare_llm_panels(panel(.5), panel(.2), "mteb")
    assert len(rows) == 45
    assert len(folds) == 900
    assert rows[0]["delta_r2"] == pytest.approx(-.3)
    assert rows[-1]["budget"] == 15


@pytest.mark.parametrize("mutation", ["names", "missing_fold", "nonfinite"])
def test_llm_rejects_mismatched_panels(mutation):
    candidate = panel(.2)
    if mutation == "names":
        candidate["benchmark_names"] = np.array(["b", "a"])
    elif mutation == "missing_fold":
        candidate["mi_r2"] = np.zeros((9, 15))
    else:
        candidate["mi_rmse"][0, 0] = np.nan
    with pytest.raises(ValueError):
        compare_llm_panels(panel(.5), candidate, "mteb")


def test_prediction_reconstruction_and_shape_validation(tmp_path):
    source = np.array([[1., np.nan], [np.nan, 4.]])
    path = tmp_path / "prediction.npz"
    np.savez(path, missing_values=np.array([2., 3.]), shape=np.array([2, 2]))
    assert np.array_equal(decode_prediction(path, source), np.array([[1., 2.], [3., 4.]]))
    np.savez(path, missing_values=np.array([2.]), shape=np.array([2, 2]))
    with pytest.raises(ValueError):
        decode_prediction(path, source)


def test_save_prediction_creates_nested_case_directories(tmp_path):
    source = np.array([[1., np.nan], [np.nan, 4.]])
    prediction = np.array([[1., 2.], [3., 4.]])
    path = tmp_path / "predictions" / "uci" / "dataset" / "repeat-0.npz"
    save_prediction(path, prediction, np.isnan(source))
    assert np.array_equal(decode_prediction(path, source), prediction)


def test_incomplete_group_has_no_headline_and_equal_dataset_weights(tmp_path):
    reference = [dict(case_id=str(index), suite="uci", pattern="MCAR", missingness=.1,
                      dataset="first" if index < 2 else "second", nrmse_original=1.) for index in range(3)]
    records = [{**row, "new_nrmse": 2. if index < 2 else 4.} for index, row in enumerate(reference)]
    export_tabular(tmp_path, reference, records[:2])
    assert "pending" in (tmp_path / "RESULTS.md").read_text()
    export_tabular(tmp_path, reference, records)
    assert "3.000000" in (tmp_path / "RESULTS.md").read_text()
