from types import SimpleNamespace

import numpy as np
import pytest

from evaluate_chunked_uci import configure_windows, summarize
from tabimpute.interface import ImputePFN


def test_smaller_matrices_keep_original_full_execution():
    imputer = SimpleNamespace()
    configure_windows(imputer, (1503, 5), 2000, 1000)
    assert imputer.max_num_rows is None and imputer.max_num_cols is None
    configure_windows(imputer, (20640, 9), 2000, 1000)
    assert imputer.max_num_rows == 2000 and imputer.max_num_cols is None
    assert imputer.max_num_chunks == 1
    with pytest.raises(ValueError):
        configure_windows(imputer, (5, 3), 0, 1000)


def test_windows_preserve_original_masks_and_average_overlap():
    imputer = ImputePFN.__new__(ImputePFN)
    imputer.verbose = False
    configure_windows(imputer, (5, 1), 3, 2)
    matrix = np.array([[0.0], [np.nan], [np.nan], [np.nan], [4.0]])
    windows = []

    def predict(window, num_repeats):
        windows.append(window.copy())
        prediction = np.where(np.isnan(window), np.nanmean(window), window)
        return prediction, prediction

    imputer._get_imputation_chunk = predict
    actual, _ = imputer.get_imputation(matrix.copy())
    assert len(windows) == 2
    assert np.isnan(windows[0][2, 0]) and np.isnan(windows[1][0, 0])
    np.testing.assert_array_equal(actual[:, 0], [0, 0, 2, 4, 4])
    assert np.isnan(matrix[1:4]).all()


def test_summary_keeps_checkpoint_and_geometry_comparisons_distinct(tmp_path):
    parent = [{"case_id": "case", "suite": "uci", "dataset": "example", "pattern": "MCAR", "missingness": 0.4,
               "nrmse_original": 1.0}]
    full = [{**parent[0], "nrmse_original": 1.2}]
    records = [{**parent[0], "new_nrmse": 0.9}]
    summarize(tmp_path, parent, full, records, 2000, 1000)
    text = (tmp_path / "RESULTS.md").read_text()
    assert "1.000000 | 1.200000 | 0.900000 | -25.00%" in text
    assert "no row train/test split" in text
