import csv
import json

import numpy as np
import pytest

from audit_dwip_missbench_uci import load_original
from evaluate_single_matrixformer_tabular import (
    SCORERS, checked_score, non_cli_ast, prediction_grid, read_blend_rows, verify_hashes, write_summaries,
)
from recsys_evaluation_common import sha256


def test_reference_reader_filters_method_and_rejects_duplicate_cases(tmp_path):
    path = tmp_path / "reference.csv"
    path.write_text("case_id,method,nrmse\nfirst,matrixformer,2.\nfirst,matrixformer_plus,1.\n")
    assert read_blend_rows(path) == {"first": {"case_id": "first", "method": "matrixformer_plus", "nrmse": "1."}}
    with path.open("a") as stream:
        stream.write("first,matrixformer_plus,1.\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_blend_rows(path)


@pytest.mark.parametrize("problem", (None, "shape", "coverage", "nonfinite"))
def test_prediction_reconstruction_checks_all_queries(tmp_path, problem):
    missing = np.array([[np.nan, 2.], [3., np.nan]], dtype=np.float32)
    values = np.array([1., 4.], dtype=np.float32)
    shape = missing.shape
    if problem == "shape":
        shape = (1, 4)
    elif problem == "coverage":
        values = values[:1]
    elif problem == "nonfinite":
        values[0] = np.nan
    path = tmp_path / "predictions.npz"
    np.savez(path, shape=shape, prediction=values)
    if problem is None:
        np.testing.assert_array_equal(prediction_grid(missing, path, "prediction"), [[1., 2.], [3., 4.]])
    else:
        with pytest.raises(ValueError):
            prediction_grid(missing, path, "prediction")


@pytest.mark.parametrize("suite", ("missbench", "uci"))
def test_original_nrmse_and_constant_target_conventions(suite):
    truth = np.array([[1., 2., 1.], [1., 2., 3.], [1., 2., 5.]])
    prediction = np.array([[1., 3., 2.], [1., 3., 4.], [1., 3., 6.]])
    missing = np.full_like(truth, np.nan)
    original = load_original(SCORERS[suite], "compute_normalized_rmse_columnwise")
    score, diagnostics = checked_score(truth, prediction, missing, suite, original)
    assert score == pytest.approx((1 / np.std([1., 3., 5.])) / 2)
    assert diagnostics == {"missing_cells": 9, "nrmse_scored_columns": 2, "nrmse_skipped_constant_columns": 1}


def test_summaries_use_equal_dataset_weights_and_never_hide_missing_cases(tmp_path):
    cases = [{"case_id": f"case{index}", "suite": "uci", "dataset": dataset, "pattern": "MCAR",
              "missingness": .4, "repeat": index, "nrmse_original": value}
             for index, (dataset, value) in enumerate((("small", 1.), ("large", 3.), ("large", 5.)))]
    completed = {case["case_id"]: {**case, "new_nrmse": case["nrmse_original"] - .5,
                                  "blend_nrmse": case["nrmse_original"], "delta_nrmse": -.5}
                 for case in cases}
    write_summaries(tmp_path, {"cases": cases}, completed)
    with (tmp_path / "overall.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert float(row["previous_nrmse"]) == 2.5
    assert float(row["new_nrmse"]) == 2.
    del completed["case2"]
    write_summaries(tmp_path, {"cases": cases}, completed)
    with (tmp_path / "overall.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["new_nrmse"] == ""
    assert row["complete"] == "False"


def test_source_compatibility_ignores_only_cli_main(tmp_path):
    previous, current = tmp_path / "previous.py", tmp_path / "current.py"
    previous.write_text("def infer():\n    return 1\ndef main():\n    pass\n")
    current.write_text("def infer():\n    return 1\ndef main():\n    raise ValueError()\n")
    assert non_cli_ast(previous) == non_cli_ast(current)
    current.write_text("def infer():\n    return 2\ndef main():\n    pass\n")
    assert non_cli_ast(previous) != non_cli_ast(current)


def test_changed_frozen_file_is_rejected(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"case": 1}))
    hashes = {str(path): sha256(path)}
    verify_hashes(hashes)
    path.write_text(json.dumps({"case": 2}))
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        verify_hashes(hashes)
