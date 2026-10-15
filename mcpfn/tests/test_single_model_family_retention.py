import numpy as np
import pytest

from diagnose_single_model_family_retention import column_changes, summarize_family


def test_exact_error_decomposition_and_query_only_coverage():
    truth = np.array([[0., 2., 4.], [2., 2., np.nan], [4., 2., np.nan]])
    mask = np.array([[True, True, False], [True, True, False], [False, True, False]])
    initial = np.array([[1., 3., np.nan], [3., 3., np.nan], [np.nan, 3., np.nan]])
    final = np.array([[0., 4., np.nan], [2., 4., np.nan], [np.nan, 4., np.nan]])
    rows = column_changes(truth, mask, initial, final)
    assert len(rows) == 2
    assert rows[0]["initial_common_nrmse"] == 1
    assert rows[0]["final_common_nrmse"] == 0
    assert rows[0]["mse_linear_change"] == -2
    assert rows[0]["mse_quadratic_change"] == 1
    assert not rows[1]["common_defined"]
    assert rows[1]["final_common_nrmse"] is None


@pytest.mark.parametrize("kind", ["nonfinite", "shape", "mask"])
def test_invalid_queries_rejected(kind):
    truth = np.arange(6.).reshape(3, 2)
    mask = np.ones_like(truth, dtype=bool)
    prediction = truth.copy()
    if kind == "nonfinite":
        prediction[0, 0] = np.nan
    elif kind == "shape":
        prediction = prediction[:1]
    else:
        mask = mask.astype(float)
    with pytest.raises(ValueError):
        column_changes(truth, mask, truth, prediction)


def test_signed_contributions_keep_improvements_and_nulls():
    rows = [{"index": 0, "initial_common_nrmse": 1., "final_common_nrmse": 1.4},
            {"index": 1, "initial_common_nrmse": 2., "final_common_nrmse": 1.8},
            {"index": 2, "initial_common_nrmse": None, "final_common_nrmse": None}]
    result = summarize_family(rows)
    assert result["cases"] == 3
    assert result["defined_cases"] == 2
    assert result["regressed_cases"] == 1
    assert sum(result["case_contributions"].values()) == pytest.approx(result["delta"])
    assert result["case_contributions"]["1"] < 0


def test_family_cannot_silently_change_coverage_or_duplicate_cases():
    rows = [{"index": 0, "initial_common_nrmse": 1., "final_common_nrmse": None}]
    with pytest.raises(ValueError):
        summarize_family(rows)
    rows[0]["final_common_nrmse"] = 1.1
    with pytest.raises(ValueError):
        summarize_family(rows * 2)
