import numpy as np
import pytest

from summarize_dwip_fold_regression import paired_error_summary


def test_paired_summary_retains_all_rows_and_explains_fraction_above_one():
    shared = {"target": np.zeros((2, 2)), "evaluation_mask": np.ones((2, 2), dtype=bool),
              "mean": np.zeros(2), "scale": np.ones(2)}
    before = {**shared, "prediction": np.array([[1., 1.], [1., 1.]])}
    after = {**shared, "prediction": np.array([[2., 2.], [0., 0.]])}
    result = paired_error_summary(before, after, [30, 40])
    assert result["scored_sse_delta"] == 4
    assert len(result["rows"]) == 2
    assert result["rows"][0]["dataset_row"] == 30
    assert result["rows"][0]["fraction_of_net_sse_increase"] == 1.5
    assert result["rows"][1]["scored_sse_delta"] == -2


def test_paired_summary_rejects_different_targets_and_row_identities():
    before = {"target": np.zeros((2, 2)), "evaluation_mask": np.ones((2, 2), dtype=bool),
              "mean": np.zeros(2), "scale": np.ones(2), "prediction": np.zeros((2, 2))}
    with pytest.raises(ValueError, match="inputs differ"):
        paired_error_summary(before, {**before, "target": np.ones((2, 2))}, [0, 1])
    with pytest.raises(ValueError, match="row identities"):
        paired_error_summary(before, before, [0])
