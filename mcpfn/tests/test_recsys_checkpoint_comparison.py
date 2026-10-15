import numpy as np
import pytest

from compare_recsys_checkpoint import audit_scores, comparison_record, reference_paths, summarize


def test_comparison_reports_signed_changes():
    result = comparison_record("Example", "global", {"RMSE": 1.0, "MAE": 0.8}, {"RMSE": 0.9, "MAE": 0.84})
    assert result["delta_RMSE"] == pytest.approx(-0.1)
    assert result["change_percent_RMSE"] == pytest.approx(-10.0)
    assert result["change_percent_MAE"] == pytest.approx(5.0)


def test_audit_checks_equal_user_weighting_and_archived_scores():
    split = {"test_users": np.array([0, 1, 1, 1]), "test_ratings": np.ones(4)}
    prediction = np.array([4.0, 1.0, 1.0, 1.0])
    assert audit_scores(split, prediction)["RMSE"] == 1.5
    with pytest.raises(AssertionError):
        audit_scores(split, prediction, {"RMSE": 2.0, "MAE": 1.5})
    with pytest.raises(ValueError, match="Invalid"):
        audit_scores(split, np.array([np.nan] * 4))


def test_reference_geometry_and_summary_keep_both_normalizations(tmp_path):
    assert reference_paths("100K", "global")["rows"] == 9976
    assert reference_paths("10M", "column")["rows"] == 10000
    comparisons = [comparison_record("Example", normalization, {"RMSE": 1.0, "MAE": 0.8},
                                     {"RMSE": 0.9, "MAE": 0.7}) for normalization in ("global", "column")]
    summarize(tmp_path, comparisons, 8)
    text = (tmp_path / "RESULTS.md").read_text()
    assert "Completed 2 of 8" in text
    assert "Global normalization" in text and "Column normalization" in text
    assert "-10.00%" in text
