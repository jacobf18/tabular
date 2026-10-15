import pytest

from compare_imputation_checkpoint import aggregate, write_report


def example():
    reference = [{"case_id": identifier, "suite": "uci", "pattern": "MCAR", "dataset": dataset,
                  "nrmse_original": score} for identifier, dataset, score in
                 (("first-0", "first", 1.0), ("first-1", "first", 3.0), ("second-0", "second", 10.0))]
    current = {identifier: {"new_nrmse": score} for identifier, score in
               (("first-0", 0.5), ("first-1", 1.5), ("second-0", 8.0))}
    return reference, current


def test_dataset_weighting_and_signed_changes():
    reference, current = example()
    result, = aggregate(reference, current, ("suite", "pattern"))
    assert result["previous_nrmse"] == 6.0
    assert result["new_nrmse"] == 4.5
    assert result["change_percent"] == -25.0
    assert result["completed_cases"] == result["expected_cases"] == 3
    assert result["complete"]


def test_incomplete_groups_do_not_get_a_new_headline_score(tmp_path):
    reference, current = example()
    del current["second-0"]
    result, = aggregate(reference, current, ("suite", "pattern"))
    assert result["previous_nrmse"] == 6.0
    assert result["new_nrmse"] is None and result["change_percent"] is None
    assert not result["complete"] and result["completed_cases"] == 2
    write_report(tmp_path, [result], 3, 2, 0)
    text = (tmp_path / "NRMSE_COMPARISON.md").read_text()
    assert "pending" in text and "2/3" in text
    assert "No row train/test split" in text


def test_zero_reference_score_has_no_percentage_change():
    reference, current = example()
    for row in reference:
        row["nrmse_original"] = 0.0
    result, = aggregate(reference, current, ("suite",))
    assert result["new_nrmse"] == pytest.approx(4.5)
    assert result["change_percent"] is None
