import pytest

from summarize_single_model_synthetic import summarize


def case(domain, student, teacher):
    return {"domain": domain, "prior": "low_rank", "preference_family": "clustered", "gradient_data": "synthetic",
            "parent": {"rmse": 1., "mae": 1.}, "student": {"rmse": student, "mae": student},
            "teacher": {"rmse": teacher, "mae": teacher}, "routing": {"specialist_weight": .25}}


def test_summary_uses_equal_case_weights_and_keeps_losing_teacher_cases():
    rows = summarize([case("imputation", .8, 1.1), case("imputation", 1.1, 1.2), case("recsys", .9, .8)])
    row = next(row for row in rows if row["domain"] == "imputation" and row["family"] == "ALL")
    assert row["cases"] == 2 and row["student_rmse_wins_vs_parent"] == 1
    assert row["teacher_rmse_wins_vs_parent"] == 0
    assert row["student_rmse"] == pytest.approx(.95)
    assert row["teacher_rmse_change_percent"] == pytest.approx(15.)
    assert row["mean_specialist_weight"] == .25


def test_summary_rejects_real_cases():
    altered = case("imputation", .8, .9)
    altered["gradient_data"] = "real"
    with pytest.raises(ValueError, match="synthetic"):
        summarize([altered])
