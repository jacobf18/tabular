from copy import deepcopy
import json

import pytest

import review_single_matrixformer_campaign as review
from recsys_evaluation_common import write_csv, write_json


def summaries(root):
    for stage in ("tabular", "recsys", "llm"):
        (root / stage).mkdir()
    case = {"suite": "missbench", "pattern": "MCAR", "dataset": "fixture", "missingness": .4,
            "complete": True, "completed_cases": 1, "expected_cases": 1, "new_nrmse": .9, "previous_nrmse": 1.}
    for name, count, field in (("overall", 2, "suite"), ("per_pattern", 14, "pattern"),
                                ("per_dataset", 54, "dataset"), ("per_pattern_rate", 26, "missingness")):
        write_csv(root / "tabular" / f"{name}.csv", [{**case, field: str(index)} for index in range(count)])
    write_csv(root / "recsys/comparison.csv", [{"dataset": dataset, "role": role, "RMSE": .9, "MAE": .8,
        "blend_RMSE": 1., "blend_MAE": .9} for dataset in ("MovieLens 100K", "Netflix small", "MovieLens 1M", "MovieLens 10M")
        for role in ("student", "parent_same_recipe")])
    metrics = {"student_r2": .8, "blend_r2": .7, "student_rmse": .9, "blend_rmse": 1., "complete": True}
    write_csv(root / "llm/per_dataset.csv", [{**metrics, "dataset": str(index), "panels": 45, "expected_panels": 45} for index in range(4)])
    write_csv(root / "llm/panels.csv", [{**metrics, "dataset": str(index // 45), "method": "fixture", "budget": index,
        "folds": 10, "student_r2_folds": 10, "blend_r2_folds": 10, "student_rmse_folds": 10, "blend_rmse_folds": 10}
        for index in range(180)])


@pytest.mark.parametrize("student,blend,higher,retained", [(.9, 1., False, True), (1., 1., False, True),
    (1.000000000001, 1., False, False), (.8, .7, True, True), (-.1, -.2, True, True), (-.3, -.2, True, False)])
def test_strict_metric_direction_without_rounding_margin(student, blend, higher, retained):
    result = review.compare("fixture", "fixture", "R2" if higher else "RMSE", student, blend, higher_better=higher)
    assert result["retained"] is retained
    if higher:
        assert result["relative_change_percent"] is None


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_comparisons_cannot_disappear(value):
    with pytest.raises(ValueError):
        review.compare("fixture", "fixture", "RMSE", value, 1.)


def test_review_includes_full_scope_without_claiming_goal_success(tmp_path):
    summaries(tmp_path)
    records = review.comparisons(tmp_path)
    result = review.decision(records)
    assert len(records) == 480 and result["required_comparisons"] == 392 and result["primary_comparisons"] == 18
    assert result["strict_point_estimate_parity"] and result["required_regressions"] == 0
    assert result["goal_success_not_assessed"] and not result["uncertainty_estimated_here"]
    assert not any(row["required_for_point_parity"] for row in records if row["scope"] == "recsys_parent_same_recipe")
    assert "does not establish goal completion" in review.markdown(records, result)


@pytest.mark.parametrize("scope", ["tabular_overall", "tabular_per_pattern", "recsys_student", "llm_per_dataset", "llm_panels"])
def test_each_required_scope_can_veto_aggregate_success(tmp_path, scope):
    summaries(tmp_path)
    records = review.comparisons(tmp_path)
    selected = next(row for row in records if row["scope"] == scope)
    selected["retained"] = False
    result = review.decision(records)
    assert not result["strict_point_estimate_parity"] and result["required_regressions"] == 1
    assert result["all_primary_point_estimates_retained"] is (scope in {"tabular_per_pattern", "llm_panels"})


@pytest.mark.parametrize("change", ["missing", "duplicate", "incomplete", "dropped_case", "missing_fold", "wrong_role"])
def test_missing_coverage_is_not_a_success(tmp_path, change):
    summaries(tmp_path)
    relative = "llm/panels.csv" if change == "missing_fold" else "recsys/comparison.csv" if change == "wrong_role" else "tabular/per_pattern.csv"
    path = tmp_path / relative
    records = review.rows(path)
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records[-1] = deepcopy(records[0])
    elif change == "incomplete":
        records[0]["complete"] = False
    elif change == "dropped_case":
        records[0]["completed_cases"] = 0
    elif change == "missing_fold":
        records[0]["student_r2_folds"] = 9
    else:
        records[0]["role"] = "another model"
    write_csv(path, records)
    with pytest.raises(ValueError):
        review.comparisons(tmp_path)


@pytest.mark.parametrize("stage", ["tabular", "recsys", "llm"])
def test_reaudit_writes_are_separate_from_original_artifacts(tmp_path, stage):
    source, target = tmp_path / "source", tmp_path / "reaudit"
    source.mkdir()
    write_json(source / "manifest.json", {"fixture": True})
    names = ["100K", "netflix", "1M", "10M"] if stage == "recsys" else ["predictions", "cases"] if stage == "tabular" else ["predictions"]
    if stage != "recsys":
        write_json(source / "parent_parity.json", {"passed": True})
    for name in names:
        (source / name).mkdir()
    (source / "comparison.csv").write_text("original\n")
    review.readonly_inputs(source, target, stage)
    (target / "comparison.csv").write_text("regenerated\n")
    write_json(target / "audit.json", {"complete": True})
    assert (source / "comparison.csv").read_text() == "original\n" and not (source / "audit.json").exists()
    assert (target / "manifest.json").read_bytes() == (source / "manifest.json").read_bytes()
    assert not (target / "manifest.json").is_symlink()
    assert all((target / name).resolve() == (source / name).resolve() for name in names)


def test_report_rejects_an_active_campaign_before_touching_it(tmp_path):
    write_json(tmp_path / "status.json", {"state": "evaluating"})
    write_json(tmp_path / "campaign_audit.json", {"all_evaluations_complete": False})
    write_json(tmp_path / "config.json", {})
    before = (tmp_path / "status.json").read_bytes()
    with pytest.raises(ValueError, match="complete original"):
        review.validate_campaign(tmp_path)
    assert (tmp_path / "status.json").read_bytes() == before


def test_report_cannot_be_written_inside_its_original_campaign(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        review.review(tmp_path, tmp_path / "review")
