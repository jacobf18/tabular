from copy import deepcopy
import csv
import json

import pytest

import report_single_model_structured_run as reporter


def example():
    legacy = [["imputation", name] for name in ("missbench_nrmse", "uci_nrmse", "rmse", "common_nrmse")]
    legacy += [["recsys", name] for name in ("rmse", "mae")]
    config = {"steps": 400, "validate_every": 50, "retention_metrics": legacy}
    metrics = {"imputation": {name: 1. for domain, name in legacy if domain == "imputation"},
               "recsys": {"rmse": 1., "mae": 1.},
               "structured": {"overall": {name: 1. for name in reporter.STRUCTURED_METRICS},
                              "families": {family: {"common_nrmse": 1.} for family in reporter.FAMILIES}}}
    reports = [{"step": step, "score": 1., "eligible": True, "best_step": 0, "student": deepcopy(metrics)} for step in range(0, 401, 50)]
    audit = {"complete": True, "full_400_update_log_and_validation_audited": True, "through_step": 400,
             "logged_training_updates_checked": 400, "prediction_grids_rescored": 1672,
             "research_selection_tolerance": 0, "original_artifacts_unchanged": True,
             "selection_trace": [{key: report[key] for key in ("step", "score", "eligible", "best_step")} for report in reports]}
    status = {"state": "training_complete", "step": 400, "best_step": 0}
    return reports, config, audit, status


def test_all_boundaries_and_related_definitions_are_preserved():
    summary, rows = reporter.summarize(*example())
    assert len(summary["boundaries"]) == 9
    assert len(rows) == 198
    assert len([row for row in rows if row["step"] == 400]) == 22
    assert summary["goal_success_not_assessed"]
    assert not summary["selected_trained_checkpoint"]
    text = reporter.markdown(summary, rows)
    assert "No trained checkpoint was selected" in text
    assert "not MissBench/UCI/recommender benchmark measurements" in text


def test_tiny_regression_is_not_hidden_by_rounding():
    reports, config, audit, status = example()
    reports[-1]["student"]["recsys"]["mae"] += 1e-12
    reports[-1]["eligible"] = False
    audit["selection_trace"][-1]["eligible"] = False
    summary, rows = reporter.summarize(reports, config, audit, status)
    assert summary["boundaries"][-1]["regressions"] == ["recsys/mae"]
    assert any(row["step"] == 400 and 0 < row["delta"] < 1e-11 and not row["retained"] for row in rows)


def test_improved_minimax_alone_does_not_select_a_regressed_candidate():
    reports, config, audit, status = example()
    reports[-1].update(score=.8, eligible=False)
    reports[-1]["student"]["structured"]["families"][reporter.FAMILIES[0]]["common_nrmse"] = 1.1
    audit["selection_trace"][-1].update(score=.8, eligible=False)
    summary, rows = reporter.summarize(reports, config, audit, status)
    assert summary["selected_step"] == 0
    assert not rows[-12]["retained"]


def test_selected_nonzero_checkpoint_is_reported_without_claiming_success():
    reports, config, audit, status = example()
    reports[-1].update(score=.9, best_step=400)
    audit["selection_trace"][-1].update(score=.9, best_step=400)
    status["best_step"] = 400
    summary, rows = reporter.summarize(reports, config, audit, status)
    assert summary["selected_trained_checkpoint"]
    assert summary["goal_success_not_assessed"]
    assert "No trained checkpoint was selected" not in reporter.markdown(summary, rows)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1.])
def test_undefined_nonfinite_or_negative_metrics_cannot_disappear(value):
    reports, config, audit, status = example()
    reports[-1]["student"]["recsys"]["mae"] = value
    with pytest.raises(ValueError, match="undefined or nonfinite"):
        reporter.summarize(reports, config, audit, status)


def test_zero_baseline_keeps_absolute_comparison_and_null_relative_change():
    reports, config, audit, status = example()
    for record in reports:
        record["student"]["recsys"]["mae"] = 0.
    summary, rows = reporter.summarize(reports, config, audit, status)
    zero_rows = [row for row in rows if row["group"] == "recsys" and row["metric"] == "mae"]
    assert len(zero_rows) == 9
    assert all(row["retained"] and row["relative_percent"] is None for row in zero_rows)
    assert "undefined (zero baseline)" in reporter.markdown(summary, rows)


@pytest.mark.parametrize("mutation", ["prefix", "missing_boundary", "missing_family", "changed_trace", "wrong_status"])
def test_incomplete_or_mismatched_evidence_is_rejected(mutation):
    reports, config, audit, status = example()
    if mutation == "prefix":
        audit["through_step"] = 50
    elif mutation == "missing_boundary":
        reports.pop(2)
    elif mutation == "missing_family":
        del reports[-1]["student"]["structured"]["families"][reporter.FAMILIES[0]]
    elif mutation == "changed_trace":
        audit["selection_trace"][-1]["best_step"] = 400
    elif mutation == "wrong_status":
        status["best_step"] = 400
    with pytest.raises(ValueError):
        reporter.summarize(reports, config, audit, status)


def saved_example(tmp_path):
    reports, config, audit, status = example()
    run, audited = tmp_path / "run", tmp_path / "audit"
    run.mkdir()
    audited.mkdir()
    reporter.write_json(run / "config.json", config)
    reporter.write_json(run / "status.json", status)
    for record in reports:
        reporter.write_json(run / f'validation_{record["step"]}.json', record)
    audit["original_artifact_sha256"] = {str(path): reporter.sha256(path) for path in run.glob("*.json") if path.name != "status.json"}
    reporter.write_json(audited / "audit.json", audit)
    return run, audited


def test_disk_report_preserves_originals_and_refuses_overwrite(tmp_path):
    run, audited = saved_example(tmp_path)
    paths = list(run.glob("*.json")) + list(audited.glob("*.json"))
    original = {str(path): reporter.sha256(path) for path in paths}
    output = tmp_path / "report"
    reporter.report(run, audited, output)
    assert {str(path): reporter.sha256(path) for path in paths} == original
    with (output / "retention.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 198
    assert json.loads((output / "summary.json").read_text())["selected_step"] == 0
    with pytest.raises(FileExistsError):
        reporter.report(run, audited, output)


def test_changed_original_results_cannot_be_reported_as_audited(tmp_path):
    run, audited = saved_example(tmp_path)
    path = run / "validation_400.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises((RuntimeError, ValueError)):
        reporter.report(run, audited, tmp_path / "report")
    assert not (tmp_path / "report").exists()
