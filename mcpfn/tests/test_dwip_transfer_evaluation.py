import json

import numpy as np
import pytest

import run_dwip_transfer_evaluation as transfer


def fold_fixture():
    return {"benchmark_names": np.array(["first", "second"]),
            **{f"{method}_{metric}": np.full((10, 15), .5) for method in transfer.METHODS for metric in ("r2", "rmse")}}


def test_transfer_environment_is_matched_and_excludes_inherited_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHSELECT_ROW_LIMIT", "1")
    monkeypatch.setenv("BENCHSELECT_IMPUTER_MAX_COLS", "2")
    environment = transfer.evaluation_environment(tmp_path, tmp_path / "checkpoint.pth", "baseline", "merged")
    assert "BENCHSELECT_ROW_LIMIT" not in environment
    assert "BENCHSELECT_IMPUTER_MAX_COLS" not in environment
    assert environment["BENCHSELECT_TABIMPUTE_PRECISION_MODE"] == "amp_bfloat16"
    assert environment["BENCHSELECT_TABIMPUTE_MODEL_FAMILY"] == "new_stable"
    assert environment["BENCHSELECT_METHODS"] == "entropy,mi,random"
    assert environment["BENCHSELECT_MAX_ROWS"] == "4000"
    assert "mteb" not in transfer.DATASETS


def test_paired_fold_comparison_uses_correct_improvement_directions():
    baseline, selected = fold_fixture(), fold_fixture()
    selected["mi_r2"] += .1
    selected["mi_rmse"] -= .2
    result = transfer.compare_fold_metrics(baseline, selected)
    assert result["mi_r2_5"]["delta"] == pytest.approx(.1)
    assert result["mi_rmse_15"]["delta"] == pytest.approx(-.2)
    assert result["mi_r2_5"]["folds_improved"] == result["mi_rmse_15"]["folds_improved"] == 10
    assert result["entropy_r2_10"]["folds_improved"] == 0


@pytest.mark.parametrize("change", ["names", "shape", "nonfinite"])
def test_fold_comparison_rejects_mismatched_or_incomplete_results(change):
    baseline, selected = fold_fixture(), fold_fixture()
    if change == "names":
        selected["benchmark_names"] = np.array(["wrong", "second"])
    elif change == "shape":
        selected["mi_r2"] = np.ones((9, 15))
    else:
        selected["mi_r2"][2, 0] = np.nan
    with pytest.raises(ValueError):
        transfer.compare_fold_metrics(baseline, selected)


def test_completed_dependency_requires_actual_result(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"state": "complete"}))
    with pytest.raises(RuntimeError, match="without its result"):
        transfer.dependency_state(tmp_path, 123, 456.)
    (tmp_path / "dwip1e5_result.json").write_text("{}")
    assert transfer.dependency_state(tmp_path, 123, 456.)


def test_cancelled_dependency_does_not_restart(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"state": "cancelled_by_user"}))
    with pytest.raises(RuntimeError, match="terminated"):
        transfer.dependency_state(tmp_path, 123, 456.)


def test_reused_launcher_pid_is_not_treated_as_live_dependency(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(json.dumps({"state": "training"}))
    class ReplacementProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 789.

    monkeypatch.setattr(transfer.psutil, "Process", ReplacementProcess)
    with pytest.raises(RuntimeError, match="no longer live"):
        transfer.dependency_state(tmp_path, 123, 456.)
