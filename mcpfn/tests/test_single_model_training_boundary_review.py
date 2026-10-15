from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import review_single_model_training_boundaries as review


@pytest.fixture
def saved_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(review.shutil, "disk_usage", lambda path: SimpleNamespace(free=60 * 1024**3))
    run = tmp_path / "run"
    run.mkdir()
    original = Path("/mnt/checkpoints_200/single_model_goal_20261005/all_imputation_column_mean_v52/validation_0.json")
    initial = review.read(original)
    candidate = deepcopy(initial)
    candidate.update(step=10, accepted_updates=8, best_step=10, selected_checkpoint_sha256="0" * 64)
    (run / "config.json").write_text(json.dumps({"boundaries": [0, 10, 25, 50, 64]}))
    digest = review.digest(run / "config.json")
    (run / "registration.json").write_text(json.dumps({"config_sha256": digest}))
    (run / "validation_0.json").write_text(json.dumps(initial))
    (run / "validation_10.json").write_text(json.dumps(candidate))
    records = [{"step": step, "accepted_updates": max(0, step - 2),
        "reason": "finite_metrics_rejected" if step <= 2 else "accepted"} for step in range(1, 11)]
    (run / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    return run, tmp_path / "review", digest


def test_complete_review_preserves_every_metric_guard_and_rejection(saved_boundary):
    run, output, digest = saved_boundary
    review.review(run, 10, output, digest)
    receipt = review.read(output / "recheck.json")
    assert receipt["case_metric_rows"] == 712
    assert receipt["summary_metric_rows"] == 72
    assert receipt["included_boundaries"] == [0, 10]
    assert receipt["case_aggregation_checks"] == 144
    assert receipt["guard_checks"] == 44
    assert receipt["original_guards"] == 22
    assert receipt["accepted_updates"] == 8
    assert [row["step"] for row in receipt["nonaccepted_attempts"]] == [1, 2]
    assert receipt["failed_guards"] == []
    assert all(receipt[key] is False for key in ("prediction_grids_recomputed", "model_inference_performed",
        "training_modified", "gate_changed", "real_data_used"))
    assert str(Path(review.__file__).resolve()) in receipt["input_sha256"]
    assert str(Path(__file__).resolve()) in receipt["input_sha256"]


@pytest.mark.parametrize("mutation", ["case", "guard", "coverage", "score", "aggregate", "seed", "updates", "initial_selection"])
def test_review_rejects_incomplete_or_inconsistent_saved_metrics(saved_boundary, mutation):
    run, output, digest = saved_boundary
    path = run / ("validation_0.json" if mutation == "initial_selection" else "validation_10.json")
    report = review.read(path)
    if mutation == "case":
        report["cases"].pop()
    elif mutation == "guard":
        report["decision"]["comparisons"].pop()
    elif mutation == "coverage":
        report["cases"][0]["student"]["column_coverage"]["queried_columns"] += 1
    elif mutation == "score":
        report["decision"]["score"] += .1
    elif mutation == "aggregate":
        report["student"]["imputation"]["rmse"] += .1
    elif mutation == "seed":
        report["cases"][0]["seed"] += 1
    elif mutation == "updates":
        report["accepted_updates"] = 9
    else:
        report["selected_checkpoint_sha256"] = "0" * 64
    path.write_text(json.dumps(report))
    with pytest.raises(AssertionError):
        review.review(run, 10, output, digest)
    assert not output.exists()


def test_training_prefix_ignores_a_live_later_partial_record(saved_boundary):
    run, output, digest = saved_boundary
    before = review.prefix(run, 10)
    with (run / "training.jsonl").open("a") as stream:
        stream.write('{"step":11')
    assert review.prefix(run, 10) == before
    review.review(run, 10, output, digest)


def test_review_refuses_wrong_identity_overwrite_gpu_visibility_and_low_capacity(saved_boundary, monkeypatch):
    run, output, digest = saved_boundary
    with pytest.raises(AssertionError):
        review.review(run, 10, output, "changed")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError):
        review.review(run, 10, output, digest)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(review.shutil, "disk_usage", lambda path: SimpleNamespace(free=50 * 1024**3))
    with pytest.raises(OSError):
        review.review(run, 10, output, digest)
    output.mkdir()
    with pytest.raises(ValueError):
        review.review(run, 10, output, digest)
