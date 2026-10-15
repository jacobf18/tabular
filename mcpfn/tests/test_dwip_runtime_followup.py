from copy import deepcopy
from types import SimpleNamespace

import pytest

import evaluate_dwip_runtime_followup as evaluation


def measurements():
    return [{"variant": variant, "size": size, "status": "ok", "seconds": [1., 2., 3.],
        "warmup": 1, "repeats": 3, "precision": "amp_bfloat16", "input_sha256": f"input_{size}",
        "rank": 10, "missing_cells": int(size * size * .8), "peak_allocated_gib": 2., "peak_reserved_gib": 3.}
        for size in evaluation.SIZES for variant in evaluation.VARIANTS]


def test_runtime_summary_uses_matched_raw_medians():
    rows = measurements()
    for row in rows:
        if row["variant"] == "selected":
            row["seconds"] = [2., 4., 6.]
    result = evaluation.summarize(rows)
    assert set(result) == {"100", "500", "1000"}
    assert result["500"]["latency_ratio"] == 2.
    assert result["500"]["original_seconds"] == 2.
    assert result["500"]["selected_seconds"] == 4.


@pytest.mark.parametrize("field,value", [("input_sha256", "changed"), ("rank", 9),
    ("missing_cells", 0), ("status", "out_of_memory"), ("warmup", 0), ("repeats", 2),
    ("seconds", [1., 2.]), ("seconds", [float("nan"), 2., 3.]), ("peak_allocated_gib", float("inf")),
    ("peak_reserved_gib", 0.), ("precision", "full_bfloat16")])
def test_runtime_rejects_invalid_or_unmatched_measurements(field, value):
    rows = measurements()
    rows[0][field] = value
    with pytest.raises(ValueError):
        evaluation.summarize(rows)


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_runtime_requires_exact_result_population(change):
    rows = measurements()
    if change == "missing":
        rows.pop()
    else:
        rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError, match="one matched result"):
        evaluation.summarize(rows)


def test_runtime_worker_uses_variant_manifest_and_frozen_protocol(tmp_path):
    command = evaluation.worker_command(tmp_path, "selected", 1000)
    assert command[command.index("--dwip-manifest") + 1] == str(tmp_path / "selected_models.json")
    assert command[command.index("--worker-size") + 1] == "1000"
    assert command[command.index("--precision") + 1] == "amp_bfloat16"
    assert "--allow-shared-gpu" not in command


def test_runtime_dependency_checks_every_completed_stage(tmp_path):
    retention = tmp_path / "retention_evaluation"
    retention.mkdir()
    evaluation.write_json(retention / "status.json", {"state": "complete"})
    evaluation.write_json(tmp_path / "status.json", {"state": "training"})
    with pytest.raises(RuntimeError, match="completion lacks"):
        evaluation.dependency_ready(tmp_path, {"pid": 123, "created": 456})
    transfer = tmp_path / "transfer_evaluation"
    transfer.mkdir()
    for folder in (tmp_path, transfer, retention):
        evaluation.write_json(folder / "status.json", {"state": "complete"})
        evaluation.write_json(folder / "results.json", {})
    assert evaluation.dependency_ready(tmp_path, {"pid": 123, "created": 456})


def test_runtime_dependency_rejects_recycled_pid(tmp_path, monkeypatch):
    retention = tmp_path / "retention_evaluation"
    retention.mkdir()
    evaluation.write_json(retention / "status.json", {"state": "waiting_for_transfer"})
    monkeypatch.setattr(evaluation.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 789.))
    with pytest.raises(RuntimeError, match="no longer live"):
        evaluation.dependency_ready(tmp_path, {"pid": 123, "created": 456.})
