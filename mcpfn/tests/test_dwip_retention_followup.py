from copy import deepcopy

import pytest

import evaluate_dwip_retention_followup as evaluation


def records():
    return [{"id": "first", "group": "group", "n": 2, "sse": 2., "raw_sse": 2.},
            {"id": "second", "group": "group", "n": 4, "sse": 4., "raw_sse": 4.}]


def test_retention_followup_reports_matched_frozen_cases():
    baseline, selected = records(), records()
    for record in selected:
        record["sse"] *= .25
    result = evaluation.compare_records(baseline, selected)["group"]
    assert result["macro_case_rmse_ratio"] == .5
    assert result["cases_improved"] == result["case_count"] == 2


@pytest.mark.parametrize("change", ["id", "group", "count", "duplicate", "nonfinite"])
def test_retention_comparison_rejects_changed_populations(change):
    baseline, selected = records(), records()
    if change in {"id", "group"}:
        selected[0][change] = "changed"
    elif change == "count":
        selected[0]["n"] += 1
    elif change == "duplicate":
        selected.append(deepcopy(selected[0]))
    else:
        selected[0]["sse"] = float("nan")
    with pytest.raises(ValueError):
        evaluation.compare_records(baseline, selected)


def test_retention_wait_requires_completed_transfer_results(tmp_path):
    transfer = tmp_path / "transfer_evaluation"
    transfer.mkdir()
    evaluation.write_json(transfer / "status.json", {"state": "complete"})
    with pytest.raises(RuntimeError, match="completion lacks"):
        evaluation.dependency_ready(tmp_path, {"pid": 123, "created": 456.})
    evaluation.write_json(transfer / "results.json", {})
    evaluation.write_json(tmp_path / "status.json", {"state": "complete"})
    assert evaluation.dependency_ready(tmp_path, {"pid": 123, "created": 456.})


def test_retention_wait_does_not_accept_recycled_pid(tmp_path, monkeypatch):
    transfer = tmp_path / "transfer_evaluation"
    transfer.mkdir()
    evaluation.write_json(transfer / "status.json", {"state": "waiting_for_training_queue"})

    class ReplacementProcess:
        def create_time(self):
            return 789.

    monkeypatch.setattr(evaluation.psutil, "Process", lambda pid: ReplacementProcess())
    with pytest.raises(RuntimeError, match="no longer live"):
        evaluation.dependency_ready(tmp_path, {"pid": 123, "created": 456.})


def test_retention_followup_registration_and_input_hashes_are_immutable(tmp_path):
    path = tmp_path / "manifest.json"
    evaluation.immutable_json(path, {"checkpoint": "first"})
    hashes = {str(path): evaluation.sha256(path)}
    evaluation.verify_hashes(hashes)
    with pytest.raises(ValueError, match="Registered evaluation changed"):
        evaluation.immutable_json(path, {"checkpoint": "second"})
    path.write_text("tampered")
    with pytest.raises(ValueError, match="Frozen evaluation input changed"):
        evaluation.verify_hashes(hashes)


def test_retention_uses_the_same_selected_weights_as_completed_transfer(tmp_path):
    (tmp_path / "transfer_evaluation").mkdir()
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"selected weights")
    evaluation.write_json(tmp_path / "transfer_evaluation/checkpoint_sha256.json",
                          {"selected": evaluation.sha256(checkpoint)})
    evaluation.verify_selected_checkpoint(tmp_path, checkpoint)
    checkpoint.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="differ from the completed"):
        evaluation.verify_selected_checkpoint(tmp_path, checkpoint)
