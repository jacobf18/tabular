import json

import pytest

from diagnose_dwip_matched_objective_gradients import matched_commands, require_completed_stages, verify_hashes
from run_dwip_transfer_evaluation import sha256


def test_matched_commands_only_normalizes_output_paths():
    control = ["trainer", "--run-name", "control", "--fixed-validation-cache", "control.pth",
               "--point-loss-weight", "0.2", "--decay-steps", "63400"]
    candidate = ["trainer", "--run-name", "candidate", "--fixed-validation-cache", "candidate.pth",
                 "--point-loss-weight", "2.0", "--decay-steps", "63400"]
    matched_commands(control, candidate)
    assert control[2] == "control"
    assert candidate[2] == "candidate"
    candidate[-1] = "47400"
    with pytest.raises(ValueError):
        matched_commands(control, candidate)


def test_requires_all_four_complete_stages_and_results(tmp_path):
    folders = [tmp_path, *(tmp_path / f"{stage}_evaluation" for stage in ("transfer", "retention", "runtime"))]
    for folder in folders:
        folder.mkdir(exist_ok=True)
        (folder / "status.json").write_text(json.dumps({"state": "complete"}))
        (folder / "results.json").write_text("{}")
    require_completed_stages(tmp_path)
    (folders[-1] / "status.json").write_text(json.dumps({"state": "evaluating"}))
    with pytest.raises(ValueError):
        require_completed_stages(tmp_path)
    (folders[-1] / "status.json").write_text(json.dumps({"state": "complete"}))
    (folders[-1] / "results.json").unlink()
    with pytest.raises(ValueError):
        require_completed_stages(tmp_path)


def test_hash_verification_rejects_changed_inputs(tmp_path):
    path = tmp_path / "input"
    path.write_text("original")
    hashes = {str(path): sha256(path)}
    verify_hashes(hashes)
    path.write_text("changed")
    with pytest.raises(ValueError):
        verify_hashes(hashes)
