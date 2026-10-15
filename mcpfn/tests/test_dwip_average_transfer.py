import json

import numpy as np
import pytest

import run_dwip_average_transfer as transfer


def gate_fixture(root, average=.9, seed1=1., seed2=.95):
    scores = {"average": average, "seed1": seed1, "seed2": seed2}
    accepted = average < min(seed1, seed2)
    selected = "average" if accepted else min(("seed1", "seed2"), key=scores.get)
    transfer.write_json(root / "status.json", {
        "state": "synthetic_evaluation_complete", "selected_variant": selected,
        "realworld_evaluation_allowed": accepted,
    })
    transfer.write_json(root / "selection.json", {
        "selected_variant": selected, "average_accepted": accepted, "macro_rmse": scores,
    })


@pytest.mark.parametrize("average,expected", [(.9, True), (.95, False), (.97, False), (1.1, False)])
def test_gate_requires_strict_improvement_over_both_raw_models(tmp_path, average, expected):
    gate_fixture(tmp_path, average)
    assert transfer.accepted_average(tmp_path) is expected


@pytest.mark.parametrize("average", [float("nan"), float("inf"), -.1])
def test_nonfinite_or_negative_synthetic_scores_fail_closed(tmp_path, average):
    gate_fixture(tmp_path, average)
    with pytest.raises(ValueError, match="finite nonnegative"):
        transfer.accepted_average(tmp_path)


def test_disagreeing_permission_fails_closed(tmp_path):
    gate_fixture(tmp_path)
    status = transfer.read_json(tmp_path / "status.json")
    status["realworld_evaluation_allowed"] = False
    transfer.write_json(tmp_path / "status.json", status)
    with pytest.raises(ValueError, match="disagree"):
        transfer.accepted_average(tmp_path)


def test_unfinished_gate_waits_and_failed_gate_raises(tmp_path):
    transfer.write_json(tmp_path / "status.json", {"state": "evaluating_synthetic"})
    assert transfer.accepted_average(tmp_path) is None
    transfer.write_json(tmp_path / "status.json", {"state": "error"})
    with pytest.raises(RuntimeError, match="terminated"):
        transfer.accepted_average(tmp_path)


def test_rejection_never_prepares_panel_or_queries_gpu(tmp_path, monkeypatch):
    gate_fixture(tmp_path, average=.97)
    monkeypatch.setattr(transfer, "sha256", lambda path: "fixture")

    def forbidden(*args):
        pytest.fail("A rejected average must never prepare or evaluate the benchmark panel")

    monkeypatch.setattr(transfer, "prepare_panel", forbidden)
    monkeypatch.setattr(transfer, "gpu_processes", forbidden)
    transfer.run(tmp_path, {"pid": 123, "created": 456.})
    status = transfer.read_json(tmp_path / "transfer_evaluation/status.json")
    assert status["state"] == "skipped_rejected_average"
    assert status["benchmark_evaluation_started"] is False


def test_registration_and_frozen_hashes_reject_changes(tmp_path):
    path = tmp_path / "registration.json"
    transfer.immutable_json(path, {"input": "first"})
    transfer.immutable_json(path, {"input": "first"})
    with pytest.raises(ValueError, match="Registered inputs changed"):
        transfer.immutable_json(path, {"input": "second"})
    hashes = {str(path): transfer.sha256(path)}
    transfer.verify_hashes(hashes)
    path.write_text(json.dumps({"input": "tampered"}))
    with pytest.raises(ValueError, match="Frozen input changed"):
        transfer.verify_hashes(hashes)


def test_recycled_dependency_pid_is_rejected(monkeypatch):
    class ReplacementProcess:
        def create_time(self):
            return 789.

    monkeypatch.setattr(transfer.psutil, "Process", lambda pid: ReplacementProcess())
    with pytest.raises(RuntimeError, match="no longer live"):
        transfer.require_live({"pid": 123, "created": 456.})


@pytest.fixture
def accepted_panel(tmp_path, monkeypatch):
    gate_fixture(tmp_path)
    (tmp_path / "candidate").mkdir()
    (tmp_path / "transfer_evaluation").mkdir()
    checkpoint = tmp_path / "candidate/checkpoint.pth"
    checkpoint.write_bytes(b"averaged weights")
    transfer.write_json(tmp_path / "candidate_sha256.json", {"sha256": transfer.sha256(checkpoint)})
    config = {"model": {"embedding_size": 768}}
    transfer.write_json(tmp_path / "candidate/config.json", config)
    transfer.write_json(tmp_path / "prespecification.json", {"weights": [.5, .5]})
    for domain in ("spectral", "general"):
        transfer.write_json(tmp_path / f"average_{domain}.json", {"macro_rmse": 1.1})
    scores = transfer.read_json(tmp_path / "selection.json")["macro_rmse"]
    for label, score in scores.items():
        transfer.write_json(tmp_path / f"{label}_validation.json", {"macro_rmse": score})
    work = tmp_path / "work"
    monkeypatch.setattr(transfer, "WORK", work)
    checkpoints = {}
    queues = {"seed1": "first", "seed2": "second"}
    for label, queue in queues.items():
        model_dir = work / f"tabimpute-{queue}-dwip1e5"
        model_dir.mkdir(parents=True)
        raw = model_dir / "best_validation_checkpoint.pth"
        raw.write_bytes(label.encode())
        checkpoints[label] = transfer.sha256(raw)
        transfer.write_json(model_dir / "config.json", config)
    transfer.write_json(tmp_path / "manifest.json", {
        "source_sha256": {}, "queues": queues, "checkpoint_sha256": checkpoints,
        "prespecification_sha256": transfer.sha256(tmp_path / "prespecification.json"),
    })
    reference = tmp_path / "reference.npz"
    np.savez(reference, benchmark_names=np.array(["first", "second"]), **{
        f"{method}_{metric}": np.full((10, 15), .5)
        for method in ("mi", "entropy", "random") for metric in ("r2", "rmse")
    })
    monkeypatch.setattr(transfer, "baseline_cache", lambda dataset: reference)
    monkeypatch.setattr(transfer, "raw_cache", lambda queue, dataset: reference)
    return tmp_path


def test_accepted_panel_preserves_all_four_datasets_and_both_raw_references(accepted_panel):
    checkpoint, references, hashes = transfer.prepare_panel(accepted_panel, {"input_sha256": {}})
    assert checkpoint == accepted_panel / "candidate/checkpoint.pth"
    assert tuple(references) == ("mteb", "mmlu", "merged", "benchpress")
    assert all(set(paths) == {"baseline", "seed1", "seed2"} for paths in references.values())
    transfer.verify_hashes(hashes)


@pytest.mark.parametrize("tampering", ["checkpoint", "fresh_score", "architecture"])
def test_accepted_panel_rejects_inconsistent_artifacts(accepted_panel, tampering):
    if tampering == "checkpoint":
        (accepted_panel / "candidate/checkpoint.pth").write_bytes(b"replacement weights")
    elif tampering == "fresh_score":
        transfer.write_json(accepted_panel / "average_validation.json", {"macro_rmse": .99})
    else:
        transfer.write_json(accepted_panel / "candidate/config.json", {"model": {"embedding_size": 256}})
    with pytest.raises(ValueError):
        transfer.prepare_panel(accepted_panel, {"input_sha256": {}})
