import fcntl
import numpy as np
import pytest

from recsys_evaluation_common import sha256, write_json
import reuse_single_matrixformer_parent_controls as reuse


def fixture(root, monkeypatch):
    source, protocol_path, candidate, confirmation = [root / name for name in ("source", "protocol", "candidate", "confirmation")]
    candidate.mkdir()
    checkpoint = candidate / "best_validation_checkpoint.pth"
    checkpoint.write_bytes(b"frozen candidate")
    write_json(candidate / "config.json", {"model": "fixture"})
    parent = root / "parent.pth"
    parent.write_bytes(b"frozen general parent")
    specialist = root / "specialist.pth"
    specialist.write_bytes(b"frozen specialist")
    monkeypatch.setattr(reuse.evaluation, "GENERAL", parent)
    monkeypatch.setattr(reuse.evaluation, "SPECIALIST", specialist)
    split = {"test_users": np.array([0, 0, 1]), "test_items": np.array([0, 1, 1]),
             "test_ratings": np.array([1., 2., 4.])}
    split_path = root / "split.npz"
    np.savez(split_path, **split)
    prediction = np.array([1.5, 2.5, 3.5])
    metrics = reuse.evaluation.audit_scores(split, prediction)
    cases = [{"dataset": dataset, "shape": [2, 2], "split": str(split_path), "rating_range": [.5, 5.]}
             for dataset in ("100K", "netflix", "1M", "10M")]
    protocol = {"cases": cases}
    write_json(protocol_path / "protocol.json", protocol)
    monkeypatch.setattr(reuse.evaluation, "load_protocol", lambda path: protocol)
    manifest = {"candidate_sha256": {str(checkpoint): sha256(checkpoint)}, "protocol": str(protocol_path),
                "protocol_sha256": sha256(protocol_path / "protocol.json")}
    write_json(source / "manifest.json", manifest)
    audit = {"complete": True, "manifest_sha256": sha256(source / "manifest.json"),
             "cases": [{"dataset": case["dataset"], "role": role, "queries": 3, **metrics}
                       for case in cases for role in ("student", "parent_same_recipe")]}
    write_json(source / "audit.json", audit)
    write_json(source / "status.json", {"state": "complete"})
    for case in cases:
        directory = source / case["dataset"]
        directory.mkdir()
        path = directory / "parent_same_recipe.npz"
        np.savez(path, prediction=prediction, counts=np.ones(3, dtype=np.uint8))
        write_json(path.with_suffix(".json"), {"dataset": case["dataset"], "role": "parent_same_recipe", "metrics": metrics,
                                               "manifest_sha256": sha256(source / "manifest.json"), "checkpoint_sha256": sha256(parent),
                                               "prediction_sha256": sha256(path), "seconds": 12.})
        write_json(directory / "specialist_parity.json", {"passed": True, "checkpoint_sha256": sha256(specialist)})
    write_json(confirmation / "status.json", {"state": "complete"})
    write_json(confirmation / "gate.json", {"passed": True, "selected_step": 350, "checkpoint_sha256": sha256(checkpoint)})
    return source, protocol_path, checkpoint, confirmation


def test_copy_preserves_bytes_and_requires_fresh_student_and_parity(tmp_path, monkeypatch):
    source, protocol, checkpoint, confirmation = fixture(tmp_path, monkeypatch)
    original_hashes = {path: sha256(path) for path in source.rglob("*") if path.is_file()}
    output = tmp_path / "future_campaign/recsys"
    report = reuse.prepare(source, protocol, checkpoint, confirmation, output)
    assert len(report["controls"]) == 4 and not report["full_candidate_audit_complete"]
    assert not report["student_predictions_copied"] and not report["specialist_parity_copied"]
    for control in report["controls"]:
        directory = output / control["dataset"]
        assert sha256(directory / "parent_same_recipe.npz") == control["prediction_sha256"]
        record = reuse.read(directory / "parent_same_recipe.json")
        assert record["manifest_sha256"] == sha256(output / "manifest.json")
        assert record["seconds"] == 0. and record["original_inference_seconds"] == 12.
        assert not record["inference_performed_in_this_campaign"]
        assert not (directory / "student.npz").exists() and not (directory / "student.json").exists()
        assert not (directory / "specialist_parity.json").exists()
    assert {path: sha256(path) for path in original_hashes} == original_hashes
    with pytest.raises(FileExistsError):
        reuse.prepare(source, protocol, checkpoint, confirmation, output)


def test_unchanged_evaluator_accepts_the_import_manifest_before_gpu_work(tmp_path, monkeypatch):
    source, protocol, checkpoint, confirmation = fixture(tmp_path, monkeypatch)
    output = tmp_path / "future_campaign/recsys"
    reuse.prepare(source, protocol, checkpoint, confirmation, output)

    class BeforeGPU(Exception):
        pass

    monkeypatch.setattr(reuse.evaluation, "wait_for_gpu", lambda *args: (_ for _ in ()).throw(BeforeGPU()))
    with pytest.raises(BeforeGPU):
        reuse.evaluation.evaluate(output, protocol, checkpoint, confirmation)


@pytest.mark.parametrize("failure", ["incomplete", "checkpoint", "bytes", "protocol", "audit"])
def test_changed_source_is_rejected_without_creating_output(tmp_path, monkeypatch, failure):
    source, protocol, checkpoint, confirmation = fixture(tmp_path, monkeypatch)
    if failure == "incomplete":
        write_json(source / "status.json", {"state": "evaluating"})
    elif failure == "checkpoint":
        reuse.evaluation.GENERAL.write_bytes(b"different parent")
    elif failure == "bytes":
        with (source / "100K/parent_same_recipe.npz").open("ab") as handle:
            handle.write(b"changed")
    elif failure == "protocol":
        write_json(protocol / "protocol.json", {"changed": True})
    else:
        audit = reuse.read(source / "audit.json")
        audit["cases"] = audit["cases"][:-1]
        write_json(source / "audit.json", audit)
    output = tmp_path / "future_campaign/recsys"
    with pytest.raises(ValueError):
        reuse.prepare(source, protocol, checkpoint, confirmation, output)
    assert not output.exists()


def test_changed_overlap_counts_are_independently_rejected(tmp_path, monkeypatch):
    source, protocol, checkpoint, confirmation = fixture(tmp_path, monkeypatch)
    path = source / "10M/parent_same_recipe.npz"
    np.savez(path, prediction=np.array([1.5, 2.5, 3.5]), counts=np.array([1, 1, 2], dtype=np.uint8))
    record = reuse.read(path.with_suffix(".json"))
    record["prediction_sha256"] = sha256(path)
    write_json(path.with_suffix(".json"), record)
    with pytest.raises(AssertionError):
        reuse.prepare(source, protocol, checkpoint, confirmation, tmp_path / "future_campaign/recsys")


def test_cache_import_cannot_mutate_a_running_campaign(tmp_path, monkeypatch):
    source, protocol, checkpoint, confirmation = fixture(tmp_path, monkeypatch)
    output = tmp_path / "future_campaign/recsys"
    output.parent.mkdir()
    with (output.parent / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            reuse.prepare(source, protocol, checkpoint, confirmation, output)
    assert not output.exists()


def test_unconfirmed_candidate_cannot_import_controls(tmp_path, monkeypatch):
    source, protocol, checkpoint, confirmation = fixture(tmp_path, monkeypatch)
    write_json(confirmation / "gate.json", {"passed": False})
    with pytest.raises(ValueError, match="synthetic-confirmed"):
        reuse.prepare(source, protocol, checkpoint, confirmation, tmp_path / "future_campaign/recsys")
