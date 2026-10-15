import json
from pathlib import Path

import pytest

import relocate_single_model_mounted_archives as relocation


@pytest.mark.parametrize("script", ["train_single_matrixformer_structured_teacher.py", "audit_single_model_structured_teacher.py",
                                  "run_single_model_structured_teacher_pipeline.py", "compare_single_model_point_proposals.py"])
def test_scientific_workers_are_rejected(script):
    assert relocation.scientific_arguments(["python", "scripts/" + script])


def test_only_actual_script_tokens_are_workers():
    assert not relocation.scientific_arguments(["bash", "-c", "python scripts/train_single_matrixformer_structured_teacher.py > run.log"])
    assert not relocation.scientific_arguments(["python", relocation.__file__])


@pytest.mark.parametrize("state,pid", [("running", 99999999), ("complete", 1), ("complete", -1), ("complete", None)])
def test_terminal_requires_completed_absent_producer(tmp_path, state, pid):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"state": state, "pid": pid}))
    with pytest.raises(ValueError, match="terminal producers"):
        relocation.terminal(path)


def test_terminal_accepts_complete_absent_producer(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"state": "training_complete", "pid": 99999999}))
    assert relocation.terminal(path) == relocation.sha256(path)


def test_archive_allowlist_is_fixed():
    with pytest.raises(ValueError, match="three fixed"):
        relocation.bind_archive("unrelated_archive")


def mount_fixture(monkeypatch, tmp_path, *, options="ro,relatime", backing=None):
    image = tmp_path / "archive.squashfs"
    image.write_bytes(b"archive")
    record = {"unit": str(tmp_path / "unit.mount"), "mount": str(tmp_path / "files"), "image": str(image)}
    report = {"filesystems": [{"source": "/dev/loop99", "target": record["mount"], "fstype": "squashfs", "options": options}]}
    calls = []
    monkeypatch.setattr(relocation.subprocess, "run", lambda args, **kwargs: calls.append(args))
    monkeypatch.setattr(relocation.subprocess, "check_output",
                        lambda args, **kwargs: json.dumps(report) if args[0] == "findmnt" else str(backing or image) + "\n")
    return record, calls


def test_mount_verifies_enabled_read_only_and_actual_backing(monkeypatch, tmp_path):
    record, calls = mount_fixture(monkeypatch, tmp_path)
    assert relocation.check_mount(record)["read_only"]
    assert [call[1] for call in calls] == ["is-active", "is-enabled"]


@pytest.mark.parametrize("options", ["rw,relatime", "relatime"])
def test_writable_mount_is_rejected(monkeypatch, tmp_path, options):
    record, _ = mount_fixture(monkeypatch, tmp_path, options=options)
    with pytest.raises(ValueError, match="read-only"):
        relocation.check_mount(record)


@pytest.mark.parametrize("backing", ["/old/archive.squashfs (deleted)", "/wrong/archive.squashfs"])
def test_pinned_or_wrong_loop_inode_is_rejected(monkeypatch, tmp_path, backing):
    record, _ = mount_fixture(monkeypatch, tmp_path, backing=backing)
    with pytest.raises(ValueError, match="old inode"):
        relocation.check_mount(record)


def test_durable_copy_precedes_unchanged_mount_restart(monkeypatch, tmp_path):
    record, calls = mount_fixture(monkeypatch, tmp_path)
    record.update({"protected": [], "input_sha256": {}, "destination": str(tmp_path / "destination.squashfs"),
                   "image_sha256": "digest", "records": [{"source": "original.pth"}], "name": "selected"})
    events = []
    monkeypatch.setattr(relocation, "OUTPUT", tmp_path)
    monkeypatch.setattr(relocation, "require_idle", lambda paths: events.append("idle"))
    monkeypatch.setattr(relocation, "verify_hashes", lambda hashes: events.append("verify"))
    monkeypatch.setattr(relocation, "check_mount", lambda *args: {"checked": True})
    monkeypatch.setattr(relocation.durable, "relocate", lambda *args, **kwargs: events.append("copy") or {})
    monkeypatch.setattr(relocation, "write_json", lambda *args: None)
    monkeypatch.setattr(relocation.subprocess, "run", lambda args, **kwargs: events.append(args[1]))
    monkeypatch.setattr(relocation, "sha256", lambda path: "digest")
    monkeypatch.setattr(relocation.torch, "load", lambda *args, **kwargs: events.append("mmap") or {})
    result = relocation.move_one(record)
    assert events.index("copy") < events.index("restart") < events.index("mmap")
    assert result["all_original_artifact_hashes_unchanged"]
    assert result["mmap_records_checked"] == 1


def test_capacity_gate_requires_successful_completed_archive(monkeypatch, tmp_path):
    monkeypatch.setattr(relocation, "CAPACITY", tmp_path)
    monkeypatch.setattr(relocation, "terminal", lambda path: "digest")
    (tmp_path / "independent_recheck.json").write_text(json.dumps({"complete": True, "capacity_target_reached": False}))
    with pytest.raises(ValueError, match="successful lossless"):
        relocation.capacity_ready()


def test_existing_output_is_never_restarted(monkeypatch, tmp_path):
    monkeypatch.setattr(relocation, "OUTPUT", tmp_path)
    with pytest.raises(FileExistsError, match="Never restart"):
        relocation.main()
