import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import relocate_single_model_idle_diagnostics_to_root as relocation


def test_allowlist_excludes_selected_teachers_and_active_trial():
    assert len(relocation.ARTIFACTS) == len(set(relocation.ARTIFACTS)) == 6
    assert all("top_block_v24" not in name and "selected_checkpoint" not in name
               and "best_validation_checkpoint" not in name for name in relocation.ARTIFACTS)
    assert relocation.DESTINATION.is_relative_to(relocation.REPO / "runs")
    assert "unrelated live training" in relocation.POLICY.lower()


@pytest.fixture
def idle_process(tmp_path):
    artifact = tmp_path / "artifact.pt"
    artifact.write_bytes(b"original")
    proc = tmp_path / "proc"
    process = proc / "123"
    (process / "fd").mkdir(parents=True)
    (process / "maps").write_text("")
    (process / "cmdline").write_bytes(b"python\0train_single_matrixformer_top_block.py\0--stage\0train")
    return artifact, proc, process


def test_specific_reader_check_allows_unrelated_live_training(idle_process):
    artifact, proc, process = idle_process
    relocation.require_no_artifact_readers([artifact], proc)
    (process / "fd/3").symlink_to(artifact)
    with pytest.raises(ValueError, match="descriptor"):
        relocation.require_no_artifact_readers([artifact], proc)


def test_specific_reader_check_detects_aliased_mmap_inode(idle_process):
    artifact, proc, process = idle_process
    current = artifact.stat()
    (process / "maps").write_text(
        f"1000-2000 r--p 00000000 {os.major(current.st_dev):02x}:{os.minor(current.st_dev):02x} "
        f"{current.st_ino} /different/alias/name\n")
    with pytest.raises(ValueError, match="mmap"):
        relocation.require_no_artifact_readers([artifact], proc)


def test_stopped_owner_guard_rejects_live_or_wrong_state(tmp_path, monkeypatch):
    monkeypatch.setattr(relocation, "ROOT", tmp_path)
    monkeypatch.setattr(relocation, "OWNERS", {"owner": "complete"})
    relocation.write_json(tmp_path / "owner/status.json", {"state": "complete", "pid": os.getpid()})
    with pytest.raises(ValueError, match="stopped"):
        relocation.require_stopped_owners()
    relocation.write_json(tmp_path / "owner/status.json", {"state": "training", "pid": 999999999})
    with pytest.raises(ValueError, match="stopped"):
        relocation.require_stopped_owners()
    relocation.write_json(tmp_path / "owner/status.json", {"state": "complete", "pid": 999999999})
    relocation.require_stopped_owners()


def test_preflight_and_tail_cannot_become_selected():
    preflight = {"preflight_only": True, "selection_qualified": False,
                 "gradient_updates_performed": 0, "model": {"weight": torch.ones(2)}}
    relocation.check_payload(preflight, relocation.ARTIFACTS[0])
    for key, wrong in {"preflight_only": False, "selection_qualified": True, "gradient_updates_performed": 1}.items():
        with pytest.raises(ValueError, match="nonpromotable"):
            relocation.check_payload({**preflight, key: wrong}, relocation.ARTIFACTS[0])
    tail = {"step": 96, "best_step": 32, "gradient_updates_performed": 67, "independent_confirmation_passed": False,
            "model": {"weight": torch.ones(2)}, "optimizer": {"state": {}}}
    relocation.check_payload(tail, relocation.ARTIFACTS[-1])
    with pytest.raises(ValueError, match="terminal tail"):
        relocation.check_payload({**tail, "step": 32}, relocation.ARTIFACTS[-1])
    with pytest.raises(ValueError, match="six exact"):
        relocation.check_payload(tail, "top_block_v24/latest_checkpoint.pth")
    with pytest.raises(ValueError, match="rate candidate"):
        relocation.check_payload(torch.zeros(3), relocation.ARTIFACTS[1])


@pytest.mark.parametrize("path", ["/", relocation.ROOT, "/mnt/matrixformer_scratch"])
def test_capacity_budget_respects_all_reserves(monkeypatch, path):
    total = 10000
    capacity = {"/": 6 * 1024**3 + total + 1024**2, relocation.ROOT: 18 * 1024**3,
                "/mnt/matrixformer_scratch": 50 * 1024**3}
    monkeypatch.setattr(relocation.shutil, "disk_usage", lambda target: SimpleNamespace(free=capacity[target]))
    relocation.require_capacity(total)
    capacity[path] -= 1
    with pytest.raises(OSError, match="reserves"):
        relocation.require_capacity(total)


def test_copy_publish_preserves_bytes_and_mmap_and_cannot_repeat(tmp_path):
    source, destination = tmp_path / "source.pth", tmp_path / "target/checkpoint.pth"
    payload = {"preflight_only": True, "selection_qualified": False,
               "gradient_updates_performed": 0, "model": {"weight": torch.tensor([1., -0.])}}
    torch.save(payload, source)
    row = {"name": relocation.ARTIFACTS[0], "source": str(source), "destination": str(destination),
           "sha256": relocation.sha256(source), "bytes": source.stat().st_size}
    relocation.copy_and_verify(row)
    relocation.publish(row)
    assert source.is_symlink() and source.resolve() == destination
    assert relocation.sha256(source) == relocation.sha256(destination) == row["sha256"]
    restored = torch.load(source, weights_only=True, mmap=True)
    assert torch.equal(restored["model"]["weight"].view(torch.int32), payload["model"]["weight"].view(torch.int32))
    del restored
    with pytest.raises(ValueError, match="untouched original"):
        relocation.publish(row)


def test_publish_rejects_changed_copy_without_removing_source(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"original")
    destination.write_bytes(b"modified")
    row = {"source": str(source), "destination": str(destination), "sha256": relocation.sha256(source)}
    with pytest.raises(ValueError, match="verified complete copy"):
        relocation.publish(row)
    assert source.read_bytes() == b"original" and not source.is_symlink()
