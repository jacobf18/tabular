from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import single_model_compressed_endpoint as endpoint
import single_model_composite_gradient_storage as composite


def payload():
    gradients = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    gradients[0, 0] = -0.0
    complete, snapshot = composite.share(gradients, {"task_gradients": gradients[:2].clone()}, 2)
    return {"gradients": complete, "snapshot": snapshot,
            "record": {"composite_tasks": 2, "protected_tasks": 6}, "extra": torch.tensor(float("nan"))}


@pytest.fixture
def mock_archive(monkeypatch):
    images = {}

    def compress(source, image, log):
        images["raw"] = source.read_bytes()
        image.write_bytes(b"compressed-test-image")
        log.write_text("mock compression\n")

    def mount(image, mountpoint):
        (mountpoint / "endpoint.pt").write_bytes(images["raw"])
        return {"unit": "/fake/unit.mount", "unit_sha256": "fake", "persistent_mount_enabled": True}

    monkeypatch.setattr(endpoint, "compress", compress)
    monkeypatch.setattr(endpoint, "mount_unit", mount)
    monkeypatch.setattr(endpoint, "verify_mount", lambda directory, mounted: {})
    monkeypatch.setattr(endpoint, "filesystem", lambda path: "tmpfs")
    monkeypatch.setattr(endpoint.shutil, "disk_usage", lambda path: SimpleNamespace(free=64 * 1024**3))
    return images


def test_roundtrip_preserves_complete_gradients_and_shorter_alias(tmp_path, mock_archive):
    path = tmp_path / "endpoint.pt"
    original = payload()
    digest = endpoint.save(original, path)
    restored = torch.load(path, weights_only=True, mmap=True)
    assert path.is_symlink() and endpoint.sha256(path) == digest
    composite.check_alias(restored)
    assert composite.byte_equal(original["gradients"], restored["gradients"])
    assert composite.byte_equal(original["extra"], restored["extra"])
    assert endpoint.audit(restored, path)
    assert not path.with_suffix(".write.json").exists()


@pytest.mark.parametrize("target", ["canonical", "directory", "receipt", "broken_link"])
def test_refuses_any_existing_evidence(tmp_path, mock_archive, target):
    path = tmp_path / "endpoint.pt"
    if target == "canonical":
        path.touch()
    elif target == "directory":
        path.with_suffix(".compressed").mkdir()
    elif target == "receipt":
        path.with_suffix(".compressed-write.json").touch()
    else:
        path.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError, match="Never replace"):
        endpoint.save(payload(), path)


@pytest.mark.parametrize("kind", ["ram", "root", "durable", "not_tmpfs"])
def test_checks_reserves_before_serialization(tmp_path, mock_archive, monkeypatch, kind):
    path = tmp_path / "endpoint.pt"

    def free(current):
        name = str(current)
        low = ((kind == "ram" and name == "/run") or (kind == "root" and name == "/")
               or (kind == "durable" and name == str(tmp_path)))
        return SimpleNamespace(free=0 if low else 64 * 1024**3)

    monkeypatch.setattr(endpoint.shutil, "disk_usage", free)
    if kind == "not_tmpfs":
        monkeypatch.setattr(endpoint, "filesystem", lambda current: "ext4")
    with pytest.raises(OSError):
        endpoint.save(payload(), path)
    assert not path.with_suffix(".compressed").exists()


def test_checks_actual_compressed_size_before_durable_copy(tmp_path, mock_archive, monkeypatch):
    path = tmp_path / "endpoint.pt"
    free = endpoint.MINIMUM_FREE + endpoint.ALLOWANCE + 1
    monkeypatch.setattr(endpoint.shutil, "disk_usage", lambda current: SimpleNamespace(
        free=free if Path(current) == tmp_path else 64 * 1024**3))
    with pytest.raises(OSError, match="Actual compressed image"):
        endpoint.save(payload(), path)
    assert not path.exists()
    assert not (path.with_suffix(".compressed") / "endpoint.squashfs").exists()


@pytest.mark.parametrize("field", ["sha256", "tensor_storage_bytes", "file_bytes", "image_sha256", "image_bytes",
    "composite_tasks", "protected_tasks", "free_before_image", "free_after", "minimum_free_bytes",
    "all_tensor_bytes_retained", "composite_proposal_alias_preserved", "raw_staging_filesystem"])
def test_audit_rejects_false_receipt(tmp_path, mock_archive, field):
    path = tmp_path / "endpoint.pt"
    endpoint.save(payload(), path)
    receipt = path.with_suffix(".compressed-write.json")
    report = json.loads(receipt.read_text())
    report[field] = False if isinstance(report[field], bool) else 0 if isinstance(report[field], int) else "wrong"
    receipt.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="receipt"):
        endpoint.audit(torch.load(path, weights_only=True, mmap=True), path)


def test_rejects_changed_archived_bytes(tmp_path, mock_archive):
    path = tmp_path / "endpoint.pt"
    endpoint.save(payload(), path)
    restored = torch.load(path, weights_only=True, mmap=True)
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="receipt"):
        endpoint.audit(restored, path)


def test_rejects_independent_proposal_storage(tmp_path, mock_archive):
    current = deepcopy(payload())
    current["snapshot"]["task_gradients"] = current["snapshot"]["task_gradients"].clone()
    with pytest.raises(ValueError, match="exact shorter proposal"):
        endpoint.save(current, tmp_path / "endpoint.pt")
