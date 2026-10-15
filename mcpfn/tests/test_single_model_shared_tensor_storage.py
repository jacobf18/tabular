import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import single_model_shared_tensor_storage as storage


def test_aliases_count_once_and_views_account_full_backing_storage():
    source = torch.arange(4096, dtype=torch.float32)
    assert storage.storage_bytes({"source": source, "slice": source[:3], "nested": [source]}) == source.numel() * 4
    assert storage.storage_bytes({"slice": source[:3]}) == source.numel() * 4
    assert storage.storage_bytes([source, source.clone()]) == source.numel() * 8


def test_unsupported_payload_cannot_hide_bytes():
    with pytest.raises(ValueError, match="unsupported"):
        storage.storage_bytes({"array": np.zeros(128)})
    with pytest.raises(ValueError, match="CPU strided"):
        storage.storage_bytes(torch.empty(2, device="meta"))


def test_exact_boundary_save_audit_and_tamper_rejection(tmp_path, monkeypatch):
    source = torch.arange(4096, dtype=torch.float32)
    payload = {"gradients": source, "snapshot": {"task_gradients": source}, "step": 1}
    required = storage.storage_bytes(payload) + storage.METADATA_ALLOWANCE
    free = storage.MINIMUM_FREE + required
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=free))
    path = tmp_path / "endpoint.pt"
    digest = storage.save(payload, path)
    saved = torch.load(path, weights_only=True)
    assert saved["gradients"].data_ptr() == saved["snapshot"]["task_gradients"].data_ptr()
    assert digest == storage.sha256(path)
    assert storage.audit(saved, path)
    with pytest.raises(FileExistsError):
        storage.save(payload, path)
    receipt = path.with_suffix(".write.json")
    report = json.loads(receipt.read_text())
    report["free_before"] -= 1
    receipt.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="receipt"):
        storage.audit(saved, path)


def test_reserve_rejection_occurs_before_any_write(tmp_path, monkeypatch):
    payload = {"value": torch.ones(32)}
    free = storage.MINIMUM_FREE + storage.storage_bytes(payload) + storage.METADATA_ALLOWANCE - 1
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: SimpleNamespace(free=free))
    path = tmp_path / "endpoint.pt"
    with pytest.raises(OSError, match="18 GiB"):
        storage.save(payload, path)
    assert not path.exists() and not path.with_suffix(".write.json").exists()
