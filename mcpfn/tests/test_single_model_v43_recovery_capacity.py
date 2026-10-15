from types import SimpleNamespace

import pytest
import torch

import relocate_single_model_v43_recovery_capacity as relocation


@pytest.mark.parametrize("root_free,total,checkpoint_free,scratch_free,passed", [
    (5.7, 1.0, 20, 60, True), (2.9, 5.0, 20, 60, False), (5.0, 0.5, 20, 60, False),
    (5.7, 1.0, 18, 60, False), (5.7, 1.0, 20, 50, False),
])
def test_capacity_restores_original_reserve_without_requiring_it_before_relocation(
        monkeypatch, root_free, total, checkpoint_free, scratch_free, passed):
    free = {"/": root_free, str(relocation.ROOT): checkpoint_free, str(relocation.copying.SCRATCH): scratch_free}
    monkeypatch.setattr(relocation.shutil, "disk_usage", lambda path: SimpleNamespace(free=int(free[str(path)] * 1024**3)))
    if passed:
        relocation.check_capacity(int(total * 1024**3))
    else:
        with pytest.raises(OSError):
            relocation.check_capacity(int(total * 1024**3))


def test_copy_and_atomic_link_keep_both_paths_and_exact_serialized_bytes(tmp_path, monkeypatch):
    source, destination = tmp_path / "source.pt", tmp_path / "scratch" / "copy.pt"
    torch.save({"model": {"weight": torch.arange(12)}}, source)
    original_bytes = source.read_bytes()
    row = {"source": str(source), "destination": str(destination), "bytes": source.stat().st_size,
           "sha256": relocation.sha256(source), "name": "test"}
    inspected = []
    monkeypatch.setattr(relocation.previous, "check_payload", lambda payload, name: inspected.append(name))
    relocation.copy_and_verify(row)
    assert source.is_file() and not source.is_symlink()
    relocation.copying.publish(row)
    assert inspected == ["test"]
    assert source.is_symlink() and source.resolve() == destination
    assert source.read_bytes() == destination.read_bytes() == original_bytes
    assert torch.equal(torch.load(source, mmap=True, weights_only=True)["model"]["weight"], torch.arange(12))


def test_wrong_copy_digest_is_rejected_without_publishing(tmp_path, monkeypatch):
    source, destination = tmp_path / "source.pt", tmp_path / "copy.pt"
    torch.save({"model": {}}, source)
    row = {"source": str(source), "destination": str(destination), "bytes": source.stat().st_size,
           "sha256": "incorrect", "name": "test"}
    with pytest.raises(ValueError, match="inexact"):
        relocation.copy_and_verify(row)
    assert source.is_file() and not source.is_symlink()


def test_existing_operation_is_never_restarted(tmp_path, monkeypatch):
    monkeypatch.setattr(relocation, "OUTPUT", tmp_path)
    monkeypatch.setattr(relocation, "inventory", lambda: pytest.fail("Do not begin another inventory"))
    with pytest.raises(FileExistsError):
        relocation.main()
