import inspect

import pytest
import torch

import relocate_single_model_idle_diagnostics_to_scratch as relocation


def test_scope_excludes_every_baseline_and_live_trial():
    assert relocation.OWNERS == {"family_projection_v12_control": "failed", "family_projection_v12_projected": "failed",
                                 "query_budget_v8_tail_average": "complete"}
    assert "machine-local scratch" in relocation.POLICY and "may not survive VM replacement" in relocation.POLICY
    assert "provider-persistent" in relocation.POLICY and "without formatting" in relocation.POLICY
    assert relocation.SCRATCH != relocation.ROOT and not relocation.SCRATCH.is_relative_to(relocation.ROOT)
    source = inspect.getsource(relocation.inventory)
    assert 'report["full_100_update_audit_complete"] is not False' in source
    assert 'report["input_sha256"]' in source and 'row["sha256"]' in source
    assert "range(300, 401, 10)" in source


def record(tmp_path):
    source, destination = tmp_path / "original.pth", tmp_path / "scratch/copy.pth"
    torch.save({"model": {"weights": torch.tensor([1., -0., float("nan")])}, "optimizer": {"step": 10}}, source)
    return {"source": str(source), "destination": str(destination), "sha256": relocation.sha256(source),
            "bytes": source.stat().st_size}


def test_copy_fsync_verify_and_canonical_mmap_roundtrip(tmp_path):
    row = record(tmp_path)
    relocation.copy_and_verify(row)
    source, destination = relocation.Path(row["source"]), relocation.Path(row["destination"])
    assert source.is_file() and not source.is_symlink()
    assert relocation.sha256(destination) == row["sha256"]
    relocation.publish(row)
    assert source.is_symlink() and source.resolve() == destination
    assert relocation.sha256(source) == row["sha256"]
    restored = torch.load(source, weights_only=True, mmap=True)
    assert restored["optimizer"]["step"] == 10 and torch.signbit(restored["model"]["weights"][1])
    with pytest.raises(ValueError, match="prior incomplete"):
        relocation.publish(row)


@pytest.mark.parametrize("mutation", ["source", "destination", "staging_link"])
def test_never_publishes_changed_or_incomplete_data(tmp_path, mutation):
    row = record(tmp_path)
    relocation.copy_and_verify(row)
    source, destination = relocation.Path(row["source"]), relocation.Path(row["destination"])
    if mutation == "source":
        source.write_bytes(b"changed")
    elif mutation == "destination":
        destination.write_bytes(b"changed")
    else:
        source.with_suffix(".pth.scratch-link").symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="exact copied bytes"):
        relocation.publish(row)
    assert not source.is_symlink()


def test_copy_never_overwrites_existing_destination(tmp_path):
    row = record(tmp_path)
    relocation.copy_and_verify(row)
    with pytest.raises(FileExistsError):
        relocation.copy_and_verify(row)
