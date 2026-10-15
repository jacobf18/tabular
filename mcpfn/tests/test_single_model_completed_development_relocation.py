import numpy as np
import pytest

import relocate_single_model_completed_development_banks as relocation


def record(tmp_path):
    source = tmp_path / "grid.npz"
    np.savez_compressed(source, values=np.array([1., -0., np.nan]))
    return {"source": str(source), "destination": str(tmp_path / "scratch/grid.npz"),
            "sha256": relocation.sha256(source), "bytes": source.stat().st_size}


def test_scope_is_only_all_completed_development_grids():
    assert relocation.OWNERS == ("structured_teacher_v20", "point_inclusive_v21")
    assert relocation.BOUNDARIES == (0, 10, 25, 50, 64)
    assert "all1520" in relocation.POLICY and "not provider-persistent" in relocation.POLICY


def test_exact_copy_and_canonical_readability(tmp_path):
    row = record(tmp_path)
    relocation.copy_and_verify(row)
    relocation.storage.publish(row)
    assert relocation.Path(row["source"]).is_symlink()
    assert relocation.sha256(row["source"]) == row["sha256"]
    with np.load(row["source"], allow_pickle=False) as payload:
        assert np.signbit(payload["values"][1]) and np.isnan(payload["values"][2])
    with pytest.raises(FileExistsError):
        relocation.copy_and_verify(row)


@pytest.mark.parametrize("mutation", ["source", "destination"])
def test_changed_bytes_cannot_be_published(tmp_path, mutation):
    row = record(tmp_path)
    relocation.copy_and_verify(row)
    relocation.Path(row[mutation]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="exact copied bytes"):
        relocation.storage.publish(row)
    assert not relocation.Path(row["source"]).is_symlink()


def test_changed_copy_digest_is_rejected(tmp_path):
    row = record(tmp_path)
    row["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact serialized copy"):
        relocation.copy_and_verify(row)


@pytest.mark.parametrize("missing", [False, True])
def test_inventory_binds_every_grid_and_original_audit(tmp_path, monkeypatch, missing):
    monkeypatch.setattr(relocation, "ROOT", tmp_path)
    monkeypatch.setattr(relocation, "OWNERS", ("completed",))
    monkeypatch.setattr(relocation, "BOUNDARIES", (10,))
    monkeypatch.setattr(relocation.storage, "require_mount", lambda: None)
    monkeypatch.setattr(relocation, "require_idle", lambda: None)
    directory = tmp_path / "completed/validation_010/student_predictions_000"
    directory.mkdir(parents=True)
    hashes = {}
    for index in range(152):
        path = directory / f"case_{index:03d}.npz"
        np.savez_compressed(path, values=np.ones(2))
        hashes[str(path)] = relocation.sha256(path)
    for suffix in ("", "_pipeline"):
        owner = tmp_path / ("completed" + suffix)
        owner.mkdir(exist_ok=True)
        for name in ("config.json", "status.json", "registration.json"):
            relocation.write_json(owner / name, {})
    audit = tmp_path / "completed_audit"
    audit.mkdir()
    relocation.write_json(audit / "audit.json", {"complete": True, "complete_training_audit": True,
                          "through_step": 64, "artifact_sha256": hashes})
    if missing:
        (directory / "case_151.npz").unlink()
        with pytest.raises(ValueError, match="every original development grid"):
            relocation.inventory()
    else:
        records, inputs = relocation.inventory()
        assert len(records) == 152 and len(inputs) == 7
        assert {row["source"]: row["sha256"] for row in records} == hashes
