from pathlib import Path

import numpy as np
import pytest

import archive_single_model_duplicate_predictions as archive
from recsys_evaluation_common import atomic_npz


def files(tmp_path):
    paths = [tmp_path / name for name in ("first.npz", "same.npz", "different.npz")]
    for index, path in enumerate(paths):
        atomic_npz(path, prediction=np.full((2, 3), int(index == 2), dtype=np.float32))
    return paths


def test_only_identical_bytes_are_eligible(tmp_path):
    paths = files(tmp_path)
    records = archive.duplicate_records(paths)
    assert {row["source"] for row in records} == {str(path) for path in paths[:2]}
    assert len({row["archive_name"] for row in records}) == 1
    assert all(row["sha256"] == archive.sha256(paths[0]) for row in records)


def test_one_entry_preserves_multiple_original_file_reads(tmp_path):
    paths = files(tmp_path)
    records = archive.duplicate_records(paths)
    group = tmp_path / "group"
    group.mkdir()
    inputs = archive.stage_inputs(group, records)
    assert len(list(inputs.iterdir())) == 1
    mounted = group / "files"
    mounted.mkdir()
    target = mounted / records[0]["archive_name"]
    target.write_bytes(paths[0].read_bytes())
    archive.validate_npz(target)
    for row in records:
        archive.archive.replace_with_verified_link(row["source"], target, row["sha256"])
    archive.checkpoints.remove_staging_links(group, records)
    assert not inputs.exists()
    assert all(archive.sha256(path) == records[0]["sha256"] for path in paths[:2])
    assert not paths[2].is_symlink()


@pytest.mark.parametrize("failure", ["name", "hash", "changed", "unique"])
def test_coalescing_never_accepts_approximate_or_substituted_bytes(tmp_path, failure):
    paths = files(tmp_path)
    records = archive.duplicate_records(paths)
    group = tmp_path / "group"
    group.mkdir()
    if failure == "name":
        records[0]["archive_name"] = "../escape.npz"
    elif failure == "hash":
        records[0]["sha256"] = "wrong"
    elif failure == "changed":
        atomic_npz(paths[0], prediction=np.ones((2, 3), dtype=np.float32))
    else:
        records = records[:1]
    with pytest.raises(ValueError):
        archive.stage_inputs(group, records)
    assert not (group / "named_inputs").exists()


def test_object_arrays_are_not_accepted(tmp_path):
    path = tmp_path / "objects.npz"
    atomic_npz(path, objects=np.array([{}], dtype=object))
    with pytest.raises(ValueError):
        archive.validate_npz(path)


def test_read_only_scope_and_original_concurrent_reserve():
    assert archive.compressor().__globals__["MINIMUM_FREE"] == 21 * 1024**3
    assert archive.archive.MINIMUM_FREE == 21 * 1024**3
    assert "torch_mmap_readable" not in archive.COMPRESSOR_SOURCE
    assert "validate_npz(archived)" in archive.COMPRESSOR_SOURCE
    assert len(archive.RUNS) == 3 and all("v20" not in name for name in archive.RUNS)
