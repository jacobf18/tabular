from pathlib import Path

import pytest

import archive_single_model_completed_task_groups as groups


def test_partition_keeps_every_batch_whole_and_each_group_bounded(tmp_path):
    records = {}
    for index in range(5):
        directory = tmp_path / f"step_{index:03d}"
        directory.mkdir()
        for name in ("task.pt", "manifest.json"):
            path = directory / name
            path.write_bytes(b"12345")
            records[str(path)] = name
    result = groups.partition(records, limit=20)
    assert list(map(len, result)) == [4, 4, 2]
    assert {row["source"]: row["sha256"] for group in result for row in group} == records
    ownership = {}
    for index, group in enumerate(result):
        assert sum(Path(row["source"]).stat().st_size for row in group) <= 20
        for row in group:
            assert ownership.setdefault(Path(row["source"]).parent, index) == index


def test_an_oversized_batch_is_never_silently_split(tmp_path):
    path = tmp_path / "task.pt"
    path.write_bytes(b"1234")
    with pytest.raises(ValueError, match="complete audited batch"):
        groups.partition({str(path): "hash"}, limit=3)


def test_inventory_is_explicit_and_preexisting_reserves_are_unchanged():
    assert len(groups.RUNS) == 5
    assert sum(row[1] for row in groups.RUNS) == 328
    assert groups.MAXIMUM_GROUP == 384 * 1024**2
    assert groups.MINIMUM_FREE == 18 * 1024**3
    assert groups.archive.MINIMUM_FREE == 21 * 1024**3
