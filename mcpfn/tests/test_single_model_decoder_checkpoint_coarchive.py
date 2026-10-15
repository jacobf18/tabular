from pathlib import Path

import pytest

import archive_single_model_decoder_checkpoints as coarchive


def originals(tmp_path):
    records = []
    for index in range(2):
        directory = tmp_path / f"source_{index}"
        directory.mkdir()
        source = directory / "latest_checkpoint.pth"
        source.write_bytes(b"original immutable model and optimizer" + bytes([index]))
        records.append({"source": str(source), "sha256": coarchive.sha256(source), "archive_name": f"source_{index}.pth"})
    group = tmp_path / "group"
    group.mkdir()
    return group, records


def test_scoped_compressor_keeps_concurrent_writer_reserve():
    assert coarchive.compressor().__globals__["MINIMUM_FREE"] == 21 * 1024**3
    assert coarchive.archive.MINIMUM_FREE == 21 * 1024**3
    assert 'row["archive_name"]' in coarchive.COMPRESSOR_SOURCE
    assert 'remove_staging_links(directory, records)' in coarchive.COMPRESSOR_SOURCE
    assert len(coarchive.TRAINING_RUNS) == len(coarchive.PREFLIGHT_RUNS) == 6
    assert all("v20" not in name for name, count in coarchive.TRAINING_RUNS)


def test_named_inputs_preserve_originals_until_durable_replacements(tmp_path):
    group, records = originals(tmp_path)
    inputs = coarchive.stage_inputs(group, records)
    for row in records:
        assert Path(row["source"]).stat().st_nlink == 2
        assert coarchive.sha256(inputs / row["archive_name"]) == row["sha256"]
    with pytest.raises(ValueError, match="before durable"):
        coarchive.remove_staging_links(group, records)
    (group / "files").mkdir()
    for row in records:
        archived = group / "files" / row["archive_name"]
        archived.write_bytes(Path(row["source"]).read_bytes())
        coarchive.archive.replace_with_verified_link(row["source"], archived, row["sha256"])
    coarchive.remove_staging_links(group, records)
    assert not inputs.exists()
    assert all(coarchive.sha256(row["source"]) == row["sha256"] for row in records)


@pytest.mark.parametrize("failure", ["collision", "escape", "changed", "linked"])
def test_invalid_inputs_do_not_create_staging(tmp_path, failure):
    group, records = originals(tmp_path)
    if failure == "collision":
        records[1]["archive_name"] = records[0]["archive_name"]
    elif failure == "escape":
        records[1]["archive_name"] = "../escape.pth"
    elif failure == "changed":
        Path(records[0]["source"]).write_bytes(b"changed")
    else:
        __import__("os").link(records[0]["source"], tmp_path / "other-link")
    with pytest.raises(ValueError):
        coarchive.stage_inputs(group, records)
    assert not (group / "named_inputs").exists()


def test_unaccounted_staging_file_is_never_deleted(tmp_path):
    group, records = originals(tmp_path)
    inputs = coarchive.stage_inputs(group, records)
    (inputs / "unaccounted").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="unaccounted"):
        coarchive.remove_staging_links(group, records)
    assert (inputs / "unaccounted").read_bytes() == b"preserve"


@pytest.mark.parametrize("state,pid", [("training", 999999999), ("training_complete", 1)])
def test_no_live_source_can_move(monkeypatch, state, pid):
    monkeypatch.setattr(coarchive, "read", lambda path: {"state": state, "pid": pid})
    with pytest.raises(ValueError, match="completed idle"):
        coarchive.require_terminal()
