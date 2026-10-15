import pytest

import archive_single_model_idle_evidence as evidence


def test_idle_floor_changes_no_original_archive_safety_code():
    compressor = evidence.idle_compressor()
    assert compressor.__code__ is evidence.archive.compress_group.__code__
    assert compressor.__globals__["MINIMUM_FREE"] == 18 * 1024**3
    assert evidence.archive.MINIMUM_FREE == 21 * 1024**3
    assert compressor.__globals__["replace_with_verified_link"] is evidence.archive.replace_with_verified_link


def test_only_unchanged_regular_originals_can_be_planned(tmp_path):
    source = tmp_path / "evidence.pt"
    source.write_bytes(b"original evidence")
    expected = evidence.sha256(source)
    assert evidence.original_record(source, expected) == {"source": str(source), "sha256": expected}
    with pytest.raises(ValueError, match="registered hash"):
        evidence.original_record(source, "changed")
    alias = tmp_path / "alias.pt"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="regular original"):
        evidence.original_record(alias, expected)


def test_a_live_source_process_blocks_archival(monkeypatch):
    monkeypatch.setattr(evidence, "read", lambda path: {"state": "complete", "pid": 1})
    with pytest.raises(ValueError, match="idle scientific"):
        evidence.require_idle()
