import pytest

import archive_single_model_paired_capacity as capacity


def test_only_declared_artifacts_and_explicit_volume_reserves_are_supported():
    with pytest.raises(ValueError, match="two explicit"):
        capacity.plan("other")
    assert capacity.CHECKPOINT_RESERVE == 18 * 1024**3
    assert capacity.ROOT_RESERVE == 3 * 1024**3
    assert capacity.archive.MINIMUM_FREE == 21 * 1024**3


def test_unknown_or_linked_staging_is_rejected_without_creating_an_archive(tmp_path):
    source = tmp_path / "original.pt"
    source.write_bytes(b"original")
    staging = tmp_path / "untrusted.squashfs"
    staging.write_bytes(b"image")
    output = tmp_path / "archive"
    with pytest.raises(ValueError, match="owned RAM-staged"):
        capacity.publish_staged(output, {"source": str(source), "sha256": "hash"}, staging)
    assert not output.exists()
    assert source.read_bytes() == b"original"
