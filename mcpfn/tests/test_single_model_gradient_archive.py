import pytest

from relocate_single_model_gradient_archive import relocate, sha256


def test_atomic_relocation_preserves_canonical_bytes_and_is_idempotent(tmp_path):
    source, destination = tmp_path / "original.pt", tmp_path / "archive" / "original.pt"
    source.write_bytes(b"owned synthetic gradient bytes")
    digest = sha256(source)
    result = relocate(source, destination, digest, minimum_free=0)
    assert source.is_symlink() and source.read_bytes() == destination.read_bytes()
    assert result["canonical_path_preserved"] and result["durable_copy_fsynced"]
    assert relocate(source, destination, digest, minimum_free=0)["already_linked"]


def test_bad_source_or_destination_never_replaces_original(tmp_path):
    source, destination = tmp_path / "original.pt", tmp_path / "archive.pt"
    source.write_bytes(b"source")
    with pytest.raises(ValueError, match="changed contents"):
        relocate(source, destination, "incorrect", minimum_free=0)
    destination.write_bytes(b"unrelated")
    with pytest.raises(FileExistsError, match="unrelated"):
        relocate(source, destination, sha256(source), minimum_free=0)
    assert not source.is_symlink() and source.read_bytes() == b"source"


def test_destination_reserve_is_checked_before_copying(tmp_path):
    source, destination = tmp_path / "original.pt", tmp_path / "archive.pt"
    source.write_bytes(b"source")
    with pytest.raises(OSError, match="reserve"):
        relocate(source, destination, sha256(source), minimum_free=10**20)
    assert source.is_file() and not source.is_symlink() and not destination.exists()
