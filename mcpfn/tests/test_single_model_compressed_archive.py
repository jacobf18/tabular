import pytest

from compress_single_model_gradient_archive import replace_with_verified_link
from relocate_single_model_gradient_archive import sha256


def test_verified_replacement_preserves_canonical_and_open_reader(tmp_path):
    original, archived = tmp_path / "original.pt", tmp_path / "archived.pt"
    payload = b"identical scientific gradient evidence" * 200
    original.write_bytes(payload)
    archived.write_bytes(payload)
    with original.open("rb") as reader:
        result = replace_with_verified_link(original, archived, sha256(original))
        assert reader.read() == payload
    assert original.is_symlink()
    assert original.read_bytes() == payload
    assert result["canonical_path_preserved"]
    assert result["bytes_archived"] == len(payload)


@pytest.mark.parametrize("mutation", ["source", "archive", "missing", "linked", "staging"])
def test_replacement_preserves_original_on_preflight_failure(tmp_path, mutation):
    original, archived = tmp_path / "original.pt", tmp_path / "archived.pt"
    original.write_bytes(b"evidence")
    archived.write_bytes(b"evidence")
    expected = sha256(original)
    if mutation == "source":
        original.write_bytes(b"changed")
    elif mutation == "archive":
        archived.write_bytes(b"changed")
    elif mutation == "missing":
        archived.unlink()
    elif mutation == "linked":
        original.unlink()
        original.symlink_to(archived)
    else:
        original.with_suffix(".pt.compressed-link").symlink_to(archived)
    before = original.read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        replace_with_verified_link(original, archived, expected)
    assert original.read_bytes() == before
