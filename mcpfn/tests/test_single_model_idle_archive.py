import pytest

import extend_single_model_idle_archive as extension


def test_idle_reserve_does_not_mutate_original_archive_policy():
    original = extension.archive.MINIMUM_FREE
    compressor = extension.idle_group_compressor()
    assert extension.MINIMUM_FREE == 18 * 1024**3
    assert extension.GROUP_SIZE == 2
    assert compressor.__globals__["MINIMUM_FREE"] == extension.MINIMUM_FREE
    assert extension.archive.MINIMUM_FREE == original == 21 * 1024**3
    assert compressor.__code__ is extension.archive.compress_group.__code__
    assert compressor.__globals__["replace_with_verified_link"] is extension.archive.replace_with_verified_link
    assert compressor.__globals__["mount_unit"] is extension.archive.mount_unit


def test_active_writer_blocks_archive_even_with_terminal_state(monkeypatch):
    monkeypatch.setattr(extension, "read", lambda path: {"state": "complete", "pid": 1})
    with pytest.raises(ValueError, match="handles absent"):
        extension.require_idle()


def test_nonterminal_writer_blocks_archive(monkeypatch):
    monkeypatch.setattr(extension, "read", lambda path: {"state": "running", "pid": 999999999})
    with pytest.raises(ValueError, match="terminal"):
        extension.require_idle()
