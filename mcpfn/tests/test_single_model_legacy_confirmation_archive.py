import pytest

import archive_single_model_legacy_confirmation_banks as archive


@pytest.mark.parametrize("state,pid", [("building", 999999999), ("complete", 1)])
def test_never_archive_active_confirmation(monkeypatch, state, pid):
    monkeypatch.setattr(archive, "read", lambda path: {"state": state, "pid": pid})
    with pytest.raises(ValueError, match="completed idle"):
        archive.require_terminal()


def test_bounded_inventory_and_unchanged_reserve():
    assert len(archive.RUNS) == 4
    assert len(archive.ALL_RUNS) == 7 and len(archive.DEVELOPMENT_RUNS) == 3
    assert all("confirmation" in name for name in archive.RUNS)
    assert all("v20" not in name for name in archive.RUNS)
    assert archive.MINIMUM_FREE == 18 * 1024**3
    assert archive.archive.MINIMUM_FREE == 21 * 1024**3
