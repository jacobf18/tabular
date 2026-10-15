import pytest

import archive_single_model_early_banks as archive


@pytest.mark.parametrize("state,pid", [("training", 999999999), ("training_complete", 1)])
def test_early_sources_must_be_terminal_and_idle(monkeypatch, state, pid):
    monkeypatch.setattr(archive, "read", lambda path: {"state": state, "pid": pid})
    with pytest.raises(ValueError, match="terminal idle"):
        archive.require_terminal()


def test_scope_does_not_include_current_or_future_trial():
    assert len(archive.RUNS) == 5
    assert len(archive.CONFIRMATION_RUNS) == 2
    assert all("v20" not in name for name in archive.RUNS)
    assert archive.MINIMUM_FREE == 18 * 1024**3
    assert "not a new independent performance audit" in archive.ADAPTED_SOURCE
