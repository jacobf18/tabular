import pytest

import archive_single_model_replicated_evidence as evidence


def test_only_declared_artifacts_are_allowed_and_shared_floors_are_unchanged():
    with pytest.raises(ValueError, match="explicitly declared"):
        evidence.plan("arbitrary")
    assert evidence.archive.MINIMUM_FREE == 21 * 1024**3
    assert evidence.idle.idle_compressor().__globals__["MINIMUM_FREE"] == 18 * 1024**3


@pytest.mark.parametrize("kind", ["preflight", "endpoint"])
@pytest.mark.parametrize("state,pid,complete", [("capturing_gradients", 999999999, True),
                                               ("terminal", 1, True), ("terminal", 999999999, False)])
def test_live_or_unaudited_sources_cannot_move(monkeypatch, kind, state, pid, complete):
    expected = "complete" if kind == "preflight" else "training_complete"
    monkeypatch.setattr(evidence, "read", lambda path: {"state": expected if state == "terminal" else state, "pid": pid}
                        if path.name == "status.json" else {"complete": complete})
    with pytest.raises(ValueError, match="live, incomplete or unaudited"):
        evidence.plan(kind)
