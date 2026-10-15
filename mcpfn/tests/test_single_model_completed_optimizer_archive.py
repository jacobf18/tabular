import pytest

import archive_single_model_completed_optimizer_evidence as evidence


def test_new_archive_keeps_the_verified_compressor_and_original_floor_unchanged():
    compressor = evidence.idle_compressor()
    assert compressor.__code__ is evidence.archive.compress_group.__code__
    assert compressor.__globals__["MINIMUM_FREE"] == 18 * 1024**3
    assert evidence.archive.MINIMUM_FREE == 21 * 1024**3
    assert evidence.OUTPUT == evidence.ROOT / "completed_optimizer_archive_v1"
    assert '"canonical_artifacts_checked": 6' in evidence.ADAPTED_SOURCE
    assert '"canonical_artifacts": 6' in evidence.ADAPTED_SOURCE


@pytest.mark.parametrize("state,pid", [("initializing_decoder_training", 999999999), ("complete", 1)])
def test_archive_refuses_live_or_nonterminal_preflight_sources(monkeypatch, state, pid):
    monkeypatch.setattr(evidence, "read", lambda path: {"state": state, "pid": pid})
    with pytest.raises(ValueError, match="terminal idle"):
        evidence.require_completed()


@pytest.mark.parametrize("field,value", [("complete", False), ("training_tasks", 13), ("prediction_grids", 27)])
def test_archive_requires_complete_independent_preflight_evidence(monkeypatch, field, value):
    report = {"complete": True, "training_tasks": 14, "prediction_grids": 28}
    report[field] = value
    monkeypatch.setattr(evidence, "read", lambda path: report)
    with pytest.raises(ValueError, match="completed independent audit"):
        evidence.completed_inventory()
