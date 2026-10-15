import json

import pytest

import run_single_matrixformer_benchmarks as campaign


def test_wait_requires_a_live_confirmation_handle(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(json.dumps({"state": "waiting_for_training", "pid": 123}))
    checked = []
    monkeypatch.setattr(campaign.os, "kill", lambda process, signal: checked.append((process, signal)))
    assert campaign.confirmation_state(tmp_path) is None
    assert checked == [(123, 0)]


def test_stale_confirmation_status_is_not_a_verified_wait(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(json.dumps({"state": "evaluating", "pid": 123}))

    def missing(process, signal):
        raise ProcessLookupError()

    monkeypatch.setattr(campaign.os, "kill", missing)
    with pytest.raises(RuntimeError, match="missing"):
        campaign.confirmation_state(tmp_path)


def test_complete_confirmation_returns_gate_without_requiring_old_process(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"state": "complete"}))
    gate = {"passed": False, "reason": "step zero"}
    (tmp_path / "gate.json").write_text(json.dumps(gate))
    assert campaign.confirmation_state(tmp_path) == gate


@pytest.mark.parametrize("stage,audit", (("tabular", {"complete": True, "cases": 462}),
                                        ("llm", {"complete": True, "fold_settings": 180, "panels": 180}),
                                        ("recsys", {"complete": True, "cases": []})))
def test_partial_benchmarks_cannot_complete_campaign(stage, audit):
    with pytest.raises(ValueError):
        campaign.require_full_audit(stage, audit)


def test_campaign_accepts_only_full_separate_ratings_roles():
    cases = [{"dataset": dataset, "role": role} for dataset in ("100K", "netflix", "1M", "10M")
             for role in ("student", "parent_same_recipe")]
    campaign.require_full_audit("recsys", {"complete": True, "cases": cases})
    campaign.require_full_audit("tabular", {"complete": True, "cases": 2262})
    campaign.require_full_audit("llm", {"complete": True, "fold_settings": 1800, "panels": 180})
    with pytest.raises(ValueError):
        campaign.require_full_audit("recsys", {"complete": True, "cases": cases[:-1] + cases[:1]})
