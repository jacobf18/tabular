from copy import deepcopy

import pytest

import preflight_single_model_targeted_decoder as original
import preflight_single_model_v8_decoder as preflight


def test_parent_is_v8_not_an_unqualified_tail_or_probe_endpoint():
    assert preflight.PARENT == preflight.training.alignment.fixed.PARENT
    assert preflight.PARENT.parent.name == "query_budget_v8"
    assert original.PARENT.parent.name == "decoder_repair_v14"
    assert preflight.WAIT_FOR.name == "stronger_decoder_v16_pipeline"
    assert preflight.EVIDENCE.name == "targeted_decoder_v15_progress_100"
    assert preflight.RATE == 2e-5 and original.RATE == 2e-6


def test_prior_and_objective_inputs_do_not_change_with_the_new_backbone():
    current, previous = preflight.batch(), original.batch()
    restored = deepcopy(current)
    for task, reference in zip(restored["tasks"], previous["tasks"]):
        task["gradient_seed"] = reference["gradient_seed"]
        task["specification"].get("structured", task["specification"])["seed"] = reference["specification"].get("structured", reference["specification"])["seed"]
    assert restored == previous
    seeds = preflight.training.alignment.fixed.seeds_in
    assert len(seeds(current)) == 28
    assert seeds(current).isdisjoint(seeds(previous))
    assert seeds(current).isdisjoint(seeds([preflight.preceding.training.batches(), preflight.preceding.training.confirmation_cases()]))
    assert preflight.training is original.training


@pytest.mark.parametrize("qualified,expected", [(True, "skip_qualified_v16"), (False, "run_preflight")])
def test_wait_decision_never_interferes_with_a_qualified_model(qualified, expected):
    assert preflight.after_v16({"complete": True, "gate": {"passed": qualified}}) == expected
    with pytest.raises(ValueError, match="completed unchanged"):
        preflight.after_v16({"complete": False, "gate": {"passed": qualified}})


def test_no_changed_direction_metric_or_optimizer_audit_in_the_new_preflight():
    source = preflight.ADAPTED_SOURCE
    assert 'training.guarded.protected_update(view, optimizer, gradients[:len(entries)]' in source
    assert 'checks.check_endpoint(payload, record' in source
    assert 'compare_nested(decision, record["finite"])' in source
    assert 'if read(output / "status.json")["state"] != "ready_for_preflight"' in source
    assert '"preflight_only": True' in source
    assert '"selection_qualified": False' in source
    assert '"parent_selection_qualified": False' not in source


def test_artifact_reserve_is_checked_again_after_wait(monkeypatch):
    class Usage:
        free = 22 * 1024**3 - 1

    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda path: Usage())
    with pytest.raises(OSError, match="18 GiB"):
        preflight.require_space()
    Usage.free += 1
    preflight.require_space()


@pytest.mark.parametrize("qualified", [True, False])
def test_actual_waiter_checks_live_handle_and_never_runs_before_terminal_gate(monkeypatch, tmp_path, qualified):
    output = tmp_path / "preflight"
    events = []
    pending = iter([{"state": "waiting_for_training", "pid": 42}, {"state": "complete", "pid": 42}])

    def read(path):
        if path == output / "status.json":
            return {"state": "registered"}
        if path == preflight.WAIT_FOR / "status.json":
            return next(pending)
        if path == preflight.WAIT_FOR / "result.json":
            return {"complete": True, "gate": {"passed": qualified}}
        if path == preflight.preceding.REPLAY / "audit.json":
            return {"complete": True, "artifact_sha256": {}}
        raise AssertionError(path)

    def handle(pid, start):
        assert pid == 42 and start == "original"
        events.append("live")

    monkeypatch.setattr(preflight, "verify", lambda path: {"wait_handle": {"pid": 42, "start_ticks": "original"}})
    monkeypatch.setattr(preflight, "read", read)
    monkeypatch.setattr(preflight, "pipeline_handle", handle)
    monkeypatch.setattr(preflight.time, "sleep", lambda seconds: events.append("wait"))
    monkeypatch.setattr(preflight, "status", lambda path, state, **fields: events.append(state))
    monkeypatch.setattr(preflight.preceding, "verify", lambda path: events.append("verify_pipeline"))
    monkeypatch.setattr(preflight.preceding, "prerequisite", lambda path: events.append("full_audit"))
    monkeypatch.setattr(preflight, "verify_hashes", lambda hashes: events.append("replay_audit"))
    monkeypatch.setattr(preflight, "write_json", lambda path, data: events.append("wait_evidence"))
    monkeypatch.setattr(preflight, "sha256", lambda path: "hash")
    monkeypatch.setattr(preflight, "require_space", lambda: events.append("space"))
    monkeypatch.setattr(preflight, "run", lambda path: events.append("run"))
    preflight.wait_and_run(output)
    assert events[:5] == ["live", "waiting_for_v16", "wait", "verify_pipeline", "wait_evidence"]
    if qualified:
        assert events[5:] == ["skipped_qualified_v16"]
    else:
        assert events[5:] == ["full_audit", "replay_audit", "space", "ready_for_preflight", "run"]


def test_failed_pipeline_cannot_be_bypassed_by_the_new_preflight(monkeypatch, tmp_path):
    output = tmp_path / "preflight"
    monkeypatch.setattr(preflight, "verify", lambda path: {"wait_handle": {"pid": 42, "start_ticks": "original"}})
    monkeypatch.setattr(preflight, "read", lambda path: {"state": "registered" if path == output / "status.json" else "failed"})
    monkeypatch.setattr(preflight, "run", lambda path: pytest.fail("Never start a model after an unaudited failure"))
    with pytest.raises(RuntimeError, match="do not bypass"):
        preflight.wait_and_run(output)
