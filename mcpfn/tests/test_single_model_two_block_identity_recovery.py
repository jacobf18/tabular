from copy import deepcopy

import pytest

import recover_single_model_two_block_pipeline_identity as recovery


def identities():
    expected = {"pid": 101, "start_ticks": 202,
                "command": "bash -c set -e; python -u scripts/train_single_matrixformer_top_two_blocks.py --stage train; " + recovery.AUDITOR_COMMAND}
    observed = {**expected, "command": recovery.AUDITOR_COMMAND}
    current = {"pid": 101, "state": "initializing_independent_task_replay"}
    return expected, observed, current


def test_exact_exec_handoff_is_not_misclassified_as_a_stopped_process():
    expected, observed, current = identities()
    assert expected != observed
    assert (expected["pid"], expected["start_ticks"]) == (observed["pid"], observed["start_ticks"])
    recovery.validate_handoff(expected, observed, recovery.REPO, current)


@pytest.mark.parametrize("mutation", ["missing", "pid", "ticks", "command", "cwd", "state", "owner", "suffix", "trainer"])
def test_handoff_refuses_pid_reuse_substitution_and_unrelated_processes(mutation):
    expected, observed, current = identities()
    cwd = recovery.REPO
    if mutation == "missing":
        observed = None
    elif mutation == "pid":
        observed["pid"] += 1
    elif mutation == "ticks":
        observed["start_ticks"] += 1
    elif mutation == "command":
        observed["command"] += " --different-run"
    elif mutation == "cwd":
        cwd = recovery.REPO.parent
    elif mutation == "state":
        current["state"] = "failed"
    elif mutation == "owner":
        current["pid"] += 1
    elif mutation == "suffix":
        expected["command"] += "; different-worker"
    else:
        expected["command"] = "bash -c echo wrong; " + recovery.AUDITOR_COMMAND
    with pytest.raises(ValueError, match="same original process"):
        recovery.validate_handoff(expected, observed, cwd, current)


@pytest.mark.parametrize("mutation", [None, "error", "coordinator_live", "training_live", "updates", "selection", "state"])
def test_only_exact_failed_orchestration_and_finished_training_can_be_recovered(monkeypatch, mutation):
    failed = {"state": "failed", "error": recovery.FAILURE, "pid": 401}
    trained = {"state": "training_complete", "step": 64, "accepted_updates": 64, "best": {"step": 64}, "pid": 402}
    if mutation == "error":
        failed["error"] = "a real scientific failure"
    elif mutation == "updates":
        trained["accepted_updates"] = 63
    elif mutation == "selection":
        trained["best"]["step"] = 50
    elif mutation == "state":
        trained["state"] = "running"
    monkeypatch.setattr(recovery.original, "verify", lambda output: {"registered": True})
    monkeypatch.setattr(recovery, "read", lambda path: failed if path.parent == recovery.original.OUTPUT else trained)
    live = 401 if mutation == "coordinator_live" else 402 if mutation == "training_live" else None
    monkeypatch.setattr(recovery.original, "process_identity", lambda pid: {"pid": pid} if pid == live else None)
    if mutation is None:
        assert recovery.require_original_failure() == {"registered": True}
    else:
        with pytest.raises(ValueError, match="exact stopped coordinator"):
            recovery.require_original_failure()


@pytest.mark.parametrize("mutation", [None, "started", "payload", "campaign", "published"])
def test_registration_never_repeats_any_reserved_scientific_stage(tmp_path, monkeypatch, mutation):
    for name in ("REPLAY", "CONFIRMATION", "CAMPAIGN", "REVIEW"):
        monkeypatch.setattr(recovery.original, name, tmp_path / name)
    monkeypatch.setattr(recovery.original.training, "OUTPUT", tmp_path / "training")
    for directory in (recovery.original.REPLAY, recovery.original.CONFIRMATION):
        directory.mkdir()
        for name in ("config.json", "registration.json", "status.json"):
            recovery.write_json(directory / name, {"state": "registered"})
    if mutation == "started":
        recovery.write_json(recovery.original.REPLAY / "status.json", {"state": "awaiting_independent_audit"})
    elif mutation == "payload":
        (recovery.original.CONFIRMATION / "synthetic_bank.pt").touch()
    elif mutation == "campaign":
        recovery.original.CAMPAIGN.mkdir()
    elif mutation == "published":
        recovery.original.training.OUTPUT.mkdir()
        (recovery.original.training.OUTPUT / "best_validation_checkpoint.pth").touch()
    if mutation is None:
        recovery.require_unstarted_reservations()
    else:
        with pytest.raises(ValueError):
            recovery.require_unstarted_reservations()


def test_already_complete_original_audit_requires_full_prerequisite_without_restarting(monkeypatch):
    expected, observed, _ = identities()
    calls = []
    monkeypatch.setattr(recovery, "audit_state", lambda: {"state": "complete", "pid": expected["pid"]})
    monkeypatch.setattr(recovery.original, "prerequisite", lambda output: calls.append(output))
    monkeypatch.setattr(recovery.original, "process_identity", lambda pid: pytest.fail("No worker restart or live dependency"))
    recovery.wait_for_original_audit({"original_chain": expected, "observed_auditor": observed})
    assert calls == [recovery.original.OUTPUT]


def test_wait_follows_only_original_live_auditor_until_its_complete_audit(monkeypatch):
    expected, observed, current = identities()
    states = iter([current, {"state": "complete", "pid": expected["pid"]}])
    calls = []
    monkeypatch.setattr(recovery, "audit_state", lambda: next(states))
    monkeypatch.setattr(recovery.original, "process_identity", lambda pid: deepcopy(observed))
    monkeypatch.setattr(recovery, "validate_handoff", lambda *arguments: calls.append("handoff"))
    monkeypatch.setattr(recovery, "status", lambda *arguments, **keywords: calls.append("waiting"))
    monkeypatch.setattr(recovery.time, "sleep", lambda seconds: calls.append(seconds))
    monkeypatch.setattr(recovery.original, "prerequisite", lambda output: calls.append("full_audit"))
    recovery.wait_for_original_audit({"original_chain": expected, "observed_auditor": observed})
    assert calls == ["handoff", "waiting", 30, "full_audit"]


@pytest.mark.parametrize("mutation", ["replacement", "failed", "identity"])
def test_audit_wait_failure_never_falls_back_or_launches_another_worker(monkeypatch, mutation):
    expected, observed, current = identities()
    config = {"original_chain": expected, "observed_auditor": observed}
    if mutation == "replacement":
        current["pid"] += 1
    elif mutation == "failed":
        current["state"] = "failed"
    else:
        config["observed_auditor"] = {**observed, "start_ticks": 203}
    monkeypatch.setattr(recovery, "audit_state", lambda: current)
    monkeypatch.setattr(recovery.original, "process_identity", lambda pid: observed)
    monkeypatch.setattr(recovery, "validate_handoff", lambda *arguments: None)
    monkeypatch.setattr(recovery.time, "sleep", lambda seconds: pytest.fail("Never wait through an authoritative failure"))
    with pytest.raises(RuntimeError):
        recovery.wait_for_original_audit(config)


@pytest.mark.parametrize("passed", [False, True])
def test_orchestration_uses_original_stages_gates_locks_capacity_and_full_real_commands(tmp_path, monkeypatch, passed):
    monkeypatch.setattr(recovery, "OUTPUT", tmp_path / "recovery")
    monkeypatch.setattr(recovery, "verify", lambda: {})
    monkeypatch.setattr(recovery, "wait_for_original_audit", lambda config: None)
    monkeypatch.setattr(recovery, "sha256", lambda path: "preserved-hash")
    monkeypatch.setattr(recovery, "status", lambda *arguments, **keywords: None)
    monkeypatch.setattr(recovery, "read", lambda path: {"state": "awaiting_independent_audit"}
                        if path.name == "status.json" else {"passed": passed})
    children = []
    monkeypatch.setattr(recovery.original, "run_child", lambda *arguments, **keywords: children.append((arguments, keywords)))
    recovery.run()
    labels = [arguments[1] for arguments, keywords in children]
    assert labels == ["replay", "audit-replay", "confirm", "audit-confirmation"] + (["benchmarks", "review"] if passed else [])
    for arguments, keywords in children[:4]:
        assert arguments[0] == recovery.OUTPUT
        assert arguments[2] == recovery.Path(recovery.original.__file__).resolve()
        assert arguments[3] == ["--stage", arguments[1], "--output", str(recovery.original.OUTPUT)]
        assert keywords["cpu"] == arguments[1].startswith("audit-")
    if passed:
        assert children[4][0][2] == recovery.REPO / "scripts/run_single_matrixformer_cached_benchmarks.py"
        assert children[4][0][3] == ["--run", str(recovery.original.training.OUTPUT), "--confirmation", str(recovery.original.CONFIRMATION),
            "--source", str(recovery.original.PARENT_CACHE), "--output", str(recovery.original.CAMPAIGN)]
        assert children[5][0][2] == recovery.REPO / "scripts/review_single_matrixformer_campaign.py"
    result = recovery.original.read(recovery.OUTPUT / "result.json")
    assert result["scientific_stages_repeated"] is False
    assert result["original_failed_pipeline"] == str(recovery.original.OUTPUT)
    assert not (recovery.OUTPUT / "best_validation_checkpoint.pth").exists()
