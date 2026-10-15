import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch

import recover_single_model_stronger_mean_idle_owner as recovery


@pytest.mark.parametrize("hidden,initialized", [(True, False), (False, True)])
def test_coordinator_rejects_hidden_children_or_own_cuda_context(monkeypatch, hidden, initialized):
    if hidden:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    else:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    with pytest.raises(ValueError, match="CUDA-uninitialized"):
        recovery.require_cpu_coordinator()


def test_coordinator_admission_does_not_touch_cuda(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("Do not initialize CUDA in a launcher"))
    recovery.require_cpu_coordinator()


def test_real_cuda_smoke_in_exited_child_leaves_parent_uninitialized():
    assert not torch.cuda.is_initialized()
    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    source = ("import json,os,torch; value=torch.ones(1,device='cuda'); torch.cuda.synchronize(); "
              "print(json.dumps({'pid':os.getpid(),'value':value.item(),'initialized':torch.cuda.is_initialized()}))")
    result = subprocess.run([sys.executable, "-c", source], env=environment,
                            check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["value"] == 1 and report["initialized"] is True
    assert not torch.cuda.is_initialized()
    assert not Path(f'/proc/{report["pid"]}').exists()
    assert report["pid"] not in recovery.gpu_owners()
    assert os.getpid() not in recovery.gpu_owners()


def test_process_identity_binds_current_process_and_missing_owner():
    current = recovery.identity(os.getpid())
    assert current["pid"] == os.getpid()
    assert current["cwd"] == os.getcwd()
    assert current["start_ticks"].isdigit()
    assert recovery.identity(99999999) is None


def test_static_recovery_globals_are_resolved():
    import inspect

    for value in vars(recovery).values():
        if inspect.isfunction(value) and value.__globals__ is vars(recovery):
            assert recovery.previous.unresolved_globals(value) == []


@pytest.fixture
def pair(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "OUTPUT", tmp_path)
    coordinator = {"pid": 100, "start_ticks": "123", "command": ["python"], "cwd": str(tmp_path)}
    child = {"pid": 101, "start_ticks": "124", "command": ["python"], "cwd": str(tmp_path)}
    states = {100: coordinator, 101: child}
    monkeypatch.setattr(recovery, "identity", lambda pid: states.get(pid))
    monkeypatch.setattr(recovery, "require_waiting_pair", lambda parent, current: None)
    monkeypatch.setattr(recovery.os, "pidfd_open", lambda pid: 12345)
    monkeypatch.setattr(recovery.os, "close", lambda descriptor: None)
    monkeypatch.setattr(recovery, "gpu_owners", lambda: [100] if 100 in states else [])
    return {"coordinator": coordinator, "child": child}, states


def test_release_signals_only_checked_coordinator_and_retains_child(pair, monkeypatch):
    config, states = pair
    signals = []

    def send(descriptor, number):
        signals.append((descriptor, number))
        states.pop(100)

    monkeypatch.setattr(recovery.signal, "pidfd_send_signal", send)
    recovery.release_parent(config)
    assert signals == [(12345, signal.SIGTERM)]
    assert states == {101: config["child"]}
    report = recovery.original.read(recovery.OUTPUT / "release.json")
    assert report["child_retained"] == config["child"]
    assert report["scientific_work_repeated"] is False


def test_reused_parent_pid_is_never_signaled(pair, monkeypatch):
    config, states = pair
    states[100] = {**states[100], "start_ticks": "replacement"}
    monkeypatch.setattr(recovery.signal, "pidfd_send_signal", lambda *args: pytest.fail("Never signal another owner"))
    with pytest.raises(ValueError, match="substituted"):
        recovery.release_parent(config)


@pytest.mark.parametrize("passed", [False, True])
def test_no_relaunch_of_replay_and_complete_unchanged_downstream_scope(tmp_path, monkeypatch, passed):
    monkeypatch.setattr(recovery, "OUTPUT", tmp_path)
    monkeypatch.setattr(recovery.original, "CONFIRMATION", tmp_path / "confirmation")
    recovery.original.CONFIRMATION.mkdir()
    monkeypatch.setattr(recovery, "verify", lambda: {"child": {"pid": 101}})
    events = []
    monkeypatch.setattr(recovery, "release_parent", lambda config: events.append("release"))
    monkeypatch.setattr(recovery, "wait_for_original_replay", lambda config: events.append("observe"))
    monkeypatch.setattr(recovery, "sha256", lambda path: "digest")
    calls = []

    def child(output, label, script, arguments, cpu=False):
        calls.append((label, script, arguments, cpu))
        if label == "confirm":
            recovery.write_json(recovery.original.CONFIRMATION / "status.json", {"state": "awaiting_independent_audit"})
        if label == "audit-confirmation":
            recovery.write_json(recovery.original.CONFIRMATION / "gate.json", {"passed": passed})

    monkeypatch.setattr(recovery.original, "run_child", child)
    recovery.run()
    assert events == ["release", "observe"]
    assert [entry[0] for entry in calls] == ["audit-replay", "confirm", "audit-confirmation"] + (
        ["benchmarks", "review"] if passed else [])
    assert all(entry[1] == Path(recovery.previous.__file__).resolve() for entry in calls[:3])
    assert [entry[3] for entry in calls[:3]] == [True, False, True]
    assert not recovery.original.read(tmp_path / "result.json")["cuda_initialized"]
