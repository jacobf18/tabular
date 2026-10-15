from types import SimpleNamespace

import pytest

import supervise_dwip_teacher_mean_horizon_storage as supervisor
from run_dwip_budgeted_queue_v1 import check_storage


def budget():
    return {"reserve_bytes": 21474836480, "peak_additional_bytes": 2433744896, "atomic_temporary_bytes": 985661440}


def test_original_poll_can_double_reserve_an_in_progress_temporary():
    allocated = 312107295 + 2 * 936303027
    free = 24016000000 - allocated
    with pytest.raises(RuntimeError, match="storage guard"):
        check_storage(budget(), free, allocated)
    assert supervisor.check_live_allocation(budget(), free, allocated)["free_bytes"] > budget()["reserve_bytes"]


@pytest.mark.parametrize("free,allocated", ((21474836479, 100), (22000000000, 2433744897)))
def test_true_reserve_and_peak_violations_are_still_rejected(free, allocated):
    with pytest.raises(RuntimeError, match="Actual disk reserve"):
        supervisor.check_live_allocation(budget(), free, allocated)


@pytest.mark.parametrize("change", ("queue", "identity", "command", "cwd", "gpu"))
def test_unrelated_or_reused_processes_cannot_be_suspended(monkeypatch, change):
    root = supervisor.REPO / "runs/objective_queues" / supervisor.QUEUE
    process = SimpleNamespace(cmdline=lambda: ["python", "scripts/run_dwip_teacher_mean_horizon.py", "run"],
                              cwd=lambda: str(supervisor.REPO), environ=lambda: {"CUDA_VISIBLE_DEVICES": ""})
    monkeypatch.setattr(supervisor, "original_live", lambda handle: change != "identity")
    monkeypatch.setattr(supervisor.psutil, "Process", lambda pid: process)
    if change == "queue":
        root = root.parent / "other"
    elif change == "command":
        process.cmdline = lambda: ["python", "another.py"]
    elif change == "cwd":
        process.cwd = lambda: "/tmp"
    elif change == "gpu":
        process.environ = lambda: {"CUDA_VISIBLE_DEVICES": "0"}
    with pytest.raises(ValueError):
        supervisor.verify_owned_controller({"pid": 42}, root)


def test_only_exact_live_cpu_controller_is_accepted(monkeypatch):
    root = supervisor.REPO / "runs/objective_queues" / supervisor.QUEUE
    process = SimpleNamespace(cmdline=lambda: ["python", "scripts/run_dwip_teacher_mean_horizon.py", "run"],
                              cwd=lambda: str(supervisor.REPO), environ=lambda: {"CUDA_VISIBLE_DEVICES": ""})
    monkeypatch.setattr(supervisor, "original_live", lambda handle: True)
    monkeypatch.setattr(supervisor.psutil, "Process", lambda pid: process)
    assert supervisor.verify_owned_controller({"pid": 42}, root) is process


def test_error_after_suspend_always_resumes_same_controller(tmp_path, monkeypatch):
    root = tmp_path / "runs/objective_queues" / supervisor.QUEUE
    (root / "storage_supervision").mkdir(parents=True)
    records = {"registration.json": {"input_sha256": {}}, "pipeline_status.json": {"state": "training"},
               "controller_identity.json": {"pid": 100, "created": 1.}, "training.json": {"pid": 200, "created": 2.},
               "pipeline_launch.json": {"pid": 300, "created": 0.}, "artifact_budget.json": budget()}
    monkeypatch.setattr(supervisor, "REPO", tmp_path)
    monkeypatch.setattr(supervisor, "read_json", lambda path: records[path.name])
    monkeypatch.setattr(supervisor, "verify_hashes", lambda values: None)
    monkeypatch.setattr(supervisor, "sha256", lambda path: "frozen")
    monkeypatch.setattr(supervisor, "original_live", lambda handle: True)
    controller = SimpleNamespace(pid=100, ppid=lambda: 300)
    monkeypatch.setattr(supervisor, "verify_owned_controller", lambda handle, path: controller)
    monkeypatch.setattr(supervisor.psutil, "Process", lambda pid=None: SimpleNamespace(ppid=lambda: 100, create_time=lambda: 3.))
    monkeypatch.setattr(supervisor.signal, "signal", lambda *args: None)
    signals = []
    monkeypatch.setattr(supervisor.os, "kill", lambda pid, signum: signals.append((pid, signum)))
    with pytest.raises(TimeoutError, match="24h cap"):
        supervisor.run(tmp_path / "registration.json")
    assert signals == [(100, supervisor.signal.SIGSTOP), (100, supervisor.signal.SIGCONT)]
    assert (root / "storage_supervision/controller_resumed.json").exists()
