"""Release only V46's idle CUDA-owning coordinator and retain its live replay."""

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

import torch

import recover_single_model_stronger_mean_cuda_entry as previous
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


original = previous.original
OUTPUT = original.ROOT / "stronger_mean_distillation_v46_idle_owner_recovery_v1"
STAGES = (("audit-replay", True), ("confirm", False))
POLICY = (
    "Preserve the completed V46 recovered audit and original live endpoint-replay worker. "
    "The first recovery coordinator retained a616 MiB CUDA context after a scalar hardware "
    "smoke test, so the unchanged child's exclusive-GPU admission waits for its own parent. "
    "Register exact stopped-science prefix and parent/child process identities before sending "
    "SIGTERM only to that idle coordinator through a checked pidfd. Never signal or relaunch "
    "the child. Preserve the old coordinator status/config/log and bind its interruption here. "
    "Observe the same child to completion, then run only the remaining original qualification "
    "and conditional full benchmark/review stages once through the previous immutable adapter. "
    "The new coordinator must never initialize CUDA; hardware smoke tests must run in exited "
    "subprocesses, not persistent launchers. No GPU reset, admission bypass, model/task retry, "
    "training, reselection, new seeds, changed tolerance, inference ensemble or success claim."
)


def identity(pid):
    directory = Path(f"/proc/{pid}")
    try:
        fields = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": fields[19],
                "command": (directory / "cmdline").read_bytes().decode().rstrip("\0").split("\0"),
                "cwd": str((directory / "cwd").resolve())}
    except FileNotFoundError:
        return None


def require_cpu_coordinator():
    if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise ValueError("Keep the coordinator CUDA-uninitialized and GPU children visible")


def gpu_owners():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    return sorted(int(line.strip()) for line in result.stdout.splitlines() if line.strip())


def require_waiting_pair(coordinator, child):
    if coordinator is None or child is None or coordinator["pid"] == child["pid"]:
        raise ValueError("Require the two distinct original live owners")
    script = Path(previous.__file__).resolve()
    for current, stage in ((coordinator, "qualify"), (child, "replay")):
        if (len(current["command"]) != 5 or current["command"][1] != "-u"
                or Path(current["cwd"], current["command"][2]).resolve() != script
                or current["command"][3:] != ["--stage", stage]):
            raise ValueError("Bind the exact original recovery coordinator and scientific child")
    parent_state = original.read(previous.OUTPUT / "status.json")
    child_state = original.read(original.REPLAY / "status.json")
    if (parent_state["state"] != "replay" or parent_state["pid"] != coordinator["pid"]
            or parent_state["child_pid"] != child["pid"] or child_state["state"] != "waiting_for_gpu"
            or child_state["pid"] != child["pid"] or gpu_owners() != [coordinator["pid"]]):
        raise ValueError("Only the unchanged child waiting for its sole idle CUDA-owning parent is admitted")
    fields = Path(f'/proc/{child["pid"]}/stat').read_text().rsplit(") ", 1)[1].split()
    if int(fields[1]) != coordinator["pid"]:
        raise ValueError("Preserve the actual parent-child relation before releasing the parent")
    if set(path.name for path in original.REPLAY.iterdir()) != {"config.json", "registration.json", "status.json", "run.lock"}:
        raise ValueError("The waiting original replay must have no scientific output yet")
    if (set(path.name for path in original.CONFIRMATION.iterdir()) != {"config.json", "registration.json", "status.json"}
            or original.read(original.CONFIRMATION / "status.json")["state"] != "registered"
            or original.CAMPAIGN.exists() or original.REVIEW.exists()
            or (original.training.OUTPUT / "best_validation_checkpoint.pth").exists()):
        raise ValueError("Preserve every unstarted downstream original stage")


def register(coordinator_pid, child_pid):
    if OUTPUT.exists():
        raise FileExistsError("Never replace the idle-owner recovery")
    require_cpu_coordinator()
    previous.verify(previous.OUTPUT)
    previous.require_completed_audit()
    coordinator, child = identity(coordinator_pid), identity(child_pid)
    require_waiting_pair(coordinator, child)
    sources = [Path(__file__).resolve(), REPO / "tests/test_single_model_stronger_mean_idle_owner.py"]
    paths = [previous.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "worker.log")]
    paths.extend(previous.ENTRY / name for name in ("config.json", "registration.json", "status.json", "result.json"))
    paths.append(previous.AUDIT / "audit.json")
    config = {"policy": POLICY, "coordinator": coordinator, "child": child,
              "audit": str(previous.AUDIT), "qualification_stages": [list(stage) for stage in STAGES],
              "input_sha256": {str(path): sha256(path) for path in paths},
              "source_sha256": {str(path): sha256(path) for path in sources},
              "original_child_status_at_registration": original.read(original.REPLAY / "status.json"),
              "original_child_log_sha256_at_registration": sha256(previous.OUTPUT / "replay.log")}
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json"),
               "registered_unix": time.time(), "signal_sent": False, "scientific_work_repeated": False})
    original.status(OUTPUT, "registered")


def verify():
    require_cpu_coordinator()
    config = original.read(OUTPUT / "config.json")
    if (original.read(OUTPUT / "registration.json")["config_sha256"] != sha256(OUTPUT / "config.json")
            or config["policy"] != POLICY or config["audit"] != str(previous.AUDIT)
            or config["qualification_stages"] != [list(stage) for stage in STAGES]):
        raise ValueError("Preserve the exact idle-owner-only operation")
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    previous.verify(previous.OUTPUT)
    previous.require_completed_audit()
    return config


def release_parent(config):
    expected, child = config["coordinator"], config["child"]
    descriptor = os.pidfd_open(expected["pid"])
    try:
        if identity(expected["pid"]) != expected or identity(child["pid"]) != child:
            raise ValueError("Never signal a substituted or reused process")
        require_waiting_pair(expected, child)
        write_json(OUTPUT / "signal_intent.json", {"target": expected, "signal": "SIGTERM", "child_signaled": False})
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
    finally:
        os.close(descriptor)
    deadline = time.monotonic() + 60
    while identity(expected["pid"]) is not None:
        if time.monotonic() > deadline:
            raise RuntimeError("The idle parent did not stop; do not escalate or repeat the signal")
        time.sleep(1)
    if identity(child["pid"]) != child or expected["pid"] in gpu_owners():
        raise RuntimeError("Require the exact surviving worker and release of its parent's CUDA context")
    write_json(OUTPUT / "release.json", {"coordinator_stopped": True, "child_retained": child,
               "scientific_work_repeated": False, "original_coordinator_status_preserved": True})


def wait_for_original_replay(config):
    child = config["child"]
    while True:
        current = original.read(original.REPLAY / "status.json")
        observed = identity(child["pid"])
        if current["pid"] != child["pid"] or current["state"] == "failed":
            raise RuntimeError("The original replay failed or was replaced; never restart it")
        if observed is None:
            if current["state"] != "awaiting_independent_audit" or not (original.REPLAY / "result.json").is_file():
                raise RuntimeError("Original replay stopped without its complete result")
            return
        if observed != child:
            raise RuntimeError("Never adopt a substituted replay worker")
        original.status(OUTPUT, "observing_original_replay", child_pid=child["pid"], replay_state=current["state"])
        time.sleep(30)


def run():
    config = verify()
    release_parent(config)
    wait_for_original_replay(config)
    for stage, cpu in STAGES:
        verify()
        original.run_child(OUTPUT, stage, Path(previous.__file__).resolve(), ["--stage", stage], cpu=cpu)
    if original.read(original.CONFIRMATION / "status.json")["state"] == "awaiting_independent_audit":
        original.run_child(OUTPUT, "audit-confirmation", Path(previous.__file__).resolve(), ["--stage", "audit-confirmation"], cpu=True)
    gate = original.read(original.CONFIRMATION / "gate.json")
    if gate["passed"]:
        original.run_child(OUTPUT, "benchmarks", REPO / "scripts/run_single_matrixformer_cached_benchmarks.py",
                           ["--run", str(original.training.OUTPUT), "--confirmation", str(original.CONFIRMATION),
                            "--source", str(original.PARENT_CACHE), "--output", str(original.CAMPAIGN)])
        original.run_child(OUTPUT, "review", REPO / "scripts/review_single_matrixformer_campaign.py",
                           ["--campaign", str(original.CAMPAIGN), "--output", str(original.REVIEW)], cpu=True)
    verify()
    write_json(OUTPUT / "result.json", {"complete": True, "gate": gate,
               "audit_sha256": sha256(previous.AUDIT / "audit.json"), "original_replay_child": config["child"],
               "training_repeated": False, "scientific_stages_repeated": False,
               "campaign": str(original.CAMPAIGN) if gate["passed"] else None,
               "review": str(original.REVIEW) if gate["passed"] else None,
               "cuda_initialized": torch.cuda.is_initialized(), "goal_success_not_assessed": True})
    original.status(OUTPUT, "complete", confirmed=gate["passed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    parser.add_argument("--coordinator-pid", type=int)
    parser.add_argument("--child-pid", type=int)
    arguments = parser.parse_args()
    with original.arithmetic_recovery.previous.original_arithmetic():
        if arguments.stage == "register":
            if arguments.coordinator_pid is None or arguments.child_pid is None:
                raise ValueError("Registration requires both exact live owner PIDs")
            register(arguments.coordinator_pid, arguments.child_pid)
            return
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if original.read(OUTPUT / "status.json")["state"] != "registered":
                raise ValueError("Never restart the idle-owner recovery")
            try:
                run()
            except BaseException as error:
                original.status(OUTPUT, "failed", error=repr(error))
                raise


if __name__ == "__main__":
    main()
