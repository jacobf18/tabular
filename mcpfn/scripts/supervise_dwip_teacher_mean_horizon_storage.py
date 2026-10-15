"""Preserve one live trainer while avoiding duplicate temporary-space reservations."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import signal
import time

import psutil

from audit_dwip_continuous_completion import save_new
from dwip_teacher_mean_horizon import QUEUE
from dwip_teacher_mean_training import read_json, verify_hashes
from run_dwip_budgeted_queue_v1 import regular_bytes
from run_dwip_teacher_mean_horizon import original_live
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import BENCH, REPO, WORK, write_json


def check_live_allocation(budget, free_bytes, allocated_bytes):
    if free_bytes < budget["reserve_bytes"] or allocated_bytes > budget["peak_additional_bytes"]:
        raise RuntimeError("Actual disk reserve or registered peak allocation violated")
    return {"free_bytes": free_bytes, "allocated_bytes": allocated_bytes,
            "reserve_bytes": budget["reserve_bytes"], "peak_additional_bytes": budget["peak_additional_bytes"]}


def verify_owned_controller(handle, root):
    if root.name != QUEUE or not original_live(handle):
        raise ValueError("Require this exact live horizon controller")
    process = psutil.Process(handle["pid"])
    command = process.cmdline()
    if command[1:] != ["scripts/run_dwip_teacher_mean_horizon.py", "run"] or process.cwd() != str(REPO):
        raise ValueError("Refuse to signal an unrelated controller")
    if process.environ().get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Only the CPU-only controller may be suspended")
    return process


def run(registration_path):
    registration = read_json(registration_path)
    verify_hashes(registration["input_sha256"])
    root = REPO / "runs/objective_queues" / QUEUE
    if read_json(root / "pipeline_status.json")["state"] != "training":
        raise ValueError("Supervision may only begin while the original trainer is running")
    controller = read_json(root / "controller_identity.json")
    training = read_json(root / "stage_handles/training.json")
    outer = read_json(root / "pipeline_launch.json")
    if not original_live(training) or not original_live(outer):
        raise ValueError("Require original live trainer launcher and24h controller timeout")
    process = verify_owned_controller(controller, root)
    if process.ppid() != outer["pid"] or psutil.Process(training["pid"]).ppid() != process.pid:
        raise ValueError("Registered original process ancestry changed")
    folder = root / "storage_supervision"
    identity = {"pid": os.getpid(), "created": psutil.Process().create_time()}
    save_new(folder / "started.json", {"supervisor": identity, "controller": controller, "training": training,
             "utc": datetime.now(timezone.utc).isoformat(), "registration_sha256": sha256(registration_path)})
    budget = read_json(root / "artifact_budget.json")
    work = WORK / f"tabimpute-{QUEUE}-dwip1e5"

    def interrupted(signum, frame):
        raise InterruptedError(f"Supervisor received{signum}; resume the same owned controller")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    suspended = False
    try:
        suspended = True
        os.kill(process.pid, signal.SIGSTOP)
        write_json(folder / "status.json", {"state": "supervising_training", "controller_suspended": True,
                   "trainer_suspended": False, "updated_unix": time.time()})
        while original_live(training):
            verify_owned_controller(controller, root)
            if not original_live(outer) or time.time() - outer["created"] >= 23.5 * 3600:
                raise TimeoutError("Resume the original controller before its unchanged24h cap")
            external = list((BENCH / "logs").glob(f"tabimpute_{QUEUE}_*.npz"))
            allocated = sum(regular_bytes(path) for path in (root, work, *external))
            storage = check_live_allocation(budget, shutil.disk_usage(WORK).free, allocated)
            write_json(folder / "storage_status.json", {**storage, "updated_unix": time.time(),
                       "checkpoint_prewrite_and_postwrite_guards": "unchanged in the running trainer"})
            time.sleep(2)
        verify_hashes(registration["input_sha256"])
        write_json(folder / "status.json", {"state": "training_exited_resuming_controller",
                   "updated_unix": time.time(), "goal_status": "active"})
    except BaseException as error:
        write_json(folder / "status.json", {"state": "error_resuming_controller", "error": repr(error),
                   "updated_unix": time.time()})
        raise
    finally:
        if suspended and original_live(controller):
            verify_owned_controller(controller, root)
            os.kill(controller["pid"], signal.SIGCONT)
            save_new(folder / "controller_resumed.json", {"controller": controller,
                     "utc": datetime.now(timezone.utc).isoformat(), "restart": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    run(args.registration)


if __name__ == "__main__":
    main()
