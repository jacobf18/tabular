"""Confirm the prospective column-aligned recipe on its complete fresh synthetic bank."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

import torch

from confirm_single_matrixformer_continuation import training_ready
from recsys_evaluation_common import sha256, write_json
import single_model_column_objective as column
from train_single_matrixformer import setup, storage_guard
from train_single_matrixformer_column_aligned import validate_recipe
from train_single_matrixformer_continuation import read, verify_run
from train_streamed_large_matrices import wait_for_gpu


@torch.no_grad()
def confirm(run, output):
    config = read(run / "config.json")
    verify_run(config, run)
    validate_recipe(config)
    if read(run / "status.json")["state"] != "training_complete":
        raise ValueError("Only completed training may enter independent confirmation")
    checkpoint = run / "best_validation_checkpoint.pth"
    fingerprint = sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["config_sha256"] != sha256(run / "config.json"):
        raise ValueError("Checkpoint/config mismatch")
    manifest = {"checkpoint": str(checkpoint), "checkpoint_sha256": fingerprint,
                "config_sha256": state["config_sha256"], "selected_step": state["step"],
                "seed_start": config["confirmation_seed"], "cases": config["confirmation_cases"],
                "source_sha256": sha256(Path(__file__)), "training_performed": False,
                "policy": config["confirmation"], "purpose": "Independent synthetic screen, not benchmark parity"}
    write_json(output / "manifest.json", manifest)
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    if state["step"] == 0:
        write_json(output / "gate.json", {"passed": False, "reason": "Continuation selected unchanged initialization",
                                         "checkpoint_sha256": fingerprint, "selected_step": 0, "real_benchmark_success": False})
        return
    student, optimizer, criterion, teachers, sampler = setup(config)
    del optimizer
    with column.evaluation_metrics():
        bank, reference = column.continuation.prepare_bank(sampler, teachers, criterion, config["confirmation_cases"], output)
        initial_metrics, initial_cases = column.continuation.evaluate_bank(student, criterion, bank)
        student.load_state_dict(state["model"], strict=True)
        del state
        student_metrics, student_cases = column.continuation.evaluate_bank(student, criterion, bank)
    write_json(output / "initialization.json", {"metrics": initial_metrics, "cases": initial_cases})
    write_json(output / "student.json", {"metrics": student_metrics, "cases": student_cases})
    gate = {**column.confirmation_gate(student_metrics, initial_metrics, reference["general"], reference["references"]),
            "checkpoint_sha256": fingerprint, "selected_step": manifest["selected_step"],
            "bank_sha256": reference["bank_sha256"], "cases": len(bank),
            "metrics": {"student": student_metrics, "initialization": initial_metrics,
                        "general": reference["general"], "teacher": reference["teacher"]}}
    if len(bank) != 92 or sha256(checkpoint) != fingerprint or sha256(output / "synthetic_bank.pt") != reference["bank_sha256"]:
        raise ValueError("Checkpoint or complete confirmation bank changed")
    verify_run(config, run)
    write_json(output / "gate.json", gate)
    print(json.dumps({"gate": gate}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-for-training", action="store_true")
    arguments = parser.parse_args()
    run, output = arguments.run.resolve(), arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            storage_guard(output)
            while not training_ready(run):
                if not arguments.wait_for_training:
                    raise RuntimeError("Training is still running")
                write_json(output / "status.json", {"state": "waiting_for_training", "pid": os.getpid(), "updated_unix": time.time()})
                time.sleep(30)
            wait_for_gpu(output, arguments.wait_for_training, False)
            confirm(run, output)
            write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
