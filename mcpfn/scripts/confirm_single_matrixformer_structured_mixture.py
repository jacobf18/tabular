"""Independent full legacy and structured-prior confirmation, fixed before V9 training."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import sha256, write_json
import single_model_structured_mixture as mixture
from train_single_matrixformer_structured_mixture import read, validate_recipe
from confirm_single_matrixformer_continuation import training_ready
from train_streamed_large_matrices import wait_for_gpu


def register(run, output):
    if output.exists():
        raise FileExistsError("Never replace registered independent confirmation")
    config = read(run / "config.json")
    validate_recipe(config)
    mixture.backend.verify_run(config, run)
    output.mkdir(parents=True)
    write_json(output / "config.json", {"run": str(run), "run_config_sha256": sha256(run / "config.json"),
        "cases": config["confirmation_cases"], "policy": mixture.POLICY, "registered_unix": time.time(),
        "training_performed": False, "checkpoint_reselection_performed": False,
        "source_sha256": {str(Path(__file__).resolve()): sha256(Path(__file__))}})
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "student_evaluated": False})
    write_json(output / "status.json", {"state": "registered", "pid": os.getpid(), "updated_unix": time.time()})


@torch.no_grad()
def confirm(run, output):
    config = read(run / "config.json")
    registration = read(output / "config.json")
    if (registration["run"] != str(run) or registration["run_config_sha256"] != sha256(run / "config.json")
            or registration["cases"] != config["confirmation_cases"] or registration["policy"] != mixture.POLICY
            or read(output / "registration.json")["config_sha256"] != sha256(output / "config.json")):
        raise ValueError("Independent confirmation registration changed")
    verify_hashes(registration["source_sha256"])
    validate_recipe(config)
    mixture.backend.verify_run(config, run)
    checkpoint = run / "best_validation_checkpoint.pth"
    fingerprint = sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["config_sha256"] != sha256(run / "config.json") or state["step"] != read(run / "status.json")["best_step"]:
        raise ValueError("Require the actually selected training checkpoint")
    write_json(output / "manifest.json", {"checkpoint": str(checkpoint), "checkpoint_sha256": fingerprint,
        "selected_step": state["step"], "cases": config["confirmation_cases"], "run_config_sha256": sha256(run / "config.json"),
        "policy": mixture.POLICY, "training_performed": False})
    if state["step"] == 0:
        write_json(output / "gate.json", {"passed": False, "selected_step": 0, "checkpoint_sha256": fingerprint,
            "reason": "No trained checkpoint retained every prospectively registered selection constraint", "real_benchmark_success": False})
        return
    with mixture.runtime(config, output):
        student, optimizer, criterion, teachers, sampler = mixture.backend.setup(config)
        del optimizer
        bank, reference = mixture.backend.prepare_bank(sampler, teachers, criterion, config["confirmation_cases"], output)
        initial_metrics, initial_cases = mixture.backend.evaluate_bank(student, criterion, bank)
        student.load_state_dict(state["model"], strict=True)
        selected_step = state["step"]
        del state
        student_metrics, student_cases = mixture.backend.evaluate_bank(student, criterion, bank)
    if len(bank) != 284 or len(initial_cases) != 284 or len(student_cases) != 284:
        raise ValueError("Never omit any original92 or structured192 confirmation case")
    coverage_rows = []
    for index in range(len(bank)):
        coverage_rows.append({"domain": reference["cases"][index]["domain"],
            "general": reference["cases"][index]["general"], "teacher": reference["cases"][index]["teacher"],
            "initialization": initial_cases[index]["student"], "student": student_cases[index]["student"]})
    mixture.require_truth_coverage(coverage_rows, ("general", "teacher", "initialization", "student"))
    write_json(output / "initialization.json", {"metrics": initial_metrics, "cases": initial_cases})
    write_json(output / "student.json", {"metrics": student_metrics, "cases": student_cases})
    gate = {**mixture.confirmation_gate(student_metrics, initial_metrics, reference["general"], reference["references"]),
        "checkpoint_sha256": fingerprint, "selected_step": selected_step, "cases": len(bank),
        "legacy_cases": 92, "structured_cases": 192, "bank_sha256": reference["bank_sha256"],
        "metrics": {"student": student_metrics, "initialization": initial_metrics, "general": reference["general"], "teacher": reference["teacher"]}}
    mixture.backend.verify_run(config, run)
    verify_hashes({str(checkpoint): fingerprint, str(output / "synthetic_bank.pt"): reference["bank_sha256"]})
    write_json(output / "gate.json", gate)
    print(json.dumps({"gate": gate}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "confirm"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if args.stage == "register":
        register(run, output)
        return
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(output / "status.json")["state"] != "registered":
            raise ValueError("Never repeat a started independent confirmation")
        try:
            write_json(output / "status.json", {"state": "waiting_for_training", "pid": os.getpid(), "updated_unix": time.time()})
            while not training_ready(run):
                time.sleep(30)
            wait_for_gpu(output, True, False)
            confirm(run, output)
            write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
