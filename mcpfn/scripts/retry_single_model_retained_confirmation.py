"""Retry only the average's unchanged gate after its ordering prerequisite stopped."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import torch

import confirm_single_model_retained_average as original
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import sha256, write_json
from train_single_matrixformer import storage_guard
from train_single_matrixformer_continuation import read
from train_streamed_large_matrices import wait_for_gpu


FAILURE = "Every imputation validation case requires a common nonconstant column"


def require_terminal(directory, expected):
    status = read(directory / "status.json")
    if status["state"] != expected:
        raise ValueError(f"Prerequisite is not the expected terminal state: {directory}")
    if "pid" in status:
        try:
            os.kill(status["pid"], 0)
            state = Path(f"/proc/{status['pid']}/stat").read_text().split(") ", 1)[1].split()[0]
            if state != "Z":
                raise RuntimeError(f"Terminal status still has a live process: {directory}")
        except (ProcessLookupError, FileNotFoundError):
            pass
    return status


def verify_coverage_records(probe, records):
    expected = [(bank, index, case["seed"]) for bank, cases in probe["banks"].items()
                for index, case in enumerate(cases) if case["domain"] == "imputation"]
    actual = [(row["bank"], row["index"], row["seed"]) for row in records]
    if len(expected) != 120 or actual != expected:
        raise ValueError("Require all 120 unchanged ordered imputation draws")
    for row in records:
        if row["area_prior"]["requested_regime"] != "legacy" or row["area_prior"]["applied"]:
            raise ValueError("Coverage probe changed a confirmation input regime")
        if row["bank"] == "average" and row["common_nonconstant_columns"] <= 0:
            raise ValueError("This retry must not alter the average's unchanged common-column guard")
    empty = [row for row in records if row["common_nonconstant_columns"] == 0]
    if (len(empty) != 1 or empty[0]["bank"] != "selected" or empty[0]["index"] != 6
            or empty[0]["seed"] != 54100507 or empty[0]["query_count_min"] != 1 or empty[0]["query_count_max"] != 1):
        raise ValueError("Require the specific diagnosed singleton-query failure")


def recovery_proof(average, failed, coverage):
    previous = read(failed / "config.json")
    original.verify_registration(previous, failed)
    if Path(previous["average"]).resolve() != average.resolve():
        raise ValueError("Never substitute another averaged candidate")
    previous_status = require_terminal(failed, "failed")
    if "Prerequisite failed; never bypass it" not in previous_status.get("error", ""):
        raise ValueError("Only a pre-inference ordering failure may use this retry")
    if any((failed / name).exists() for name in ("manifest.json", "candidate_audit.json", "synthetic_bank.pt", "student.json", "gate.json")):
        raise ValueError("Never retry an already evaluated or rejected average")
    campaign = Path(previous["after"])
    campaign_status = require_terminal(campaign, "failed")
    if "Synthetic confirmation stopped" not in campaign_status.get("error", ""):
        raise ValueError("Original campaign failed for a different reason")
    launcher = read(campaign / "cache_launcher_config.json")
    confirmation = Path(launcher["confirmation"])
    producer = Path(previous["producer"])
    if Path(launcher["run"]).resolve() != producer.resolve():
        raise ValueError("Original failure must belong to this exact producer")
    status = require_terminal(confirmation, "failed")
    if FAILURE not in status.get("error", "") or (confirmation / "gate.json").exists() or (confirmation / "student.json").exists():
        raise ValueError("Require the documented failure before student evaluation")
    producer_status = require_terminal(producer, "training_complete")
    if producer_status["step"] != 400 or read(average / "status.json")["state"] != "complete":
        raise ValueError("Original training and every averaging contribution must be complete")
    probe, summary = read(coverage / "config.json"), read(coverage / "summary.json")
    if (not summary["complete"] or summary["config_sha256"] != sha256(coverage / "config.json")
            or summary["by_bank"] != {"selected": 1, "average": 0} or summary["imputation_cases"] != 120
            or summary["model_inference_performed"] or summary["case_replacement_performed"] or summary["gate_changed"]
            or Path(probe["producer"]).resolve() != producer.resolve()
            or probe["banks"]["average"] != previous["cases"]):
        raise ValueError("Require the complete model-free diagnosis of the unchanged two banks")
    verify_hashes(probe["input_sha256"])
    verify_hashes(probe["source_sha256"])
    verify_coverage_records(probe, read(coverage / "cases.json"))
    artifacts = [directory / "status.json" for directory in (failed, campaign, confirmation, producer, average)]
    artifacts += [failed / "config.json", failed / "registration.json", campaign / "cache_launcher_config.json"]
    artifacts += [coverage / name for name in ("config.json", "registration.json", "summary.json", "cases.json", "empty_common_cases.json")]
    return {"previous_qualification": str(failed), "original_campaign": str(campaign), "source_failure": str(confirmation),
            "failure_reason": FAILURE, "all_prerequisites_terminal": True,
            "qualification_cases_changed": False, "qualification_gate_changed": False,
            "candidate_changed": False, "previous_average_evaluation_performed": False,
            "ordering_recovery": "The original campaign is terminal-failed, satisfying the prospectively frozen terminal-before-idle ordering; retain the failed attempts separately",
            "input_sha256": {str(path): sha256(path) for path in artifacts}}


def confirm_after_terminal_failure(config, output, proof):
    verify_hashes(proof["input_sha256"])
    previous_ready = original.prerequisite_ready

    def ready(directory, *, campaign=False):
        if campaign and directory.resolve() == Path(proof["original_campaign"]).resolve():
            verify_hashes(proof["input_sha256"])
            require_terminal(directory, "failed")
            return True
        return previous_ready(directory, campaign=campaign)

    original.prerequisite_ready = ready
    try:
        original.confirm(config, output)
    finally:
        original.prerequisite_ready = previous_ready
    verify_hashes(proof["input_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--average", type=Path, required=True)
    parser.add_argument("--failed-confirmation", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    average, failed, coverage, output = [path.resolve() for path in
        (arguments.average, arguments.failed_confirmation, arguments.coverage, arguments.output)]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            torch.set_num_threads(4)
            storage_guard(output)
            write_json(output / "status.json", {"state": "registering", "pid": os.getpid(), "updated_unix": time.time()})
            proof = recovery_proof(average, failed, coverage)
            write_json(output / "recovery.json", proof)
            write_json(output / "recovery_registration.json", {"sha256": sha256(output / "recovery.json"), "registered_unix": time.time()})
            config = original.register(average, output)
            wait_for_gpu(output, True, False)
            if sha256(output / "recovery.json") != read(output / "recovery_registration.json")["sha256"]:
                raise ValueError("Recovery proof changed")
            confirm_after_terminal_failure(config, output, proof)
            write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
