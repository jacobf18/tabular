"""Independently confirm a registered training-tail average without model selection."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch

from confirm_single_matrixformer_continuation import confirmation_gate
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from train_single_matrixformer import setup, storage_guard
from train_single_matrixformer_continuation import bank_specs, evaluate_bank, prepare_bank, read, verify_run
from train_streamed_large_matrices import wait_for_gpu


ROOT = REPO / "runs/single_model_goal_20261005"


def ready(directory):
    status = read(directory / "status.json")
    if status["state"] == "complete":
        return True
    if status["state"] in ("failed", "interrupted", "not_promoted"):
        raise RuntimeError(f"Prerequisite did not complete: {directory}: {status['state']}")
    try:
        os.kill(status["pid"], 0)
        if Path(f"/proc/{status['pid']}/stat").read_text().split(") ", 1)[1].split()[0] == "Z":
            raise ProcessLookupError()
    except (ProcessLookupError, FileNotFoundError):
        if read(directory / "status.json")["state"] == "complete":
            return True
        raise RuntimeError(f"Prerequisite handle is missing or terminal: {directory}; never restart it here") from None
    return False


def inventory(state, original):
    if tuple(state) != tuple(original):
        raise ValueError("Average must contain exactly one original network inventory")
    changed = 0
    for name, value in state.items():
        if value.shape != original[name].shape or value.dtype != original[name].dtype or not torch.isfinite(value).all():
            raise ValueError(f"Invalid averaged tensor: {name}")
        changed += not torch.equal(value, original[name])
    return {"tensors": len(state), "elements": sum(value.numel() for value in state.values()), "changed_tensors": changed}


def check_window(config, result, state, receipts):
    expected = config["steps"]
    if expected != list(range(300, 401, 10)) or config["registered_at_producer_step"] >= expected[0]:
        raise ValueError("Require the prospectively registered eleven-state window")
    actual = receipts["snapshots"]
    if receipts["expected_steps"] != expected or [row["step"] for row in actual] != expected:
        raise ValueError("Incomplete or reordered averaging receipts")
    for row in actual:
        digest = row["checkpoint_sha256"]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("Invalid source-checkpoint byte hash")
        if row["captured_unix"] < config["registered_unix"]:
            raise ValueError("Contribution precedes registration")
    if result["source_updates"] != expected or result["snapshots"] != len(expected) or not result["single_parameter_set"]:
        raise ValueError("Average result differs from its registered window")
    if (state["source_updates"] != expected or state["step"] != expected[-1] or not state["derived_checkpoint"]
            or state["gradient_updates_performed_here"] != 0):
        raise ValueError("Derived checkpoint has inconsistent update semantics")


def register(average, output):
    average_config = read(average / "config.json")
    producer = Path(average_config["producer"])
    base = read(producer / "config.json")
    verify_run(base, producer)
    verify_hashes(average_config["source_sha256"])
    verify_hashes({str(producer / "config.json"): average_config["producer_config_sha256"]})
    dependency = ROOT / "full_gradient_diagnostic_v1"
    sources = {Path(__file__).resolve(), REPO / "src/tabimpute/data/borders.pt"}
    sources.update((REPO / "src/tabimpute").rglob("*.py"))
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(Path(module.__file__).resolve()).startswith(str(REPO))
                   and str(module.__file__).endswith(".py") and Path(module.__file__).is_file())
    cases = bank_specs(20100501, confirmation=True)
    previous_seeds = {case["seed"] for case in base["validation_cases"] + base["confirmation_cases"]}
    if previous_seeds.intersection(case["seed"] for case in cases):
        raise ValueError("Require a new independent synthetic bank")
    config = {"average": str(average), "average_config_sha256": sha256(average / "config.json"),
              "producer": str(producer), "after": str(dependency), "seed_start": 20100501,
              "cases": cases, "base": base,
              "input_sha256": {str(average / "config.json"): sha256(average / "config.json"),
                               str(producer / "config.json"): average_config["producer_config_sha256"],
                               str(dependency / "config.json"): sha256(dependency / "config.json"), **base["input_sha256"]},
              "source_sha256": {str(path): sha256(path) for path in sorted(sources)},
              "policy": "Same strict continuation gate: all three metrics <= initial pilot and general control, and strict minimax improvement over initial pilot",
              "selection_performed": False, "training_performed": False, "real_benchmark_inputs_used": False,
              "candidate": "The one predetermined 300:10:400 weight average; no intermediate average or replacement window",
              "inference": "One ordinary network state; no teachers or averaging of outputs",
              "checkpoint_step_semantics": "Last contributing producer update, not new gradient steps here",
              "ordering": "Wait for average completion and the existing full-gradient diagnostic, then require an idle GPU"}
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"),
                                              "registered_unix": time.time(), "candidate_evaluated": False})
    for source in sources:
        destination = output / "sources" / source.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return config


def verify_registration(config, output):
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    verify_run(config["base"], Path(config["producer"]))
    if sha256(output / "config.json") != read(output / "registration.json")["config_sha256"]:
        raise ValueError("Confirmation registration changed")


def audit_candidate(config, output):
    average = Path(config["average"])
    if read(average / "status.json")["state"] != "complete":
        raise ValueError("Average is not complete")
    source_config = read(average / "config.json")
    if source_config["model"] != config["base"]["model"]:
        raise ValueError("Averaged candidate architecture differs from its producer")
    result, receipts = read(average / "result.json"), read(average / "receipts.json")
    checkpoint = average / "averaged_checkpoint.pth"
    if not result["complete"] or Path(result["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError("Average result does not bind its checkpoint")
    fingerprint = sha256(checkpoint)
    if fingerprint != result["checkpoint_sha256"] or result["config_sha256"] != config["average_config_sha256"]:
        raise ValueError("Average bytes or configuration changed")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["config_sha256"] != config["average_config_sha256"]:
        raise ValueError("Checkpoint/config mismatch")
    receipt_hash = sha256(average / "receipts.json")
    if state["receipts_sha256"] != receipt_hash or result["receipts_sha256"] != receipt_hash:
        raise ValueError("Averaging receipts changed")
    check_window(source_config, result, state, receipts)
    producer_status = read(Path(config["producer"]) / "status.json")
    if producer_status["state"] != "training_complete" or producer_status["step"] != state["step"]:
        raise ValueError("Producer did not complete the required training")
    original = torch.load(config["base"]["parent_checkpoint"], map_location="cpu", weights_only=True)["model"]
    audit = {"checkpoint": str(checkpoint), "checkpoint_sha256": fingerprint, "selected_step": state["step"],
             **inventory(state["model"], original), "source_updates": state["source_updates"],
             "derived_checkpoint": True, "gradient_updates_performed_here": 0,
             "arithmetic_independently_reconstructed": False,
             "provenance_limit": "Source snapshots were overwritten by their trainer; provenance binds the tested collector and exact-byte receipts, not a second reconstruction of their mean"}
    write_json(output / "candidate_audit.json", audit)
    return state, audit


def publish_alias(average, gate):
    if not gate["passed"]:
        raise ValueError("Never publish a failed candidate for benchmark evaluation")
    checkpoint = average / "averaged_checkpoint.pth"
    if sha256(checkpoint) != gate["checkpoint_sha256"]:
        raise ValueError("Qualified average changed")
    alias = average / "best_validation_checkpoint.pth"
    if alias.exists() or alias.is_symlink():
        if alias.resolve() != checkpoint.resolve():
            raise ValueError("Refuse to overwrite an unrelated checkpoint alias")
    else:
        alias.symlink_to(checkpoint.name)


@torch.no_grad()
def confirm(config, output):
    verify_registration(config, output)
    state, audit = audit_candidate(config, output)
    write_json(output / "manifest.json", {**audit, "config_sha256": sha256(output / "config.json"),
                                          "cases": config["cases"], "seed_start": config["seed_start"],
                                          "training_performed": False, "policy": config["policy"]})
    model, optimizer, criterion, teachers, sampler = setup(config["base"])
    del optimizer
    bank, reference = prepare_bank(sampler, teachers, criterion, config["cases"], output)
    initial, initial_cases = evaluate_bank(model, criterion, bank)
    model.load_state_dict(state["model"], strict=True)
    del state
    student, student_cases = evaluate_bank(model, criterion, bank)
    write_json(output / "initialization.json", {"metrics": initial, "cases": initial_cases})
    write_json(output / "student.json", {"metrics": student, "cases": student_cases})
    gate = {**confirmation_gate(student, initial, reference["general"], reference["references"]),
            "checkpoint_sha256": audit["checkpoint_sha256"], "selected_step": audit["selected_step"],
            "derived_checkpoint": True, "bank_sha256": reference["bank_sha256"], "cases": len(bank),
            "metrics": {"student": student, "initialization": initial, "general": reference["general"], "teacher": reference["teacher"]}}
    verify_hashes({audit["checkpoint"]: audit["checkpoint_sha256"],
                   str(output / "synthetic_bank.pt"): reference["bank_sha256"]})
    verify_registration(config, output)
    if gate["passed"]:
        publish_alias(Path(config["average"]), gate)
    write_json(output / "gate.json", gate)
    print(json.dumps({"gate": gate}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--average", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-for-ready", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(output / "status.json", {"state": "registering", "pid": os.getpid(), "updated_unix": time.time()})
            storage_guard(output)
            config = register(arguments.average.resolve(), output)
            while not (ready(Path(config["average"])) and ready(Path(config["after"]))):
                if not arguments.wait_for_ready:
                    raise RuntimeError("Confirmation prerequisites are still running")
                write_json(output / "status.json", {"state": "waiting_for_prerequisites", "pid": os.getpid(), "updated_unix": time.time()})
                time.sleep(30)
            wait_for_gpu(output, arguments.wait_for_ready, False)
            confirm(config, output)
            write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
