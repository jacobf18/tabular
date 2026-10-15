"""Qualify the one retained V8 tail average on its preregistered full-query bank."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch

from average_single_model_retained_tail import qualification_plan
from confirm_single_matrixformer_query_budget import verify_legacy_cases
from confirm_single_model_weight_average import check_window, inventory, publish_alias, verify_registration
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_area_prior import evaluation_setup
import single_model_column_objective as column
from train_single_matrixformer import storage_guard
from train_single_matrixformer_continuation import read, verify_run
from train_single_matrixformer_query_budget import validate_recipe
from train_streamed_large_matrices import wait_for_gpu


def verify_plan(config, base):
    expected = json.loads(json.dumps(qualification_plan(base)))
    if config["qualification"] != expected:
        raise ValueError("Never change the prospectively registered 92-case six-metric plan")
    if config["steps"] != list(range(300, 401, 10)) or config["registered_at_producer_step"] >= 300:
        raise ValueError("Require the unique prospectively registered eleven-state candidate")
    if config["model"] != base["model"]:
        raise ValueError("Candidate must use the producer's original network")


def prerequisite_ready(directory, *, campaign=False):
    status = read(directory / "status.json")
    terminal = {"complete", "not_promoted"} if campaign else {"complete"}
    if status["state"] in terminal:
        return True
    if status["state"] in {"failed", "interrupted", "not_promoted"}:
        raise RuntimeError(f"Prerequisite failed; never bypass it: {directory}: {status['state']}")
    try:
        os.kill(status["pid"], 0)
        if Path(f"/proc/{status['pid']}/stat").read_text().split(") ", 1)[1].split()[0] == "Z":
            raise ProcessLookupError()
    except (ProcessLookupError, FileNotFoundError):
        if read(directory / "status.json")["state"] in terminal:
            return True
        raise RuntimeError(f"Prerequisite process disappeared: {directory}; never restart it here") from None
    return False


def register(average, output):
    source = read(average / "config.json")
    producer = Path(source["producer"])
    base = read(producer / "config.json")
    verify_run(base, producer)
    validate_recipe(base)
    verify_plan(source, base)
    verify_hashes(source["source_sha256"])
    if sha256(average / "config.json") != read(average / "registration.json")["config_sha256"]:
        raise ValueError("Average registration changed")
    campaign = Path(source["after_campaign"])
    if read(campaign / "cache_launcher_config.json")["run"] != str(producer):
        raise ValueError("Must wait for this producer's original benchmark campaign")
    sources = {Path(__file__).resolve(), REPO / "src/tabimpute/data/borders.pt"}
    sources.update((REPO / "src/tabimpute").rglob("*.py"))
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(Path(module.__file__).resolve()).startswith(str(REPO))
                   and str(module.__file__).endswith(".py") and Path(module.__file__).is_file())
    config = {"average": str(average), "average_config_sha256": sha256(average / "config.json"),
              "producer": str(producer), "after": str(campaign), "base": base, "qualification": source["qualification"],
              "cases": source["qualification"]["cases"], "seed_start": source["qualification"]["seed_start"],
              "policy": source["qualification"]["rule"], "training_performed": False, "selection_performed": False,
              "real_benchmark_inputs_used": False, "candidate": "Only the fixed retained 300:10:400 parameter average",
              "reconstruction": "Independent FP64 sum divided by eleven; original dtypes, at most one output ULP for online-mean rounding",
              "input_sha256": {**base["input_sha256"], str(average / "config.json"): sha256(average / "config.json"),
                               str(average / "registration.json"): sha256(average / "registration.json"),
                               str(producer / "config.json"): source["producer_config_sha256"],
                               str(campaign / "cache_launcher_config.json"): source["after_campaign_config_sha256"]},
              "source_sha256": {**source["source_sha256"], **{str(path): sha256(path) for path in sorted(sources)}}}
    verify_hashes(config["input_sha256"])
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"),
                                              "registered_unix": time.time(), "candidate_evaluated": False})
    for path in sources:
        destination = output / "sources" / path.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return config


def reconstruct(average, source, candidate, receipts):
    retained = read(average / "retained_states.json")
    result = read(average / "retention_result.json")
    steps = source["steps"]
    if (not result["complete"] or result["states"] != len(steps) or result["source_updates"] != steps
            or result["retained_states_sha256"] != sha256(average / "retained_states.json")
            or retained["expected_steps"] != steps or [row["step"] for row in retained["states"]] != steps):
        raise ValueError("Require all eleven immutable retained states")
    if len(receipts["snapshots"]) != len(steps):
        raise ValueError("Incomplete original checkpoint receipts")
    sums = {}
    hashes = {str(average / "retained_states.json"): result["retained_states_sha256"],
              str(average / "retention_result.json"): sha256(average / "retention_result.json")}
    for record, receipt in zip(retained["states"], receipts["snapshots"]):
        path = average / "contributing_states" / f"step_{record['step']}.pth"
        if Path(record["path"]).resolve() != path.resolve() or sha256(path) != record["sha256"]:
            raise ValueError("Retained contribution path or bytes changed")
        if (record["producer_checkpoint_sha256"] != receipt["checkpoint_sha256"] or record["step"] != receipt["step"]
                or record["captured_unix"] < source["registered_unix"]):
            raise ValueError("Retained contribution does not bind the original prospective receipt")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload["step"] != record["step"] or payload["producer_checkpoint_sha256"] != receipt["checkpoint_sha256"]
                or payload["producer_config_sha256"] != source["producer_config_sha256"]):
            raise ValueError("Retained payload provenance mismatch")
        inventory(payload["model"], candidate)
        for name, value in payload["model"].items():
            if name not in sums:
                sums[name] = value.double().clone() if value.is_floating_point() else value.clone()
            elif value.is_floating_point():
                sums[name].add_(value.double())
            elif not torch.equal(value, sums[name]):
                raise ValueError(f"Nonfloating contribution differs: {name}")
        hashes[str(path)] = record["sha256"]
        del payload
    mismatches, maximum = 0, 0.
    for name, value in candidate.items():
        if value.is_floating_point():
            expected = (sums.pop(name) / len(steps)).to(value.dtype)
            lower = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
            upper = torch.nextafter(expected, torch.full_like(expected, torch.inf))
            if not torch.isfinite(expected).all() or not ((value >= lower) & (value <= upper)).all():
                raise ValueError(f"Averaged parameter disagrees with independently reconstructed mean: {name}")
            mismatches += int((value != expected).sum())
            maximum = max(maximum, float((value.double() - expected.double()).abs().max()))
        elif not torch.equal(value, sums.pop(name)):
            raise ValueError(f"Averaged nonfloating state differs: {name}")
    verify_hashes(hashes)
    return {"arithmetic_independently_reconstructed": True, "states": len(steps),
            "method": "Independent FP64 sum/division, not the collector's running mean",
            "allowed_output_ulps": 1, "non_bitwise_elements": mismatches, "maximum_absolute_difference": maximum,
            "retained_artifact_sha256": hashes}


def audit_candidate(config, output):
    average, producer = Path(config["average"]), Path(config["producer"])
    if read(average / "status.json")["state"] != "complete":
        raise ValueError("Only the complete retained average can be qualified")
    source = read(average / "config.json")
    verify_plan(source, config["base"])
    result, receipts = read(average / "result.json"), read(average / "receipts.json")
    checkpoint = average / "averaged_checkpoint.pth"
    fingerprint = sha256(checkpoint)
    if (not result["complete"] or Path(result["checkpoint"]).resolve() != checkpoint.resolve()
            or result["checkpoint_sha256"] != fingerprint or result["config_sha256"] != config["average_config_sha256"]):
        raise ValueError("Average result does not bind the registered checkpoint")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (state["config_sha256"] != config["average_config_sha256"]
            or state["receipts_sha256"] != sha256(average / "receipts.json")
            or result["receipts_sha256"] != state["receipts_sha256"]):
        raise ValueError("Checkpoint configuration or original receipts changed")
    check_window(source, result, state, receipts)
    status = read(producer / "status.json")
    if status["state"] != "training_complete" or status["step"] != 400:
        raise ValueError("Producer must finish its unchanged 400-update schedule")
    original = torch.load(config["base"]["parent_checkpoint"], map_location="cpu", weights_only=True)["model"]
    audit = {"checkpoint": str(checkpoint), "checkpoint_sha256": fingerprint, "selected_step": state["step"],
             **inventory(state["model"], original), "source_updates": state["source_updates"],
             "derived_checkpoint": True, "gradient_updates_performed_here": 0,
             **reconstruct(average, source, state["model"], receipts),
             "provenance_limit": "Original full optimizer states are byte-bound by receipts; retained model states permit independent parameter-mean reconstruction"}
    del original
    write_json(output / "candidate_audit.json", audit)
    return state, audit


@torch.no_grad()
def confirm(config, output):
    verify_registration(config, output)
    if not prerequisite_ready(Path(config["after"]), campaign=True):
        raise ValueError("Original benchmark campaign is still active")
    state, audit = audit_candidate(config, output)
    write_json(output / "manifest.json", {**audit, "config_sha256": sha256(output / "config.json"),
        "cases": config["cases"], "seed_start": config["seed_start"], "training_performed": False,
        "policy": config["policy"], "prior_regime": "legacy", "query_budget_applied": False})
    model, optimizer, criterion, teachers, sampler = evaluation_setup(config["base"])
    del optimizer
    with column.evaluation_metrics():
        bank, reference = column.continuation.prepare_bank(sampler, teachers, criterion, config["cases"], output)
        verify_legacy_cases(bank)
        if [case["metadata"]["seed"] for case in bank] != [case["seed"] for case in config["cases"]]:
            raise ValueError("Confirmation bank differs from the prospectively registered fresh seeds")
        initial, initial_cases = column.continuation.evaluate_bank(model, criterion, bank)
        model.load_state_dict(state["model"], strict=True)
        del state
        student, student_cases = column.continuation.evaluate_bank(model, criterion, bank)
    write_json(output / "initialization.json", {"metrics": initial, "cases": initial_cases})
    write_json(output / "student.json", {"metrics": student, "cases": student_cases})
    gate = {**column.confirmation_gate(student, initial, reference["general"], reference["references"]),
            "checkpoint_sha256": audit["checkpoint_sha256"], "selected_step": audit["selected_step"],
            "derived_checkpoint": True, "arithmetic_independently_reconstructed": True,
            "bank_sha256": reference["bank_sha256"], "cases": len(bank), "prior_regime": "legacy", "query_budget_applied": False,
            "metrics": {"student": student, "initialization": initial, "general": reference["general"], "teacher": reference["teacher"]}}
    verify_hashes({audit["checkpoint"]: audit["checkpoint_sha256"],
                   str(output / "synthetic_bank.pt"): reference["bank_sha256"], **audit["retained_artifact_sha256"]})
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
            torch.set_num_threads(4)
            write_json(output / "status.json", {"state": "registering", "pid": os.getpid(), "updated_unix": time.time()})
            storage_guard(output)
            config = register(arguments.average.resolve(), output)
            while not (prerequisite_ready(Path(config["average"])) and prerequisite_ready(Path(config["after"]), campaign=True)):
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
