"""Prospectively average V8's future tail, retaining every contributing model state."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch

import average_single_model_training_tail as base
from recsys_evaluation_common import REPO, sha256, write_json
from train_single_matrixformer import storage_guard
from train_single_matrixformer_query_budget import validate_recipe
from train_single_matrixformer_continuation import bank_specs, verify_run
import single_model_column_objective as column


STEPS = list(range(300, 401, 10))
CONFIRMATION_SEED = 58100501


def qualification_plan(config):
    cases = [case for replica in range(2) for case in bank_specs(CONFIRMATION_SEED + 1000 * replica, confirmation=True)]
    seeds = {case["seed"] for case in cases}
    previous = {case["seed"] for case in config["validation_cases"] + config["confirmation_cases"]}
    if len(cases) != 92 or len(seeds) != 92 or seeds & previous:
        raise ValueError("Require 92 fresh independent confirmation cases")
    return {"seed_start": CONFIRMATION_SEED, "replicas": 2, "cases": cases, "prior_regime": "legacy",
            "query_budget_applied": False, "initialization_checkpoint": config["parent_checkpoint"],
            "primary_metrics": [list(pair) for pair in column.PRIMARY_METRICS],
            "retention_metrics": [list(pair) for pair in column.RETENTION_METRICS],
            "rule": "Retain initialization and original general control on all six original metrics; strictly improve four-metric minimax versus initialization",
            "ordering": "Wait for the original producer benchmark campaign to be terminal, then require an idle GPU",
            "checkpoint_selection_performed": False, "fallback_to_producer_or_partial_average": False}


def prepare(producer, campaign, output):
    config = base.read(producer / "config.json")
    verify_run(config, producer)
    validate_recipe(config)
    if base.read(campaign / "cache_launcher_config.json")["run"] != str(producer):
        raise ValueError("Require this producer's original campaign for sequential GPU ordering")
    status = base.read(producer / "status.json")
    base.require_live(status)
    if status["state"] != "training" or status.get("step", 0) >= STEPS[0]:
        raise ValueError("Register before update300 of the live original 400-update producer")
    sources = {Path(__file__).resolve(), Path(base.__file__).resolve(), REPO / "scripts/recsys_evaluation_common.py"}
    plan = {"model": config["model"], "producer": str(producer), "producer_config_sha256": sha256(producer / "config.json"),
            "producer_inputs": config["input_sha256"], "producer_sources": config["source_sha256"],
            "producer_teacher_configs": config["teacher_config_sha256"],
            "source_sha256": {str(path): sha256(path) for path in sorted(sources)},
            "registered_at_producer_step": status["step"], "registered_unix": time.time(), "steps": STEPS,
            "rule": "Uniform arithmetic mean of every complete state at300,310,...,400; FP64 accumulator and original output dtypes",
            "source_retention": "Retain all eleven model-only states with original optimizer-checkpoint hashes, allowing independent mean reconstruction",
            "qualification": qualification_plan(config), "after_campaign": str(campaign),
            "after_campaign_config_sha256": sha256(campaign / "cache_launcher_config.json"),
            "training_performed_here": False, "real_benchmark_inputs_used": False, "selection_performed": False,
            "inference": "One original parameter set, not averaged predictions or multiple inference models",
            "rationale": "V8's synthetic selection oscillates between150 and200 under noisy query-budget gradients; test a fixed future late-training mean without changing its selector",
            "evaluation": "Separate unqualified candidate; fresh preregistered92-case six-metric gate, then all frozen benchmarks if qualified. Never replace the producer's selected checkpoint.",
            "step_semantics": "Last contributing producer update, not new optimizer updates in this process"}
    status = base.read(producer / "status.json")
    base.require_live(status)
    if status["state"] != "training" or status.get("step", 0) >= STEPS[0]:
        raise ValueError("Prospective registration window closed before publication")
    plan["registered_at_producer_step"] = status["step"]
    storage_guard(output)
    write_json(output / "config.json", plan)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"),
                                              "registered_unix": time.time(), "candidate_evaluated": False})
    shutil.copy2(producer / "config.json", output / "producer_config.json")
    for source in sources:
        destination = output / "sources" / source.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return plan


def retain_payload(payload, digest, config, output, records):
    expected = config["steps"][len(records)]
    if payload["step"] != expected or payload["config_sha256"] != config["producer_config_sha256"]:
        raise ValueError("Never retain a partial or replacement averaging window")
    directory = output / "contributing_states"
    directory.mkdir(exist_ok=True)
    path = directory / f"step_{expected}.pth"
    if path.exists():
        raise ValueError("Never overwrite a contributing state")
    storage_guard(output)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save({"model": payload["model"], "step": expected, "producer_config_sha256": config["producer_config_sha256"],
                    "producer_checkpoint_sha256": digest}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    records.append({"step": expected, "path": str(path), "sha256": sha256(path),
                    "producer_checkpoint_sha256": digest, "captured_unix": time.time()})
    write_json(output / "retained_states.json", {"expected_steps": config["steps"], "states": records})


def execute(config, output):
    if sha256(output / "config.json") != base.read(output / "registration.json")["config_sha256"]:
        raise ValueError("Registered average changed")
    records = []
    original = base.load_stable
    original_writer = base.write_json
    completed = False

    def retaining_loader(path):
        payload, digest = original(path)
        retain_payload(payload, digest, config, output, records)
        return payload, digest

    def defer_completion(path, payload):
        nonlocal completed
        if path == output / "status.json" and payload.get("state") == "complete":
            completed = True
            return
        original_writer(path, payload)

    base.load_stable = retaining_loader
    base.write_json = defer_completion
    try:
        base.execute(config, output)
    finally:
        base.load_stable = original
        base.write_json = original_writer
    if not completed:
        raise ValueError("Original averaging collector did not complete")
    receipts = base.read(output / "receipts.json")["snapshots"]
    if ([record["step"] for record in records] != config["steps"]
            or [record["producer_checkpoint_sha256"] for record in records] != [row["checkpoint_sha256"] for row in receipts]):
        raise ValueError("Retained model states do not bind every original averaging receipt")
    write_json(output / "retention_result.json", {"complete": True, "states": len(records),
        "retained_states_sha256": sha256(output / "retained_states.json"), "source_updates": config["steps"],
        "independent_reconstruction_performed": False, "training_performed": False, "real_benchmark_success": False})
    write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--after-campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_available():
                raise RuntimeError("Require CPU-only collection")
            torch.set_num_threads(2)
            config = prepare(arguments.producer.resolve(), arguments.after_campaign.resolve(), output)
            execute(config, output)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
