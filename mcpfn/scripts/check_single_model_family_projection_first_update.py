"""Preserve a CPU-only first-update audit while the registered paired trial continues."""

import os
from pathlib import Path
import time

import torch

import recover_single_model_family_projection_pipeline as recovery
from recsys_evaluation_common import sha256, write_json


OUTPUT = recovery.pipeline.ROOT / "family_projection_v12_first_update_recovery1"


def execute(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("The independent first-update check must be CPU-only")
    training = recovery.audit.training
    config = training.verify_registration(training.PAIR)
    manifest_path = training.PAIR / "training_batches/step_001/manifest.json"
    manifest = training.read(manifest_path)
    rows = {name: recovery.audit.read_rows(arm / "training.jsonl")[0] for name, arm in training.ARMS.items()}
    sources = {**recovery.SOURCES, str(Path(__file__).resolve()): sha256(Path(__file__)),
               str(Path(recovery.__file__).resolve()): sha256(Path(recovery.__file__))}
    inputs = {str(training.PAIR / "config.json"): sha256(training.PAIR / "config.json"), str(manifest_path): sha256(manifest_path)}
    inputs.update({task["artifact"]: task["sha256"] for task in manifest["tasks"]})
    inputs.update({row["raw_snapshot"]: row["raw_snapshot_sha256"] for row in rows.values()})
    write_json(output / "config.json", {"scope": "Actual first update only, not the full trial, selection, confirmation or benchmarks",
        "source_sha256": sources, "input_sha256": inputs, "training_restarted": False, "autograd_replayed": False})
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"), "registered_unix": time.time()})
    recovery.verify_hashes(inputs)
    snapshots, results = {}, {}
    with recovery.runtime():
        for task, logged in zip(config["batches"][0]["tasks"], manifest["tasks"]):
            recovery.audit.check_episode(torch.load(logged["artifact"], map_location="cpu", weights_only=True), task, logged)
        sizes = [record[2] for record in training.read(training.PAIR / "parameter_inventory.json")]
        for name, row in rows.items():
            snapshot = torch.load(row["raw_snapshot"], map_location="cpu", weights_only=True, mmap=True)
            results[name] = {**recovery.audit.check_snapshot(snapshot, row, sizes, 1, config["base"]["weight_decay"]),
                "adverse_tasks_before": row["adverse_tasks_before"], "adverse_tasks_after": row["adverse_tasks_after"],
                "proposed_step_norm": row["proposed_step_norm"], "actual_step_norm": row["actual_step_norm"]}
            snapshots[name] = snapshot
            print(f"Checked {name}: {results[name]}", flush=True)
        diagnostics = recovery.pair_diagnostics(snapshots)
    diagnostics["first_forward_losses_exactly_equal"] = all(task["losses"]["control"] == task["losses"]["projected"] for task in manifest["tasks"])
    diagnostics["first_objective_diagnostics_exactly_equal"] = all(task["diagnostics"]["control"] == task["diagnostics"]["projected"] for task in manifest["tasks"])
    if any(recovery.audit.read_rows(arm / "training.jsonl")[0] != rows[name] for name, arm in training.ARMS.items()):
        raise ValueError("Completed first-update records changed during the independent audit")
    recovery.verify_hashes(inputs)
    recovery.verify_hashes(sources)
    write_json(output / "audit.json", {"complete": True, "scope": "first update only", "arms": results,
        "cached_training_episodes_checked": len(manifest["tasks"]), "raw_gradient_vectors_remeasured": 28,
        "initial_pair_diagnostics": diagnostics, "source_sha256": sources, "input_sha256": inputs,
        "source_mask_recovery": recovery.REASON, "optimizer_arithmetic_recovery": recovery.OPTIMIZER_REASON,
        "cuda_initialized": torch.cuda.is_initialized(), "autograd_replayed": False, "fused_cuda_adamw_replayed": False,
        "per_arm_roundoff_tolerances_changed": False, "research_selection_tolerance_changed": False,
        "training_restarted": False, "validation_improvement_not_assessed": True, "goal_success_not_assessed": True})


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    try:
        recovery.pipeline.status(OUTPUT, "auditing")
        execute(OUTPUT)
        recovery.pipeline.status(OUTPUT, "complete")
    except BaseException as error:
        recovery.pipeline.status(OUTPUT, "failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
