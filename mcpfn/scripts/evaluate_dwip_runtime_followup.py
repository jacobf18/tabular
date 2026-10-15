"""Report matched public-interface runtime for the fixed-selected DWIP after retention."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import statistics
import sys
import time

import psutil

from evaluate_dwip_retention_followup import verify_selected_checkpoint
from run_dwip_budgeted_queue_v1 import immutable_json, run_guarded, storage_guard
from run_dwip_spectral_geometry import dwip_spec
from run_dwip_transfer_evaluation import gpu_processes, sha256
from run_rowcol_controlled_experiments import REPO, WORK, write_json

SIZES = (100, 500, 1000)
VARIANTS = ("original", "selected")
WORKER = "scripts/benchmark_dwip_geometry_runtime.py"


def read_json(path):
    return json.loads(path.read_text())


def dependency_ready(root, handle):
    retention = root / "retention_evaluation"
    status = read_json(retention / "status.json")
    if status["state"] == "complete":
        for folder in (root, root / "transfer_evaluation", retention):
            if read_json(folder / "status.json")["state"] != "complete" or not (folder / "results.json").exists():
                raise RuntimeError("Runtime dependency completion lacks completed stages/results")
        return True
    if status["state"] in {"error", "cancelled_by_user"}:
        raise RuntimeError(f"Retention dependency failed: {status}")
    try:
        process = psutil.Process(handle["pid"])
        if process.create_time() != handle["created"] or process.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError("Original retention handle no longer live; no restart attempted")
    except psutil.NoSuchProcess as error:
        raise RuntimeError("Original retention handle absent; no restart attempted") from error
    return False


def worker_command(output, variant, size):
    return [sys.executable, WORKER, "--dwip-manifest", str(output / f"{variant}_models.json"),
        "--worker-model", "dwip", "--worker-size", str(size), "--worker-output", str(output / f"{variant}_{size}.json"),
        "--repeats", "3", "--warmup", "1", "--precision", "amp_bfloat16", "--missing-fraction", "0.8",
        "--seed", "20260915", "--threads", "4", "--memory-fraction", "0.9"]


def summarize(rows):
    indexed = {(row["variant"], row["size"]): row for row in rows}
    expected = {(variant, size) for variant in VARIANTS for size in SIZES}
    if len(indexed) != len(rows) or indexed.keys() != expected:
        raise ValueError("Require one matched result per variant and size")
    result = {}
    for size in SIZES:
        original, selected = (indexed[(variant, size)] for variant in VARIANTS)
        for row in (original, selected):
            if row["status"] != "ok" or len(row["seconds"]) != 3:
                raise ValueError("Require three successful timed public calls")
            values = [*row["seconds"], row["peak_allocated_gib"], row["peak_reserved_gib"]]
            if not all(math.isfinite(value) and value > 0 for value in values):
                raise ValueError("Nonfinite or nonpositive runtime/memory measurement")
            if row["warmup"] != 1 or row["repeats"] != 3 or row["precision"] != "amp_bfloat16":
                raise ValueError("Runtime protocol differs from registration")
        if not original["input_sha256"] or original["input_sha256"] != selected["input_sha256"]:
            raise ValueError("Matched runtime input hashes differ")
        if original["missing_cells"] != selected["missing_cells"] or original["rank"] != selected["rank"]:
            raise ValueError("Matched runtime input populations differ")
        before = statistics.median(original["seconds"])
        after = statistics.median(selected["seconds"])
        result[str(size)] = {"original_seconds": before, "selected_seconds": after,
            "latency_ratio": after / before, "original_peak_allocated_gib": original["peak_allocated_gib"],
            "selected_peak_allocated_gib": selected["peak_allocated_gib"],
            "input_sha256": original["input_sha256"], "missing_cells": original["missing_cells"]}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="dwip_spectral_anchor_mix25_64k_20260916")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = REPO / "runs/objective_queues" / args.queue
    work = WORK / f"tabimpute-{args.queue}-dwip1e5"
    output = root / "runtime_evaluation"
    output.mkdir(exist_ok=True)
    with (output / "evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "results.json").exists():
            raise ValueError("Runtime results already exist; inspect instead of overwriting")
        retention_launch = read_json(root / "retention_evaluation/launch.json")
        handle = {key: retention_launch[key] for key in ("pid", "created")}
        spec = dwip_spec()
        checkpoints = {"original": Path(spec["checkpoint_path"]), "selected": work / "best_validation_checkpoint.pth"}
        sources = [WORKER, "scripts/evaluate_dwip_runtime_followup.py", "scripts/evaluate_dwip_retention_followup.py",
            "scripts/run_dwip_budgeted_queue_v1.py", "scripts/run_dwip_transfer_evaluation.py",
            "scripts/run_dwip_spectral_geometry.py", "src/tabimpute/interface.py", "src/tabimpute/model/model_new_stable.py"]
        registration = {"retention_handle": handle, "queue_manifest_sha256": sha256(root / "manifest.json"),
            "source_sha256": {path: sha256(REPO / path) for path in sources},
            "initial_checkpoint_sha256": sha256(checkpoints["original"]), "model_config": spec["model_config"],
            "commands": {f"{variant}_{size}": worker_command(output, variant, size) for size in SIZES for variant in VARIANTS},
            "protocol": "Separate processes per variant/size; alternate model order by size; one warmup/three timed public calls, seed20260915,80% missing, rank10, default public SDPA, FP32 weights/BF16 compute. No windowing.100/500/1000 square matrices match the earlier runtime protocol.",
            "selection": "Only fixed-synthetic-selected weights after full transfer and frozen retention; runtime and finite-output checks are report-only. No benchmark-based model/checkpoint choice.",
            "scope": "Complete public impute latency includes normalization/transfers/CPU output; excludes load/warmup/copy. Three repeats are descriptive, not a significance or accuracy test. All artifacts remain inside the existing2GiB non-checkpoint allowance."}
        immutable_json(output / "registration.json", registration)
        if args.dry_run:
            print(json.dumps(registration, indent=2))
            return

        def status(state, **extra):
            write_json(output / "status.json", {"state": state, "updated_unix": time.time(), **extra})

        def verify_sources():
            if any(sha256(REPO / path) != expected for path, expected in registration["source_sha256"].items()):
                raise ValueError("Registered runtime source changed")
            if sha256(root / "manifest.json") != registration["queue_manifest_sha256"]:
                raise ValueError("Registered training manifest changed")

        try:
            while not dependency_ready(root, handle):
                status("waiting_for_retention", verified_live_launcher=handle["pid"])
                time.sleep(120)
            verify_sources()
            verify_selected_checkpoint(root, checkpoints["selected"])
            hashes = {name: sha256(path) for name, path in checkpoints.items()}
            if hashes["original"] != registration["initial_checkpoint_sha256"]:
                raise ValueError("Original checkpoint changed")
            if hashes["selected"] != read_json(root / "retention_evaluation/status.json")["checkpoint_sha256"]:
                raise ValueError("Selected weights differ from the completed retention panel")
            config = read_json(work / "config.json")
            if config["model_family"] != "new_stable" or config["model"] != spec["model_config"]:
                raise ValueError("Selected architecture differs from original DWIP")
            immutable_json(output / "checkpoint_sha256.json", hashes)
            for variant, checkpoint in checkpoints.items():
                immutable_json(output / f"{variant}_models.json", {"models": [{"name": "dwip-v2-2",
                    "checkpoint_path": str(checkpoint), "model_config": spec["model_config"], "model_family": "new_stable"}]})
            budget = read_json(root / "artifact_budget.json")
            rows = []
            for index, size in enumerate(SIZES):
                for variant in VARIANTS if index % 2 == 0 else reversed(VARIANTS):
                    while active := gpu_processes():
                        status("waiting_for_idle_gpu", observed_gpu_pids=active)
                        time.sleep(60)
                    verify_sources()
                    status("evaluating", variant=variant, size=size)
                    target = output / f"{variant}_{size}.json"
                    if target.exists():
                        raise ValueError("Partial runtime output exists; inspect instead of overwriting")
                    command = ["timeout", "--signal=TERM", "--kill-after=120s", "15m", *worker_command(output, variant, size)]
                    run_guarded(command, output / f"{variant}_{size}.log", lambda: storage_guard(root, work, budget))
                    rows.append({**read_json(target), "variant": variant})
                    write_json(output / "partial_results.json", rows)
            verify_sources()
            if any(sha256(path) != hashes[name] for name, path in checkpoints.items()):
                raise ValueError("Checkpoint changed during runtime measurement")
            result = {"rows": rows, "comparisons": summarize(rows), "scope": registration["scope"]}
            write_json(output / "results.json", result)
            status("complete", checkpoint_sha256=hashes["selected"], timed_calls=18, warmup_calls=6)
            print(json.dumps(result["comparisons"], indent=2))
        except Exception as error:
            status("error", error=repr(error))
            raise


if __name__ == "__main__":
    main()
