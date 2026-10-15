"""Matched, report-only real-world transfer after the unchanged DWIP training queue."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil

from run_rowcol_controlled_experiments import REPO, WORK, BENCH, write_json

DATASETS = ("mmlu", "merged", "benchpress")
METHODS = ("entropy", "mi", "random")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluation_environment(output, checkpoint, label, dataset):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("BENCHSELECT_")}
    environment.update({"PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}", "WANDB_MODE": "disabled",
        "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "BENCHSELECT_DATA_DIR": str(BENCH / "data"), "BENCHSELECT_EXPERIMENT_ROOT": str(output),
        "BENCHSELECT_TABIMPUTE_MODEL": "direct_checkpoint", "BENCHSELECT_TABIMPUTE_MODEL_FAMILY": "new_stable",
        "BENCHSELECT_TABIMPUTE_CHECKPOINT": str(checkpoint), "BENCHSELECT_TABIMPUTE_LABEL": label,
        "BENCHSELECT_TABIMPUTE_DEVICE": "cuda", "BENCHSELECT_TABIMPUTE_PRECISION_MODE": "amp_bfloat16",
        "BENCHSELECT_TABIMPUTE_PREDICTION_STAT": "mean", "BENCHSELECT_DATASETS": dataset,
        "BENCHSELECT_METHODS": ",".join(METHODS), "BENCHSELECT_N_FOLDS": "10",
        "BENCHSELECT_HOLDOUT_FRAC": "0.1", "BENCHSELECT_K_MAX": "15",
        "BENCHSELECT_MAX_ROWS": "4000", "BENCHSELECT_TEST_BATCH": "200"})
    return environment


def compare_fold_metrics(baseline, selected):
    if not np.array_equal(baseline["benchmark_names"], selected["benchmark_names"]):
        raise ValueError("Benchmark identities differ")
    result = {}
    for method in METHODS:
        for metric in ("r2", "rmse"):
            key = f"{method}_{metric}"
            before, after = np.asarray(baseline[key]), np.asarray(selected[key])
            if before.shape != (10, 15) or after.shape != before.shape:
                raise ValueError(f"Expected matched 10-fold, 15-budget results: {key}")
            if not np.isfinite(before).all() or not np.isfinite(after).all():
                raise ValueError(f"Incomplete or nonfinite folds: {key}")
            for budget in (5, 10, 15):
                baseline_values, selected_values = before[:, budget - 1], after[:, budget - 1]
                delta = selected_values - baseline_values
                result[f"{key}_{budget}"] = {
                    "baseline": float(baseline_values.mean()), "selected": float(selected_values.mean()),
                    "delta": float(delta.mean()), "fold_deltas": delta.tolist(),
                    "folds_improved": int((delta > 0 if metric == "r2" else delta < 0).sum()),
                    "fold_count": 10,
                }
    return result


def dependency_state(root, expected_pid, expected_created):
    state = json.loads((root / "status.json").read_text())
    if state["state"] == "complete":
        if not (root / "dwip1e5_result.json").exists():
            raise RuntimeError("Dependency claims completion without its result")
        return True
    if state["state"] in {"error", "cancelled_by_user"}:
        raise RuntimeError(f"Dependency terminated: {state}")
    try:
        process = psutil.Process(expected_pid)
        if process.create_time() != expected_created or process.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError("Original DWIP launcher is no longer live; no restart attempted")
    except psutil.NoSuchProcess as error:
        raise RuntimeError("Original DWIP launcher is absent; no restart attempted") from error
    return False


def gpu_processes():
    return subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                   text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="dwip_spectral_20260915")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = REPO / "runs/objective_queues" / args.queue
    output = root / "transfer_evaluation"
    output.mkdir(exist_ok=True)
    lock = (output / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    work = WORK / f"tabimpute-{args.queue}-dwip1e5"
    checkpoints = {"baseline": root / "baseline/checkpoint.pth", "selected": work / "best_validation_checkpoint.pth"}
    sources = {"evaluator": BENCH / "code/eval_tabimpute.py", "interface": REPO / "src/tabimpute/interface.py",
               "model": REPO / "src/tabimpute/model/model_new_stable.py", "driver": Path(__file__)}
    manifest = {"datasets": list(DATASETS), "data_sha256": {
        name: sha256(BENCH / "data" / f"{name}.matrix.npz") for name in DATASETS},
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
        "checkpoints": {label: str(path) for label, path in checkpoints.items()},
        "selection": "Original synthetic-selected checkpoint only; all real-world scores report-only.",
        "protocol": "10 folds, seed42, holdout0.1, k1-15, entropy/MI/random, max4000 rows, testbatch200",
        "precision": "FP32 weights, BF16 autocast, mean predictions; unchanged DWIP architecture",
        "gpu_policy": "Wait for original queue completion and an idle GPU; do not stop other jobs."}
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Transfer evaluation manifest changed; do not reuse old results")
    write_json(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    def status(state, **extra):
        write_json(output / "status.json", {"state": state, "updated_unix": time.time(), **extra})

    try:
        launcher_pid = int((root / "launcher.pid").read_text())
        queue_complete = json.loads((root / "status.json").read_text())["state"] == "complete"
        launcher_created = None if queue_complete else psutil.Process(launcher_pid).create_time()
        while not dependency_state(root, launcher_pid, launcher_created):
            status("waiting_for_training_queue", verified_live_launcher=launcher_pid)
            time.sleep(60)
        if json.loads((checkpoints["baseline"].parent / "config.json").read_text())["model"] != json.loads(
                (checkpoints["selected"].parent / "config.json").read_text())["model"]:
            raise ValueError("Baseline and selected model architectures differ")
        checkpoint_hashes = {label: sha256(path) for label, path in checkpoints.items()}
        hash_path = output / "checkpoint_sha256.json"
        if hash_path.exists() and json.loads(hash_path.read_text()) != checkpoint_hashes:
            raise ValueError("Checkpoint weights changed since the cached evaluation")
        write_json(hash_path, checkpoint_hashes)
        results = {}
        for dataset in DATASETS:
            data_path = BENCH / "data" / f"{dataset}.matrix.npz"
            if sha256(data_path) != manifest["data_sha256"][dataset]:
                raise RuntimeError("Evaluation data changed after preregistration")
            cached = {}
            for label, checkpoint in checkpoints.items():
                trial_label = f"dwip_transfer_{label}"
                cache = output / "logs" / f"tabimpute_{trial_label}_{dataset}.npz"
                if not cache.exists():
                    while active := gpu_processes():
                        status("waiting_for_idle_gpu", dataset=dataset, variant=label, observed_gpu_pids=active)
                        time.sleep(60)
                    if any(sha256(path) != manifest["source_sha256"][name] for name, path in sources.items()):
                        raise RuntimeError("Evaluation source changed after preregistration")
                    status("evaluating", dataset=dataset, variant=label)
                    started = time.monotonic()
                    with (output / f"{dataset}_{label}.log").open("a") as log:
                        subprocess.run([sys.executable, "code/eval_tabimpute.py"], cwd=BENCH,
                            env=evaluation_environment(output, checkpoint, trial_label, dataset),
                            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=6 * 3600)
                    write_json(output / f"{dataset}_{label}_runtime.json", {"seconds": time.monotonic() - started})
                if sha256(checkpoint) != checkpoint_hashes[label]:
                    raise RuntimeError("Checkpoint changed during evaluation")
                if sha256(data_path) != manifest["data_sha256"][dataset]:
                    raise RuntimeError("Evaluation data changed during evaluation")
                with np.load(cache, allow_pickle=True) as data:
                    cached[label] = {key: data[key] for key in data.files}
            results[dataset] = compare_fold_metrics(cached["baseline"], cached["selected"])
            write_json(output / "results.json", results)
        status("complete", datasets=list(results))
    except Exception as error:
        status("error", error=repr(error))
        raise


if __name__ == "__main__":
    main()
