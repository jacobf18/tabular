"""Report the full canonical panel only after a cross-seed average passes its gate."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil

from run_dwip_transfer_evaluation import (
    BENCH, REPO, WORK, compare_fold_metrics, evaluation_environment, gpu_processes,
    sha256, write_json,
)

DATASETS = ("mteb", "mmlu", "merged", "benchpress")
BASELINE_QUEUE = "dwip_spectral_20260915"


def read_json(path):
    return json.loads(path.read_text())


def accepted_average(root):
    status = read_json(root / "status.json")
    if status["state"] in {"error", "superseded_without_evaluation", "cancelled_by_user"}:
        raise RuntimeError(f"Average dependency terminated: {status}")
    if status["state"] != "synthetic_evaluation_complete":
        return None
    selection = read_json(root / "selection.json")
    scores = selection["macro_rmse"]
    if set(scores) != {"seed1", "seed2", "average"} or not all(
            math.isfinite(value) and value >= 0 for value in scores.values()):
        raise ValueError("Require finite nonnegative scores for both raw models and the average")
    accepted = scores["average"] < min(scores["seed1"], scores["seed2"])
    expected = "average" if accepted else min(("seed1", "seed2"), key=scores.get)
    if (selection.get("average_accepted") is not accepted
            or status.get("realworld_evaluation_allowed") is not accepted
            or selection.get("selected_variant") != expected
            or status.get("selected_variant") != expected):
        raise ValueError("Synthetic gate decision and benchmark permission disagree")
    return accepted


def require_live(handle):
    try:
        process = psutil.Process(handle["pid"])
        if process.create_time() != handle["created"] or process.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError("Average launcher is no longer live; no restart attempted")
    except psutil.NoSuchProcess as error:
        raise RuntimeError("Average launcher is absent; no restart attempted") from error


def immutable_json(path, content):
    if path.exists() and read_json(path) != content:
        raise ValueError(f"Registered inputs changed: {path}")
    write_json(path, content)


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Frozen input changed: {name}")


def baseline_cache(dataset):
    if dataset == "mteb":
        return BENCH / "logs" / f"tabimpute_{BASELINE_QUEUE}_baseline_mteb.npz"
    return (REPO / "runs/objective_queues" / BASELINE_QUEUE / "transfer_evaluation/logs"
            / f"tabimpute_dwip_transfer_baseline_{dataset}.npz")


def raw_cache(queue, dataset):
    if dataset == "mteb":
        return BENCH / "logs" / f"tabimpute_{queue}_dwip1e5_mteb.npz"
    return (REPO / "runs/objective_queues" / queue / "transfer_evaluation/logs"
            / f"tabimpute_dwip_transfer_selected_{dataset}.npz")


def load_cache(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def prepare_panel(root, registration):
    if accepted_average(root) is not True:
        raise RuntimeError("No benchmark evaluation without synthetic acceptance")
    verify_hashes(registration["input_sha256"])
    manifest = read_json(root / "manifest.json")
    verify_hashes(manifest["source_sha256"])
    if manifest["prespecification_sha256"] != sha256(root / "prespecification.json"):
        raise ValueError("Average prespecification changed")
    checkpoint = root / "candidate/checkpoint.pth"
    if sha256(checkpoint) != read_json(root / "candidate_sha256.json")["sha256"]:
        raise ValueError("Average checkpoint integrity mismatch")
    files = [root / name for name in (
        "prespecification.json", "manifest.json", "selection.json", "status.json",
        "candidate_sha256.json", "candidate/checkpoint.pth", "candidate/config.json",
        "average_spectral.json", "average_general.json",
    )]
    decision = read_json(root / "selection.json")
    for label, score in decision["macro_rmse"].items():
        path = root / f"{label}_validation.json"
        if read_json(path)["macro_rmse"] != score:
            raise ValueError("Decision does not match fresh synthetic scores")
        files.append(path)
    references = {dataset: {"baseline": baseline_cache(dataset), **{
        label: raw_cache(queue, dataset) for label, queue in manifest["queues"].items()
    }} for dataset in DATASETS}
    for label, queue in manifest["queues"].items():
        work = WORK / f"tabimpute-{queue}-dwip1e5"
        raw = work / "best_validation_checkpoint.pth"
        if sha256(raw) != manifest["checkpoint_sha256"][label]:
            raise ValueError("Raw selected checkpoint changed")
        if read_json(root / "candidate/config.json")["model"] != read_json(work / "config.json")["model"]:
            raise ValueError("Candidate and raw architectures differ")
        files.extend((raw, work / "config.json"))
    for paths in references.values():
        for path in paths.values():
            cached = load_cache(path)
            compare_fold_metrics(cached, cached)
            files.append(path)
    hashes = {**registration["input_sha256"], **manifest["source_sha256"],
              **{str(path): sha256(path) for path in files}}
    immutable_json(root / "transfer_evaluation/manifest.json", {
        "datasets": list(DATASETS), "input_sha256": hashes,
        "protocol": "Canonical 10 folds, seed42, holdout0.1, k1-15, entropy/mi/random; FP32/BF16 mean",
        "selection": "Synthetic-accepted equal parameter average only; all real-world scores report-only",
    })
    return checkpoint, references, hashes


def run(root, handle):
    output = root / "transfer_evaluation"
    output.mkdir(exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def status(state, **extra):
            write_json(output / "status.json", {"state": state, "updated_unix": time.time(), **extra})

        try:
            sources = [Path(__file__), REPO / "scripts/run_dwip_transfer_evaluation.py",
                       BENCH / "code/eval_tabimpute.py", REPO / "src/tabimpute/interface.py",
                       REPO / "src/tabimpute/model/model_new_stable.py",
                       REPO / "src/tabimpute/model/bar_distribution.py", root / "prespecification.json"]
            sources.extend(BENCH / "data" / f"{dataset}.matrix.npz" for dataset in DATASETS)
            registration = {"dependency": handle, "datasets": list(DATASETS),
                            "input_sha256": {str(path): sha256(path) for path in sources}}
            immutable_json(output / "registration.json", registration)
            while (accepted := accepted_average(root)) is None:
                require_live(handle)
                status("waiting_for_synthetic_gate", verified_live_launcher=handle["pid"])
                time.sleep(120)
            if not accepted:
                status("skipped_rejected_average", benchmark_evaluation_started=False)
                return
            checkpoint, references, hashes = prepare_panel(root, registration)
            results = {}
            for dataset in DATASETS:
                label = "dwip_cross_seed_average"
                cache = output / "logs" / f"tabimpute_{label}_{dataset}.npz"
                if not cache.exists():
                    while active := gpu_processes():
                        status("waiting_for_idle_gpu", dataset=dataset, observed_gpu_pids=active)
                        time.sleep(60)
                    verify_hashes(hashes)
                    status("evaluating", dataset=dataset)
                    started = time.monotonic()
                    with (output / f"{dataset}_average.log").open("a") as log:
                        subprocess.run([sys.executable, "code/eval_tabimpute.py"], cwd=BENCH,
                            env=evaluation_environment(output, checkpoint, label, dataset),
                            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=6 * 3600)
                    write_json(output / f"{dataset}_runtime.json", {"seconds": time.monotonic() - started})
                verify_hashes(hashes)
                selected = load_cache(cache)
                results[dataset] = {name: compare_fold_metrics(load_cache(path), selected)
                                    for name, path in references[dataset].items()}
                write_json(output / "results.json", results)
            status("complete", datasets=list(results), benchmark_evaluation_started=True)
        except Exception as error:
            status("error", error=repr(error))
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="dwip_point2_cross_seed_average_20260916_v2")
    parser.add_argument("--dependency-pid", type=int, required=True)
    parser.add_argument("--dependency-created", type=float, required=True)
    args = parser.parse_args()
    run(REPO / "runs/objective_queues" / args.experiment,
        {"pid": args.dependency_pid, "created": args.dependency_created})


if __name__ == "__main__":
    main()
