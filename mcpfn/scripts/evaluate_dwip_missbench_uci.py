"""Evaluate a fixed DWIP checkpoint on the original frozen MissBench and UCI masks."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np


REPO = Path(__file__).resolve().parents[1]
PATTERNS = ("MCAR", "MAR", "MNAR", "MAR_Neural", "MAR_BlockNeural", "MNARPolarizationPattern",
            "MNARSoftPolarizationPattern", "MNARLatentFactorPattern", "MNARClusterLevelPattern",
            "MNARTwoPhaseSubsetPattern", "MNARCensoringPattern")
CHECKPOINT = REPO / "src/tabimpute/workdir/tabimpute-dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918-dwip1e5/best_validation_checkpoint.pth"
METRICS = ("nrmse_original", "corr2_original", "r2_columnwise", "rmse", "mae", "rmse_observed_standardized")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def discover_cases(data_root):
    cases = []
    for suite, folder, expected in (("missbench", "openml", 42), ("uci", "uci", 12)):
        directories = sorted(path for path in (data_root / folder).iterdir() if path.is_dir())
        if len(directories) != expected:
            raise ValueError(f"Expected {expected} {suite} datasets, found {len(directories)}")
        for directory in directories:
            settings = [(pattern, 0.4, None, directory / f"{pattern}_0.4") for pattern in PATTERNS] if suite == "missbench" else [
                (pattern, rate / 10, repeat, directory / pattern / f"missingness-{rate / 10}" / f"repeat-{repeat}")
                for pattern in ("MCAR", "MAR", "MNAR") for rate in range(1, 6) for repeat in range(10)]
            for pattern, rate, repeat, path in settings:
                for filename in ("missing.npy", "true.npy"):
                    if not (path / filename).is_file():
                        raise FileNotFoundError(path / filename)
                key = f"{suite}/{directory.name}/{pattern}_{rate}" + (f"/repeat-{repeat}" if repeat is not None else "")
                cases.append({"case_id": key, "suite": suite, "dataset": directory.name, "pattern": pattern,
                              "missingness": rate, "repeat": repeat, "source_directory": str(path)})
    return cases


def score_predictions(true, prediction, missing, suite):
    if true.shape != prediction.shape or true.shape != missing.shape:
        raise ValueError("Mismatched prediction/data shapes")
    if not np.isfinite(true).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite ground truth or predictions")
    mask = np.isnan(missing)
    if not mask.any():
        raise ValueError("No held-out cells")
    nrmse, correlations, conventional = [], [], []
    nrmse_skipped = 0
    threshold = 1e-3 if suite == "missbench" else 1e-4
    for column in range(true.shape[1]):
        selected = mask[:, column]
        if not selected.any():
            continue
        truth = true[selected, column]
        predicted = prediction[selected, column]
        error = predicted - truth
        rmse = np.sqrt(np.mean(error**2))
        scale = np.std(truth)
        if scale < threshold:
            if rmse < threshold:
                nrmse.append(0.0)
            else:
                nrmse_skipped += 1
        else:
            nrmse.append(float(rmse / scale))
        if selected.sum() < 2:
            continue
        if np.isclose(scale, 0, atol=1e-3) or np.isclose(np.std(predicted), 0, atol=1e-3):
            correlations.append(0.0)
        else:
            correlation = np.corrcoef(predicted, truth)[0, 1] ** 2
            correlations.append(float(correlation) if np.isfinite(correlation) else 0.0)
        truth64 = truth.astype(np.float64)
        squared_error = float(np.sum((predicted.astype(np.float64) - truth64)**2))
        denominator = float(np.sum((truth64 - truth64.mean())**2))
        conventional.append(1.0 - squared_error / denominator if denominator > 0 else float(squared_error == 0))
    error = prediction[mask].astype(np.float64) - true[mask].astype(np.float64)
    observed_scale = np.nanstd(missing.astype(np.float64), axis=0)
    observed_scale = np.where(np.isfinite(observed_scale) & (observed_scale > 1e-12), observed_scale, 1.0)
    standardized_error = ((prediction - true) / observed_scale)[mask]
    return {"nrmse_original": float(np.mean(nrmse)) if nrmse else 0.0,
            "corr2_original": float(np.mean(correlations)) if correlations else (0.0 if suite == "missbench" else None),
            "r2_columnwise": float(np.mean(conventional)) if conventional else None,
            "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error))),
            "rmse_observed_standardized": float(np.sqrt(np.mean(standardized_error**2))),
            "missing_cells": int(mask.sum()), "nrmse_scored_columns": len(nrmse),
            "nrmse_skipped_constant_columns": nrmse_skipped, "r2_scored_columns": len(conventional)}


def standardize_uci(missing):
    mean = np.nanmean(missing, axis=0)
    scale = np.nanstd(missing, axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("UCI preprocessing has undefined mean/std")
    constant_columns = np.flatnonzero(scale == 0).tolist()
    scale = np.where(scale == 0, 1.0, scale)
    return (missing - mean) / scale, mean, scale, constant_columns


def summarize(output, rows, expected):
    successful = [row for row in rows if row["status"] == "ok"]
    groups = defaultdict(list)
    for row in successful:
        groups[row["suite"], "overall", "all", "all"].append(row)
        groups[row["suite"], "pattern", row["pattern"], "all"].append(row)
        groups[row["suite"], "pattern_rate", row["pattern"], str(row["missingness"])].append(row)
    summaries = []
    for (suite, level, pattern, rate), group in sorted(groups.items()):
        result = {"suite": suite, "level": level, "pattern": pattern, "missingness": rate,
                  "cases": len(group), "datasets": len({row["dataset"] for row in group})}
        for metric in METRICS:
            per_dataset = defaultdict(list)
            for row in group:
                if row[metric] is not None and np.isfinite(row[metric]):
                    per_dataset[row["dataset"]].append(row[metric])
            values = [float(np.mean(values)) for values in per_dataset.values()]
            result[f"mean_{metric}"] = float(np.mean(values)) if values else None
            result[f"median_{metric}"] = float(np.median(values)) if values else None
        summaries.append(result)
    for filename, records in (("per_case_metrics.csv", rows), ("summary_metrics.csv", summaries)):
        if records:
            fields = list(dict.fromkeys(key for row in records for key in row))
            temporary = output / f"{filename}.tmp"
            with temporary.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
            temporary.replace(output / filename)
    sections = ["# New DWIP: MissBench and UCI\n",
                f"Successful cases: {len(successful)}/{expected}. Failed cases: {len(rows) - len(successful)}.\n",
                "Fixed synthetic-selected step-98000 checkpoint from the 128000-step run; report-only evaluation. "
                "No MCAR rate sweep on MissBench. Original frozen masks and ground truth are unchanged.\n",
                "NRMSE is the original mean columnwise masked-cell RMSE divided by masked ground-truth standard deviation. "
                "The original scripts call squared Pearson correlation ‘R²’; it is explicitly labeled Corr² here. "
                "Conventional R² is also computed per column as 1 − SSE/SST. "
                "Aggregate metrics first average repeats/settings within each dataset, then weight datasets equally. "
                "Raw-scale RMSE/MAE are in the CSV and should not be compared across datasets with different units.\n",
                "| Suite | Group | Missing fraction | Cases | NRMSE ↓ | Original Corr² ↑ | Conventional R² ↑ |",
                "|---|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        if row["level"] == "pattern_rate" and row["suite"] == "missbench":
            continue
        def display(value):
            return "NA" if value is None else f"{value:.6f}"
        sections.append(f"| {row['suite']} | {row['level']}: {row['pattern']} | {row['missingness']} | {row['cases']} | "
                        f"{display(row['mean_nrmse_original'])} | {display(row['mean_corr2_original'])} | {display(row['mean_r2_columnwise'])} |")
    (output / "RESULTS.md").write_text("\n".join(sections) + "\n")
    return summaries


def run(args):
    import torch
    from tabimpute.interface import ImputePFN

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".evaluation.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cases = discover_cases(args.data_root.resolve())
    if args.datasets:
        cases = [case for case in cases if case["dataset"] in args.datasets]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("No selected cases")
    checkpoint_hash = sha256(args.checkpoint)
    config_path = args.checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text())
    protocol = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
                "config_sha256": sha256(config_path), "script_sha256": sha256(Path(__file__)),
                "model_family": "new_stable", "prediction_stat": "mean", "precision": "amp_bfloat16",
                "max_num_rows": None, "max_num_cols": None, "preprocessors": None,
                "uci_external_standardization": True, "missbench_external_standardization": False,
                "seed": 42, "cudnn_sdpa": False, "cases": cases,
                "prediction_format": "Compressed NPZ: missing_values in C-order mask traversal, plus shape; reconstruct observed cells from missing.npy",
                "scope": "MissBench: 42 datasets x 11 mechanisms at 0.4; UCI: 12 datasets x 3 mechanisms x 5 rates x 10 repeats"}
    protocol["uci_zero_std_policy"] = "Use scale 1 for exactly zero observed-column std; no change to masks or scoring"
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["protocol"] != protocol:
        prior_manifest = json.loads(manifest_path.read_text())
        comparable = dict(protocol)
        comparable.pop("uci_zero_std_policy")
        comparable["script_sha256"] = prior_manifest["protocol"]["script_sha256"]
        archive = output / "initial_attempt"
        if not args.resume_after_zero_std_fix or prior_manifest["protocol"] != comparable:
            raise ValueError("Existing output has a different protocol; use a new output directory")
        if sha256(archive / "evaluate_dwip_missbench_uci.py") != prior_manifest["protocol"]["script_sha256"]:
            raise ValueError("Original evaluator archive is missing or changed")
        failures = [json.loads(path.read_text()) for path in (output / "cases").rglob("*.json")
                    if json.loads(path.read_text())["status"] != "ok"]
        if not failures or any(row.get("error") != "Original UCI preprocessing has undefined mean/std" for row in failures):
            raise ValueError("This migration only handles the recorded zero-std preprocessing failures")
        prior_manifest.update({"protocol": protocol, "revision_utc": datetime.now(timezone.utc).isoformat(),
                               "previous_manifest": "initial_attempt/manifest.json",
                               "revision_reason": "Safe scale=1 for exactly constant observed UCI columns; successful nonconstant cases unchanged"})
        write_json(manifest_path, prior_manifest)
    if not manifest_path.exists():
        write_json(manifest_path, {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
                                  "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                                  "gpu": torch.cuda.get_device_name(), "disk_reserve_gib": args.disk_reserve_gib})
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(0.9)
    imputer = ImputePFN(device="cuda", checkpoint_path=str(args.checkpoint), entry_wise_features=False,
                        json_config={"config": config.get("model", config.get("config"))}, model_family="new_stable",
                        precision_mode="amp_bfloat16", prediction_stat="mean", max_num_rows=None, max_num_cols=None)
    started = time.time()
    rows = []
    for index, case in enumerate(cases):
        if shutil.disk_usage(output).free < args.disk_reserve_gib * 1024**3:
            raise RuntimeError("Disk reserve reached; evaluation stopped without deleting artifacts")
        result_path = output / "cases" / (case["case_id"] + ".json")
        prediction_path = output / "predictions" / (case["case_id"] + ".npz")
        source = Path(case["source_directory"])
        input_hashes = {"missing_sha256": sha256(source / "missing.npy"), "true_sha256": sha256(source / "true.npy")}
        if result_path.exists():
            prior = json.loads(result_path.read_text())
            if prior["status"] == "ok":
                if any(prior[key] != value for key, value in input_hashes.items()) or sha256(prediction_path) != prior["prediction_sha256"]:
                    raise ValueError(f"Changed cached inputs/predictions: {case['case_id']}")
                rows.append(prior)
                continue
        write_json(output / "status.json", {"state": "running", "pid": os.getpid(), "completed": len(rows),
                   "expected": len(cases), "current": case["case_id"], "elapsed_seconds": time.time() - started})
        row = {**case, **input_hashes, "status": "ok"}
        try:
            missing = np.load(source / "missing.npy", allow_pickle=False)
            true = np.load(source / "true.npy", allow_pickle=False)
            if not np.isfinite(true).all() or np.isinf(missing).any():
                raise ValueError("Invalid source arrays")
            if case["suite"] == "uci":
                model_input, mean, scale, constant_columns = standardize_uci(missing)
                row["zero_std_columns"] = constant_columns
            else:
                model_input = missing.copy()
            torch.manual_seed(42)
            np.random.seed(42)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            case_started = time.perf_counter()
            prediction = imputer.impute(model_input.copy())
            torch.cuda.synchronize()
            row["inference_seconds"] = time.perf_counter() - case_started
            row["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
            if case["suite"] == "uci":
                prediction = prediction * scale + mean
            row.update(score_predictions(true, prediction, missing, case["suite"]))
            row.update({"rows": int(true.shape[0]), "columns": int(true.shape[1]),
                        "observed_max_abs_change": float(np.max(np.abs(prediction[~np.isnan(missing)] - missing[~np.isnan(missing)])))})
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = prediction_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, missing_values=prediction[np.isnan(missing)], shape=np.asarray(missing.shape))
            temporary.replace(prediction_path)
            row.update({"prediction_path": str(prediction_path.relative_to(output)), "prediction_sha256": sha256(prediction_path)})
        except Exception as error:
            row.update({"status": "failed", "error": str(error), "traceback": traceback.format_exc()})
            torch.cuda.empty_cache()
        write_json(result_path, row)
        rows.append(row)
        print(f"[{index + 1}/{len(cases)}] {case['case_id']} {row['status']} "
              f"seconds={row.get('inference_seconds', 0):.3f} nrmse={row.get('nrmse_original')}", flush=True)
        if (index + 1) % 25 == 0:
            summarize(output, rows, len(cases))
    summarize(output, rows, len(cases))
    if sha256(args.checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed during evaluation")
    failures = sum(row["status"] != "ok" for row in rows)
    write_json(output / "status.json", {"state": "complete" if failures == 0 else "complete_with_failures", "pid": os.getpid(),
               "completed": len(rows), "successful": len(rows) - failures, "failed": failures, "expected": len(cases),
               "elapsed_seconds": time.time() - started, "checkpoint_sha256": checkpoint_hash,
               "free_disk_gib": shutil.disk_usage(output).free / 1024**3})
    if failures:
        raise RuntimeError(f"{failures} evaluation cases failed; see per-case records")
    if (output / "fatal_error.json").exists() and (output / "initial_attempt").is_dir():
        (output / "fatal_error.json").replace(output / "initial_attempt/fatal_error_prior_to_successful_resume.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=REPO / "benchmark/datasets")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--disk-reserve-gib", type=float, default=20.0)
    parser.add_argument("--resume-after-zero-std-fix", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        write_json(args.output / "fatal_error.json", {"error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
