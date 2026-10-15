"""Frozen MissBench/UCI and LLM transfer comparison for the recsys specialist."""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from audit_dwip_missbench_uci import load_original
from compare_imputation_checkpoint import aggregate
from compare_llm_checkpoint import load_panel
from evaluate_chunked_uci import configure_windows
from evaluate_dwip_missbench_uci import PATTERNS, REPO, score_predictions, sha256, standardize_uci, write_json
from recsys_evaluation_common import atomic_npz, write_csv
from run_dwip_transfer_evaluation import BENCH, METHODS, evaluation_environment


ROOT = REPO / "runs/benchmark_evaluations"
CHECKPOINT = REPO / "runs/recsys_goal_20260920/synthetic_user_scaled_parent_risk_400/best_validation_checkpoint.pth"
GENERAL = REPO / "runs/streamed_training/large_matrices_98k_20260930/best_validation_checkpoint.pth"
FULL = ROOT / "matrixformer_large_trained_missbench_uci_20260930"
CHUNKED = ROOT / "matrixformer_uci_chunks2000_20260930"
LLM = ROOT / "matrixformer_large_trained_llm_20261001"
OUTPUT = ROOT / "recsys_specialist_transfer_20261002"
DATASETS = ("mteb", "mmlu", "merged", "benchpress")
LABEL = "recsys_specialist_transfer"


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_hashes(hashes):
    for filename, expected in hashes.items():
        if sha256(Path(filename)) != expected:
            raise ValueError(f"Frozen artifact changed: {filename}")


def register(output):
    full = read_json(FULL / "manifest.json")["protocol"]
    chunked = read_json(CHUNKED / "manifest.json")
    llm = read_json(LLM / "manifest.json")
    general_hash = sha256(GENERAL)
    if not general_hash == full["checkpoint_sha256"] == chunked["checkpoint_sha256"] == llm["checkpoint_sha256"]:
        raise ValueError("Reference suites do not use the same general checkpoint")
    if chunked["context_shape"] != [2000, 1000]:
        raise ValueError("Unexpected archived UCI context")
    specialist_config = read_json(CHECKPOINT.parent / "config.json")
    if specialist_config["model"] != read_json(GENERAL.parent / "config.json")["model"]:
        raise ValueError("Checkpoint architecture mismatch")
    for dataset in DATASETS:
        path = BENCH / "data" / f"{dataset}.matrix.npz"
        if sha256(path) != llm["input_sha256"][str(path)]:
            raise ValueError("LLM benchmark inputs changed")
    sources = [Path(__file__).resolve(), *sorted((REPO / "src/tabimpute").rglob("*.py")),
               *sorted((BENCH / "code").glob("*.py"))]
    sources.extend(REPO / "scripts" / filename for filename in (
        "audit_dwip_missbench_uci.py", "compare_imputation_checkpoint.py", "compare_llm_checkpoint.py",
        "evaluate_chunked_uci.py", "evaluate_dwip_missbench_uci.py", "recsys_evaluation_common.py",
        "run_dwip_transfer_evaluation.py", "run_rowcol_controlled_experiments.py"))
    sources.extend([REPO / "benchmark/plot_options.py", REPO / "benchmark/uci/plot_uci_negative_rmse.py"])
    inputs = [CHECKPOINT, CHECKPOINT.parent / "config.json", GENERAL, GENERAL.parent / "config.json",
              FULL / "manifest.json", CHUNKED / "manifest.json", LLM / "manifest.json", LLM / "audit.json"]
    inputs.extend(BENCH / "data" / f"{dataset}.matrix.npz" for dataset in DATASETS)
    inputs.extend(LLM / "logs" / f"tabimpute_matrixformer_large_trained_{dataset}.npz" for dataset in DATASETS)
    protocol = {
        "specialist_checkpoint": str(CHECKPOINT), "specialist_sha256": sha256(CHECKPOINT),
        "general_checkpoint": str(GENERAL), "general_sha256": general_hash,
        "input_sha256": {str(path): sha256(path) for path in inputs},
        "source_sha256": {str(path): sha256(path) for path in sources},
        "tabular_cases": full["cases"], "uci_context": [2000, 1000], "missbench_context": "full matrix",
        "normalization": "Each frozen benchmark's column normalization; no recsys user scaling or rating clipping",
        "llm": "Four datasets; ten folds; entropy/MI/random; budgets 1-15; 4000 rows and 200 test rows",
        "selection": "Previously selected checkpoints, report-only; no tuning or neural training",
        "scope": "MissBench, UCI and LLM score completion; not causal inference or the separate R-method suite",
    }
    path = output / "manifest.json"
    if path.exists():
        if read_json(path) != protocol:
            raise ValueError("Resume manifest differs from the frozen campaign")
    else:
        write_json(path, protocol)
        for source in sources:
            target = output / "source_snapshot" / source.relative_to(REPO.parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return protocol


def reference_record(case):
    root = CHUNKED if case["suite"] == "uci" else FULL
    row = read_json(root / "cases" / (case["case_id"] + ".json"))
    if row["status"] != "ok" or row["case_id"] != case["case_id"]:
        raise ValueError("Incomplete or mismatched reference case")
    row["nrmse_original"] = row["new_nrmse"] if case["suite"] == "uci" else row["nrmse_original"]
    return root, row


def decode_prediction(path, missing):
    with np.load(path, allow_pickle=False) as stored:
        mask = np.isnan(missing)
        values = stored["missing_values"]
        if tuple(stored["shape"]) != missing.shape or values.shape != (int(mask.sum()),):
            raise ValueError("Prediction shape or mask coverage mismatch")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite saved predictions")
        prediction = missing.astype(values.dtype)
        prediction[mask] = values
    return prediction


def save_prediction(path, prediction, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(path, missing_values=prediction[mask], shape=np.array(prediction.shape))


def export_tabular(output, reference, records):
    completed = {row["case_id"]: row for row in records}
    write_csv(output / "per_case_comparison.csv", records)
    patterns = None
    for label, dimensions in (("pattern", ("suite", "pattern")),
                              ("pattern_rate", ("suite", "pattern", "missingness")),
                              ("dataset", ("suite", "dataset")), ("suite", ("suite",))):
        summaries = aggregate(reference, completed, dimensions)
        write_csv(output / f"comparison_per_{label}.csv", summaries)
        if label == "pattern":
            patterns = summaries
    lines = ["# Recommender specialist: tabular transfer", "",
             f"Completed {len(records)}/{len(reference)} cases. Lower NRMSE is better.", "",
             "General = September 30 larger-matrix checkpoint; specialist = selected recsys step 100.",
             "Identical frozen masks, column normalization and contexts; no row train/test split.",
             "MissBench uses full matrices; UCI uses 2000 x 1000 windows with original small-matrix behavior.",
             "Cases are averaged within datasets, then datasets equally. Positive changes mean regression.",
             "Incomplete pattern groups have no specialist headline score.", ""]
    for suite, title in (("missbench", "MissBench: 40% missingness"), ("uci", "UCI: 10-50%, ten repeats")):
        lines += [f"## {title}", "", "| Pattern | General NRMSE | Specialist NRMSE | Change | Cases |",
                  "|---|---:|---:|---:|---:|"]
        for row in sorted((row for row in patterns if row["suite"] == suite), key=lambda row: PATTERNS.index(row["pattern"])):
            value = f"{row['new_nrmse']:.6f}" if row["complete"] else "pending"
            change = f"{row['change_percent']:+.2f}%" if row["change_percent"] is not None else "—"
            lines.append(f"| {row['pattern']} | {row['previous_nrmse']:.6f} | {value} | {change} | {row['completed_cases']}/{row['expected_cases']} |")
        lines.append("")
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def run_tabular(output, manifest):
    import torch
    from tabimpute.interface import ImputePFN

    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(.85)
    torch.manual_seed(42)
    np.random.seed(42)
    imputer = ImputePFN(device="cuda", checkpoint_path=str(CHECKPOINT), entry_wise_features=False,
                        json_config={"config": read_json(CHECKPOINT.parent / "config.json")["model"]},
                        model_family="new_stable", precision_mode="amp_bfloat16", prediction_stat="mean",
                        max_num_rows=None, max_num_cols=None, max_num_chunks=1, verbose=False)
    scorers = {suite: load_original(REPO / relative, "compute_normalized_rmse_columnwise") for suite, relative in (
        ("missbench", "benchmark/plot_options.py"), ("uci", "benchmark/uci/plot_uci_negative_rmse.py"))}
    references = [reference_record(case)[1] for case in manifest["tabular_cases"]]
    records = []
    started = time.monotonic()
    for case, reference in zip(manifest["tabular_cases"], references):
        if shutil.disk_usage(output).free < 20 * 1024**3:
            raise RuntimeError("20 GiB disk reserve reached; all artifacts preserved")
        case_id = case["case_id"]
        write_json(output / "status.json", {"state": "running", "completed": len(records),
                   "expected": len(references), "current": case_id, "elapsed_seconds": time.monotonic() - started})
        source = Path(case["source_directory"])
        hashes = {key: sha256(source / filename) for key, filename in (
            ("missing_sha256", "missing.npy"), ("true_sha256", "true.npy"))}
        if any(reference[key] != value for key, value in hashes.items()):
            raise ValueError(f"Reference input or mask changed: {case_id}")
        missing, truth = (np.load(source / filename, allow_pickle=False) for filename in ("missing.npy", "true.npy"))
        mask = np.isnan(missing)
        reference_root = CHUNKED if case["suite"] == "uci" else FULL
        reference_path = reference_root / "predictions" / (case_id + ".npz")
        if sha256(reference_path) != reference["prediction_sha256"]:
            raise ValueError("Archived reference prediction hash mismatch")
        reference_prediction = decode_prediction(reference_path, missing)
        reference_score = score_predictions(truth, reference_prediction, missing, case["suite"])["nrmse_original"]
        if not np.isclose(reference_score, reference["nrmse_original"], rtol=0, atol=1e-7):
            raise ValueError("Archived reference NRMSE does not reproduce")
        result_path = output / "cases" / (case_id + ".json")
        prediction_path = output / "predictions" / (case_id + ".npz")
        if result_path.exists():
            record = read_json(result_path)
            if record["checkpoint_sha256"] != manifest["specialist_sha256"] or any(record[key] != value for key, value in hashes.items()):
                raise ValueError("Resume case provenance mismatch")
            if sha256(prediction_path) != record["prediction_sha256"]:
                raise ValueError("Resume prediction hash mismatch")
            prediction = decode_prediction(prediction_path, missing)
        else:
            if case["suite"] == "uci":
                normalized, mean, scale, _ = standardize_uci(missing)
                configure_windows(imputer, missing.shape, 2000, 1000)
            else:
                normalized, mean, scale = missing, 0., 1.
                imputer.max_num_rows = imputer.max_num_cols = None
            torch.manual_seed(42)
            np.random.seed(42)
            torch.cuda.synchronize()
            case_started = time.monotonic()
            prediction = imputer.impute(normalized.copy()) * scale + mean
            torch.cuda.synchronize()
            elapsed = time.monotonic() - case_started
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite specialist prediction")
            save_prediction(prediction_path, prediction, mask)
            record = {**case, **hashes, "checkpoint_sha256": manifest["specialist_sha256"],
                      "prediction_sha256": sha256(prediction_path), "inference_seconds": elapsed}
        score = score_predictions(truth, prediction, missing, case["suite"])["nrmse_original"]
        original_score = float(scorers[case["suite"]](truth, prediction, mask))
        original_reference = float(scorers[case["suite"]](truth, reference_prediction, mask))
        if not np.isclose(score, original_score, atol=1e-5, rtol=1e-5) or not np.isclose(reference_score, original_reference, atol=1e-5, rtol=1e-5):
            raise ValueError("Original NRMSE implementation disagrees")
        if "new_nrmse" in record and not np.isclose(record["new_nrmse"], score, atol=1e-7, rtol=0):
            raise ValueError("Saved specialist score does not reproduce")
        record.update(status="ok", new_nrmse=score, general_nrmse=reference_score, delta_nrmse=score-reference_score,
                      rows=missing.shape[0], columns=missing.shape[1], original_scorer_difference=abs(score-original_score))
        write_json(result_path, record)
        records.append(record)
        if len(records) % 25 == 0 or len(records) == len(references):
            export_tabular(output, references, records)
            print(json.dumps({"completed": len(records), "expected": len(references), "current": case_id}), flush=True)
    write_json(output / "status.json", {"state": "complete", "completed": len(records), "expected": len(references),
               "elapsed_seconds": time.monotonic() - started, "both_models_predictions_rescored": True})


def compare_llm_panels(general, specialist, dataset):
    if not np.array_equal(general["benchmark_names"], specialist["benchmark_names"]):
        raise ValueError("LLM benchmark identities differ")
    rows, folds = [], []
    for method in METHODS:
        for budget in range(1, 16):
            row = {"dataset": dataset, "selection": method, "budget": budget}
            for metric in ("r2", "rmse"):
                before, after = general[f"{method}_{metric}"], specialist[f"{method}_{metric}"]
                if before.shape != (10, 15) or after.shape != (10, 15) or not np.isfinite(before).all() or not np.isfinite(after).all():
                    raise ValueError("Incomplete or nonfinite LLM panel")
                row.update({f"general_{metric}": float(before[:, budget-1].mean()),
                            f"specialist_{metric}": float(after[:, budget-1].mean()),
                            f"delta_{metric}": float((after[:, budget-1]-before[:, budget-1]).mean())})
                for fold in range(10):
                    folds.append({"dataset": dataset, "selection": method, "budget": budget, "fold": fold,
                                  "metric": metric, "general": float(before[fold, budget-1]), "specialist": float(after[fold, budget-1])})
            rows.append(row)
    return rows, folds


def run_llm(output, manifest):
    environment = evaluation_environment(output, CHECKPOINT, LABEL, ",".join(DATASETS))
    environment.update(CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(output / "matplotlib"))
    write_json(output / "environment.json", {key: value for key, value in environment.items() if key.startswith("BENCHSELECT_")})
    rows, folds, completed = [], [], []
    for dataset in DATASETS:
        path = output / "logs" / f"tabimpute_{LABEL}_{dataset}.npz"
        write_json(output / "status.json", {"state": "running", "current": dataset, "completed": completed})
        if not path.exists():
            environment["BENCHSELECT_DATASETS"] = dataset
            with (output / f"{dataset}.log").open("a") as log:
                subprocess.run([sys.executable, str(BENCH / "code/eval_tabimpute.py")], cwd=BENCH,
                               env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        general = load_panel(LLM / "logs" / f"tabimpute_matrixformer_large_trained_{dataset}.npz")
        specialist = load_panel(path)
        new_rows, new_folds = compare_llm_panels(general, specialist, dataset)
        rows.extend(new_rows)
        folds.extend(new_folds)
        completed.append(dataset)
        write_csv(output / "comparison.csv", rows)
        write_csv(output / "comparison_per_fold.csv", folds)
        lines = ["# Recommender specialist: LLM score completion", "", f"Completed datasets: {len(completed)}/4.",
                 "General = September 30 checkpoint; identical frozen datasets, ten folds, methods and budgets.",
                 "Higher R2 is better. Fold-level metrics are saved; the historical evaluator does not save per-entry predictions.", "",
                 "| Dataset | Selection | Budget | General R2 | Specialist R2 | Change |", "|---|---|---:|---:|---:|---:|"]
        for row in rows:
            if row["budget"] in (5, 10, 15):
                lines.append(f"| {row['dataset']} | {row['selection']} | {row['budget']} | {row['general_r2']:.6f} | {row['specialist_r2']:.6f} | {row['delta_r2']:+.6f} |")
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
        verify_hashes(manifest["input_sha256"])
        verify_hashes(manifest["source_sha256"])
        print(f"Completed LLM {dataset}: {len(completed)}/4", flush=True)
    write_json(output / "status.json", {"state": "complete", "completed": completed, "settings": len(rows),
               "fold_metric_rows": len(folds), "checkpoint_sha256": manifest["specialist_sha256"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--stage", choices=("all", "tabular", "llm"), default="all")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{arguments.stage}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = register(output)
        status_path = output / ("status.json" if arguments.stage == "all" else f"{arguments.stage}/status.json")
        try:
            if arguments.stage == "all":
                for stage in ("tabular", "llm"):
                    write_json(status_path, {"state": "running", "stage": stage, "pid": os.getpid()})
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stage", stage,
                                    "--output", str(output)], check=True)
            else:
                directory = output / arguments.stage
                directory.mkdir(exist_ok=True)
                {"tabular": run_tabular, "llm": run_llm}[arguments.stage](directory, manifest)
            verify_hashes(manifest["input_sha256"])
            verify_hashes(manifest["source_sha256"])
            if arguments.stage == "all":
                write_json(status_path, {"state": "complete", "stages": ["tabular", "llm"], "finished_unix": time.time()})
        except Exception as error:
            write_json(status_path, {"state": "failed", "error": repr(error), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
