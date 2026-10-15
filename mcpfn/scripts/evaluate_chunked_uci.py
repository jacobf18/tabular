"""Fixed training-sized UCI windows, compared with both archived full-matrix models."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from audit_dwip_missbench_uci import load_original
from compare_imputation_checkpoint import aggregate
from evaluate_dwip_missbench_uci import REPO, score_predictions, standardize_uci
from recsys_evaluation_common import atomic_npz, sha256, write_csv, write_json


ROOT = REPO / "runs/benchmark_evaluations"
PARENT = ROOT / "dwip_teacher_mean128k_selected98k_missbench_uci_20260918"
FULL = ROOT / "matrixformer_large_trained_missbench_uci_20260930"


def configure_windows(imputer, shape, rows, columns):
    if min(rows, columns) <= 0:
        raise ValueError("Context limits must be positive")
    imputer.max_num_rows = rows if shape[0] > rows else None
    imputer.max_num_cols = columns if shape[1] > columns else None
    imputer.max_num_chunks = 1


def summarize(output, parent, full, records, rows, columns):
    completed = {record["case_id"]: record for record in records}
    summaries = {}
    for name, dimensions in (("pattern", ("suite", "pattern")),
                             ("pattern_rate", ("suite", "pattern", "missingness")),
                             ("dataset", ("suite", "dataset")), ("overall", ("suite",))):
        compared_parent = aggregate(parent, completed, dimensions)
        compared_full = aggregate(full, completed, dimensions)
        summaries[name] = []
        for parent_row, full_row in zip(compared_parent, compared_full):
            summaries[name].append({
                **{key: parent_row[key] for key in dimensions},
                "parent_full_nrmse": parent_row["previous_nrmse"],
                "new_full_nrmse": full_row["previous_nrmse"],
                "new_chunked_nrmse": full_row["new_nrmse"],
                "change_percent_vs_new_full": full_row["change_percent"],
                "change_percent_vs_parent_full": parent_row["change_percent"],
                **{key: full_row[key] for key in ("completed_cases", "expected_cases", "datasets", "complete")},
            })
        write_csv(output / f"comparison_per_{name}.csv", summaries[name])
    write_csv(output / "per_case_metrics.csv", records)
    lines = ["# UCI: training-sized context windows", "",
             f"Completed {len(records)}/{len(full)} cases. Fixed maximum context: {rows} rows × {columns} columns.",
             "Same larger-matrix checkpoint, frozen masks, column normalization, precision, and NRMSE scorer as full-matrix evaluation.",
             "Whole-matrix observed-column normalization is retained; it is not re-estimated per chunk.",
             "One independent context window per model call; overlapping tail predictions are averaged.",
             "Smaller matrices keep the original unchunked execution path and must reproduce archived predictions exactly.",
             "All original rows and columns are retained; there is no row train/test split, shuffling, or padding.",
             "The 2000 × 1000 default is the upper size of the final training phase, not an exact match to every sampled training shape.",
             "NRMSE is averaged within datasets before equal dataset weighting. Negative changes are improvements.", ""]
    for name, title, field in (("pattern", "By missingness pattern", "pattern"), ("dataset", "By dataset", "dataset")):
        lines += [f"## {title}", "", "| Group | Parent full | New full | New chunked | Change vs new full | Cases |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in summaries[name]:
            current = f"{row['new_chunked_nrmse']:.6f}" if row["complete"] else "pending"
            change = f"{row['change_percent_vs_new_full']:+.2f}%" if row["change_percent_vs_new_full"] is not None else "—"
            lines.append(f"| {row[field]} | {row['parent_full_nrmse']:.6f} | {row['new_full_nrmse']:.6f} | "
                         f"{current} | {change} | {row['completed_cases']}/{row['expected_cases']} |")
        lines.append("")
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def run(arguments):
    import torch
    from tabimpute.interface import ImputePFN

    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    full_protocol = json.loads((FULL / "manifest.json").read_text())["protocol"]
    checkpoint = Path(full_protocol["checkpoint"])
    config_path = checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text())
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != full_protocol["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the full-matrix evaluation")
    cases = [case for case in full_protocol["cases"] if case["suite"] == "uci"]
    references = {}
    for name, root in (("parent", PARENT), ("full", FULL)):
        references[name] = {case["case_id"]: json.loads((root / "cases" / (case["case_id"] + ".json")).read_text())
                            for case in cases}
        if len(references[name]) != 1800 or any(row["status"] != "ok" for row in references[name].values()):
            raise ValueError("Reference UCI coverage is incomplete")
    sources = [Path(__file__), REPO / "src/tabimpute/interface.py", *sorted((REPO / "src/tabimpute/model").rglob("*.py")),
               *[Path(__file__).with_name(name) for name in ("evaluate_dwip_missbench_uci.py", "compare_imputation_checkpoint.py",
                                                            "audit_dwip_missbench_uci.py", "recsys_evaluation_common.py")],
               REPO / "benchmark/uci/plot_uci_negative_rmse.py"]
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(0.4)
    manifest = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash, "config_sha256": sha256(config_path),
        "context_shape": [arguments.rows, arguments.columns], "max_num_chunks": 1,
        "training_phases": config["phases"], "cases": cases,
        "precision": "amp_bfloat16", "prediction_stat": "mean", "seed": 42,
        "normalization": "Unchanged whole-matrix observed-column normalization; original UCI external standardization retained",
        "tail_policy": "Overlapping full-size tail windows, prediction averaging; no padding or shuffled rows",
        "selection": "One geometry chosen from training dimensions before evaluation; no test-error-based size sweep",
        "source_sha256": {str(path): sha256(path) for path in sources},
        "reference_manifest_sha256": {str(root): sha256(root / "manifest.json") for root in (PARENT, FULL)},
        "torch": torch.__version__, "numpy": np.__version__, "gpu": torch.cuda.get_device_name(),
    }
    if (output / "manifest.json").exists():
        if json.loads((output / "manifest.json").read_text()) != manifest:
            raise ValueError("Refusing to resume with changed checkpoint, sources, or context")
    else:
        write_json(output / "manifest.json", manifest)
        for path in sources:
            destination = output / "sources" / path.relative_to(REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    imputer = ImputePFN(device="cuda", checkpoint_path=str(checkpoint), entry_wise_features=False,
                        json_config={"config": config["model"]}, model_family="new_stable", precision_mode="amp_bfloat16",
                        prediction_stat="mean", max_num_rows=arguments.rows, max_num_cols=arguments.columns, max_num_chunks=1)
    shapes = []

    def check_context(module, inputs):
        shape = tuple(inputs[0].shape)
        if shape[0] != 1 or shape[1] > arguments.rows or shape[2] > arguments.columns:
            raise ValueError(f"Unexpected model input shape: {shape}")
        shapes.append(shape)

    handle = imputer.model.register_forward_pre_hook(check_context)
    original_score = load_original(REPO / "benchmark/uci/plot_uci_negative_rmse.py", "compute_normalized_rmse_columnwise")
    records = []
    started = time.monotonic()
    try:
        for case in cases:
            result_path = output / "cases" / (case["case_id"] + ".json")
            prediction_path = output / "predictions" / (case["case_id"] + ".npz")
            source = Path(case["source_directory"])
            input_hashes = {field: sha256(source / filename) for field, filename in
                            (("missing_sha256", "missing.npy"), ("true_sha256", "true.npy"))}
            for reference in references.values():
                if any(reference[case["case_id"]][field] != value for field, value in input_hashes.items()):
                    raise ValueError("Frozen input/mask mismatch")
            if result_path.exists():
                record = json.loads(result_path.read_text())
                if (record["checkpoint_sha256"] != checkpoint_hash or sha256(prediction_path) != record["prediction_sha256"]
                        or any(record[field] != value for field, value in input_hashes.items())):
                    raise ValueError("Completed case artifact mismatch")
            else:
                if shutil.disk_usage(output).free < 2 * 1024**3:
                    raise RuntimeError("Disk reserve reached")
                write_json(output / "status.json", {"state": "running", "completed": len(records), "expected": len(cases),
                                                   "current": case["case_id"], "elapsed_seconds": time.monotonic() - started})
                missing = np.load(source / "missing.npy", allow_pickle=False)
                truth = np.load(source / "true.npy", allow_pickle=False)
                normalized, mean, scale, constant_columns = standardize_uci(missing)
                configure_windows(imputer, missing.shape, arguments.rows, arguments.columns)
                torch.manual_seed(42)
                np.random.seed(42)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                shapes.clear()
                case_started = time.perf_counter()
                prediction = imputer.impute(normalized.copy()) * scale + mean
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - case_started
                expected_calls = math.ceil(missing.shape[0] / arguments.rows) * math.ceil(missing.shape[1] / arguments.columns)
                if len(shapes) != expected_calls:
                    raise ValueError("Unexpected number of model windows")
                score = score_predictions(truth, prediction, missing, "uci")["nrmse_original"]
                mask = np.isnan(missing)
                independent = float(original_score(truth, prediction, mask))
                np.testing.assert_allclose(score, independent, rtol=5e-7, atol=1e-7)
                unchunked = missing.shape[0] <= arguments.rows and missing.shape[1] <= arguments.columns
                if unchunked:
                    previous = references["full"][case["case_id"]]
                    old_path = FULL / previous["prediction_path"]
                    if sha256(old_path) != previous["prediction_sha256"]:
                        raise ValueError("Unchunked control artifact changed")
                    with np.load(old_path, allow_pickle=False) as stored:
                        np.testing.assert_array_equal(prediction[mask], stored["missing_values"])
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_npz(prediction_path, missing_values=prediction[mask], shape=np.asarray(missing.shape))
                with np.load(prediction_path, allow_pickle=False) as stored:
                    np.testing.assert_array_equal(stored["missing_values"], prediction[mask])
                record = {
                    **case, **input_hashes, "status": "ok", "new_nrmse": score,
                    "parent_full_nrmse": references["parent"][case["case_id"]]["nrmse_original"],
                    "new_full_nrmse": references["full"][case["case_id"]]["nrmse_original"],
                    "rows": missing.shape[0], "columns": missing.shape[1], "model_input_shapes": shapes.copy(),
                    "unchunked_control_bitwise_equal": True if unchunked else None,
                    "original_scorer_difference": abs(score - independent), "zero_std_columns": constant_columns,
                    "prediction_sha256": sha256(prediction_path), "checkpoint_sha256": checkpoint_hash,
                    "inference_seconds": elapsed, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
                write_json(result_path, record)
            records.append(record)
            if len(records) % 25 == 0 or len(records) == len(cases):
                summarize(output, list(references["parent"].values()), list(references["full"].values()),
                          records, arguments.rows, arguments.columns)
                print(json.dumps({"completed": len(records), "current": case["case_id"], "nrmse": record["new_nrmse"]}), flush=True)
    finally:
        handle.remove()
    if sha256(checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed during evaluation")
    write_json(output / "audit.json", {
        "cases": len(records), "same_checkpoint_and_inputs": True, "all_contexts_within_limits": True,
        "unchunked_bitwise_equal_cases": sum(record["unchunked_control_bitwise_equal"] is True for record in records),
        "maximum_original_scorer_difference": max(record["original_scorer_difference"] for record in records),
    })
    write_json(output / "status.json", {"state": "complete", "completed": len(records), "expected": len(cases),
                                       "completed_utc": datetime.now(timezone.utc).isoformat(),
                                       "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--columns", type=int, default=1000)
    arguments = parser.parse_args()
    try:
        run(arguments)
    except BlockingIOError:
        raise
    except Exception:
        write_json(arguments.output / "status.json", {"state": "failed", "error": traceback.format_exc()})
        raise
