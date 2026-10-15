"""Reproduce saved fold scores and decompose errors without changing the benchmark."""
import argparse
import fcntl
import gc
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from run_dwip_transfer_evaluation import evaluation_environment, gpu_processes, sha256
from run_rowcol_controlled_experiments import BENCH, REPO, WORK, write_json
from tabimpute.interface import ImputePFN

QUEUES = {"24k": "dwip_spectral_20260915", "48k_seed1": "dwip_spectral_horizon48k_20260916",
          "48k_seed2": "dwip_spectral_seed2_48k_20260916", "point2_24k": "dwip_spectral_point2_20260916",
          "point2_48k": "dwip_spectral_point2_horizon48k_20260916"}


class CaptureImputer:
    def __init__(self, imputer):
        self.imputer = imputer
        self.calls = []

    def impute(self, source):
        before = source.copy()
        prediction = self.imputer.impute(source)
        self.calls.append({"source": before, "prediction": prediction.copy()})
        return prediction


def decompose_scores(target, observed, prediction, selected, mean, scale):
    if target.shape != prediction.shape or observed.shape != target.shape:
        raise ValueError("Prediction, target and observation shapes must match")
    if not np.isfinite(prediction).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Nonfinite predictions or invalid scoring scales")
    mask = observed.astype(bool).copy()
    mask[:, selected] = False
    if not mask.any():
        raise ValueError("No evaluated cells")
    standardized_target, standardized_prediction = (target - mean) / scale, (prediction - mean) / scale
    clipped_target, clipped_prediction = np.clip(standardized_target, -10, 10), np.clip(standardized_prediction, -10, 10)
    squared_error = np.where(mask, (clipped_prediction - clipped_target) ** 2, 0.)
    squared_target = np.where(mask, clipped_target ** 2, 0.)
    raw_squared_error = np.where(mask, (prediction - target) ** 2, 0.)
    unbounded_error = np.where(mask, (standardized_prediction - standardized_target) ** 2, 0.)
    residual, total, count = float(squared_error.sum()), float(squared_target.sum()), int(mask.sum())
    if total < 1e-12:
        raise ValueError("Degenerate target sum of squares")
    columns = []
    for index in range(target.shape[1]):
        subset = mask[:, index]
        if not subset.any():
            continue
        columns.append({"index": index, "n_eval": int(subset.sum()), "score_scale": float(scale[index]),
            "scored_sse": float(squared_error[:, index].sum()), "scored_sst": float(squared_target[:, index].sum()),
            "raw_sse": float(raw_squared_error[:, index].sum()), "unclipped_standardized_sse": float(unbounded_error[:, index].sum()),
            "predictions_clipped": int((np.abs(standardized_prediction[subset, index]) > 10).sum()),
            "targets_clipped": int((np.abs(standardized_target[subset, index]) > 10).sum()),
            "prediction_min": float(prediction[subset, index].min()), "prediction_max": float(prediction[subset, index].max()),
            "target_min": float(target[subset, index].min()), "target_max": float(target[subset, index].max())})
    return {"metrics": {"r2": 1 - residual / total, "rmse": (residual / count) ** .5, "n_eval": count},
            "raw_rmse_report_only": float((raw_squared_error.sum() / count) ** .5),
            "unclipped_standardized_rmse_report_only": float((unbounded_error.sum() / count) ** .5),
            "scored_sse": residual, "scored_sst": total, "columns": columns, "evaluation_mask": mask}


def verify_reproduced_scores(official, cached, decomposition, label, fold, budget):
    for metric in ("r2", "rmse"):
        expected = float(cached[f"random_{metric}"][fold, budget - 1])
        actual = float(official[metric])
        if not np.isclose(actual, expected, rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"Fresh score does not reproduce cached protocol: {label},k{budget},{metric},"
                               f"fresh={actual},cached={expected}")
        if not np.isclose(actual, decomposition[metric], rtol=1e-10, atol=1e-10):
            raise RuntimeError("Error decomposition disagrees with unchanged benchmark")
    if official["n_eval"] != decomposition["n_eval"]:
        raise RuntimeError("Evaluated cell count disagrees with unchanged benchmark")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mteb", "merged"), default="mteb")
    parser.add_argument("--fold", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.fold < 10:
        raise ValueError("Fold must be in0..9")
    root = args.output or REPO / f"runs/information_flow_diagnostics/dwip_{args.dataset}_fold{args.fold}_regression_20260916_v2"
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "diagnostic.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if gpu_processes():
        raise RuntimeError("GPU is busy; no diagnostic started")
    torch.set_num_threads(4)
    baseline = REPO / "runs/objective_queues/dwip_spectral_20260915"
    checkpoints = {"original": baseline / "baseline/checkpoint.pth",
                   **{label: WORK / f"tabimpute-{name}-dwip1e5/best_validation_checkpoint.pth" for label, name in QUEUES.items()}}
    if args.dataset == "mteb":
        caches = {"original": BENCH / "logs/tabimpute_dwip_spectral_20260915_baseline_mteb.npz",
                  **{label: BENCH / f"logs/tabimpute_{name}_dwip1e5_mteb.npz" for label, name in QUEUES.items()}}
    else:
        caches = {"original": baseline / "transfer_evaluation/logs/tabimpute_dwip_transfer_baseline_merged.npz",
                  **{label: REPO / f"runs/objective_queues/{name}/transfer_evaluation/logs/tabimpute_dwip_transfer_selected_merged.npz" for label, name in QUEUES.items()}}
    environment = evaluation_environment(root, checkpoints["original"], "fold_diagnostic", args.dataset)
    for key in list(os.environ):
        if key.startswith("BENCHSELECT_"):
            del os.environ[key]
    os.environ.update(environment)
    sys.path.insert(0, str(BENCH / "code"))
    evaluator = importlib.import_module("eval_tabimpute")
    if evaluator.ROW_LIMIT is not None or (evaluator.SEED, evaluator.MAX_ROWS, evaluator.TEST_BATCH) != (42, 4000, 200):
        raise ValueError("Unexpected benchmark protocol")
    data_path = BENCH / "data" / f"{args.dataset}.matrix.npz"
    with np.load(data_path, allow_pickle=True) as data:
        values, observed, names = data["B"].astype(np.float64), data["O"].astype(np.float64), data["benchmark_names"]
    folds = evaluator.cv_folds(len(values), 10, 42)
    train, test = evaluator.cv_train_val(folds, args.fold, .1, len(values), 42)
    train_values, train_observed, target, target_observed = values[train], observed[train], values[test], observed[test]
    correlation, mean, scale = evaluator.estimate_corr(train_values, train_observed)
    selected = evaluator.random_select(correlation, 15, seed=(49, args.fold, 1000), benchmark_names=names)["selected"]
    context_count = min(len(train), evaluator.MAX_ROWS - evaluator.TEST_BATCH)
    counts = train_observed.sum(0)
    empirical_std = np.sqrt(((train_values - mean) ** 2 * train_observed).sum(0) / counts.clip(1))
    source_paths = [Path(__file__), BENCH / "code/eval_tabimpute.py", BENCH / "code/cv_split.py", BENCH / "code/greedy_select.py",
                    REPO / "src/tabimpute/interface.py", REPO / "src/tabimpute/model/model_new_stable.py"]
    manifest = {"dataset": args.dataset, "fold": args.fold, "budgets": [10, 15], "method": "random",
        "train_indices": train.tolist(), "test_indices": test.tolist(), "selected": [int(index) for index in selected],
        "data_sha256": sha256(data_path), "checkpoint_sha256": {label: sha256(path) for label, path in checkpoints.items()},
        "cached_scores_sha256": {label: sha256(path) for label, path in caches.items()},
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
                    "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
                    "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                    "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled()},
        "protocol": "Canonical unchanged impute_tabimpute(),10 folds,seed42,holdout0.1,max4000,testbatch200,FP32 weights/BF16 autocast,mean",
        "purpose": "Post-hoc diagnosis of concentrated errors, not checkpoint selection, prior fitting, metric replacement or exclusion of difficult cells"}
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Diagnostic inputs changed; use a new output directory")
    write_json(manifest_path, manifest)
    results = {}
    for label, checkpoint in checkpoints.items():
        write_json(root / "status.json", {"state": "diagnosing", "model": label})
        config = json.loads((checkpoint.parent / "config.json").read_text())
        imputer = ImputePFN(device="cuda", checkpoint_path=str(checkpoint), entry_wise_features=False,
                           json_config={"config": config["model"]}, model_family="new_stable",
                           precision_mode="amp_bfloat16", prediction_stat="mean", max_num_rows=None, max_num_cols=None)
        results[label] = {}
        with np.load(caches[label], allow_pickle=True) as cached:
            if not np.array_equal(cached["benchmark_names"], names):
                raise ValueError("Cached benchmark identities differ")
            for budget in (10, 15):
                capture = CaptureImputer(imputer)
                official = evaluator.impute_tabimpute(capture, train_values, train_observed, target, target_observed,
                                                       selected[:budget], mean, scale, values.shape[1])
                prediction = np.concatenate([call["prediction"][context_count:] for call in capture.calls])
                report = decompose_scores(target, target_observed, prediction, selected[:budget], mean, scale)
                verify_reproduced_scores(official, cached, report["metrics"], label, args.fold, budget)
                for column in report["columns"]:
                    index = column["index"]
                    column.update(name=str(names[index]), train_observed_count=int(counts[index]),
                                  train_empirical_std=float(empirical_std[index]))
                np.savez_compressed(root / f"{label}_k{budget}_predictions.npz", prediction=prediction, target=target,
                                    evaluation_mask=report.pop("evaluation_mask"), mean=mean, scale=scale,
                                    **{f"source_{index}": call["source"] for index, call in enumerate(capture.calls)})
                report["columns"].sort(key=lambda column: column["scored_sse"], reverse=True)
                results[label][str(budget)] = report
                write_json(root / "results.json", results)
                print(label, budget, report["metrics"], "raw_rmse", report["raw_rmse_report_only"], flush=True)
        if sha256(checkpoint) != manifest["checkpoint_sha256"][label]:
            raise RuntimeError("Source checkpoint changed during diagnosis")
        del capture, imputer
        gc.collect()
        torch.cuda.empty_cache()
    write_json(root / "status.json", {"state": "complete", "cached_scores_reproduced": True})


if __name__ == "__main__":
    main()
