"""Frozen, support-matched benchmark completion without neural training."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np


REPO = Path(__file__).resolve().parents[1]
BENCH = REPO.parent / "benchmark-selection"
OUTPUT = REPO / "runs/benchmark_evaluations/matched_llm_20261005"
GENERAL = REPO / "runs/streamed_training/large_matrices_98k_20260930/best_validation_checkpoint.pth"
SPECIALIST = REPO / "runs/recsys_goal_20260920/synthetic_user_scaled_parent_risk_400/best_validation_checkpoint.pth"
DATASETS = ("mteb", "mmlu", "merged", "benchpress")
METHODS = ("entropy", "mi", "random")
VARIANTS = ("general", "specialist", "observed_blend", "gaussian")
CONTEXT_ROWS = 3800
BATCH_ROWS = 200


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def write_csv(path, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def visible_input(context_scores, context_observed, test_scores, test_observed, selected):
    context = np.where(context_observed, context_scores, np.nan)
    visible = np.zeros_like(test_observed, dtype=bool)
    visible[:, selected] = test_observed[:, selected]
    return context, np.where(visible, test_scores, np.nan)


def target_mask(observed, selected):
    mask = np.asarray(observed, dtype=bool).copy()
    mask[:, selected] = False
    return mask


def score_standardized(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Targets and predictions must be finite with identical shapes")
    target = np.clip(target, -10., 10.)
    prediction = np.clip(prediction, -10., 10.)
    residual = float(np.sum((target - prediction) ** 2))
    total = float(np.sum(target ** 2))
    return {"r2": 1. - residual / total if total > 1e-12 else None,
            "rmse": float(np.sqrt(residual / target.size)) if target.size else None,
            "n_eval": int(target.size), "ss_res": residual, "ss_tot": total}


def gaussian_prediction(visible, correlation, mean, scale):
    standardized = (visible - mean) / scale
    predictions = np.zeros_like(visible, dtype=np.float64)
    for row_index, row in enumerate(standardized):
        available = np.flatnonzero(np.isfinite(row))
        if available.size:
            covariance = correlation[np.ix_(available, available)] + .01 * np.eye(available.size)
            predictions[row_index] = correlation[:, available] @ np.linalg.solve(covariance, row[available])
    return predictions


def build_manifest(output):
    import scipy
    import platform

    sources = [Path(__file__).resolve(), *sorted((REPO / "src/tabimpute").rglob("*.py")),
               *sorted((BENCH / "code").glob("*.py"))]
    sources += sorted(path for path in (REPO / "src/tabimpute/data").iterdir() if path.is_file())
    snapshot = output / "source_snapshot"
    source_hashes = {}
    for path in sources:
        relative = path.relative_to(REPO.parent)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256(destination) != sha256(path):
            raise ValueError("Source snapshot changed; choose a new output directory")
        if not destination.exists():
            shutil.copyfile(path, destination)
        source_hashes[str(relative)] = sha256(destination)
    config_hashes = {}
    for name, checkpoint in (("general", GENERAL), ("specialist", SPECIALIST)):
        destination = output / f"{name}_config.json"
        source_config = checkpoint.parent / "config.json"
        if destination.exists() and sha256(destination) != sha256(source_config):
            raise ValueError("Model configuration changed; choose a new output directory")
        if not destination.exists():
            shutil.copyfile(source_config, destination)
        config_hashes[name] = sha256(destination)
    return {
        "protocol_version": 1, "datasets": list(DATASETS), "methods": list(METHODS),
        "variants": list(VARIANTS), "folds": 10, "budgets": list(range(1, 16)), "seed": 42,
        "context_rows": CONTEXT_ROWS, "test_batch_rows": BATCH_ROWS,
        "context": "Same deterministic training-context subset for every predictor; selection, covariance, and scoring statistics use that subset",
        "selection": "Released algorithms, computed once and frozen; random seed (49, fold, 1000); no reselection during inference",
        "targets": "Every originally observed unselected test entry, including rows without selected observations",
        "clipping": "Raw context and exposed test inputs are unmodified; all targets and predictions clipped to [-10,10] after common training-context column standardization",
        "gaussian": "Released context covariance; ridge 0.01; standardized prediction zero for rows without selected observations",
        "metric": "Mean fold R2 = 1-SSE/sum(z_target^2), and mean fold standardized RMSE; identical support and clipping",
        "blend": "Unchanged observed_blend; one observed-only reconstruction weight per test batch; no outer labels",
        "neural": "Frozen checkpoints, FP32 weights, BF16 autocast, mean predictions, cuDNN SDPA enabled; no optimizer or neural training",
        "checkpoint_paths": {"general": str(GENERAL), "specialist": str(SPECIALIST)},
        "checkpoint_sha256": {"general": sha256(GENERAL), "specialist": sha256(SPECIALIST)},
        "config_sha256": config_hashes,
        "data_sha256": {dataset: sha256(BENCH / "data" / f"{dataset}.matrix.npz") for dataset in DATASETS},
        "source_sha256": source_hashes,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "blas_threads": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")}},
    }


def prepare(output):
    if (output / "frozen_index.json").exists():
        verify_frozen(output)
        return
    manifest_path = output / "manifest.json"
    manifest = build_manifest(output)
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Manifest differs; choose a new output directory")
    write_json(manifest_path, manifest)
    os.environ["BENCHSELECT_EXPERIMENT_ROOT"] = str(output / "selection_scratch")
    sys.path.insert(0, str(output / "source_snapshot/benchmark-selection/code"))
    from cv_split import cv_folds, cv_train_val
    from eval_entropy_vs_mi import estimate_corr
    from greedy_select import greedy_entropy, greedy_mi, random_select

    frozen_hashes = {}
    for dataset in DATASETS:
        with np.load(BENCH / "data" / f"{dataset}.matrix.npz", allow_pickle=False) as data:
            scores, observed = data["B"].astype(np.float64), data["O"].astype(bool)
        folds = cv_folds(len(scores), 10, 42)
        for fold in range(10):
            path = output / "frozen" / dataset / f"fold_{fold:02d}.npz"
            train, test = cv_train_val(folds, fold, .1, len(scores), 42)
            context = train[np.random.default_rng(42).choice(len(train), CONTEXT_ROWS, replace=False)] if len(train) > CONTEXT_ROWS else train
            if not path.exists():
                correlation, mean, scale = estimate_corr(scores[context], observed[context].astype(float))
                orders = {
                    "entropy": greedy_entropy(correlation, 15)["selected"],
                    "mi": greedy_mi(correlation, 15)["selected"],
                    "random": random_select(correlation, 15, seed=(49, fold, 1000))["selected"],
                }
                arrays = dict(train=train, test=test, context=context, correlation=correlation, mean=mean, scale=scale)
                for method, order in orders.items():
                    arrays[f"order_{method}"] = np.array(order, dtype=np.int64)
                    for budget in range(1, 16):
                        arrays[f"mask_{method}_{budget:02d}"] = target_mask(observed[test], order[:budget])
                atomic_npz(path, **arrays)
            frozen_hashes[str(path.relative_to(output))] = sha256(path)
            print(f"Frozen {dataset} fold {fold}", flush=True)
    write_json(output / "frozen_index.json", {"manifest_sha256": sha256(manifest_path), "files_sha256": frozen_hashes})


def verify_frozen(output):
    manifest = json.loads((output / "manifest.json").read_text())
    if sha256(Path(__file__)) != manifest["source_sha256"]["mcpfn/scripts/evaluate_matched_llm.py"]:
        raise ValueError("Evaluation script differs from the frozen source snapshot")
    index = json.loads((output / "frozen_index.json").read_text())
    if sha256(output / "manifest.json") != index["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    for relative, digest in index["files_sha256"].items():
        if sha256(output / relative) != digest:
            raise ValueError(f"Frozen fold changed: {relative}")
    for dataset, digest in manifest["data_sha256"].items():
        if sha256(BENCH / "data" / f"{dataset}.matrix.npz") != digest:
            raise ValueError(f"Input data changed: {dataset}")
    for relative, digest in manifest["source_sha256"].items():
        if sha256(output / "source_snapshot" / relative) != digest:
            raise ValueError(f"Snapshot source changed: {relative}")
    for name, digest in manifest["config_sha256"].items():
        if sha256(output / f"{name}_config.json") != digest:
            raise ValueError(f"Model config changed: {name}")
    for name, digest in manifest["checkpoint_sha256"].items():
        if sha256(manifest["checkpoint_paths"][name]) != digest:
            raise ValueError(f"Checkpoint changed: {name}")
    return manifest


def make_router(output, manifest):
    import torch
    from tabimpute.interface import ImputePFN
    from tabimpute.routing import TwoExpertRouter

    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.enable_cudnn_sdp(True)
    torch.cuda.set_per_process_memory_fraction(.9)
    experts = []
    for name in ("general", "specialist"):
        configuration = json.loads((output / f"{name}_config.json").read_text())["model"]
        expert = ImputePFN(device="cuda", checkpoint_path=manifest["checkpoint_paths"][name],
                          entry_wise_features=False, json_config={"config": configuration},
                          model_family="new_stable", precision_mode="amp_bfloat16", prediction_stat="mean",
                          max_num_rows=None, max_num_cols=None, max_num_chunks=1, verbose=False)
        expert.model.eval()
        expert.model.requires_grad_(False)
        experts.append(expert)
    return TwoExpertRouter(*experts, policy="observed_blend")


def evaluate(output, datasets, limit):
    import torch

    manifest = verify_frozen(output)
    router = make_router(output, manifest)
    completed = 0
    started = time.monotonic()
    for dataset in datasets:
        with np.load(BENCH / "data" / f"{dataset}.matrix.npz", allow_pickle=False) as data:
            scores, observed = data["B"].astype(np.float64), data["O"].astype(bool)
        for fold in range(10):
            fold_path = output / "frozen" / dataset / f"fold_{fold:02d}.npz"
            with np.load(fold_path) as frozen:
                context, test = frozen["context"], frozen["test"]
                mean, scale, correlation = frozen["mean"], frozen["scale"], frozen["correlation"]
                for method in METHODS:
                    for budget in range(1, 16):
                        name = f"{method}_{budget:02d}"
                        destination = output / "predictions" / dataset / f"fold_{fold:02d}" / f"{name}.npz"
                        record_path = destination.with_suffix(".json")
                        if record_path.exists():
                            record = json.loads(record_path.read_text())
                            if record["prediction_sha256"] != sha256(destination) or record["fold_sha256"] != sha256(fold_path):
                                raise ValueError("Cached prediction changed")
                            continue
                        if shutil.disk_usage(output).free < 10 * 1024**3:
                            raise RuntimeError("10 GiB free-space reserve reached")
                        selected = frozen[f"order_{method}"][:budget]
                        mask = frozen[f"mask_{method}_{budget:02d}"]
                        np.testing.assert_array_equal(mask, target_mask(observed[test], selected))
                        context_input, test_input = visible_input(scores[context], observed[context], scores[test], observed[test], selected)
                        assert not np.isfinite(test_input[mask]).any()
                        truth = ((scores[test] - mean) / scale)[mask]
                        predictions = {"gaussian": gaussian_prediction(test_input, correlation, mean, scale)[mask]}
                        neural = {variant: np.empty(test_input.shape, dtype=np.float64) for variant in VARIANTS if variant != "gaussian"}
                        decisions = []
                        setting_started = time.monotonic()
                        for start in range(0, len(test), BATCH_ROWS):
                            stop = min(start + BATCH_ROWS, len(test))
                            batch = np.vstack([context_input, test_input[start:stop]])
                            with torch.inference_mode():
                                result = router.predict_all(batch, include_holdout=False)
                            for variant in neural:
                                neural[variant][start:stop] = result.predictions[variant][len(context):]
                            decisions.append(result.decisions["observed"])
                        for variant, prediction in neural.items():
                            predictions[variant] = ((prediction - mean) / scale)[mask]
                        metrics = {variant: score_standardized(truth, prediction) for variant, prediction in predictions.items()}
                        query_rows, query_columns = np.where(mask)
                        atomic_npz(destination, target_z=truth, query_rows=query_rows, query_columns=query_columns,
                                   **{f"{variant}_z": prediction for variant, prediction in predictions.items()})
                        record = {"dataset": dataset, "fold": fold, "method": method, "budget": budget,
                                  "fold_sha256": sha256(fold_path), "prediction_sha256": sha256(destination),
                                  "metrics": metrics, "blend_decisions": decisions,
                                  "seconds": time.monotonic() - setting_started}
                        write_json(record_path, record)
                        completed += 1
                        write_json(output / "status.json", {"state": "running", "last": [dataset, fold, method, budget],
                                   "new_settings_completed": completed, "elapsed_seconds": time.monotonic() - started})
                        print(f"{dataset} fold={fold} {method} k={budget} general={metrics['general']['r2']:.5f} plus={metrics['observed_blend']['r2']:.5f} gaussian={metrics['gaussian']['r2']:.5f} seconds={record['seconds']:.2f}", flush=True)
                        if limit is not None and completed >= limit:
                            return
    verify_frozen(output)


def audit(output):
    verify_frozen(output)
    records, grouped, prediction_hashes = [], {}, {}
    for dataset in DATASETS:
        with np.load(BENCH / "data" / f"{dataset}.matrix.npz") as data:
            scores, observed = data["B"], data["O"].astype(bool)
        for fold in range(10):
            fold_path = output / "frozen" / dataset / f"fold_{fold:02d}.npz"
            with np.load(fold_path) as frozen:
                assert not np.intersect1d(frozen["context"], frozen["test"]).size
                for method in METHODS:
                    for budget in range(1, 16):
                        path = output / "predictions" / dataset / f"fold_{fold:02d}" / f"{method}_{budget:02d}.npz"
                        record = json.loads(path.with_suffix(".json").read_text())
                        assert record["prediction_sha256"] == sha256(path)
                        assert record["fold_sha256"] == sha256(fold_path)
                        with np.load(path) as predictions:
                            mask = frozen[f"mask_{method}_{budget:02d}"]
                            np.testing.assert_array_equal(mask, target_mask(observed[frozen["test"]], frozen[f"order_{method}"][:budget]))
                            query_rows, query_columns = np.where(mask)
                            np.testing.assert_array_equal(predictions["query_rows"], query_rows)
                            np.testing.assert_array_equal(predictions["query_columns"], query_columns)
                            truth = ((scores[frozen["test"]] - frozen["mean"]) / frozen["scale"])[mask]
                            np.testing.assert_array_equal(predictions["target_z"], truth)
                            for variant in VARIANTS:
                                metrics = score_standardized(truth, predictions[f"{variant}_z"])
                                assert metrics == record["metrics"][variant]
                                records.append({"dataset": dataset, "method": method, "budget": budget, "fold": fold,
                                                "variant": variant, **metrics})
                                grouped.setdefault((dataset, method, budget, variant), []).append(metrics)
                        prediction_hashes[str(path.relative_to(output))] = record["prediction_sha256"]
    summary = []
    for (dataset, method, budget, variant), metrics in grouped.items():
        row = {"dataset": dataset, "method": method, "budget": budget, "variant": variant,
               "n_eval": sum(metric["n_eval"] for metric in metrics)}
        for metric_name in ("r2", "rmse"):
            values = [metric[metric_name] for metric in metrics if metric[metric_name] is not None]
            row[metric_name] = float(np.mean(values)) if values else None
            row[f"{metric_name}_folds"] = len(values)
        summary.append(row)
    write_csv(output / "per_fold.csv", records)
    write_csv(output / "summary.csv", summary)
    write_json(output / "prediction_index.json", prediction_hashes)
    write_json(output / "final_audit.json", {"state": "complete", "fold_settings": len(prediction_hashes),
               "settings": len(summary) // len(VARIANTS), "variants": list(VARIANTS),
               "all_predictions_rescored": True, "shared_masks_and_targets_verified": True,
               "checkpoints_unchanged": True, "neural_training_performed": False,
               "summary_sha256": sha256(output / "summary.csv"), "per_fold_sha256": sha256(output / "per_fold.csv"),
               "frozen_index_sha256": sha256(output / "frozen_index.json"),
               "prediction_index_sha256": sha256(output / "prediction_index.json")})
    write_json(output / "status.json", {"state": "complete", "fold_settings": len(prediction_hashes)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--stage", choices=("prepare", "evaluate", "audit"), required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--limit-settings", type=int)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if arguments.stage == "prepare":
                prepare(output)
            elif arguments.stage == "evaluate":
                snapshot = output / "source_snapshot/mcpfn/src"
                sys.path.insert(0, str(snapshot))
                evaluate(output, arguments.datasets, arguments.limit_settings)
            else:
                audit(output)
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "stage": arguments.stage, "error": repr(error)})
            raise


if __name__ == "__main__":
    main()
