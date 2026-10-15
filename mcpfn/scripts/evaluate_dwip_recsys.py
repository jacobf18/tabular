"""Evaluate the fixed selected DWIP checkpoint on the existing Cornac rec-sys benchmarks."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time

import cornac
from cornac.eval_methods import RatioSplit
from cornac.models import Recommender
from cornac.exception import ScoreException
from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP
import numpy as np

from evaluate_dwip_missbench_uci import CHECKPOINT, REPO, sha256, write_json


BENCH = Path("/root/matrix-completion-benchmarks")
HARNESS = BENCH / "tests/rec_sys/test_cornac.py"
OUTPUT = REPO / "runs/benchmark_evaluations/dwip_teacher_mean128k_selected98k_recsys_20260919"
METRIC_NAMES = ("MAE", "RMSE", "AUC", "MAP", "NDCG@10", "Precision@10", "Recall@10")
DATASETS = {
    "movielens:100K": {"label": "MovieLens 100K", "folder": "movielens_100k", "source": Path("/root/.cornac/ml-100k/u.data"),
                       "shape": (943, 1656), "train": 80000, "test": 19971,
                       "baselines": BENCH / "tests/rec_sys/results/movielens100k_plus_recsys_clean_comparison.csv"},
    "netflix:small": {"label": "Netflix small", "folder": "netflix_small", "source": Path("/root/.cornac/netflix/data_small.csv"),
                      "shape": (9976, 4895), "train": 486242, "test": 121470,
                      "baselines": BENCH / "tests/rec_sys/results/netflix_small_baselines_plus_compressor_outer300x100.csv"},
}


def metrics():
    return [MAE(), RMSE(), Precision(k=10), Recall(k=10), NDCG(k=10), AUC(), MAP()]


def normalize_training_ratings(matrix):
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    stds = np.where(~np.isfinite(stds) | np.isclose(stds, 0, atol=1e-4), 1.0, stds)
    return means, stds


def user_rating_metrics(users, truth, prediction):
    unique, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse)
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    mae = np.bincount(inverse, weights=np.abs(error)) / counts
    rmse = np.sqrt(np.bincount(inverse, weights=error**2) / counts)
    return {"MAE": float(mae.mean()), "RMSE": float(rmse.mean()), "users": int(unique.size)}


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


class FrozenRatings(Recommender):
    def __init__(self, imputer=None, prediction=None):
        super().__init__(name="DWIP_teacher_mean128k_selected98k", trainable=True, verbose=False)
        self.imputer = imputer
        self.pred_matrix = prediction
        self.normalization_means = None
        self.normalization_stds = None
        self.imputation_seconds = None

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        if self.pred_matrix is not None:
            if self.pred_matrix.shape != (train_set.num_users, train_set.num_items):
                raise ValueError("Saved predictions do not match the train mapping")
            return self
        import torch

        observed = np.full((train_set.num_users, train_set.num_items), np.nan, dtype=np.float64)
        users, items, values = train_set.uir_tuple
        observed[users, items] = values
        means, stds = normalize_training_ratings(observed)
        self.normalization_means, self.normalization_stds = means, stds
        torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = self.imputer.impute(observed, means=means, stds=stds)
        torch.cuda.synchronize()
        self.imputation_seconds = time.perf_counter() - started
        if prediction.shape != observed.shape or not np.isfinite(prediction).all():
            raise ValueError("Invalid completed rating matrix")
        self.pred_matrix = np.clip(prediction, train_set.min_rating, train_set.max_rating)
        if not np.allclose(self.pred_matrix[users, items], values, rtol=1e-12, atol=1e-12):
            raise ValueError("Observed training ratings changed")
        return self

    def score(self, user_idx, item_idx=None):
        if not self.knows_user(user_idx):
            raise ScoreException(f"Unknown user: {user_idx}")
        if self.pred_matrix is None:
            raise ScoreException("No rating predictions")
        if item_idx is None:
            return self.pred_matrix[user_idx]
        if not self.knows_item(item_idx):
            raise ScoreException(f"Unknown item: {item_idx}")
        return self.pred_matrix[user_idx, item_idx]


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(output):
    import torch
    from tabimpute.interface import ImputePFN

    sys.path.insert(0, str(HARNESS.parent))
    from test_cornac import load_feedback_from_spec

    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".evaluation.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(0.9)
    config_path = CHECKPOINT.parent / "config.json"
    config = json.loads(config_path.read_text())["model"]
    protocol = {"checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT), "config_sha256": sha256(config_path),
                "script_sha256": sha256(Path(__file__)), "harness_sha256": sha256(HARNESS),
                "interface_sha256": sha256(REPO / "src/tabimpute/interface.py"),
                "datasets": list(DATASETS), "dataset_sources_sha256": {name: sha256(spec["source"]) for name, spec in DATASETS.items()},
                "test_fraction": 0.2, "split_seed": 123, "exclude_unknowns": True, "rating_threshold": 4.0,
                "user_based": True, "metrics": list(METRIC_NAMES), "prediction_stat": "mean", "precision": "amp_bfloat16",
                "max_num_rows": 1000, "max_num_cols": 1000, "max_num_chunks": 1, "preprocessors": None,
                "normalization": "Existing Cornac adapter: observed-training column means/stds; near-zero std threshold 1e-4",
                "clipping": "Training rating range [1,5], matching the existing adapter", "cudnn_sdpa": False,
                "model_seed": 42, "training": "None; Cornac fit computes zero-shot imputation only",
                "selection": "Fixed synthetic-selected checkpoint and fixed block size, no benchmark tuning or data subsampling"}
    if protocol["checkpoint_sha256"] != "1448ca69d603ac62bb6e9372e5e02980c969062de7cf099e14e46f0449d5d43d":
        raise ValueError("Unexpected selected checkpoint")
    if (output / "manifest.json").exists():
        if json.loads((output / "manifest.json").read_text())["protocol"] != protocol:
            raise ValueError("Existing run has different provenance")
    else:
        write_json(output / "manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
                                             "cornac": cornac.__version__, "torch": torch.__version__, "numpy": np.__version__,
                                             "gpu": torch.cuda.get_device_name(), "disk_reserve_gib": 20})
    imputer = ImputePFN(device="cuda", checkpoint_path=str(CHECKPOINT), entry_wise_features=False,
                        json_config={"config": config}, model_family="new_stable", prediction_stat="mean",
                        precision_mode="amp_bfloat16", max_num_rows=1000, max_num_cols=1000, max_num_chunks=1, verbose=True)
    summaries, historical = [], []
    artifact_hashes = {}
    started = time.time()
    for name, spec in DATASETS.items():
        if shutil.disk_usage(output).free < 20 * 1024**3:
            raise RuntimeError("20 GiB disk reserve reached")
        directory = output / spec["folder"]
        directory.mkdir(exist_ok=True)
        write_json(output / "status.json", {"state": "running", "dataset": name, "phase": "load_split", "completed": len(summaries)})
        feedback = load_feedback_from_spec(name)
        split = RatioSplit(data=feedback, test_size=0.2, rating_threshold=4.0, seed=123, exclude_unknowns=True)
        train_users, train_items, train_values = split.train_set.uir_tuple
        test_users, test_items, test_values = split.test_set.uir_tuple
        shape = (split.train_set.num_users, split.train_set.num_items)
        assert shape == spec["shape"] and len(train_values) == spec["train"] and len(test_values) == spec["test"]
        overlap = np.intersect1d(train_users * shape[1] + train_items, test_users * shape[1] + test_items)
        if overlap.size:
            raise ValueError("Train/test rating-cell overlap")
        atomic_npz(directory / "split.npz", train_users=train_users, train_items=train_items, train_ratings=train_values,
                   test_users=test_users, test_items=test_items, test_ratings=test_values, shape=np.asarray(shape))
        write_json(directory / "id_mappings.json", {"users": {str(key): int(value) for key, value in split.train_set.uid_map.items()},
                                                    "items": {str(key): int(value) for key, value in split.train_set.iid_map.items()}})
        write_json(output / "status.json", {"state": "running", "dataset": name, "phase": "impute_and_evaluate", "completed": len(summaries)})
        prediction_path = directory / "predictions.npz"
        result_path = directory / "metrics.json"
        if result_path.exists():
            prior = json.loads(result_path.read_text())
            assert sha256(prediction_path) == prior["prediction_sha256"]
            with np.load(prediction_path, allow_pickle=False) as stored:
                model = FrozenRatings(prediction=stored["ratings"])
            imputation_seconds = prior["imputation_seconds"]
        else:
            torch.cuda.reset_peak_memory_stats()
            model = FrozenRatings(imputer=imputer)
            imputation_seconds = None
        result, _ = split.evaluate(model=model, metrics=metrics(), user_based=True)
        values = {metric: float(result.metric_avg_results[metric]) for metric in METRIC_NAMES}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("Nonfinite evaluation metric")
        if imputation_seconds is None:
            imputation_seconds = model.imputation_seconds
        prediction = model.pred_matrix
        if not prediction_path.exists():
            atomic_npz(prediction_path, ratings=prediction)
        atomic_npz(directory / "test_predictions.npz", users=test_users, items=test_items, truth=test_values,
                   prediction=prediction[test_users, test_items])
        if model.normalization_means is not None:
            atomic_npz(directory / "normalization.npz", means=model.normalization_means, stds=model.normalization_stds)
        row = {"dataset": spec["label"], "dataset_spec": name, "users": shape[0], "items": shape[1],
               "train_ratings": len(train_values), "test_ratings": len(test_values), "source_ratings": len(feedback),
               "excluded_unknown_test_ratings": len(feedback) - len(train_values) - len(test_values), **values,
               "imputation_seconds": imputation_seconds, "cornac_fit_seconds": float(result.metric_avg_results["Train (s)"]),
               "cornac_evaluation_seconds": float(result.metric_avg_results["Test (s)"]),
               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "prediction_sha256": sha256(prediction_path)}
        user_rows = []
        per_user = result.metric_user_results
        all_users = sorted({int(user) for scores in per_user.values() for user in scores})
        for user in all_users:
            user_rows.append({"user_index": user, **{metric: float(per_user[metric][user]) if user in per_user[metric] else None for metric in METRIC_NAMES}})
        write_csv(directory / "per_user_metrics.csv", user_rows)
        row["users_scored_per_metric"] = {metric: len(per_user[metric]) for metric in METRIC_NAMES}
        write_json(result_path, row)
        write_json(output / "status.json", {"state": "running", "dataset": name, "phase": "independent_metric_replay", "completed": len(summaries)})
        independent = user_rating_metrics(test_users, test_values, prediction[test_users, test_items])
        for metric in ("MAE", "RMSE"):
            assert np.isclose(independent[metric], values[metric], rtol=1e-12, atol=1e-12)
        del model, prediction
        with np.load(prediction_path, allow_pickle=False) as stored:
            replay_model = FrozenRatings(prediction=stored["ratings"])
        replay, _ = split.evaluate(model=replay_model, metrics=metrics(), user_based=True)
        differences = {metric: abs(float(replay.metric_avg_results[metric]) - values[metric]) for metric in METRIC_NAMES}
        assert all(difference < 1e-12 for difference in differences.values())
        write_json(directory / "audit.json", {"status": "passed", "source_sha256": sha256(spec["source"]),
                   "split_sha256": sha256(directory / "split.npz"), "prediction_sha256": row["prediction_sha256"],
                   "train_test_overlap_cells": int(overlap.size), "rating_metrics_independent_numpy": independent,
                   "saved_prediction_replay_metric_differences": differences, "shape_matches_existing_benchmark": True})
        del replay_model, replay, split, feedback
        summaries.append(row)
        with spec["baselines"].open(newline="") as handle:
            for baseline in csv.DictReader(handle):
                if baseline["model"] in ("MF", "SVD", "PMF", "BPR", "UserKNN", "ItemKNN"):
                    historical.append({"dataset": spec["label"], "model": baseline["model"], "source": str(spec["baselines"]),
                                       "source_sha256": sha256(spec["baselines"]), **{metric: float(baseline[metric]) for metric in METRIC_NAMES}})
        for artifact in directory.iterdir():
            if artifact.is_file():
                artifact_hashes[str(artifact.relative_to(output))] = sha256(artifact)
        print(json.dumps(row), flush=True)
    write_csv(output / "results.csv", summaries)
    write_csv(output / "historical_baselines.csv", historical)
    sections = ["# New DWIP recommender-system results\n",
                "Fixed step-98000 checkpoint from the completed 128000-step teacher-mean run. No model training or benchmark tuning.\n",
                "Existing Cornac protocol: MovieLens 100K and Netflix small, 80/20 rating holdout, seed 123, "
                "unknown users/items excluded, relevant rating >=4, rating metrics averaged over users. "
                "Predictions clipped to training rating range [1,5], as in the existing adapter.\n",
                "Inference uses predictive mean, FP32 weights/BF16 autocast, no permutation ensemble, global training-column "
                "normalization, and fixed 1000×1000 blocks with the public interface's overlapping tail averaging. "
                "Every training user and item is retained; this is not a subset evaluation.\n",
                "| Dataset | MAE ↓ | RMSE ↓ | AUC ↑ | MAP ↑ | NDCG@10 ↑ | Precision@10 ↑ | Recall@10 ↑ |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        sections.append("| " + row["dataset"] + " | " + " | ".join(f"{row[metric]:.6f}" for metric in METRIC_NAMES) + " |")
    sections.extend(["\n## Existing baseline context\n",
                     "These baseline scores are from saved prior runs of the same documented Cornac protocol, "
                     "not rerun here. Their source files report four-decimal precision; checkpoint/block settings "
                     "for earlier TabImpute variants are not substituted for this model.\n",
                     "| Dataset | Model | MAE ↓ | RMSE ↓ | AUC ↑ | MAP ↑ | NDCG@10 ↑ |",
                     "|---|---|---:|---:|---:|---:|---:|"])
    for row in historical:
        sections.append("| " + row["dataset"] + " | " + row["model"] + " | " + " | ".join(f"{row[metric]:.4f}" for metric in ("MAE", "RMSE", "AUC", "MAP", "NDCG@10")) + " |")
    sections.extend(["\n## Saved artifacts\n",
                     "`results.csv`: full-precision metrics and split coverage. `manifest.json`: model/protocol/source hashes. "
                     "Each dataset directory contains frozen `split.npz`, raw-to-internal ID maps, `predictions.npz` "
                     "(complete clipped rating matrix), held-out `test_predictions.npz`, observed-column normalization, "
                     "per-user metric CSV, and an independent `audit.json`.\n",
                     "MAE/RMSE are Cornac's user-averaged rating metrics, not pooled rating errors or MissBench NRMSE. "
                     "Users without relevant test items are excluded from ranking averages by Cornac; counts are saved per metric. "
                     "A single established split is evaluated; no multi-seed confidence claim is made."])
    (output / "RESULTS.md").write_text("\n".join(sections) + "\n")
    assert sha256(CHECKPOINT) == protocol["checkpoint_sha256"]
    assert all(sha256(spec["source"]) == protocol["dataset_sources_sha256"][name] for name, spec in DATASETS.items())
    write_json(output / "status.json", {"state": "complete", "datasets_completed": len(summaries), "failed": 0,
               "elapsed_seconds": time.time() - started, "free_disk_gib": shutil.disk_usage(output).free / 1024**3})
    for filename in ("results.csv", "RESULTS.md", "manifest.json", "status.json", "historical_baselines.csv"):
        artifact_hashes[filename] = sha256(output / filename)
    write_json(output / "artifact_sha256.json", artifact_hashes)
    write_json(output / "completion_audit.json", {"status": "passed", "datasets": list(DATASETS),
               "checkpoint_sha256": protocol["checkpoint_sha256"], "artifact_count": len(artifact_hashes),
               "source_files_unchanged": True, "prediction_replay": "All seven metrics match saved-matrix replay within 1e-12",
               "rating_metric_audit": "Independent NumPy per-user MAE and RMSE match within 1e-12"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        run(args.output.resolve())
    except Exception as error:
        import traceback
        write_json(args.output / "fatal_error.json", {"error": str(error), "traceback": traceback.format_exc()})
        raise
