"""Evaluate the paper checkpoint on frozen rating splits without saving dense matrices."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np

from recsys_evaluation_common import (
    CHECKPOINT, DATASETS, OUTPUT as FROZEN_SOURCE, REPO, atomic_npz,
    normalize_training_ratings, sha256, user_rating_metrics, write_csv, write_json,
)


CHECKPOINT_SHA256 = "1448ca69d603ac62bb6e9372e5e02980c969062de7cf099e14e46f0449d5d43d"
DEFAULT_OUTPUT = REPO / "runs/benchmark_evaluations/matrixformer_recsys_global_20260930"


def build_observed(split):
    shape = tuple(split["shape"])
    train_users, train_items = split["train_users"], split["train_items"]
    test_users, test_items = split["test_users"], split["test_items"]
    if not np.isfinite(split["train_ratings"]).all():
        raise ValueError("Nonfinite training ratings")
    if np.intersect1d(train_users * shape[1] + train_items, test_users * shape[1] + test_items).size:
        raise ValueError("Train/test rating-cell overlap")
    observed = np.full(shape, np.nan, dtype=np.float64)
    observed[train_users, train_items] = split["train_ratings"]
    return observed


def normalization_parameters(observed, mode):
    if mode == "column":
        return normalize_training_ratings(observed)
    if mode != "global":
        raise ValueError("Unknown normalization mode")
    values = observed[np.isfinite(observed)]
    if not values.size:
        raise ValueError("No observed training ratings")
    scale = values.std(ddof=0)
    if np.isclose(scale, 0, atol=1e-4):
        scale = 1.0
    return np.full(observed.shape[1], values.mean()), np.full(observed.shape[1], scale)


def block_parameters(max_num_rows, max_num_cols):
    if max_num_rows <= 0 or max_num_cols <= 0:
        raise ValueError("Block dimensions must be positive")
    return {"max_num_rows": max_num_rows, "max_num_cols": max_num_cols, "max_num_chunks": 1}


def run(output, modes, max_num_rows=1000, max_num_cols=1000, streamed=False):
    import torch
    from tabimpute.streamed_inference import DenseReferenceImputer, StreamedImputePFN

    blocks = block_parameters(max_num_rows, max_num_cols)
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("Unexpected paper checkpoint")
    if shutil.disk_usage(output.parent).free < 256 * 1024**2:
        raise RuntimeError("Need 256 MiB free for compact held-out prediction artifacts")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.cuda.set_per_process_memory_fraction(0.9)
    config_path = CHECKPOINT.parent / "config.json"
    config = json.loads(config_path.read_text())["model"]
    source_hashes = json.loads((FROZEN_SOURCE / "artifact_sha256.json").read_text())
    source_paths = [Path(__file__), Path(__file__).with_name("recsys_evaluation_common.py"),
                    REPO / "src/tabimpute/streamed_inference.py", *sorted((REPO / "src/tabimpute/model").rglob("*.py"))]
    if streamed:
        imputer_type = StreamedImputePFN
    else:
        imputer_type = DenseReferenceImputer
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA256, "config_sha256": sha256(config_path),
        "source_files_sha256": {str(path): sha256(path) for path in source_paths},
        "frozen_source": str(FROZEN_SOURCE), "frozen_manifest_sha256": sha256(FROZEN_SOURCE / "manifest.json"),
        "normalizations": modes, "primary_normalization": "global", "normalization_data": "observed outer-training ratings only",
        "global_definition": "One population mean and standard deviation over all observed training ratings, shared by all columns and blocks",
        "split_seed": 123, "test_fraction": 0.2, "exclude_unknowns": True,
        "metric_aggregation": "Equal-user mean MAE and equal-user mean within-user RMSE; not pooled rating RMSE",
        "prediction_stat": "mean", "precision": "amp_bfloat16", "block_shape": [max_num_rows, max_num_cols],
        "max_num_chunks": 1, "preprocessors": None, "model_seed": 42,
        "clipping": "Observed training rating range [1,5]", "training": "None",
        "selection": "Paper checkpoint fixed in advance; global normalization requested before this rerun; column mode is a control",
        "geometry_selection": "Block dimensions specified before evaluation; no geometry sweep or per-dataset selection",
        "inference_execution": "streamed" if streamed else "dense",
        "streaming_batches": {"attention": 32, "cells": 8192, "encoder_rows": 32} if streamed else None,
        "test_caveat": "Previously inspected benchmark split, not a new blind test",
        "torch": torch.__version__, "numpy": np.__version__, "gpu": torch.cuda.get_device_name(),
        "splits_sha256": {},
    }
    write_json(output / "manifest.json", manifest)
    shutil.copy2(Path(__file__), output / "evaluator_snapshot.py")
    imputer = imputer_type(device="cuda", checkpoint_path=str(CHECKPOINT), entry_wise_features=False,
                        json_config={"config": config}, model_family="new_stable", prediction_stat="mean",
                        precision_mode="amp_bfloat16", preprocessors=None, verbose=False, **blocks)
    rows = []
    for dataset, specification in DATASETS.items():
        folder = specification["folder"]
        source = FROZEN_SOURCE / folder / "split.npz"
        expected_hash = source_hashes[f"{folder}/split.npz"]
        if sha256(source) != expected_hash:
            raise ValueError("Frozen split checksum mismatch")
        manifest["splits_sha256"][folder] = expected_hash
        with np.load(source, allow_pickle=False) as stored:
            split = dict(stored)
        observed = build_observed(split)
        assert observed.shape == specification["shape"]
        assert len(split["train_ratings"]) == specification["train"]
        assert len(split["test_ratings"]) == specification["test"]
        for mode in modes:
            directory = output / folder / mode
            directory.mkdir(parents=True)
            write_json(output / "status.json", {"state": "running", "dataset": dataset, "normalization": mode})
            means, stds = normalization_parameters(observed, mode)
            torch.manual_seed(42)
            np.random.seed(42)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            prediction = imputer.impute(observed.copy(), means=means, stds=stds)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if prediction.shape != observed.shape or not np.isfinite(prediction).all():
                raise ValueError("Invalid imputed matrix")
            prediction = np.clip(prediction, split["train_ratings"].min(), split["train_ratings"].max())
            if not np.allclose(prediction[split["train_users"], split["train_items"]], split["train_ratings"], rtol=1e-12, atol=1e-12):
                raise ValueError("Observed ratings changed")
            heldout = prediction[split["test_users"], split["test_items"]]
            scores = user_rating_metrics(split["test_users"], split["test_ratings"], heldout)
            prediction_path = directory / "test_predictions.npz"
            atomic_npz(prediction_path, users=split["test_users"], items=split["test_items"],
                       truth=split["test_ratings"], prediction=heldout)
            atomic_npz(directory / "normalization.npz", means=means, stds=stds)
            row = {"dataset": specification["label"], "dataset_spec": dataset, "normalization": mode,
                   "block_rows": max_num_rows, "block_cols": max_num_cols,
                   "inference_execution": "streamed" if streamed else "dense",
                   **scores, "train_ratings": len(split["train_ratings"]), "test_ratings": len(heldout),
                   "imputation_seconds": elapsed, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "split_sha256": expected_hash, "prediction_sha256": sha256(prediction_path)}
            write_json(directory / "metrics.json", row)
            rows.append(row)
            write_csv(output / "results.csv", rows)
            print(json.dumps(row), flush=True)
            del prediction, heldout
            torch.cuda.empty_cache()
        del observed
    write_json(output / "manifest.json", manifest)
    write_json(output / "status.json", {"state": "complete", "completed": len(rows)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--normalizations", nargs="+", choices=("global", "column"), default=["global", "column"])
    parser.add_argument("--max-num-rows", type=int, default=1000)
    parser.add_argument("--max-num-cols", type=int, default=1000)
    parser.add_argument("--streamed", action="store_true")
    arguments = parser.parse_args()
    run(arguments.output, arguments.normalizations, arguments.max_num_rows, arguments.max_num_cols, arguments.streamed)
