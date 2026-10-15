"""Small, standalone helpers shared by the fixed-checkpoint rating evaluations."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "src/tabimpute/workdir/tabimpute-dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918-dwip1e5/best_validation_checkpoint.pth"
OUTPUT = REPO / "runs/benchmark_evaluations/dwip_teacher_mean128k_selected98k_recsys_20260919"
DATASETS = {
    "movielens:100K": {"label": "MovieLens 100K", "folder": "movielens_100k", "shape": (943, 1656), "train": 80000, "test": 19971},
    "netflix:small": {"label": "Netflix small", "folder": "netflix_small", "shape": (9976, 4895), "train": 486242, "test": 121470},
}


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


def write_csv(path, rows):
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def normalize_training_ratings(matrix):
    means, stds = np.nanmean(matrix, axis=0), np.nanstd(matrix, axis=0)
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
