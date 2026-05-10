"""Score existing baseline .npy predictions on the EXACT same 50x50 random
patches used by run_rfm_frozen.py (matched RNG seed). This isolates the
head-swap effect from any protocol-mismatch artifact.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


METHODS = [
    "column_mean",
    "knn",
    "softimpute",
    "mice",
    "missforest",
    "ice",
    "gain",
    "miwae",
    "notmiwae",
    "hyperimpute",
    "xrfm",
    "dwip_v3r1_t002_ctl-trial_001",
    "dwip_v3r1_t002_mmix-trial_001",
    "dwip_v3r1_t003_ctl-trial_001",
    "dwip_v3r1_local14_ctl-trial_001",
]


def standardize(missing: np.ndarray):
    mu = np.nanmean(missing, axis=0)
    sigma = np.nanstd(missing, axis=0)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    return mu, sigma


def random_subsample(rng: np.random.Generator, R: int, C: int, max_R: int, max_C: int):
    rows = rng.permutation(R)[:min(R, max_R)]
    cols = rng.permutation(C)[:min(C, max_C)]
    rows.sort()
    cols.sort()
    return rows, cols


def score_method_on_patches(
    method: str, missing: np.ndarray, true: np.ndarray, cfg: Path,
    num_subsamples: int, max_rows: int, max_cols: int, seed: int,
):
    pred_path = cfg / f"{method}.npy"
    if not pred_path.is_file():
        return None
    pred = np.load(pred_path).astype(np.float64)
    if pred.shape != missing.shape:
        return None
    mu, sigma = standardize(missing)
    pred_std = (pred - mu) / sigma
    true_std = (true - mu) / sigma
    miss_mask_full = np.isnan(missing)

    rng = np.random.default_rng(seed)
    sum_sq = 0.0
    sum_abs = 0.0
    n = 0
    for _ in range(num_subsamples):
        rows, cols = random_subsample(rng, missing.shape[0], missing.shape[1], max_rows, max_cols)
        sub_pred = pred_std[np.ix_(rows, cols)]
        sub_true = true_std[np.ix_(rows, cols)]
        sub_mask = miss_mask_full[np.ix_(rows, cols)]
        if not sub_mask.any():
            continue
        err = sub_pred[sub_mask] - sub_true[sub_mask]
        sum_sq += float((err ** 2).sum())
        sum_abs += float(np.abs(err).sum())
        n += int(sub_mask.sum())

    if n == 0:
        return None
    return {
        "rmse_std": math.sqrt(sum_sq / n),
        "mae_std": sum_abs / n,
        "n_missing": n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path,
                   default=Path("/root/tabular/mcpfn/benchmark/datasets/uci"))
    p.add_argument("--pattern", default="MCAR")
    p.add_argument("--missingness", type=float, default=0.4)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--num_subsamples", type=int, default=20)
    p.add_argument("--max_rows", type=int, default=50)
    p.add_argument("--max_cols", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--methods", nargs="*", default=METHODS)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    datasets = sorted(d.name for d in args.data_root.iterdir() if d.is_dir())

    rows_by_method = {}
    for name in datasets:
        cfg = args.data_root / name / args.pattern / f"missingness-{args.missingness}" / f"repeat-{args.repeat}"
        miss_p = cfg / "missing.npy"
        true_p = cfg / "true.npy"
        if not miss_p.is_file() or not true_p.is_file():
            continue
        missing = np.load(miss_p).astype(np.float64)
        true = np.load(true_p).astype(np.float64)
        for method in args.methods:
            scored = score_method_on_patches(
                method, missing, true, cfg,
                num_subsamples=args.num_subsamples,
                max_rows=args.max_rows, max_cols=args.max_cols,
                seed=args.seed,
            )
            if scored is None:
                continue
            rows_by_method.setdefault(method, {})[name] = scored

    summary = {}
    for method, perd in rows_by_method.items():
        rmses = [d["rmse_std"] for d in perd.values() if math.isfinite(d["rmse_std"])]
        summary[method] = {
            "median_rmse_std": float(np.median(rmses)) if rmses else float("nan"),
            "mean_rmse_std": float(np.mean(rmses)) if rmses else float("nan"),
            "n_datasets": len(rmses),
        }

    out = {
        "pattern": args.pattern,
        "missingness": args.missingness,
        "repeat": args.repeat,
        "num_subsamples": args.num_subsamples,
        "max_rows": args.max_rows,
        "max_cols": args.max_cols,
        "seed": args.seed,
        "per_dataset": rows_by_method,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    methods_sorted = sorted(summary.keys(), key=lambda m: summary[m]["median_rmse_std"])
    print(f"\n=== patch-protocol  {args.pattern} p={args.missingness} repeat={args.repeat} "
          f"({args.num_subsamples}x {args.max_rows}x{args.max_cols} seed={args.seed}) ===")
    print(f"{'method':40s}  {'median_rmse_std':>16}  {'mean_rmse_std':>14}  {'n':>3}")
    for m in methods_sorted:
        s = summary[m]
        print(f"{m:40s}  {s['median_rmse_std']:16.4f}  {s['mean_rmse_std']:14.4f}  {s['n_datasets']:3d}")
    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
