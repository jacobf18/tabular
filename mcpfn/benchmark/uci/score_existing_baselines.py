"""Score every <method>.npy saved under the UCI eval tree on standardized RMSE
over the missing cells. Format-matched to the run_rfm_frozen.py output so we
can compare frozen-RFM head against existing methods (incl. the bar-head dwip).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


# Methods worth comparing against (subset: classic baselines + the same
# dwip_v3r1_t002_ctl checkpoint with its trained bar head, plus xRFM).
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
    "dwip_v3r1_t002_ctl-trial_001",     # SAME checkpoint we're using, with trained bar head
    "dwip_v3r1_t002_mmix-trial_001",    # mmix variant
    "dwip_v3r1_t003_ctl-trial_001",
    "dwip_v3r1_local14_ctl-trial_001",
]


def standardize(missing: np.ndarray):
    mu = np.nanmean(missing, axis=0)
    sigma = np.nanstd(missing, axis=0)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    return mu, sigma


def score_one(pred_path: Path, missing: np.ndarray, true: np.ndarray):
    pred = np.load(pred_path).astype(np.float64)
    if pred.shape != missing.shape:
        return None
    mu, sigma = standardize(missing)
    pred_std = (pred - mu) / sigma
    true_std = (true - mu) / sigma
    mask = np.isnan(missing)
    if not mask.any():
        return None
    err = pred_std[mask] - true_std[mask]
    return {
        "rmse_std": float(math.sqrt(np.mean(err ** 2))),
        "mae_std": float(np.mean(np.abs(err))),
        "n_missing": int(mask.sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path,
                   default=Path("/root/tabular/mcpfn/benchmark/datasets/uci"))
    p.add_argument("--pattern", default="MCAR")
    p.add_argument("--missingness", type=float, default=0.4)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--methods", nargs="*", default=METHODS)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.datasets is None:
        datasets = sorted(d.name for d in args.data_root.iterdir() if d.is_dir())
    else:
        datasets = args.datasets

    rows = {}  # rows[method][dataset] = {rmse_std, mae_std, n_missing}
    for name in datasets:
        cfg = args.data_root / name / args.pattern / f"missingness-{args.missingness}" / f"repeat-{args.repeat}"
        miss_p = cfg / "missing.npy"
        true_p = cfg / "true.npy"
        if not miss_p.is_file() or not true_p.is_file():
            continue
        missing = np.load(miss_p).astype(np.float64)
        true = np.load(true_p).astype(np.float64)
        for method in args.methods:
            pred_p = cfg / f"{method}.npy"
            if not pred_p.is_file():
                continue
            scored = score_one(pred_p, missing, true)
            if scored is None:
                continue
            rows.setdefault(method, {})[name] = scored

    # summary: per-method median across datasets
    summary = {}
    for method, perd in rows.items():
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
        "per_dataset": rows,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    # console pretty-print
    methods_sorted = sorted(summary.keys(), key=lambda m: summary[m]["median_rmse_std"])
    print(f"\n=== {args.pattern} p={args.missingness} repeat={args.repeat} ===")
    print(f"{'method':40s}  {'median_rmse_std':>16}  {'mean_rmse_std':>14}  {'n':>3}")
    for m in methods_sorted:
        s = summary[m]
        print(f"{m:40s}  {s['median_rmse_std']:16.4f}  {s['mean_rmse_std']:14.4f}  {s['n_datasets']:3d}")

    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
