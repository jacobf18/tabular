#!/usr/bin/env python3
"""Run a single dwip anchor model on the transpose of each draw, then average
the original and transposed completions to form an ensemble.

Adds two new methods to results/per_draw_metrics.csv:
    {method}-T     -- model run on observed.T (per-row standardization)
    {method}-avg   -- 0.5 * (existing {method}.npy + new {method}-T.npy)

The original {method}.npy must already exist (produced by eval_*_anchors.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

BENCH_DIR = Path(__file__).resolve().parent
CPS_DIR = BENCH_DIR / "CPS"
for extra in (BENCH_DIR, CPS_DIR):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from eval_cps_causal import (  # noqa: E402
    ROUND2_T2_CONFIG,
    compute_metrics,
    make_mcpfn_imputer,
)


ANCHOR_CKPT_DIR = Path("/root/tabimpute_anchor3_named")
V1_CONFIG = {"config": {**ROUND2_T2_CONFIG["config"], "rope_fraction": 0.75}}

ANCHOR_SPECS_BY_NAME: dict[str, dict] = {
    "dwip-v2-1": {
        "name": "dwip-v2-1",
        "model_family": "new_stable",
        "checkpoint_path": str(ANCHOR_CKPT_DIR / "dwip_v2_t002_checkpoint_17000.pth"),
        "json_config": ROUND2_T2_CONFIG,
    },
    "dwip-v2-2": {
        "name": "dwip-v2-2",
        "model_family": "new_stable",
        "checkpoint_path": str(ANCHOR_CKPT_DIR / "dwip_v2_t003_checkpoint_17000.pth"),
        "json_config": ROUND2_T2_CONFIG,
    },
    "dwip-v1": {
        "name": "dwip-v1",
        "model_family": "new",
        "checkpoint_path": str(ANCHOR_CKPT_DIR / "dwip_v1_local14_checkpoint_10000.pth"),
        "json_config": V1_CONFIG,
    },
}

DWIP_SPECTRAL_TM128K_DIR = Path(
    "/root/tabular/mcpfn/src/tabimpute/workdir/"
    "tabimpute-dwip_spectral_anchor_mix25_teacher_mean_huber1_128k_20260918-dwip1e5"
)
DWIP_SPECTRAL_TM128K_CONFIG = {
    "config": json.loads((DWIP_SPECTRAL_TM128K_DIR / "config.json").read_text())["model"]
}
for _name, _precision, _stat in (
    ("dwip-spectral-tm128k", "amp_bfloat16", "mean"),
    ("dwip-spectral-tm128k-median", "full_bfloat16", "median"),
):
    ANCHOR_SPECS_BY_NAME[_name] = {
        "name": _name,
        "model_family": "new_stable",
        "checkpoint_path": str(DWIP_SPECTRAL_TM128K_DIR / "best_validation_checkpoint.pth"),
        "json_config": DWIP_SPECTRAL_TM128K_CONFIG,
        "precision_mode": _precision,
        "prediction_stat": _stat,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-dir", type=Path, required=True,
                   help="Directory with draw_* subfolders.")
    p.add_argument("--method", type=str, required=True,
                   choices=sorted(ANCHOR_SPECS_BY_NAME.keys()),
                   help="Which dwip anchor to run on the transpose.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force-rerun", action="store_true",
                   help="Overwrite existing -T / -avg outputs.")
    return p.parse_args()


def standardize_columns(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    col_mean = np.nanmean(x, axis=0)
    col_std = np.nanstd(x, axis=0)
    col_std = np.where((col_std == 0) | np.isnan(col_std), 1.0, col_std)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    return (x - col_mean) / col_std, col_mean, col_std


def impute_transpose(impute_fn, observed: np.ndarray, missing_mask: np.ndarray,
                     standardize: bool) -> np.ndarray:
    """Run impute_fn on observed.T, return completion in original orientation."""
    obs_t = observed.T
    if standardize:
        obs_t_std, mean_t, std_t = standardize_columns(obs_t)
    else:
        obs_t_std = obs_t
        mean_t = np.zeros(obs_t.shape[1], dtype=obs_t.dtype)
        std_t = np.ones(obs_t.shape[1], dtype=obs_t.dtype)
    completed_t_std = impute_fn(obs_t_std.copy()).astype(np.float32)
    completed_t = (completed_t_std * std_t) + mean_t
    completed = completed_t.T.astype(np.float32)
    completed[~missing_mask] = observed[~missing_mask]
    return completed


def main() -> None:
    args = parse_args()
    if not args.benchmark_dir.exists():
        raise SystemExit(f"Benchmark directory does not exist: {args.benchmark_dir}")

    spec = ANCHOR_SPECS_BY_NAME[args.method]
    if not Path(spec["checkpoint_path"]).exists():
        raise SystemExit(f"Missing checkpoint: {spec['checkpoint_path']}")

    print(f"Benchmark: {args.benchmark_dir}")
    print(f"Loading anchor model: {args.method}  <-  {spec['checkpoint_path']}")
    impute_fn = make_mcpfn_imputer(spec, args.device)

    draw_dirs = sorted(
        p for p in args.benchmark_dir.iterdir()
        if p.is_dir() and p.name.startswith("draw_")
    )
    if args.limit is not None:
        draw_dirs = draw_dirs[: args.limit]

    results_dir = args.benchmark_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    method_t = f"{args.method}-T"
    method_avg = f"{args.method}-avg"

    metrics_rows: list[dict] = []
    failures: list[dict] = []
    standardize = not args.no_standardize

    for draw_dir in tqdm(draw_dirs, desc=f"{args.benchmark_dir.name}:{args.method}-T+avg"):
        true = np.load(draw_dir / "true.npy")
        observed = np.load(draw_dir / "observed.npy")
        treatment_mask = np.load(draw_dir / "treatment_mask.npy")
        metadata = json.loads((draw_dir / "metadata.json").read_text(encoding="utf-8"))
        missing_mask = np.isnan(observed)

        orig_path = draw_dir / f"{args.method}.npy"
        if not orig_path.exists():
            failures.append({"draw_name": draw_dir.name, "method": args.method,
                             "error": f"original {orig_path.name} missing — run eval_*_anchors.py first"})
            continue
        completed_orig = np.load(orig_path)

        # Transpose run
        t_path = draw_dir / f"{method_t}.npy"
        if t_path.exists() and not args.force_rerun:
            completed_t = np.load(t_path)
            runtime_t = float("nan")
        else:
            start = time.time()
            try:
                completed_t = impute_transpose(impute_fn, observed, missing_mask, standardize)
            except Exception as exc:
                failures.append({"draw_name": draw_dir.name, "method": method_t, "error": repr(exc)})
                continue
            runtime_t = time.time() - start
            np.save(t_path, completed_t)

        # Average ensemble
        avg_path = draw_dir / f"{method_avg}.npy"
        if avg_path.exists() and not args.force_rerun:
            completed_avg = np.load(avg_path)
        else:
            completed_avg = (0.5 * (completed_orig.astype(np.float32) + completed_t.astype(np.float32))).astype(np.float32)
            completed_avg[~missing_mask] = observed[~missing_mask]
            np.save(avg_path, completed_avg)

        # Pull through whatever optional metadata fields the existing CSV has;
        # prop99 includes design/Nt/T0/N/T while CPS/kansas don't.
        extra = {k: metadata.get(k) for k in ("design", "Nt", "T0", "N", "T") if k in metadata}

        for name, completed, rt in (
            (method_t, completed_t, runtime_t),
            (method_avg, completed_avg, float("nan")),
        ):
            metrics = compute_metrics(true, completed, treatment_mask)
            metrics_rows.append({
                "draw_name": draw_dir.name,
                "draw_id": int(metadata["draw_id"]),
                "method": name,
                "runtime_seconds": rt,
                **metrics,
                **extra,
            })

    per_draw_path = results_dir / "per_draw_metrics.csv"
    new_df = pd.DataFrame(metrics_rows)
    if per_draw_path.exists() and not new_df.empty:
        old_df = pd.read_csv(per_draw_path)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["draw_name", "method"], keep="last")
        merged.to_csv(per_draw_path, index=False)
    else:
        new_df.to_csv(per_draw_path, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(results_dir / "failures_transpose.csv", index=False)
        print(f"Encountered {len(failures)} failures; see results/failures_transpose.csv")

    print(f"Wrote {len(metrics_rows)} rows ({method_t}, {method_avg}) to {per_draw_path}")


if __name__ == "__main__":
    main()
