#!/usr/bin/env python3
"""Aggregate prop99 per-draw metrics across draws.

Two summaries are produced:
  results/summary.csv             — per method (overall RMSE / bias / runtime)
  results/summary_by_design.csv   — per (design, Nt, T0, N, T, method) — only when those
                                     columns are present in per_draw_metrics.csv

Mirrors kansas/summarize_kansas_causal.py and kansas's compute style. The
paper's headline metric (RMSE on hidden entries) is reported as
mean_treated_block_rmse.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, required=True,
                   help="Directory containing per_draw_metrics.csv.")
    return p.parse_args()


def aggregate(g: pd.DataFrame) -> dict:
    tau = g["tau_hat"].to_numpy(dtype=float)
    n = len(tau)
    bias = float(tau.mean())
    if n > 1:
        se_bias = float(tau.std(ddof=1) / np.sqrt(n))
        bias_lo, bias_hi = bias - 1.96 * se_bias, bias + 1.96 * se_bias
    else:
        se_bias = float("nan")
        bias_lo = bias_hi = float("nan")

    tau_sq = tau ** 2
    rmse = float(np.sqrt(tau_sq.mean()))
    if n > 1 and rmse > 0:
        se_m = float(tau_sq.std(ddof=1) / np.sqrt(n))
        se_rmse = se_m / (2 * rmse)
        rmse_lo, rmse_hi = rmse - 1.96 * se_rmse, rmse + 1.96 * se_rmse
    else:
        se_rmse = float("nan")
        rmse_lo = rmse_hi = float("nan")

    tb_rmse = g["treated_block_rmse"].to_numpy(dtype=float)
    tb_mean = float(tb_rmse.mean())
    tb_se = float(tb_rmse.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

    return {
        "n_draws": int(n),
        "bias_tau": bias,
        "bias_tau_se": se_bias,
        "bias_tau_lo95": bias_lo,
        "bias_tau_hi95": bias_hi,
        "rmse_tau": rmse,
        "rmse_tau_se": se_rmse,
        "rmse_tau_lo95": rmse_lo,
        "rmse_tau_hi95": rmse_hi,
        "mae_tau": float(g["tau_abs_error"].mean()),
        "mean_treated_block_rmse": tb_mean,
        "mean_treated_block_rmse_se": tb_se,
        "mean_treated_block_rmse_lo95": tb_mean - 1.96 * tb_se if n > 1 else float("nan"),
        "mean_treated_block_rmse_hi95": tb_mean + 1.96 * tb_se if n > 1 else float("nan"),
        "mean_treated_block_mae": float(g["treated_block_mae"].mean()),
        "mean_observed_block_rmse": float(g["observed_block_rmse"].mean()),
        "mean_runtime_seconds": float(g["runtime_seconds"].mean()),
    }


def main() -> None:
    args = parse_args()
    per_draw_path = args.results_dir / "per_draw_metrics.csv"
    if not per_draw_path.exists():
        raise SystemExit(f"Missing per-draw metrics file: {per_draw_path}")

    df = pd.read_csv(per_draw_path)
    if df.empty:
        raise SystemExit("Per-draw metrics file is empty.")

    overall_rows = []
    for method, g in df.groupby("method", dropna=False):
        overall_rows.append({"method": method, **aggregate(g)})
    overall = pd.DataFrame(overall_rows).sort_values("mean_treated_block_rmse", ascending=True)
    overall_path = args.results_dir / "summary.csv"
    overall.to_csv(overall_path, index=False)
    print(overall.to_string(index=False))
    print(f"\nWrote summary to {overall_path}")

    strat_cols = [c for c in ["design", "Nt", "T0", "N", "T"] if c in df.columns]
    if strat_cols:
        rows = []
        for keys, g in df.groupby(strat_cols + ["method"], dropna=False):
            row = dict(zip(strat_cols + ["method"], keys if isinstance(keys, tuple) else (keys,)))
            row.update(aggregate(g))
            rows.append(row)
        if rows:
            by_design = pd.DataFrame(rows).sort_values(strat_cols + ["mean_treated_block_rmse"])
            by_design_path = args.results_dir / "summary_by_design.csv"
            by_design.to_csv(by_design_path, index=False)
            print(f"Wrote stratified summary to {by_design_path}")


if __name__ == "__main__":
    main()
