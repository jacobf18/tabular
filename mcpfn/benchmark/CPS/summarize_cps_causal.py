#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate CPS causal benchmark results across draws."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing per_draw_metrics.csv.",
    )
    return parser.parse_args()


def aggregate_with_ci(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, g in df.groupby("method", dropna=False):
        tau = g["tau_hat"].to_numpy(dtype=float)
        n = len(tau)
        bias = float(tau.mean())
        if n > 1:
            se_bias = float(tau.std(ddof=1) / np.sqrt(n))
            bias_lo, bias_hi = bias - 1.96 * se_bias, bias + 1.96 * se_bias
        else:
            se_bias = float("nan")
            bias_lo = bias_hi = float("nan")

        # RMSE(tau) = sqrt(mean(tau^2)). Delta-method SE via the sample variance of tau^2.
        tau_sq = tau ** 2
        m = float(tau_sq.mean())
        rmse = float(np.sqrt(m))
        if n > 1 and rmse > 0:
            se_m = float(tau_sq.std(ddof=1) / np.sqrt(n))
            se_rmse = se_m / (2 * rmse)
            rmse_lo, rmse_hi = rmse - 1.96 * se_rmse, rmse + 1.96 * se_rmse
        else:
            se_rmse = float("nan")
            rmse_lo = rmse_hi = float("nan")

        rows.append({
            "method": method,
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
            "mean_treated_block_rmse": float(g["treated_block_rmse"].mean()),
            "mean_treated_block_mae": float(g["treated_block_mae"].mean()),
            "mean_observed_block_rmse": float(g["observed_block_rmse"].mean()),
            "mean_runtime_seconds": float(g["runtime_seconds"].mean()),
        })
    return pd.DataFrame(rows).sort_values("rmse_tau", ascending=True)


def main() -> None:
    args = parse_args()
    per_draw_path = args.results_dir / "per_draw_metrics.csv"
    if not per_draw_path.exists():
        raise SystemExit(f"Missing per-draw metrics file: {per_draw_path}")

    df = pd.read_csv(per_draw_path)
    if df.empty:
        raise SystemExit("Per-draw metrics file is empty.")

    summary = aggregate_with_ci(df)

    summary_path = args.results_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
