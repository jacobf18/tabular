#!/usr/bin/env python3
"""CPS cross-scenario summary figure.

Loads every `<scenario>/results/per_draw_metrics.csv` under the CPS causal_eval
root, filters to the allowed method set (matching plot_cps.py), and produces a
single bar chart per metric: mean within-scenario min-max-normalized RMSE
across the 9 scenarios (lower is better, [0, 1] per scenario). Min-max
normalization is within-scenario across the kept methods.

One PDF per metric: `cps_summary_tau_rmse.pdf` and
`cps_summary_tb_rmse.pdf`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BENCH_DIR = Path(__file__).resolve().parents[1]
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from plot_options import CAUSAL_INTERSECTION_METHODS, FIGURE_SIZES, PLOT_PARAMS  # noqa: E402
from plot_cps import (  # noqa: E402
    EXCLUDE_METHODS,
    aggregate,
    bar_with_errors,
    is_other_tabimpute_variant,
)

# MatFormer + xRFM residual variants: omitted from the summary figures.
XRFM_RESIDUAL_METHODS = {
    "boost_matformer_dwip_v2_t002_a0p1",
    "boost_matformer_dwip_v2_t002_a0p05",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--causal-eval-dir",
        type=Path,
        default=THIS_DIR / "causal_eval",
        help="Root containing one subdirectory per scenario (default: CPS/causal_eval).",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=THIS_DIR / "figures",
        help="Where to write summary PDFs (default: CPS/figures/).",
    )
    return p.parse_args()


def load_scenarios(root: Path) -> dict[str, pd.DataFrame]:
    """Return {scenario_name: aggregated summary} for every scenario with a CSV."""
    out: dict[str, pd.DataFrame] = {}
    for d in sorted(root.iterdir()):
        csv = d / "results" / "per_draw_metrics.csv"
        if not csv.exists():
            continue
        s = aggregate(pd.read_csv(csv))
        s = s[~s["method"].isin(EXCLUDE_METHODS)]
        s = s[~s["method"].isin(XRFM_RESIDUAL_METHODS)]
        s = s[~s["method"].apply(is_other_tabimpute_variant)]
        s = s[s["method"].isin(CAUSAL_INTERSECTION_METHODS)].reset_index(drop=True)
        out[d.name] = s
    return out


def build_summary(scenarios: dict[str, pd.DataFrame], value_col: str) -> pd.DataFrame:
    """Per-method mean rank and mean Imputation Accuracy across scenarios.

    Only methods present in *every* scenario are included, so the averages
    compare like-for-like.
    """
    per_scenario_rows: list[pd.DataFrame] = []
    for name, s in scenarios.items():
        g = s[["method", value_col]].copy()
        g["scenario"] = name
        lo = g[value_col].min()
        hi = g[value_col].max()
        denom = hi - lo
        if denom == 0 or not np.isfinite(denom):
            g["norm_rmse"] = np.nan
        else:
            g["norm_rmse"] = (g[value_col] - lo) / denom
        per_scenario_rows.append(g)

    long = pd.concat(per_scenario_rows, ignore_index=True)
    n_sc = long["scenario"].nunique()
    coverage = long.groupby("method")["scenario"].nunique()
    keep = coverage[coverage == n_sc].index
    dropped = sorted(set(coverage.index) - set(keep))
    if dropped:
        print(f"  dropping methods missing from some scenarios: {dropped}")
    long = long[long["method"].isin(keep)]

    def se(x: pd.Series) -> float:
        x = x.dropna()
        return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")

    summary = long.groupby("method").agg(
        mean_norm_rmse=("norm_rmse", "mean"),
        norm_rmse_se=("norm_rmse", se),
        n_scenarios=("scenario", "nunique"),
    ).reset_index()
    return summary


def plot_summary(out_path: Path, summary: pd.DataFrame) -> None:
    df = summary.sort_values("mean_norm_rmse", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["standard"])
    bar_with_errors(ax, df, "mean_norm_rmse", "norm_rmse_se",
                    ylabel="Normalized RMSE (95% CI)")
    top = float(np.nanmax(df["mean_norm_rmse"].to_numpy()
                          + 1.96 * df["norm_rmse_se"].to_numpy())) * 1.05
    ax.set_ylim(0.0, top)
    fig.tight_layout()
    fig.savefig(out_path, **PLOT_PARAMS)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scenarios = load_scenarios(args.causal_eval_dir)
    if not scenarios:
        raise SystemExit(f"No scenarios with results/per_draw_metrics.csv under {args.causal_eval_dir}")
    print(f"Loaded {len(scenarios)} scenarios: {sorted(scenarios)}")

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("rmse",    "cps_summary_tau_rmse.pdf"),
        ("tb_rmse", "cps_summary_tb_rmse.pdf"),
    ]
    for value_col, fname in configs:
        print(f"\n== {fname} (metric: {value_col}) ==")
        summary = build_summary(scenarios, value_col)
        out = args.figures_dir / fname
        plot_summary(out, summary)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
