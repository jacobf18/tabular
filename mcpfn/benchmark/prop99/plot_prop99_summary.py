#!/usr/bin/env python3
"""Prop99 cross-subset summary figure (ca_simul + ca_stagg + stocks).

Mirrors CPS/plot_cps_summary.py. Reads each prop99 subset's
`<subset>/results/per_draw_metrics.csv`, filters to the intersection method set
(shared across CPS/kansas/prop99), and emits a two-panel bar chart per metric:

    Left panel  : mean rank across the 3 prop99 subsets (1 = best, lower better)
    Right panel : Imputation Accuracy = 1 - min-max-normalized RMSE,
                  averaged across the 3 subsets. Normalization is within-subset
                  across methods, matching Feitelberg et al. (2026).

One PDF per metric: `prop99_summary_tau_rmse.pdf`, `prop99_summary_tb_rmse.pdf`.

The `smoke` subset is explicitly skipped.
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

from plot_options import CAUSAL_INTERSECTION_METHODS, PLOT_PARAMS  # noqa: E402
from plot_prop99 import (  # noqa: E402
    EXCLUDE_METHODS,
    aggregate,
    color_for,
    is_other_tabimpute_variant,
    prettify,
)

# prop99's `smoke` is a test-only 5-draw tree; never include it.
SKIP_SUBSETS = {"smoke"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--causal-eval-dir",
        type=Path,
        default=THIS_DIR / "causal_eval",
        help="Root containing one subdirectory per prop99 subset (default: prop99/causal_eval).",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=THIS_DIR / "figures",
        help="Where to write summary PDFs (default: prop99/figures/).",
    )
    return p.parse_args()


def load_subsets(root: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in SKIP_SUBSETS:
            continue
        csv = d / "results" / "per_draw_metrics.csv"
        if not csv.exists():
            continue
        s = aggregate(pd.read_csv(csv))
        s = s[~s["method"].isin(EXCLUDE_METHODS)]
        s = s[~s["method"].apply(is_other_tabimpute_variant)]
        s = s[s["method"].isin(CAUSAL_INTERSECTION_METHODS)].reset_index(drop=True)
        out[d.name] = s
    return out


def build_summary(subsets: dict[str, pd.DataFrame], value_col: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for name, s in subsets.items():
        g = s[["method", value_col]].copy()
        g["subset"] = name
        g["rank"] = g[value_col].rank(method="average", ascending=True)
        lo = g[value_col].min()
        hi = g[value_col].max()
        denom = hi - lo
        if denom == 0 or not np.isfinite(denom):
            g["imp_acc"] = np.nan
        else:
            g["imp_acc"] = 1.0 - (g[value_col] - lo) / denom
        rows.append(g)
    long = pd.concat(rows, ignore_index=True)

    n_sub = long["subset"].nunique()
    coverage = long.groupby("method")["subset"].nunique()
    keep = coverage[coverage == n_sub].index
    dropped = sorted(set(coverage.index) - set(keep))
    if dropped:
        print(f"  dropping methods missing from some subsets: {dropped}")
    long = long[long["method"].isin(keep)]

    def se(x: pd.Series) -> float:
        x = x.dropna()
        return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")

    summary = long.groupby("method").agg(
        mean_rank=("rank", "mean"),
        rank_se=("rank", se),
        mean_imp_acc=("imp_acc", "mean"),
        imp_acc_se=("imp_acc", se),
        n_subsets=("subset", "nunique"),
    ).reset_index()
    return summary


def plot_summary(
    summary: pd.DataFrame,
    out_path: Path,
    suptitle: str,
) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: mean rank, sorted ascending.
    dfL = summary.sort_values("mean_rank").reset_index(drop=True)
    xL = np.arange(len(dfL))
    colorsL = [color_for(m) for m in dfL["method"]]
    labelsL = [prettify(m) for m in dfL["method"]]
    yerr_rank = 1.96 * dfL["rank_se"].to_numpy()
    ax_left.bar(xL, dfL["mean_rank"], color=colorsL, edgecolor="black", linewidth=0.4)
    ax_left.errorbar(
        xL, dfL["mean_rank"], yerr=yerr_rank,
        fmt="none", ecolor="black", capsize=2.5, linewidth=0.8,
    )
    ax_left.set_xticks(xL)
    ax_left.set_xticklabels(labelsL, rotation=45, ha="right")
    ax_left.set_ylabel("Mean rank")
    ax_left.set_title("Mean rank (lower is better)")
    ax_left.margins(x=0.01)
    ax_left.set_ylim(bottom=0)

    # Right: Imputation Accuracy, sorted descending.
    dfR = summary.sort_values("mean_imp_acc", ascending=False).reset_index(drop=True)
    xR = np.arange(len(dfR))
    colorsR = [color_for(m) for m in dfR["method"]]
    labelsR = [prettify(m) for m in dfR["method"]]
    yerr_acc = 1.96 * dfR["imp_acc_se"].to_numpy()
    ax_right.bar(xR, dfR["mean_imp_acc"], color=colorsR, edgecolor="black", linewidth=0.4)
    ax_right.errorbar(
        xR, dfR["mean_imp_acc"], yerr=yerr_acc,
        fmt="none", ecolor="black", capsize=2.5, linewidth=0.8,
    )
    ax_right.set_xticks(xR)
    ax_right.set_xticklabels(labelsR, rotation=45, ha="right")
    ax_right.set_ylabel("1 - Normalized RMSE")
    ax_right.set_title("Imputation Accuracy (higher is better)")
    ax_right.set_ylim(0.0, 1.05)
    ax_right.margins(x=0.01)

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, **PLOT_PARAMS)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    subsets = load_subsets(args.causal_eval_dir)
    if not subsets:
        raise SystemExit(f"No prop99 subsets with results/per_draw_metrics.csv under {args.causal_eval_dir}")
    print(f"Loaded {len(subsets)} subsets: {sorted(subsets)}")

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("rmse",    "prop99 summary (by RMSE of ATT estimate)",       "prop99_summary_tau_rmse.pdf"),
        ("tb_rmse", "prop99 summary (by treated-block cell RMSE)",    "prop99_summary_tb_rmse.pdf"),
    ]
    for value_col, title, fname in configs:
        print(f"\n== {fname} (metric: {value_col}) ==")
        summary = build_summary(subsets, value_col)
        out = args.figures_dir / fname
        plot_summary(summary, out, suptitle=title)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
