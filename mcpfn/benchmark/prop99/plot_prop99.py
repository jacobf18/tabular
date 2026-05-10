#!/usr/bin/env python3
"""Plot prop99 benchmark results.

Mirrors kansas/plot_kansas.py. Three bar charts (RMSE of τ̂, bias, treated-block
RMSE) plus a runtime-vs-RMSE scatter, all titled for prop99.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BENCH_DIR = Path(__file__).resolve().parents[1]
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from plot_options import (  # noqa: E402
    CAUSAL_INTERSECTION_METHODS,
    FIGURE_SIZES,
    HIGHLIGHT_COLOR,
    METHOD_NAMES,
    NEUTRAL_COLOR,
    PLOT_PARAMS,
)

# plot_options' sns.set(style="whitegrid") resets font.family to sans-serif;
# restore serif to match the CPS/kansas rendering.
matplotlib.rcParams['font.family'] = 'serif'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, required=True,
                   help="Directory containing per_draw_metrics.csv.")
    p.add_argument("--figures-dir", type=Path,
                   default=Path(__file__).resolve().parent / "figures",
                   help="Where to write PDFs (default: prop99/figures/).")
    p.add_argument("--y-max-rmse", type=float, default=None,
                   help="Optional cap on RMSE-axis bar charts.")
    return p.parse_args()


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("method", dropna=False):
        tau = g["tau_hat"].to_numpy(dtype=float)
        tb_rmse = g["treated_block_rmse"].to_numpy(dtype=float)
        runtime = g["runtime_seconds"].to_numpy(dtype=float)
        n = len(tau)

        bias = float(tau.mean())
        se_bias = float(tau.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

        tau_sq = tau ** 2
        rmse = float(np.sqrt(tau_sq.mean()))
        if n > 1 and rmse > 0:
            se_m = float(tau_sq.std(ddof=1) / np.sqrt(n))
            se_rmse = se_m / (2 * rmse)
        else:
            se_rmse = float("nan")

        tb_mean = float(tb_rmse.mean())
        se_tb = float(tb_rmse.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

        rows.append({
            "method": method,
            "n_draws": n,
            "bias": bias, "bias_se": se_bias,
            "rmse": rmse, "rmse_se": se_rmse,
            "tb_rmse": tb_mean, "tb_rmse_se": se_tb,
            "runtime": float(np.nanmean(runtime)) if n > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


TABIMPUTE_KEEP = {"dwip-v2-1", "dwip-v2-1-T", "dwip-v2-1-avg", "masters_mcar"}
MATFORMER_LABEL = "MatFormer"


def is_tabimpute(method: str) -> bool:
    return method in TABIMPUTE_KEEP


def is_other_tabimpute_variant(method: str) -> bool:
    if method in TABIMPUTE_KEEP:
        return False
    return (method.startswith("dwip_") or method.startswith("dwip-")
            or method.startswith("round2_t2"))


def color_for(method: str) -> str:
    return HIGHLIGHT_COLOR if is_tabimpute(method) else NEUTRAL_COLOR


EXTRA_LABELS = {
    "did": "DiD",
    "sdid": "SDiD",
    "sc": "SC",
    "augsynth": "AugSynth",
    "dwip-v2-1": MATFORMER_LABEL,
    "dwip-v2-1-T": f"{MATFORMER_LABEL} (transpose)",
    "dwip-v2-1-avg": f"{MATFORMER_LABEL} (avg)",
}


def prettify(method: str) -> str:
    return EXTRA_LABELS.get(method, METHOD_NAMES.get(method, method))


def bar_with_errors(ax, df_sorted, value_col, se_col, ylabel, hline=None):
    x = np.arange(len(df_sorted))
    colors = [color_for(m) for m in df_sorted["method"]]
    y = df_sorted[value_col].to_numpy()
    yerr = 1.96 * df_sorted[se_col].to_numpy()
    ax.bar(x, y, color=colors, edgecolor="#6E6E6E", linewidth=0.6)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#999999", capsize=3, linewidth=0.8)
    if hline is not None:
        ax.axhline(hline, color="black", linewidth=0.6, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([prettify(m) for m in df_sorted["method"]], rotation=45, ha="right")
    for i, label in enumerate(ax.get_xticklabels()):
        if is_tabimpute(df_sorted["method"].iloc[i]):
            label.set_color(HIGHLIGHT_COLOR)
            label.set_fontweight("bold")
            label.set_fontsize(label.get_fontsize() * 1.1)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=15)
    ax.margins(x=0.01)


EXCLUDE_METHODS = {"miracle", "hyperimpute_mean", "mice"}


def plot_rmse_tau(summary, out, y_max):
    df = summary.sort_values("rmse", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["standard"])
    bar_with_errors(ax, df, "rmse", "rmse_se",
                    ylabel=r"RMSE of $\hat\tau$ (95% CI)")
    top = float(np.nanmax(df["rmse"].to_numpy()
                          + 1.96 * df["rmse_se"].to_numpy())) * 1.05
    ax.set_ylim(top=top)
    fig.tight_layout()
    fig.savefig(out, **PLOT_PARAMS); plt.close(fig)


def plot_bias_tau(summary, out, y_max):
    df = summary.reindex(
        summary["bias"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["standard"])
    bar_with_errors(ax, df, "bias", "bias_se",
                    ylabel=r"Bias of $\hat\tau$ (95% CI)", hline=0.0)
    reach = float(np.nanmax(np.abs(df["bias"].to_numpy())
                            + 1.96 * df["bias_se"].to_numpy())) * 1.05
    ax.set_ylim(-reach, reach)
    fig.tight_layout()
    fig.savefig(out, **PLOT_PARAMS); plt.close(fig)


def plot_treated_block_rmse(summary, out, y_max):
    df = summary.sort_values("tb_rmse", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["standard"])
    bar_with_errors(ax, df, "tb_rmse", "tb_rmse_se",
                    ylabel="Treated-block cell RMSE (95% CI)")
    top = float(np.nanmax(df["tb_rmse"].to_numpy()
                          + 1.96 * df["tb_rmse_se"].to_numpy())) * 1.05
    ax.set_ylim(top=top)
    fig.tight_layout()
    fig.savefig(out, **PLOT_PARAMS); plt.close(fig)


def plot_runtime_vs_rmse(summary, out, y_max):
    df = summary.dropna(subset=["runtime", "rmse"]).copy()
    df = df[df["runtime"] > 0]
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["standard"])
    for _, r in df.iterrows():
        c = color_for(r["method"])
        ax.scatter(r["runtime"], r["rmse"], color=c, edgecolor="black", s=60, zorder=3)
        ax.annotate(prettify(r["method"]), (r["runtime"], r["rmse"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=8)
    pts = df[["runtime", "rmse", "method"]].to_numpy()
    order = np.argsort(pts[:, 0])
    front_x, front_y = [], []
    best = np.inf
    for i in order:
        rt, rm = pts[i, 0], pts[i, 1]
        if rm < best:
            front_x.append(rt); front_y.append(rm); best = rm
    ax.plot(front_x, front_y, color="black", linestyle="--", linewidth=0.8,
            label="Pareto frontier", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Mean runtime per draw (s, log)")
    ax.set_ylabel(r"RMSE of $\hat\tau$")
    ax.set_title("Prop99 placebo: speed vs accuracy")
    if y_max is not None:
        ax.set_ylim(top=y_max)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, **PLOT_PARAMS); plt.close(fig)


def main() -> None:
    args = parse_args()
    per_draw_path = args.results_dir / "per_draw_metrics.csv"
    if not per_draw_path.exists():
        raise SystemExit(f"Missing {per_draw_path}")

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(per_draw_path)
    summary = aggregate(df)
    summary = summary[~summary["method"].isin(EXCLUDE_METHODS)]
    summary = summary[~summary["method"].apply(is_other_tabimpute_variant)]
    summary = summary[summary["method"].isin(CAUSAL_INTERSECTION_METHODS)].reset_index(drop=True)

    plot_rmse_tau(summary, args.figures_dir / "prop99_rmse_tau_bars.pdf", args.y_max_rmse)
    plot_bias_tau(summary, args.figures_dir / "prop99_bias_tau_bars.pdf", args.y_max_rmse)
    plot_treated_block_rmse(summary, args.figures_dir / "prop99_treated_block_rmse_bars.pdf", args.y_max_rmse)
    plot_runtime_vs_rmse(summary, args.figures_dir / "prop99_runtime_vs_rmse.pdf", args.y_max_rmse)

    print(f"Wrote figures to {args.figures_dir}")
    for p in sorted(args.figures_dir.glob("prop99_*.pdf")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
