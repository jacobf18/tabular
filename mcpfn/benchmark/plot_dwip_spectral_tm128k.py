#!/usr/bin/env python3
"""Figures for the dwip_spectral_anchor_mix25_teacher_mean_huber1_128k checkpoint
on the CPS / Kansas / Prop99 causal benchmarks.

Reads each group's results/per_draw_metrics.csv and writes, to
figures/dwip_spectral_tm128k/ (PDF + PNG):
    ratio_vs_dwip_v2_2      -- per-group RMSE and |tau| error relative to dwip-v2-2
    {cps,kansas,prop99}_tb_rmse -- per-group treated-block RMSE, mean +/- 95% CI

The -median variants are omitted: they are within ~0.3% of the mean variants.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent

DATASETS: dict[str, list[str]] = {
    "CPS": [f"CPS/causal_eval/{o}_{t}" for o in ("hours", "log_wage", "urate")
            for t in ("abort_ban", "min_wage", "open_carry")],
    "Kansas": ["kansas/causal_eval/kansas_factor_r3_200"],
    "Prop99": ["prop99/causal_eval/ca_simul", "prop99/causal_eval/ca_stagg",
               "prop99/causal_eval/stocks"],
}

NEW = "dwip-spectral-tm128k"
NEW_METHODS = [NEW, f"{NEW}-T", f"{NEW}-avg"]
BASELINES = ["dwip-v2-2", "dwip-v2-2-avg", "dwip-v2-1", "direct_pure_rabel_causal_panel_v2",
             "sdid", "sc", "did", "softimpute"]
LABELS = {
    NEW: "MatrixFormer",
    f"{NEW}-T": "MatrixFormer (transpose)",
    f"{NEW}-avg": "MatrixFormer (avg)",
    "dwip-v2-2": "MatFormer (dwip-v2-2)",
    "dwip-v2-2-avg": "MatFormer (dwip-v2-2 avg)",
    "dwip-v2-1": "dwip-v2-1",
    "direct_pure_rabel_causal_panel_v2": "RaBEL causal-panel v2",
    "sdid": "SDiD",
    "sc": "SC",
    "did": "DiD",
    "softimpute": "SoftImpute",
}

# Validated categorical slots (all-pairs, light surface); aqua is <3:1 contrast,
# so every series also carries a distinct marker and a legend entry.
BLUE, ORANGE, AQUA = "#2A6FBB", "#eb6834", "#1baf7a"
MUTED = "#898781"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def group_label(path: str) -> str:
    name = path.split("/")[-1]
    if name.startswith("kansas"):
        return "factor r3"
    for outcome in ("hours", "log_wage", "urate"):
        if name.startswith(outcome + "_"):
            return f"{outcome.replace('_', ' ')} / {name[len(outcome) + 1:].replace('_', ' ')}"
    return name


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(BENCH_DIR / path / "results/per_draw_metrics.csv")
    df = df[df["method"].isin(NEW_METHODS + BASELINES)]

    def se(x: pd.Series) -> float:
        return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")

    return df.groupby("method").agg(
        n=("treated_block_rmse", "size"),
        rmse=("treated_block_rmse", "mean"),
        rmse_se=("treated_block_rmse", se),
        tau=("tau_abs_error", "mean"),
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def plot_ratio(figures_dir: Path) -> None:
    rows = [(ds, p, load(p)) for ds, paths in DATASETS.items() for p in paths]
    series = [
        (NEW, "MatrixFormer", BLUE, "o"),
        (f"{NEW}-avg", "MatrixFormer (avg)", BLUE, "D"),
        ("sdid", "SDiD", ORANGE, "s"),
        ("softimpute", "SoftImpute", AQUA, "^"),
    ]
    lo, hi = 0.3, 2.5
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.2), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, col, title in ((axes[0], "rmse", "Treated-block RMSE"),
                           (axes[1], "tau", r"|$\hat\tau$| error")):
        style_axis(ax)
        ax.axvline(1.0, color=INK_2, linewidth=0.9, zorder=1)
        for i, (ds, _, s) in enumerate(rows):
            if i and rows[i - 1][0] != ds:
                ax.axhline(y[i] + 0.5, color="#c3c2b7", linewidth=0.6)
        offsets = np.linspace(-0.27, 0.27, len(series))
        clipped: dict[float, list[tuple[float, float, float]]] = {}
        for (method, label, color, marker), off in zip(series, offsets):
            for yi, (_, _, s) in zip(y, rows):
                if method not in s.index:
                    continue
                r = s.loc[method, col] / s.loc["dwip-v2-2", col]
                xv = min(max(r, lo), hi)
                ax.scatter(xv, yi + off, s=38, color=color, marker=marker,
                           edgecolors="white", linewidths=0.8, zorder=3,
                           label=label if yi == y[0] else None)
                if r > hi or r < lo:
                    clipped.setdefault(yi, []).append((off, xv, r))
        # Off-scale values: label at the edge, spreading labels that share a row.
        for yi, pts in clipped.items():
            for k, (off, xv, r) in enumerate(sorted(pts)):
                nudge = (k - (len(pts) - 1) / 2) * 8
                ax.annotate(f"{r:.1f}×", (xv, yi + off), xytext=(-7 if r > hi else 7, nudge),
                            textcoords="offset points", va="center",
                            ha="right" if r > hi else "left", fontsize=7.5, color=INK_2)
        ax.set_xscale("log")
        ax.set_xlim(lo * 0.93, hi * 1.07)
        ticks = [0.3, 0.5, 0.7, 1.0, 1.5, 2.5]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}×" for t in ticks])
        ax.minorticks_off()
        ax.set_title(title, fontsize=12, color=INK, loc="left")
        ax.set_xlabel("ratio to MatFormer (dwip-v2-2), log scale — left of 1 is better")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{ds} · {group_label(p)}" for ds, p, _ in rows])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"ratio_vs_dwip_v2_2.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_group(ax: plt.Axes, path: str) -> None:
    s = load(path).sort_values("rmse")
    style_axis(ax)
    y = np.arange(len(s))[::-1]
    vals = s["rmse"].to_numpy()
    ci = 1.96 * s["rmse_se"].to_numpy()
    xmax = 2.5 * float(np.median(vals))
    xmin = float(np.min(vals - ci))
    pad = 0.04 * (min(xmax, float(np.max(vals + ci))) - xmin)
    right = min(xmax, float(np.max(vals + ci))) + pad
    for yi, (method, v, c) in zip(y, zip(s.index, vals, ci)):
        color = BLUE if method in NEW_METHODS else MUTED
        if v > right:
            ax.scatter(right, yi, marker=">", s=34, color=color, zorder=3)
            ax.annotate(f"{v:.3g}", (right, yi), xytext=(-8, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=7.5, color=INK_2)
            continue
        ax.errorbar(v, yi, xerr=c, fmt="none", ecolor=color, elinewidth=1.2, capsize=0, zorder=2)
        ax.scatter(v, yi, s=30, color=color, edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_xlim(xmin - pad, right + pad)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[m] for m in s.index], fontsize=8.5)
    for label, method in zip(ax.get_yticklabels(), s.index):
        if method in NEW_METHODS:
            label.set_color(BLUE)
            label.set_fontweight("bold")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title(f"{group_label(path)}  (n={int(s['n'].max())})", fontsize=10.5,
                 color=INK, loc="left")


def plot_dataset(figures_dir: Path, name: str, paths: list[str], ncols: int) -> None:
    nrows = int(np.ceil(len(paths) / ncols))
    header = 0.6  # inches reserved above the panels for the two-line title
    height = 3.6 * nrows + header
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, height), squeeze=False)
    for ax, path in zip(axes.flat, paths):
        plot_group(ax, path)
    for ax in list(axes.flat)[len(paths):]:
        ax.set_visible(False)
    fig.text(0.01, 1 - 0.05 / height, f"{name}: treated-block RMSE", fontsize=12, color=INK,
             ha="left", va="top")
    fig.text(0.01, 1 - 0.3 / height,
             "mean ± 95% CI · lower is better · off-scale values shown at the edge",
             fontsize=9, color=INK_2, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - header / height))
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"{name.lower()}_tb_rmse.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figures-dir", type=Path, default=BENCH_DIR / "figures/dwip_spectral_tm128k")
    args = p.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_ratio(args.figures_dir)
    plot_dataset(args.figures_dir, "CPS", DATASETS["CPS"], ncols=3)
    plot_dataset(args.figures_dir, "Kansas", DATASETS["Kansas"], ncols=1)
    plot_dataset(args.figures_dir, "Prop99", DATASETS["Prop99"], ncols=3)
    for f in sorted(args.figures_dir.glob("*.pdf")):
        print(f)


if __name__ == "__main__":
    main()
