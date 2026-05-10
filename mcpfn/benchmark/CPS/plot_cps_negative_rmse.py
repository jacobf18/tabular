#!/usr/bin/env python3
"""CPS placebo: UCI-style barplots for ATT and ITT, plus per-combo entrywise scatter.

Reads per-draw .npy files from CPS/causal_eval/<combo>/draw_XXXX/. No dependency on
per_draw_metrics.csv.

Produces PDFs in CPS/figures/:
  - cps_att_rmse_bars_entrywise.pdf     ATT RMSE (lower is better), 9-combo average
  - cps_itt_rmse_bars_entrywise.pdf     Treated-block RMSE from compute_negative_rmse
  - cps_entrywise_scatter_sdid_vs_dwip_v2_1.pdf   3x3, each cell = one treated entry
  - cps_entrywise_scatter_sdid_vs_dwip_v2_2.pdf   3x3, each cell = one treated entry
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BENCH_DIR = Path(__file__).resolve().parents[1]
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from plot_options import (  # noqa: E402
    FIGURE_SIZES,
    HIGHLIGHT_COLOR,
    NEUTRAL_COLOR,
    PLOT_PARAMS,
)
from utils import compute_negative_rmse  # noqa: E402

CAUSAL_EVAL_DIR = Path(__file__).resolve().parent / "causal_eval"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

METHODS = [
    "did", "sdid", "sc",
    "dwip-v1", "dwip-v2-1", "dwip-v2-2",
    "hyperimpute", "knn", "softimpute",
]

METHOD_LABELS = {
    "did": "DiD",
    "sdid": "SDiD",
    "sc": "SC",
    "dwip-v1": "TabImpute (v1)",
    "dwip-v2-1": "TabImpute (v2-1)",
    "dwip-v2-2": "TabImpute (v2-2)",
    "hyperimpute": "HyperImpute",
    "knn": "K-Nearest Neighbors",
    "softimpute": "SoftImpute",
}

DWIP_SCATTER_METHODS = ["dwip-v2-1", "dwip-v2-2"]


def is_tabimpute(method: str) -> bool:
    return method.startswith("dwip")


def color_for(method: str) -> str:
    return HIGHLIGHT_COLOR if is_tabimpute(method) else NEUTRAL_COLOR


def list_combos() -> list[str]:
    return sorted(p.name for p in CAUSAL_EVAL_DIR.iterdir() if p.is_dir())


def list_draws(combo_dir: Path) -> list[str]:
    return sorted(
        d.name for d in combo_dir.iterdir()
        if d.is_dir() and d.name.startswith("draw_")
    )


def compute_per_draw_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for combo in list_combos():
        combo_dir = CAUSAL_EVAL_DIR / combo
        for draw in list_draws(combo_dir):
            ddir = combo_dir / draw
            true = np.load(ddir / "true.npy")
            mask = np.load(ddir / "treatment_mask.npy").astype(bool)
            if not mask.any():
                continue
            true_mean = float(true[mask].mean())
            for method in METHODS:
                p = ddir / f"{method}.npy"
                if not p.is_file():
                    continue
                pred = np.load(p)
                tau = true_mean - float(pred[mask].mean())
                tb_rmse = -compute_negative_rmse(true, pred, mask)
                rows.append({
                    "combo": combo,
                    "draw": draw,
                    "method": method,
                    "tau_hat": float(tau),
                    "tb_rmse": float(tb_rmse),
                })
    return pd.DataFrame(rows)


def _draw_bar(ax, summary: pd.DataFrame, ylabel: str) -> None:
    x = np.arange(len(summary))
    colors = [color_for(m) for m in summary["method"]]
    ax.bar(x, summary["mean"], color=colors, edgecolor="#6E6E6E", linewidth=0.6)
    ax.errorbar(
        x, summary["mean"], yerr=1.96 * summary["sem"],
        fmt="none", ecolor="#999999", capsize=3, linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_LABELS[m] for m in summary["method"]],
        rotation=45, ha="right",
    )
    for i, label in enumerate(ax.get_xticklabels()):
        if is_tabimpute(summary["method"].iloc[i]):
            label.set_color(HIGHLIGHT_COLOR)
            label.set_fontweight("bold")
    ax.set_ylabel(ylabel)
    ax.margins(x=0.01)


def plot_att_bar(per_draw: pd.DataFrame, out: Path) -> None:
    # per combo: sqrt(mean(tau_hat^2)) across draws
    combo_rmse = (
        per_draw.groupby(["combo", "method"])["tau_hat"]
        .apply(lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2))))
        .reset_index(name="att_rmse")
    )
    summary = (
        combo_rmse.groupby("method")["att_rmse"]
        .agg(["mean", "sem"]).reset_index()
    )
    summary = summary.sort_values("mean").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["wide"])
    _draw_bar(ax, summary, r"RMSE of $\hat\tau$ (avg over 9 combos)")
    ax.set_title("CPS placebo: ATT error by method (lower is better)")
    fig.tight_layout()
    fig.savefig(out, **PLOT_PARAMS)
    plt.close(fig)


def plot_itt_bar(per_draw: pd.DataFrame, out: Path) -> None:
    # per combo: mean over draws of treated-block RMSE
    combo_mean = (
        per_draw.groupby(["combo", "method"])["tb_rmse"].mean().reset_index()
    )
    summary = (
        combo_mean.groupby("method")["tb_rmse"]
        .agg(["mean", "sem"]).reset_index()
    )
    summary = summary.sort_values("mean").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["wide"])
    _draw_bar(ax, summary, "Treated-block cell RMSE (avg over 9 combos)")
    ax.set_title("CPS placebo: ITT error by method (lower is better)")
    fig.tight_layout()
    fig.savefig(out, **PLOT_PARAMS)
    plt.close(fig)


def plot_entrywise_scatter(out: Path, dwip_method: str) -> None:
    dwip_label = METHOD_LABELS.get(dwip_method, dwip_method)
    combos = list_combos()
    n = len(combos)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 4 * nrow), squeeze=False)

    for idx, combo in enumerate(combos):
        row, col = divmod(idx, ncol)
        ax = axes[row][col]
        combo_dir = CAUSAL_EVAL_DIR / combo
        sdid_errs: list[np.ndarray] = []
        dwip_errs: list[np.ndarray] = []
        for draw in list_draws(combo_dir):
            ddir = combo_dir / draw
            sdid_path = ddir / "sdid.npy"
            dwip_path = ddir / f"{dwip_method}.npy"
            if not (sdid_path.is_file() and dwip_path.is_file()):
                continue
            true = np.load(ddir / "true.npy")
            mask = np.load(ddir / "treatment_mask.npy").astype(bool)
            sdid = np.load(sdid_path)
            dwip = np.load(dwip_path)
            sdid_errs.append((true[mask] - sdid[mask]).ravel())
            dwip_errs.append((true[mask] - dwip[mask]).ravel())

        if not sdid_errs:
            ax.set_visible(False)
            continue

        # TabImpute on x-axis, SDiD on y-axis.
        x = np.concatenate(dwip_errs)
        y = np.concatenate(sdid_errs)
        lim = float(max(np.abs(np.percentile(x, [1, 99])).max(),
                        np.abs(np.percentile(y, [1, 99])).max()))
        ax.scatter(x, y, s=4, alpha=0.08, color=HIGHLIGHT_COLOR, linewidths=0)
        ax.plot([-lim, lim], [-lim, lim], color="black",
                linewidth=0.7, linestyle="--", label="y=x")
        ax.axhline(0, color="#cccccc", linewidth=0.4, zorder=0)
        ax.axvline(0, color="#cccccc", linewidth=0.4, zorder=0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(combo.replace("_", " "), fontsize=11)

        is_bottom = row == nrow - 1
        is_left = col == 0
        if is_bottom:
            ax.set_xlabel(f"{dwip_label} error (true - pred)")
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        if is_left:
            ax.set_ylabel("SDiD error (true - pred)")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    # hide any extra axes
    for idx in range(n, nrow * ncol):
        row, col = divmod(idx, ncol)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"Entrywise treated-block errors: {dwip_label} vs SDiD",
        y=1.00, fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out, **PLOT_PARAMS)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing per-draw metrics from .npy files...")
    per_draw = compute_per_draw_metrics()
    n_rows = len(per_draw)
    n_combos = per_draw["combo"].nunique()
    n_methods = per_draw["method"].nunique()
    print(f"  rows={n_rows} combos={n_combos} methods={n_methods}")

    print("Plotting ATT barplot...")
    plot_att_bar(per_draw, FIGURES_DIR / "cps_att_rmse_bars_entrywise.pdf")

    print("Plotting ITT barplot...")
    plot_itt_bar(per_draw, FIGURES_DIR / "cps_itt_rmse_bars_entrywise.pdf")

    for method in DWIP_SCATTER_METHODS:
        slug = method.replace("-", "_")
        print(f"Plotting 3x3 entrywise scatter (sdid vs {method})...")
        plot_entrywise_scatter(
            FIGURES_DIR / f"cps_entrywise_scatter_sdid_vs_{slug}.pdf",
            method,
        )

    print(f"Wrote figures to {FIGURES_DIR}")
    for p in sorted(FIGURES_DIR.glob("cps_*entrywise*.pdf")):
        print(f"  {p.name}")
    for p in sorted(FIGURES_DIR.glob("cps_entrywise_scatter*.pdf")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
