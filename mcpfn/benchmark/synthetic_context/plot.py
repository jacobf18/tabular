"""Plot synthetic-context sweep results in the UCI barplot style.

Reads results/per_draw_metrics.csv produced by run_eval.py and writes:
  - figures/synthetic_context_rmse_overall.pdf
      Single overall barplot, methods sorted by mean RMSE (best first).
      Same style as uci/plot_uci_negative_rmse.py:uci_negative_rmse_overall.pdf.
  - figures/synthetic_context_rmse_panel.pdf
      Panel: Overall + one subplot per α-setting. Same style as UCI's
      Overall+MCAR+MAR+MNAR row.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BENCH_DIR = Path(__file__).resolve().parents[1]
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from plot_options import (  # noqa: E402
    BARPLOT_STYLE, FIGURE_SIZES, HIGHLIGHT_COLOR, NEUTRAL_COLOR, PLOT_PARAMS,
)

matplotlib.rcParams["font.family"] = "serif"

# Method whitelist (order is just internal). context_only is intentionally
# dropped — it's a separate question, not part of the boost-vs-MatFormer story.
METHODS = [
    "matformer",
    "xrfm",
    "acy_J5",
    "acy_cv",
    "acy_J17",
    "boost_no_context",
    "boost_row_context",
    "boost_row_col_context",
    "hybrid",
    "hybrid_v2",
]
METHOD_PRETTY = {
    "matformer": "MatFormer",
    "xrfm": "xRFM",
    "acy_J5":  "ACY (J=5, paper default)",
    "acy_cv":  "ACY (J cross-validated)",
    "acy_J17": "ACY (J=17, oracle basis)",
    "boost_no_context": "MatFormer + xRFM",
    "boost_row_context": "MatFormer + xRFM (+ $x_r$)",
    "boost_row_col_context": "MatFormer + xRFM (+ $x_r, z_c$)",
    "hybrid":    "Variant A: ACY-CV → Boost(residual)",
    "hybrid_v2": "Variant B: Boost on each projected piece",
}
METHOD_COLORS = {
    "MatFormer":                       NEUTRAL_COLOR,
    "xRFM":                            "#9E9E9E",
    "ACY (J=5, paper default)":        "#FFB370",
    "ACY (J cross-validated)":         "#FF9F47",
    "ACY (J=17, oracle basis)":        "#FF8F2B",
    "MatFormer + xRFM":                HIGHLIGHT_COLOR,
    "MatFormer + xRFM (+ $x_r$)":      "#5C9CE0",
    "MatFormer + xRFM (+ $x_r, z_c$)": "#7FB6E8",
    "Variant A: ACY-CV → Boost(residual)":      "#3D5A80",
    "Variant B: Boost on each projected piece": "#7E2954",
}

SETTING_ORDER = ["pure_interaction", "row_only", "col_only",
                 "context_heavy", "balanced", "low_rank_only"]
SETTING_PRETTY = {
    "pure_interaction": "Pure interaction ($\\alpha_1=1$)",
    "row_only":         "Row only ($\\alpha_2=1$)",
    "col_only":         "Col only ($\\alpha_3=1$)",
    "context_heavy":    "Context heavy",
    "balanced":         "Balanced",
    "low_rank_only":    "Low-rank only ($\\alpha_4=1$)",
}


def load_long(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df = df[df["method"].isin(METHODS)].copy()
    df["method_pretty"] = df["method"].map(METHOD_PRETTY)
    return df


def _bar(ax, data: pd.DataFrame, sorted_methods: list[str], ylabel: str,
         ymin: float | None = None) -> None:
    sns.barplot(
        data=data,
        x="method_pretty",
        y="rmse_missing",
        hue="method_pretty",
        order=sorted_methods,
        palette=METHOD_COLORS,
        legend=False,
        ax=ax,
        **BARPLOT_STYLE,
    )
    ax.set_xticks(range(len(sorted_methods)))
    ax.set_xticklabels(sorted_methods, rotation=45, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)

    means = data.groupby("method_pretty")["rmse_missing"].mean()
    means = means.reindex(sorted_methods)
    ymax_data = float(means.max())
    ymin_data = float(means.min())
    spread = max(ymax_data - ymin_data, 1e-3)
    if ymin is None:
        # Auto: leave the smallest bar visibly above the floor by ~1.5 full
        # bar-to-bar spreads for breathing room.
        ymin = max(ymin_data - 1.5 * spread, 0.0)
    # Ceiling: push well above the tallest bar so the panel has clear headroom
    # rather than the largest bar bumping against the top edge.
    ymax = ymax_data + 1.0 * (ymax_data - ymin)
    ax.set_ylim(ymin, ymax)

    # Bold/colored tick labels for "ours" methods, matching UCI plot style.
    for i, label in enumerate(ax.get_xticklabels()):
        m = sorted_methods[i]
        if m.startswith("MatFormer + xRFM"):
            label.set_color(HIGHLIGHT_COLOR)
            label.set_fontsize(label.get_fontsize() * 1.05)


def plot_overall(df: pd.DataFrame, out: Path) -> None:
    means = df.groupby("method_pretty")["rmse_missing"].mean()
    sorted_methods = means.sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["standard"])
    # Hardcoded floor for the overall plot per user request — keeps the
    # uninformative 0..0.2 range out of view.
    _bar(ax, df, sorted_methods, ylabel="Missing-block RMSE", ymin=0.20)
    fig.tight_layout()
    fig.savefig(out, **PLOT_PARAMS)
    plt.close(fig)


def plot_panel(df: pd.DataFrame, out: Path) -> None:
    settings = [s for s in SETTING_ORDER if s in df["setting"].unique()]
    n_panels = 1 + len(settings)  # Overall + per-setting

    w_std, h_std = FIGURE_SIZES["standard"]
    cols = min(4, n_panels)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * w_std, rows * h_std),
                             squeeze=False)

    panels: list[tuple[str, pd.DataFrame, float | None]] = [
        ("Overall", df, 0.20),
    ]
    for s in settings:
        panels.append((SETTING_PRETTY.get(s, s), df[df["setting"] == s], None))

    for k, (title, sub, ymin) in enumerate(panels):
        ax = axes[k // cols][k % cols]
        means = sub.groupby("method_pretty")["rmse_missing"].mean()
        sorted_methods = means.sort_values().index.tolist()
        _bar(ax, sub, sorted_methods,
             ylabel="Missing-block RMSE" if k % cols == 0 else "",
             ymin=ymin)
        ax.set_title(title, fontsize=14)

    for k in range(n_panels, rows * cols):
        axes[k // cols][k % cols].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, **PLOT_PARAMS)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", type=Path,
        default=Path("/root/tabular/mcpfn/benchmark/synthetic_context/"
                     "results/per_draw_metrics.csv"),
    )
    ap.add_argument(
        "--figures-dir", type=Path,
        default=Path("/root/tabular/mcpfn/benchmark/synthetic_context/figures"),
    )
    args = ap.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    df = load_long(args.csv)

    print("Per-method mean RMSE (across all settings × seeds):")
    print(df.groupby("method_pretty")["rmse_missing"].mean().sort_values()
          .round(4).to_string())

    plot_overall(df, args.figures_dir / "synthetic_context_rmse_overall.pdf")
    plot_panel(df,   args.figures_dir / "synthetic_context_rmse_panel.pdf")
    print(f"\nWrote figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
