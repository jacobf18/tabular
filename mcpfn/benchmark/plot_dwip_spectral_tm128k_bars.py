#!/usr/bin/env python3
"""One consistent bar-plot style for CPS, Prop99 and Kansas, with MatrixFormer
(the dwip_spectral_anchor_mix25_teacher_mean_huber1_128k checkpoint) added.

Every dataset uses the same 12 methods and the same MatFormer checkpoint
(dwip-v2-2). Each draw's metric is min-max normalized across the 12 methods
(0 = best, 1 = worst in that draw), averaged over the draws of each scenario,
then over scenarios with equal weight. The 95% CI comes from draw-to-draw
variation, so it means the same thing for all three datasets. Bar styling
comes from CPS/plot_cps.py.

Writes {cps,prop99,kansas}_bars_{tb_rmse,tau_error}.pdf to
figures/dwip_spectral_tm128k/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent
for sub in ("CPS", ""):
    path = str(BENCH_DIR / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

import plot_cps  # noqa: E402
from plot_options import FIGURE_SIZES, PLOT_PARAMS  # noqa: E402

NEW = "dwip-spectral-tm128k"
MATRIXFORMER_LABELS = {
    NEW: "MatrixFormer",
    f"{NEW}-T": "MatrixFormer (transpose)",
    f"{NEW}-avg": "MatrixFormer (avg)",
}
METHODS = [
    "did", "sdid", "sc", "hyperimpute", "knn", "softimpute",
    "dwip-v2-2", "dwip-v2-2-T", "dwip-v2-2-avg",
    *MATRIXFORMER_LABELS,
]
DATASETS = {
    "CPS": [f"CPS/causal_eval/{o}_{t}" for o in ("hours", "log_wage", "urate")
            for t in ("abort_ban", "min_wage", "open_carry")],
    "Prop99": ["prop99/causal_eval/ca_simul", "prop99/causal_eval/ca_stagg",
               "prop99/causal_eval/stocks"],
    "Kansas": ["kansas/causal_eval/kansas_factor_r3_200"],
}
# (per-draw column, file suffix, metric name)
METRICS = [
    ("treated_block_rmse", "tb_rmse", "treated-block RMSE"),
    ("tau_abs_error", "tau_error", "treatment-effect error"),
]
FIGURES_DIR = BENCH_DIR / "figures/dwip_spectral_tm128k"


def normalized_summary(paths: list[str], col: str) -> tuple[pd.DataFrame, int]:
    """Per-draw min-max normalized `col`, averaged over draws, then scenarios.

    Returns (frame with method / norm / norm_se, total draws used). The SE
    combines the per-scenario draw-level SEs for an equal-weight mean.
    """
    means, ses, n_draws = [], [], 0
    for p in paths:
        df = pd.read_csv(BENCH_DIR / p / "results/per_draw_metrics.csv")
        wide = df[df["method"].isin(METHODS)].pivot_table(
            index="draw_name", columns="method", values=col)
        missing = sorted(set(METHODS) - set(wide.columns))
        if missing:
            raise SystemExit(f"{p} is missing {missing}")
        wide = wide[METHODS].dropna()
        lo, hi = wide.min(axis=1), wide.max(axis=1)
        norm = wide.sub(lo, axis=0).div(hi - lo, axis=0)[hi > lo]
        means.append(norm.mean())
        ses.append(norm.std(ddof=1) / np.sqrt(len(norm)))
        n_draws += len(norm)
    mean = pd.concat(means, axis=1).mean(axis=1)
    se = np.sqrt((pd.concat(ses, axis=1) ** 2).sum(axis=1)) / len(paths)
    return pd.DataFrame({"method": mean.index, "norm": mean.values, "norm_se": se.values}), n_draws


def plot_bars(df: pd.DataFrame, ylabel: str, title: str, out: Path) -> None:
    df = df.sort_values("norm").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["standard"])
    plot_cps.bar_with_errors(ax, df, "norm", "norm_se", ylabel=ylabel)
    top = float(np.nanmax(df["norm"].to_numpy() + 1.96 * df["norm_se"].to_numpy())) * 1.05
    ax.set_ylim(0.0, min(top, 1.05))
    ax.set_title(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(out, **PLOT_PARAMS)
    plt.close(fig)


def main() -> None:
    plot_cps.TABIMPUTE_KEEP.update(MATRIXFORMER_LABELS)
    plot_cps.EXTRA_LABELS.update(MATRIXFORMER_LABELS)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for name, paths in DATASETS.items():
        for col, suffix, metric in METRICS:
            summary, n_draws = normalized_summary(paths, col)
            scen = f"{len(paths)} scenario{'s' if len(paths) > 1 else ''}"
            out = FIGURES_DIR / f"{name.lower()}_bars_{suffix}.pdf"
            plot_bars(summary, ylabel=f"Normalized {metric} (95% CI)",
                      title=f"{name} ({scen}, {n_draws:,} draws)", out=out)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
