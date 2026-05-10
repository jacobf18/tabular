"""
Standalone plotter: xRFM vs other methods on causal-inference benchmarks.

Reads <family>/causal_eval/<scenario>/results/per_draw_metrics.csv and
produces one figure per (family, scenario) showing per-method bar plots
of RMSE(tau), |tau|-MAE, treated-block RMSE, and observed-block RMSE.

Output goes to <family>/figures/xrfm_comparison/<scenario>.png — a new
subdirectory so existing figures are untouched.

Does not import or rely on any existing plot_*.py module beyond pulling
labels from plot_options.METHOD_NAMES (read-only).

Usage
-----
    python plot_xrfm_comparison.py --family CPS
    python plot_xrfm_comparison.py --family CPS kansas prop99
    python plot_xrfm_comparison.py --family CPS --scenarios hours_abort_ban
    python plot_xrfm_comparison.py --family CPS --top 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

try:
    from plot_options import METHOD_NAMES
except Exception:
    METHOD_NAMES = {}

OUT_METHOD = "xrfm"
HIGHLIGHT = "#2A6FBB"
NEUTRAL = "#B8B8B8"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--family", nargs="+", default=["CPS", "kansas", "prop99"])
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--skip-scenarios", nargs="*",
                   default=["smoke", "causal_eval_smoketest"])
    p.add_argument("--top", type=int, default=12,
                   help="Show the top-N methods by RMSE(tau) in each figure "
                        "(xRFM is always shown).")
    p.add_argument("--exclude-prefixes", nargs="*",
                   default=["dwip_", "round2_t2"],
                   help="Method-name prefixes to drop (variant-noise filter).")
    p.add_argument("--out-subdir", default="xrfm_comparison")
    return p.parse_args()


def label_for(method: str) -> str:
    extras = {"did": "DiD", "sdid": "SDiD", "sc": "SC",
              "augsynth": "AugSynth", "dwip-v2-2": "MatFormer",
              "xrfm": "xRFM"}
    if method in extras:
        return extras[method]
    return METHOD_NAMES.get(method, method)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method aggregation matching summarize_cps_causal style."""
    rows: list[dict] = []
    for method, g in df.groupby("method", dropna=False):
        tau = g["tau_hat"].to_numpy(dtype=float)
        n = len(tau)
        bias = float(tau.mean())
        rmse_tau = float(np.sqrt(np.mean(tau ** 2))) if n else float("nan")
        mae_tau = float(g["tau_abs_error"].mean()) if "tau_abs_error" in g else float("nan")
        treat_rmse = float(g["treated_block_rmse"].mean()) if "treated_block_rmse" in g else float("nan")
        obs_rmse = float(g["observed_block_rmse"].mean()) if "observed_block_rmse" in g else float("nan")
        rt = float(g["runtime_seconds"].mean(skipna=True)) if "runtime_seconds" in g else float("nan")
        rows.append({
            "method": method,
            "n_draws": int(n),
            "bias_tau": bias,
            "rmse_tau": rmse_tau,
            "mae_tau": mae_tau,
            "treated_block_rmse": treat_rmse,
            "observed_block_rmse": obs_rmse,
            "runtime_seconds": rt,
        })
    return pd.DataFrame(rows)


def filter_methods(summary: pd.DataFrame, exclude_prefixes: list[str],
                   top: int) -> pd.DataFrame:
    if exclude_prefixes:
        mask = ~summary["method"].apply(
            lambda m: any(m.startswith(p) for p in exclude_prefixes)
        )
        summary = summary[mask]

    summary = summary.sort_values("rmse_tau", ascending=True).reset_index(drop=True)
    keep = summary.head(top).copy()
    if OUT_METHOD in summary["method"].values and OUT_METHOD not in keep["method"].values:
        keep = pd.concat(
            [keep, summary[summary["method"] == OUT_METHOD]],
            ignore_index=True,
        )
    return keep


def plot_one(summary: pd.DataFrame, title: str, out_path: Path) -> None:
    summary = summary.sort_values("rmse_tau", ascending=True).reset_index(drop=True)
    methods = summary["method"].tolist()
    labels = [label_for(m) for m in methods]
    colors = [HIGHLIGHT if m == OUT_METHOD else NEUTRAL for m in methods]
    metrics_to_plot = [
        ("rmse_tau", r"RMSE($\hat\tau$)"),
        ("mae_tau", r"MAE($\hat\tau$)"),
        ("treated_block_rmse", "Treated-block RMSE"),
        ("observed_block_rmse", "Observed-block RMSE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (col, ylabel) in zip(axes.flat, metrics_to_plot):
        if col not in summary.columns:
            ax.set_visible(False)
            continue
        vals = summary[col].to_numpy()
        ax.bar(range(len(methods)), vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        # Highlight xRFM tick label
        for i, lab in enumerate(ax.get_xticklabels()):
            if methods[i] == OUT_METHOD:
                lab.set_fontweight("bold")
                lab.set_color(HIGHLIGHT)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    skip = set(args.skip_scenarios)

    rendered = 0
    skipped = 0

    for fam in args.family:
        family_dir = BENCHMARK_DIR / fam
        causal = family_dir / "causal_eval"
        if not causal.exists():
            print(f"  [skip] {causal} missing", file=sys.stderr)
            continue
        if args.scenarios:
            scenarios = [causal / s for s in args.scenarios if (causal / s).exists()]
        else:
            scenarios = [s for s in sorted(causal.iterdir())
                         if s.is_dir() and s.name not in skip]

        for sc in scenarios:
            csv_path = sc / "results" / "per_draw_metrics.csv"
            if not csv_path.exists():
                print(f"  [skip] no CSV: {csv_path}")
                skipped += 1
                continue

            df = pd.read_csv(csv_path)
            if OUT_METHOD not in df["method"].unique():
                print(f"  [skip] no xrfm rows in {csv_path}")
                skipped += 1
                continue

            summary = aggregate(df)
            summary = filter_methods(summary, args.exclude_prefixes, args.top)

            n_xrfm = int((df["method"] == OUT_METHOD).sum())
            title = f"{fam} / {sc.name}   (xRFM n_draws={n_xrfm})"
            out_path = family_dir / "figures" / args.out_subdir / f"{sc.name}.png"
            plot_one(summary, title, out_path)
            print(f"  -> {out_path}")
            rendered += 1

    print(f"\nDone. rendered={rendered}  skipped={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
