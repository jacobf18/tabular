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
SECONDARY_HIGHLIGHT = "#D55E00"
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
    p.add_argument("--always-include", nargs="*", default=[],
                   help="Method names to include even if they are outside top-N.")
    p.add_argument("--highlight-methods", nargs="*", default=[OUT_METHOD],
                   help="Method names to highlight in the bars/tick labels.")
    p.add_argument("--formats", nargs="*", default=["png"],
                   choices=["png", "pdf", "svg"],
                   help="Output formats to write for each figure.")
    p.add_argument("--limit-draws-per-method", type=int, default=None,
                   help="Use at most this many draws per method, ordered by draw_id/draw_name.")
    p.add_argument("--include-observed-panel", action="store_true",
                   help="Also plot observed-block RMSE. Usually zero because observed cells are preserved.")
    p.add_argument("--out-subdir", default="xrfm_comparison")
    return p.parse_args()


def label_for(method: str) -> str:
    extras = {"did": "DiD", "sdid": "SDiD", "sc": "SC",
              "augsynth": "AugSynth", "dwip-v2-2": "MatFormer",
              "xrfm": "xRFM",
              "boost_matformer_dwip_v2_t002_a0p1": "MatFormer+xRFM a0.1",
              "boost_matformer_dwip_v2_t002_a0p05": "MatFormer+xRFM a0.05",
              "direct_pure_rabel_rank10": "RaBEL rank10",
              "direct_pure_rabel_causal_panel": "RaBEL causal-panel",
              "direct_pure_rabel_causal_panel_v2": "RaBEL causal-panel v2",
              "random_rank_sigma05_20k": "RaBEL random-rank",
              "mcar_to_block_10k": "RaBEL MCAR->block",
              "mcar_blockmix_10k": "RaBEL MCAR/block",
              "blockmix25_10k": "RaBEL blockmix25",
              "soup_blockmix_block_mcar": "RaBEL soup"}
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


def limit_draws_per_method(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None:
        return df
    sort_cols = [c for c in ["method", "draw_id", "draw_name"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    return df.groupby("method", group_keys=False).head(limit).reset_index(drop=True)


def filter_methods(summary: pd.DataFrame, exclude_prefixes: list[str],
                   top: int, always_include: list[str]) -> pd.DataFrame:
    if exclude_prefixes:
        mask = ~summary["method"].apply(
            lambda m: any(m.startswith(p) for p in exclude_prefixes)
        )
        summary = summary[mask]

    summary = summary.sort_values("rmse_tau", ascending=True).reset_index(drop=True)
    keep = summary.head(top).copy()
    for method in [OUT_METHOD, *always_include]:
        if method in summary["method"].values and method not in keep["method"].values:
            keep = pd.concat(
                [keep, summary[summary["method"] == method]],
                ignore_index=True,
            )
    return keep.drop_duplicates("method")


def plot_one(
    summary: pd.DataFrame,
    title: str,
    out_path: Path,
    *,
    highlight_methods: set[str],
    formats: list[str],
    include_observed_panel: bool,
) -> None:
    summary = summary.sort_values("rmse_tau", ascending=True).reset_index(drop=True)
    methods = summary["method"].tolist()
    labels = [label_for(m) for m in methods]
    colors = [
        HIGHLIGHT if m == OUT_METHOD
        else SECONDARY_HIGHLIGHT if m in highlight_methods
        else NEUTRAL
        for m in methods
    ]
    metrics_to_plot = [
        ("rmse_tau", r"RMSE($\hat\tau$)"),
        ("mae_tau", r"MAE($\hat\tau$)"),
        ("treated_block_rmse", "Treated-block RMSE"),
    ]
    if include_observed_panel:
        metrics_to_plot.append(("observed_block_rmse", "Observed-block RMSE"))

    if len(metrics_to_plot) == 3:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, (col, ylabel) in zip(np.ravel(axes), metrics_to_plot):
        if col not in summary.columns:
            ax.set_visible(False)
            continue
        vals = summary[col].to_numpy()
        ax.bar(range(len(methods)), vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        for i, lab in enumerate(ax.get_xticklabels()):
            if methods[i] == OUT_METHOD or methods[i] in highlight_methods:
                lab.set_fontweight("bold")
                lab.set_color(HIGHLIGHT if methods[i] == OUT_METHOD else SECONDARY_HIGHLIGHT)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_path.with_suffix(f".{fmt}"), dpi=150, bbox_inches="tight")
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
            df = limit_draws_per_method(df, args.limit_draws_per_method)
            summary = aggregate(df)
            summary = filter_methods(
                summary,
                args.exclude_prefixes,
                args.top,
                args.always_include,
            )

            n_xrfm = int((df["method"] == OUT_METHOD).sum())
            draw_note = f"xRFM n_draws={n_xrfm}" if n_xrfm else f"n_methods={summary['method'].nunique()}"
            if args.limit_draws_per_method is not None:
                draw_note += f", capped at {args.limit_draws_per_method}/method"
            title = f"{fam} / {sc.name}   ({draw_note})"
            out_path = family_dir / "figures" / args.out_subdir / f"{sc.name}.png"
            plot_one(
                summary,
                title,
                out_path,
                highlight_methods=set(args.highlight_methods),
                formats=args.formats,
                include_observed_panel=args.include_observed_panel,
            )
            print(f"  -> {out_path.parent / (out_path.stem + '.' + args.formats[0])}")
            rendered += 1

    print(f"\nDone. rendered={rendered}  skipped={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
