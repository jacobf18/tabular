#!/usr/bin/env python
"""
Make the figures and tables of the TabImpute paper from the benchmark results.

Reads results/<experiment>.csv (written by `run_benchmark.py evaluate`) and writes
PDFs and LaTeX tables to figures/.

Examples:
  python make_figures.py list          # what can be made and what each one needs
  python make_figures.py all           # everything whose inputs exist
  python make_figures.py fig1 table3   # specific items
  python make_figures.py fig4a --compare tabimpute hyperimpute
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_DIR))

from missbench.plots import figures as F  # noqa: E402
from missbench.plots import tables as T  # noqa: E402
from missbench.plots.context import Context, MissingInput  # noqa: E402

# name -> (function, what it needs). "openml" means results/openml.csv, and so on.
ITEMS = {
    # Paper figures
    "fig1": (F.fig_rmse_highlights, "openml (+ TabImpute-P for censor_mnar, self_mnar)"),
    "fig3": (F.fig_mask_examples, "nothing"),
    "fig4a": (F.fig_scatter_vs_missforest, "openml: tabimpute, missforest"),
    "fig4b": (F.fig_runtime_per_entry, "openml (+ tabimpute_cpu for the CPU bar)"),
    "fig5": (F.fig_rmse_all_patterns, "openml (+ TabImpute-P models)"),
    "fig6": (F.fig_self_mnar_notmiwae, "openml: notmiwae, notmiwae_self_mnar (+ tabimpute_self_mnar)"),
    "fig7": (F.fig_mcar_sweep, "openml_mcar_sweep: tabimpute, hyperimpute, missforest, gain"),
    # Paper tables
    "table1": (T.table_rmse_main, "openml"),
    "table2": (T.table_specialists_compact, "openml: TabImpute-P models"),
    "table3": (T.table_rmse_full, "openml"),
    "table4": (T.table_r2, "openml"),
    "table5": (T.table_specialists, "openml: TabImpute-P models"),
    "table7": (T.table_nonlinear, "openml: tabimpute, tabimpute_nonlinear"),
    "table8": (T.table_pattern_parameters, "nothing"),
    "table9": (T.table_datasets, "nothing"),
    # Extra figures and tables (not in the paper)
    "runtime": (F.fig_runtime_scaling, "runtime"),
    "uci": (F.fig_uci_rmse, "uci"),
    "uci_table": (T.table_uci_rmse, "uci"),
    "r2_bars": (F.metric_bars("r2", "$R^2$", ascending=False), "openml"),
    "wasserstein_bars": (F.metric_bars("wasserstein", "Wasserstein distance", ascending=True), "openml"),
    "mae_bars": (F.metric_bars("mae", "MAE", ascending=True), "openml"),
    "win_rate": (F.fig_win_rate, "openml"),
    "win_rate_table": (T.table_win_rate, "openml"),
    "critical_difference": (F.fig_critical_difference, "openml"),
}
PAPER_ITEMS = [k for k in ITEMS if k.startswith(("fig", "table"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("items", nargs="*", default=["list"],
                        help="'list', 'all', 'paper', or item names (see `list`)")
    parser.add_argument("--results-dir", type=Path, default=BENCHMARK_DIR / "results")
    parser.add_argument("--out-dir", type=Path, default=BENCHMARK_DIR / "figures")
    parser.add_argument("--methods", nargs="+",
                        help="baselines to show (default: the paper's methods that have results)")
    parser.add_argument("--compare", nargs=2, default=["tabimpute", "missforest"], metavar=("A", "B"),
                        help="methods for the fig4a scatter (default: tabimpute missforest)")
    args = parser.parse_args()

    if args.items == ["list"]:
        print(f"{'item':<20} {'needs results from':<55} description")
        for name, (fn, needs) in ITEMS.items():
            print(f"{name:<20} {needs:<55} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return

    if args.items == ["all"]:
        names, strict = list(ITEMS), False
    elif args.items == ["paper"]:
        names, strict = PAPER_ITEMS, False
    else:
        names, strict = args.items, True
        unknown = [n for n in names if n not in ITEMS]
        if unknown:
            raise SystemExit(f"Unknown item(s) {unknown}. Run `python make_figures.py list`.")

    ctx = Context(args.results_dir, args.out_dir, methods=args.methods, compare=tuple(args.compare))
    skipped = []
    for name in names:
        fn, _ = ITEMS[name]
        try:
            paths = fn(ctx)
        except MissingInput as e:
            if strict:
                raise SystemExit(f"{name}: {e}")
            skipped.append(f"{name} ({e})")
            continue
        for path in paths:
            print(f"{name}: wrote {path}")
    if skipped:
        print("\nSkipped (inputs missing):\n  " + "\n  ".join(skipped))


if __name__ == "__main__":
    main()
