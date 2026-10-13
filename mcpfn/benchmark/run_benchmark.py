#!/usr/bin/env python
"""
Run the TabImpute imputation benchmarks end to end.

Steps (each can be run on its own, or all at once with `run`):
  download     download the datasets
  missingness  sample the missingness masks
  impute       run the imputation methods
  evaluate     score the imputations and write results/<experiment>.csv

Examples:
  python run_benchmark.py list
  python run_benchmark.py run --experiment openml --methods quick
  python run_benchmark.py run --experiment openml --methods paper
  python run_benchmark.py impute --experiment openml --methods tabimpute_self_mnar \\
      --checkpoint self_mnar=/path/to/self_mnar.ckpt
  python run_benchmark.py run --experiment openml_mcar_sweep --methods tabimpute hyperimpute missforest gain
  python run_benchmark.py run --experiment uci --methods quick
  python run_benchmark.py runtime --methods tabimpute tabimpute_v2

Then make the paper figures and tables with make_figures.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_DIR))
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
# The baseline libraries emit many deprecation and dtype warnings
warnings.filterwarnings("ignore")

from missbench import datasets as ds  # noqa: E402
from missbench import experiment as ex  # noqa: E402
from missbench.methods import CHECKPOINT_METHODS, METHOD_GROUPS, METHODS, get_method, resolve_methods, specialist_key  # noqa: E402
from missbench.patterns import PATTERNS  # noqa: E402

DEFAULT_DATA_DIR = BENCHMARK_DIR / "datasets"
DEFAULT_RESULTS_DIR = BENCHMARK_DIR / "results"


def parse_checkpoints(values: list[str]) -> dict[str, str]:
    """Parse KEY=PATH pairs. KEY is a method key or a pattern key (shorthand for its TabImpute-P)."""
    out = {}
    for value in values or []:
        key, sep, path = value.partition("=")
        if not sep:
            raise SystemExit(f"--checkpoint expects KEY=PATH, got {value!r}")
        if key in PATTERNS:
            key = specialist_key(key)
        if not Path(path).exists():
            raise SystemExit(f"Checkpoint not found: {path}")
        out[key] = path
    return out


def selected_methods(args) -> list[tuple[str, str | None]]:
    """(method key, checkpoint path or None) for each selected method."""
    checkpoints = parse_checkpoints(args.checkpoint)
    keys = resolve_methods(args.methods)
    # Models given a checkpoint are run even if not listed in --methods
    keys += [k for k in checkpoints if k not in keys]
    for k in keys:
        if k in CHECKPOINT_METHODS and k not in checkpoints:
            raise SystemExit(f"{k} needs a model checkpoint: pass --checkpoint {k}=PATH")
        get_method(k, checkpoints.get(k))  # fail early on a bad --checkpoint
    return [(k, checkpoints.get(k)) for k in keys]


def selected_cells(args) -> list:
    exp = ex.EXPERIMENTS[args.experiment]
    names = args.datasets or list(ds.SUITE_DATASETS[exp.suite])
    unknown = [n for n in names if n not in ds.SUITE_DATASETS[exp.suite]]
    if unknown:
        raise SystemExit(f"Unknown {exp.suite} dataset(s): {unknown}")
    return ex.cells(exp, names, args.patterns, args.p, args.repeats)


def cmd_list(args) -> None:
    print("Experiments (--experiment):")
    for e in ex.EXPERIMENTS.values():
        print(f"  {e.name:<18} {e.description}")
    print("\nMissingness patterns (--patterns):")
    for p in PATTERNS.values():
        print(f"  {p.key:<14} {p.display}")
    print("\nMethods (--methods):")
    for m in METHODS.values():
        note = "  [needs --checkpoint]" if m.key in CHECKPOINT_METHODS else ""
        print(f"  {m.key:<24} {m.display}{note}")
    print("\nMethod groups:")
    for name, keys in METHOD_GROUPS.items():
        print(f"  {name:<6} {' '.join(keys)}")
    print(f"\nOpenML datasets: {len(ds.OPENML_DATASETS)}, UCI datasets: {len(ds.UCI_DATASETS)}")


def cmd_download(args) -> None:
    exp = ex.EXPERIMENTS[args.experiment]
    names = args.datasets or list(ds.SUITE_DATASETS[exp.suite])
    print(f"Downloading {len(names)} {exp.suite} datasets to {args.data_dir / exp.suite}")
    ds.download(args.data_dir, exp.suite, names, overwrite=args.overwrite)


def cmd_missingness(args) -> None:
    ex.create_missingness(args.data_dir, selected_cells(args), seed=args.seed, overwrite=args.overwrite)


def cmd_impute(args) -> None:
    ex.run_imputers(args.data_dir, selected_cells(args), selected_methods(args), device=args.device,
                    overwrite=args.overwrite, fail_fast=args.fail_fast)


def cmd_evaluate(args) -> None:
    df = ex.evaluate(args.data_dir, selected_cells(args))
    if df.empty:
        raise SystemExit("No imputations found. Run the 'impute' step first.")
    path = ex.results_path(args.results_dir, args.experiment)
    ex.save_results(df, path)
    summary = df.groupby("method")[["nrmse", "r2"]].mean().sort_values("nrmse")
    print(f"\nMean scores over {df[['dataset', 'pattern', 'p', 'repeat']].drop_duplicates().shape[0]} "
          f"benchmark instances (nrmse: lower is better, r2: higher is better):")
    print(summary.round(3).to_string())
    print(f"\nSaved {len(df)} rows to {path}")


def cmd_run(args) -> None:
    # Load the methods first so bad method names or checkpoints fail before any work
    selected_methods(args)
    overwrite = args.overwrite
    args.overwrite = False
    cmd_download(args)
    cmd_missingness(args)
    args.overwrite = overwrite
    cmd_impute(args)
    cmd_evaluate(args)


def cmd_runtime(args) -> None:
    from missbench import runtime

    methods = [get_method(k, checkpoint) for k, checkpoint in selected_methods(args)]
    df = runtime.run(methods, device=args.device, rows=args.rows, num_cols=args.cols,
                     num_runs=args.runs)
    path = Path(args.results_dir) / "runtime.csv"
    ex.save_results(df, path)
    print(f"Saved {len(df)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help=f"where datasets and imputations are stored (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help=f"where result CSVs are written (default: {DEFAULT_RESULTS_DIR})")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_selection(p, methods: bool = False):
        p.add_argument("--experiment", "-e", choices=list(ex.EXPERIMENTS), default="openml")
        p.add_argument("--datasets", nargs="+", help="subset of datasets (default: all)")
        p.add_argument("--patterns", nargs="+", choices=list(PATTERNS),
                       help="subset of missingness patterns (default: the experiment's)")
        p.add_argument("--p", nargs="+", type=float, help="missingness levels (default: the experiment's)")
        p.add_argument("--repeats", type=int, help="number of masks per setting (default: the experiment's)")
        if methods:
            add_methods(p)

    def add_methods(p, default=("quick",)):
        p.add_argument("--methods", "-m", nargs="+", default=list(default),
                       help="method keys or groups (paper, quick, all); see `list`")
        p.add_argument("--checkpoint", nargs="+", metavar="KEY=PATH",
                       help="TabImpute checkpoint per model, e.g. self_mnar=ckpts/self_mnar.ckpt "
                            "(pattern-specific TabImpute-P models are not on HuggingFace)")
        p.add_argument("--device", default="auto", help="auto, cpu or cuda (default: auto)")

    sub.add_parser("list", help="list experiments, patterns and methods").set_defaults(func=cmd_list)

    p = sub.add_parser("download", help="download the datasets")
    p.add_argument("--experiment", "-e", choices=list(ex.EXPERIMENTS), default="openml")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("missingness", help="sample the missingness masks")
    add_selection(p)
    p.add_argument("--seed", type=int, default=0, help="base seed for the masks (default: 0)")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_missingness)

    p = sub.add_parser("impute", help="run imputation methods")
    add_selection(p, methods=True)
    p.add_argument("--overwrite", action="store_true", help="re-run imputations that already exist")
    p.add_argument("--fail-fast", action="store_true", help="stop at the first failed imputation")
    p.set_defaults(func=cmd_impute)

    p = sub.add_parser("evaluate", help="score imputations and write results/<experiment>.csv")
    add_selection(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("run", help="download, missingness, impute and evaluate in one go")
    add_selection(p, methods=True)
    p.add_argument("--seed", type=int, default=0, help="base seed for the masks (default: 0)")
    p.add_argument("--overwrite", action="store_true", help="re-run imputations that already exist")
    p.add_argument("--fail-fast", action="store_true", help="stop at the first failed imputation")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("runtime", help="time methods on random matrices of growing size")
    add_methods(p, default=("tabimpute", "tabimpute_v2"))
    p.add_argument("--rows", nargs="+", type=int, default=[10, 25, 50, 100, 250, 500, 1000])
    p.add_argument("--cols", type=int, default=10)
    p.add_argument("--runs", type=int, default=5)
    p.set_defaults(func=cmd_runtime)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
