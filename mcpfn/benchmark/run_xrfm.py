"""
Standalone xRFM imputation runner.

Walks the existing per-repeat dataset layout
    {base_path}/{name}/{pattern}/missingness-{level}/repeat-{r}/missing.npy
and writes
    {repeat_dir}/xrfm.npy
    {repeat_dir}/xrfm_imputation_time.txt

Does not touch the existing UCI / OpenML runners. Mirrors their
convention: standardize input by column nanmean / nanstd, impute, then
un-standardize before writing to disk so the saved file is on the same
scale as `true.npy`.

Usage
-----
    python run_xrfm.py                         # default: uci, all patterns, level 0.3
    python run_xrfm.py --base uci --level 0.3  # explicit
    python run_xrfm.py --base openml --level 0.4 --patterns MCAR
    python run_xrfm.py --base uci --datasets california_housing iris
    python run_xrfm.py --force                 # re-run even if xrfm.npy exists
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from xrfm_wrapper import XRFMImputer  # noqa: E402

OUT_NAME = "xrfm"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="uci", choices=["uci", "openml"],
                   help="Dataset family directory under benchmark/datasets/")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Subset of dataset names to run (default: all under base).")
    p.add_argument("--patterns", nargs="*",
                   default=["MCAR", "MAR", "MNAR"],
                   help="Missingness patterns to run.")
    p.add_argument("--levels", nargs="*", type=float,
                   default=[0.3],
                   help="Missingness levels (matches subdir 'missingness-{level}').")
    p.add_argument("--num-repeats", type=int, default=3)
    p.add_argument("--device", default=None,
                   help="torch device. Default: cuda if available else cpu.")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--mice-iters", type=int, default=0)
    p.add_argument("--leaf-size", type=int, default=60_000)
    p.add_argument("--refill-size", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help="Re-run even if xrfm.npy already exists.")
    p.add_argument("--verbose", action="store_true",
                   help="Forward xRFM stdout/stderr (default: suppressed).")
    return p.parse_args()


def iter_repeat_dirs(base_path: Path, datasets, patterns, levels, num_repeats):
    for name in datasets:
        for pattern, level, r in itertools.product(patterns, levels, range(num_repeats)):
            level_str = f"{level}".rstrip("0").rstrip(".") if "." in f"{level}" else f"{level}"
            # Match the existing on-disk format e.g. 'missingness-0.3'
            cfg_dir = base_path / name / pattern / f"missingness-{level}" / f"repeat-{r}"
            yield name, pattern, level, r, cfg_dir


def main() -> int:
    args = parse_args()

    base_path = BENCHMARK_DIR / "datasets" / args.base
    if not base_path.exists():
        print(f"ERROR: {base_path} does not exist", file=sys.stderr)
        return 1

    if args.datasets is None:
        datasets = sorted(
            d.name for d in base_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
    else:
        datasets = list(args.datasets)

    print(
        f"xRFM runner: base={base_path} | {len(datasets)} datasets | "
        f"patterns={args.patterns} | levels={args.levels} | repeats={args.num_repeats}"
    )

    imputer_kwargs = dict(
        device=args.device,
        val_fraction=args.val_fraction,
        mice_iters=args.mice_iters,
        leaf_size=args.leaf_size,
        refill_size=args.refill_size,
        random_state=args.seed,
        verbose=args.verbose,
    )

    n_run = 0
    n_skip = 0
    n_missing_dir = 0
    n_failed = 0

    pbar = tqdm(datasets, desc="datasets")
    for name in pbar:
        pbar.set_description(f"xrfm:{name}")
        for _, pattern, level, r, cfg_dir in iter_repeat_dirs(
            base_path, [name], args.patterns, args.levels, args.num_repeats
        ):
            missing_path = cfg_dir / "missing.npy"
            if not missing_path.exists():
                n_missing_dir += 1
                continue

            out_path = cfg_dir / f"{OUT_NAME}.npy"
            time_path = cfg_dir / f"{OUT_NAME}_imputation_time.txt"
            if out_path.exists() and not args.force:
                n_skip += 1
                continue

            X_missing = np.load(missing_path).astype(np.float32)
            col_mean = np.nanmean(X_missing, axis=0)
            col_std = np.nanstd(X_missing, axis=0)
            # Guard against zero-std columns (constant after dropping NaNs).
            col_std_safe = np.where(col_std == 0, 1.0, col_std)
            X_std = (X_missing - col_mean) / col_std_safe

            t0 = time.time()
            try:
                X_hat_std = XRFMImputer(**imputer_kwargs).fit_transform(X_std)
            except Exception as e:  # noqa: BLE001
                n_failed += 1
                tqdm.write(f"FAIL  {name}/{pattern}/{level}/repeat-{r}: {e}")
                continue
            dt = time.time() - t0

            X_hat = X_hat_std * col_std_safe + col_mean
            np.save(out_path, X_hat.astype(np.float32))
            with open(time_path, "a") as f:
                f.write(f"{dt}\n")
            n_run += 1
            tqdm.write(
                f"  {name}/{pattern}/{level}/repeat-{r}  "
                f"shape={X_missing.shape}  nan={float(np.isnan(X_missing).mean()):.2f}  "
                f"{dt:.1f}s"
            )

    print(
        f"\nDone. ran={n_run}  skipped(existing)={n_skip}  "
        f"no_missing.npy={n_missing_dir}  failed={n_failed}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
