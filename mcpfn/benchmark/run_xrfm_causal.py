"""
Standalone xRFM runner for causal-inference benchmarks (CPS / kansas / prop99).

Walks
    <family>/causal_eval/<scenario>/draw_NNNN/observed.npy
and writes per-draw
    <draw_dir>/xrfm.npy
plus rows appended to
    <scenario>/results/per_draw_metrics.csv

Schema of per_draw_metrics.csv matches eval_cps_causal.py:
    draw_name, draw_id, method, runtime_seconds,
    treated_block_rmse, treated_block_mae,
    tau_hat, tau_sq_error, tau_abs_error,
    observed_block_rmse, n_treated_cells

Does not touch any existing eval / summarize / plot script.

Usage
-----
    # CPS: all 9 scenarios
    python run_xrfm_causal.py --family CPS

    # Kansas
    python run_xrfm_causal.py --family kansas

    # Prop99: ca_simul / ca_stagg / stocks (skip 'smoke')
    python run_xrfm_causal.py --family prop99

    # Restrict to one scenario
    python run_xrfm_causal.py --family CPS --scenarios hours_abort_ban

    # All three families in one shot
    python run_xrfm_causal.py --family CPS kansas prop99
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from xrfm_wrapper import XRFMImputer  # noqa: E402

OUT_NAME = "xrfm"
DEFAULT_SCENARIO_SKIP = {"smoke", "causal_eval_smoketest"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--family", nargs="+", default=["CPS", "kansas", "prop99"],
                   help="Top-level benchmark family directory name(s).")
    p.add_argument("--scenarios", nargs="*", default=None,
                   help="Scenario subset within causal_eval/ (default: all).")
    p.add_argument("--skip-scenarios", nargs="*",
                   default=sorted(DEFAULT_SCENARIO_SKIP),
                   help="Scenario names to skip (default: smoke runs).")
    p.add_argument("--limit-draws", type=int, default=None,
                   help="Cap draws per scenario (debugging).")
    p.add_argument("--device", default=None)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--mice-iters", type=int, default=0)
    p.add_argument("--leaf-size", type=int, default=60_000)
    p.add_argument("--refill-size", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help="Re-run even if xrfm.npy already exists.")
    p.add_argument("--no-csv", action="store_true",
                   help="Skip writing/merging per_draw_metrics.csv (useful if "
                        "you only want the .npy artifacts).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def compute_metrics(true: np.ndarray, completed: np.ndarray,
                    treatment_mask: np.ndarray) -> dict[str, float]:
    """Mirrors eval_cps_causal.py:compute_metrics."""
    treated = treatment_mask.astype(bool)
    observed = ~treated
    treated_true = true[treated]
    treated_pred = completed[treated]
    treated_block_rmse = float(np.sqrt(np.mean((treated_true - treated_pred) ** 2)))
    treated_block_mae = float(np.mean(np.abs(treated_true - treated_pred)))
    tau_hat = float(np.mean(treated_true) - np.mean(treated_pred))
    observed_rmse = float(np.sqrt(np.mean((true[observed] - completed[observed]) ** 2)))
    return {
        "treated_block_rmse": treated_block_rmse,
        "treated_block_mae": treated_block_mae,
        "tau_hat": tau_hat,
        "tau_sq_error": float(tau_hat ** 2),
        "tau_abs_error": float(abs(tau_hat)),
        "observed_block_rmse": observed_rmse,
        "n_treated_cells": int(treated.sum()),
    }


def run_one_draw(draw_dir: Path, imputer_kwargs: dict,
                 force: bool) -> tuple[bool, float, dict | None]:
    """Returns (ran, runtime_seconds, metrics_dict_or_None)."""
    obs_path = draw_dir / "observed.npy"
    true_path = draw_dir / "true.npy"
    mask_path = draw_dir / "treatment_mask.npy"
    meta_path = draw_dir / "metadata.json"
    if not (obs_path.exists() and true_path.exists()
            and mask_path.exists() and meta_path.exists()):
        return False, 0.0, None

    out_path = draw_dir / f"{OUT_NAME}.npy"

    observed = np.load(obs_path).astype(np.float32)
    true = np.load(true_path).astype(np.float32)
    treatment_mask = np.load(mask_path)
    missing_mask = np.isnan(observed)

    col_mean = np.nanmean(observed, axis=0)
    col_std = np.nanstd(observed, axis=0)
    col_std = np.where((col_std == 0) | np.isnan(col_std), 1.0, col_std)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    observed_in = (observed - col_mean) / col_std

    if out_path.exists() and not force:
        completed = np.load(out_path)
        runtime = float("nan")
    else:
        t0 = time.time()
        completed_std = XRFMImputer(**imputer_kwargs).fit_transform(observed_in)
        runtime = time.time() - t0
        completed = (completed_std * col_std) + col_mean
        completed = completed.astype(np.float32)
        completed[~missing_mask] = observed[~missing_mask]
        np.save(out_path, completed)

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metrics = compute_metrics(true, completed, treatment_mask)
    metrics["draw_name"] = draw_dir.name
    metrics["draw_id"] = int(metadata.get("draw_id", -1))
    metrics["method"] = OUT_NAME
    metrics["runtime_seconds"] = runtime
    return True, runtime, metrics


def merge_metrics_csv(results_dir: Path, new_rows: list[dict]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "per_draw_metrics.csv"
    new_df = pd.DataFrame(new_rows)
    # Align column ordering with eval_cps_causal.py output if present.
    preferred_order = [
        "draw_name", "draw_id", "method", "runtime_seconds",
        "treated_block_rmse", "treated_block_mae",
        "tau_hat", "tau_sq_error", "tau_abs_error",
        "observed_block_rmse", "n_treated_cells",
    ]
    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["draw_name", "method"], keep="last")
        cols = [c for c in preferred_order if c in merged.columns]
        cols += [c for c in merged.columns if c not in cols]
        merged[cols].to_csv(csv_path, index=False)
    else:
        cols = [c for c in preferred_order if c in new_df.columns]
        cols += [c for c in new_df.columns if c not in cols]
        new_df[cols].to_csv(csv_path, index=False)
    return csv_path


def iter_scenarios(family_dir: Path, scenarios: list[str] | None,
                   skip: set[str]):
    causal = family_dir / "causal_eval"
    if not causal.exists():
        return
    if scenarios is None:
        for sc in sorted(causal.iterdir()):
            if not sc.is_dir() or sc.name in skip:
                continue
            yield sc
    else:
        for s in scenarios:
            sc = causal / s
            if sc.exists() and sc.is_dir():
                yield sc


def main() -> int:
    args = parse_args()
    skip = set(args.skip_scenarios)

    imputer_kwargs = dict(
        device=args.device,
        val_fraction=args.val_fraction,
        mice_iters=args.mice_iters,
        leaf_size=args.leaf_size,
        refill_size=args.refill_size,
        random_state=args.seed,
        verbose=args.verbose,
    )

    grand_total_run = 0
    grand_total_skip = 0
    grand_total_failed = 0

    for fam in args.family:
        family_dir = BENCHMARK_DIR / fam
        if not family_dir.exists():
            print(f"  [skip] family dir not found: {family_dir}", file=sys.stderr)
            continue
        for sc_dir in iter_scenarios(family_dir, args.scenarios, skip):
            draw_dirs = sorted(p for p in sc_dir.iterdir()
                               if p.is_dir() and p.name.startswith("draw_"))
            if args.limit_draws is not None:
                draw_dirs = draw_dirs[: args.limit_draws]
            if not draw_dirs:
                continue

            metrics_rows: list[dict] = []
            n_run = 0
            n_skip = 0
            n_fail = 0
            pbar = tqdm(draw_dirs, desc=f"{fam}/{sc_dir.name}", unit="draw")
            for d in pbar:
                out_path = d / f"{OUT_NAME}.npy"
                already = out_path.exists() and not args.force
                try:
                    ok, runtime, metrics = run_one_draw(d, imputer_kwargs, args.force)
                except Exception as e:  # noqa: BLE001
                    n_fail += 1
                    tqdm.write(f"FAIL {fam}/{sc_dir.name}/{d.name}: {e}")
                    continue
                if not ok:
                    continue
                if metrics is not None:
                    metrics_rows.append(metrics)
                if already:
                    n_skip += 1
                else:
                    n_run += 1
                pbar.set_postfix(run=n_run, skip=n_skip, fail=n_fail)

            if metrics_rows and not args.no_csv:
                csv_path = merge_metrics_csv(sc_dir / "results", metrics_rows)
                tqdm.write(f"  -> wrote {len(metrics_rows)} rows to {csv_path}")

            grand_total_run += n_run
            grand_total_skip += n_skip
            grand_total_failed += n_fail

    print(
        f"\nDone. ran={grand_total_run}  "
        f"skipped(existing)={grand_total_skip}  failed={grand_total_failed}"
    )
    return 0 if grand_total_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
