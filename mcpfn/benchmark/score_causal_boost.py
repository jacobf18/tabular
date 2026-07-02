"""Score the boost predictions on CPS / Kansas / Prop99 and append rows to
the existing per_draw_metrics.csv files so the plot scripts pick them up.

Mirrors compute_metrics() from CPS/eval_cps_causal.py exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK_DIR = Path("/root/tabular/mcpfn/benchmark")
BOOST_NAMES = [
    "boost_matformer_dwip_v2_t002_a0p1",
    "boost_matformer_dwip_v2_t002_a0p05",
]


def compute_metrics(true: np.ndarray, completed: np.ndarray,
                    treatment_mask: np.ndarray) -> dict[str, float]:
    """Identical to eval_cps_causal.compute_metrics."""
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


def process_results_dir(results_dir: Path, benchmark: str, dry_run: bool) -> tuple[int, int]:
    csv_path = results_dir / "per_draw_metrics.csv"
    if not csv_path.is_file():
        return 0, 0

    df = pd.read_csv(csv_path)

    # Remove any pre-existing boost rows (in case we re-run).
    df = df[~df["method"].isin(BOOST_NAMES)].copy()

    # The results dir sits alongside draw_* directories under
    # causal_eval/<subgroup>/. Find the draws.
    parent = results_dir.parent     # e.g. .../causal_eval/log_wage_open_carry/
    draws = sorted([p for p in parent.iterdir() if p.is_dir() and p.name.startswith("draw_")])

    new_rows = []
    for draw in draws:
        obs_p = draw / "observed.npy"
        tru_p = draw / "true.npy"
        tm_p = draw / "treatment_mask.npy"
        meta_p = draw / "metadata.json"
        if not (obs_p.is_file() and tru_p.is_file()):
            continue
        try:
            tru = np.load(tru_p)
            if tm_p.is_file():
                tmask = np.load(tm_p).astype(bool)
            else:
                tmask = np.isnan(np.load(obs_p))
        except Exception as e:
            print(f"  ERROR loading {draw}: {e}")
            continue

        # Pull T0 from metadata.json if present.
        T0 = None
        if meta_p.is_file():
            try:
                m = json.loads(meta_p.read_text())
                T0 = m.get("T0", m.get("post_col_start"))
            except Exception:
                pass

        for boost_name in BOOST_NAMES:
            boost_p = draw / f"{boost_name}.npy"
            if not boost_p.is_file():
                continue
            try:
                boost = np.load(boost_p)
                metrics = compute_metrics(tru, boost, tmask)
            except Exception as e:
                print(f"  ERROR scoring {draw}/{boost_name}: {e}")
                continue
            new_rows.append({
                "draw_name": draw.name,
                "draw_id": int(draw.name.split("_")[-1].lstrip("0") or 0),
                "method": boost_name,
                "runtime_seconds": 0.0,
                "treated_block_rmse": metrics["treated_block_rmse"],
                "treated_block_mae": metrics["treated_block_mae"],
                "tau_hat": metrics["tau_hat"],
                "tau_sq_error": metrics["tau_sq_error"],
                "tau_abs_error": metrics["tau_abs_error"],
                "observed_block_rmse": metrics["observed_block_rmse"],
                "n_treated_cells": metrics["n_treated_cells"],
                "T0": T0,
            })

    if not new_rows:
        return 0, 0

    if dry_run:
        return len(new_rows), 0

    new_df = pd.DataFrame(new_rows)
    # Align columns to the existing CSV.
    for col in df.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan
    new_df = new_df[df.columns]

    out_df = pd.concat([df, new_df], ignore_index=True)
    out_df.to_csv(csv_path, index=False)
    return len(new_rows), 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["CPS", "kansas", "prop99", "all"],
                    default="all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bms = ["CPS", "kansas", "prop99"] if args.benchmark == "all" else [args.benchmark]
    for bm in bms:
        base = BENCHMARK_DIR / bm / "causal_eval"
        results_dirs = sorted(base.rglob("results"))
        # Filter to actual directories (not files named "results").
        results_dirs = [r for r in results_dirs if r.is_dir()]
        total_rows = 0
        total_files = 0
        for r in results_dirs:
            n_rows, n_files = process_results_dir(r, bm, args.dry_run)
            total_rows += n_rows
            total_files += n_files
            print(f"  [{bm}] {r.parent.name:40s}  +{n_rows} rows")
        print(f"[{bm}] total rows added: {total_rows}  csvs updated: {total_files}")


if __name__ == "__main__":
    main()
