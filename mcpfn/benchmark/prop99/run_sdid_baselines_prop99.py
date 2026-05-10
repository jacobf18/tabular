#!/usr/bin/env python3
"""SDID / SC / DiD (Arkhangelsky et al., 2021) baselines for prop99.

Same mask-driven logic as kansas/run_sdid_baselines_kansas.py — treated rows
and T0 are inferred directly from treatment_mask.

Design notes:
  - 'simultaneous' (CA-smoking block) and 'stock_block_second_half' are clean
    block designs: all treated units share the same first post-period; omega
    and lambda are fit once per draw.
  - 'staggered' sets T0 = min(post_col) for the synthdid fit. For cells
    between a unit's T0 and its own adoption time, the observed value is
    restored at the end (completed[~missing_mask] = observed[~missing_mask]),
    so staggered results are best-effort, not a proper cohort-synthdid.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

BENCH_DIR = Path(__file__).resolve().parents[1]
CPS_DIR = BENCH_DIR / "CPS"
if str(CPS_DIR) not in sys.path:
    sys.path.insert(0, str(CPS_DIR))

from eval_cps_causal import compute_metrics  # noqa: E402


TAU_ONLY_METHODS = {
    "sdid": {},
    "sc": {"synth": True},
    "did": {"did": True},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-dir", type=Path, required=True)
    p.add_argument("--methods", type=str, default=",".join(TAU_ONLY_METHODS.keys()))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--skip-staggered", action="store_true",
                   help="Skip draws whose metadata.design == 'staggered'.")
    return p.parse_args()


def infer_treated_geometry(treatment_mask: np.ndarray) -> tuple[np.ndarray, int]:
    treated_rows = np.flatnonzero(treatment_mask.any(axis=1))
    if treated_rows.size == 0:
        raise ValueError("No treated cells in treatment_mask.")
    post_cols = np.flatnonzero(treatment_mask.any(axis=0))
    T0 = int(post_cols.min())
    return treated_rows, T0


def long_format(y: np.ndarray, treatment_mask: np.ndarray, treated_rows: np.ndarray, T0: int) -> pd.DataFrame:
    N, T = y.shape
    treated_unit = np.zeros(N, dtype=bool)
    treated_unit[treated_rows] = True
    tyear = np.where(treated_unit, T0 + 1, 0)

    rows = []
    for i in range(N):
        for t in range(T):
            rows.append({
                "unit": i + 1,
                "time": t + 1,
                "outcome": float(y[i, t]),
                "treated": int(treated_unit[i]),
                "tyear": int(tyear[i]),
                "treatment": int(treatment_mask[i, t]),
            })
    return pd.DataFrame(rows)


def run_estimator(df: pd.DataFrame, kwargs: dict):
    from synthdid.synthdid import Synthdid

    model = Synthdid(df, unit="unit", time="time", treatment="treatment", outcome="outcome")
    model.fit(**kwargs)
    omega = np.asarray(model.weights["omega"][0], dtype=np.float64)
    lam = np.asarray(model.weights["lambda"][0], dtype=np.float64)
    return float(model.att), omega, lam


def reconstruct_counterfactual(
    observed: np.ndarray,
    treatment_mask: np.ndarray,
    treated_rows: np.ndarray,
    T0: int,
    omega: np.ndarray,
    lam: np.ndarray,
) -> np.ndarray:
    N = observed.shape[0]
    ctrl_mask = np.ones(N, dtype=bool)
    ctrl_mask[treated_rows] = False
    ctrl_idx = np.flatnonzero(ctrl_mask)

    Y = observed.copy()
    Y[treatment_mask.astype(bool)] = 0.0

    Y_ctrl_post = Y[ctrl_idx][:, T0:]
    Y_ctrl_pre = Y[ctrl_idx][:, :T0]

    sc_path_post = omega @ Y_ctrl_post
    double_avg = omega @ Y_ctrl_pre @ lam

    completed = observed.astype(np.float64).copy()
    for i in treated_rows:
        trt_pre_i = Y[i, :T0]
        lambda_avg_i = trt_pre_i @ lam
        completed[i, T0:] = sc_path_post + (lambda_avg_i - double_avg)

    return completed.astype(np.float32)


def main() -> None:
    args = parse_args()
    if not args.benchmark_dir.exists():
        raise SystemExit(f"Benchmark directory does not exist: {args.benchmark_dir}")

    requested = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in requested if m not in TAU_ONLY_METHODS]
    if unknown:
        raise SystemExit(f"Unknown tau-only methods: {unknown}")

    draw_dirs = sorted(
        p for p in args.benchmark_dir.iterdir()
        if p.is_dir() and p.name.startswith("draw_")
    )
    if args.limit is not None:
        draw_dirs = draw_dirs[: args.limit]

    results_dir = args.benchmark_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict] = []
    failures: list[dict] = []
    skipped = 0

    work_pbar = tqdm(total=len(draw_dirs) * len(requested),
                     desc=f"{args.benchmark_dir.name}: tau-only methods",
                     unit="step")

    for draw_dir in draw_dirs:
        true_matrix = np.load(draw_dir / "true.npy").astype(np.float64)
        observed = np.load(draw_dir / "observed.npy")
        treatment_mask = np.load(draw_dir / "treatment_mask.npy")
        metadata = json.loads((draw_dir / "metadata.json").read_text(encoding="utf-8"))

        if args.skip_staggered and metadata.get("design") == "staggered":
            skipped += 1
            work_pbar.update(len(requested))
            continue

        treated_rows, T0 = infer_treated_geometry(treatment_mask)
        df = long_format(true_matrix, treatment_mask, treated_rows, T0)

        for method in requested:
            work_pbar.set_postfix_str(f"{draw_dir.name}:{method}")
            out_path = draw_dir / f"{method}.npy"
            if out_path.exists() and not args.force_rerun:
                completed = np.load(out_path)
                runtime_seconds = float("nan")
            else:
                kwargs = TAU_ONLY_METHODS[method]
                start = time.time()
                try:
                    _att, omega, lam = run_estimator(df, kwargs)
                    completed = reconstruct_counterfactual(
                        true_matrix, treatment_mask, treated_rows, T0, omega, lam
                    )
                except Exception as exc:
                    failures.append({"draw_name": draw_dir.name, "method": method, "error": repr(exc)})
                    work_pbar.update(1)
                    continue
                runtime_seconds = time.time() - start
                missing_mask = np.isnan(observed)
                completed = completed.astype(np.float32)
                completed[~missing_mask] = observed[~missing_mask]
                np.save(out_path, completed)

            metrics = compute_metrics(true_matrix.astype(np.float32), completed, treatment_mask)
            metrics_rows.append({
                "draw_name": draw_dir.name,
                "draw_id": int(metadata["draw_id"]),
                "design": metadata.get("design"),
                "Nt": metadata.get("Nt"),
                "T0": metadata.get("T0"),
                "N": metadata.get("N"),
                "T": metadata.get("T"),
                "method": method,
                "runtime_seconds": runtime_seconds,
                **metrics,
            })
            work_pbar.update(1)

    work_pbar.close()

    per_draw_path = results_dir / "per_draw_metrics.csv"
    new_df = pd.DataFrame(metrics_rows)
    if per_draw_path.exists() and not new_df.empty:
        old_df = pd.read_csv(per_draw_path)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["draw_name", "method"], keep="last")
        merged.to_csv(per_draw_path, index=False)
    else:
        new_df.to_csv(per_draw_path, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(results_dir / "failures_tau_only.csv", index=False)
        print(f"Encountered {len(failures)} failures; see results/failures_tau_only.csv")

    print(f"Wrote {len(metrics_rows)} tau-only rows to {per_draw_path}"
          + (f" (skipped {skipped} staggered draws)" if skipped else ""))


if __name__ == "__main__":
    main()
