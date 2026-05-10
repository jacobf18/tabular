#!/usr/bin/env python3
"""Kansas causal-panel imputation evaluation.

Structurally mirrors CPS/eval_cps_causal.py but driven entirely off the
per-draw treatment_mask, so a single treated row anywhere in the matrix works
without modification. The imputer registry and metric functions are imported
from the CPS eval script to avoid duplication.
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

BENCH_DIR = Path(__file__).resolve().parents[1]
CPS_DIR = BENCH_DIR / "CPS"
for extra in (BENCH_DIR, CPS_DIR):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from eval_cps_causal import (  # noqa: E402
    DWIP_V3R1_MANIFEST_PATH,
    ROUND2_T2_SPECS,
    build_method_registry,
    compute_metrics,
    expand_method_list,
    load_manifest_mcpfn_specs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the Kansas causal benchmark by imputing the treated-post block."
    )
    p.add_argument("--benchmark-dir", type=Path, required=True,
                   help="Directory created by create_kansas_benchmark.py.")
    p.add_argument("--methods", type=str, default="knn,softimpute,tabimpute_all",
                   help="Comma-separated method ids; 'tabimpute_all', 'hyperimpute_all', 'baselines_all', or 'all' expand.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--manifest-path", type=Path, default=DWIP_V3R1_MANIFEST_PATH)
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force-rerun", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.benchmark_dir.exists():
        raise SystemExit(f"Benchmark directory does not exist: {args.benchmark_dir}")

    specs = load_manifest_mcpfn_specs(args.manifest_path)
    specs.extend(ROUND2_T2_SPECS)

    raw_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    requested_methods = expand_method_list(raw_methods, specs)
    if not requested_methods:
        raise SystemExit("No methods requested after expansion.")
    registry = build_method_registry(args, requested_methods, specs)

    draw_dirs = sorted(
        p for p in args.benchmark_dir.iterdir()
        if p.is_dir() and p.name.startswith("draw_")
    )
    if args.limit is not None:
        draw_dirs = draw_dirs[: args.limit]

    results_dir = args.benchmark_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    standardize = not args.no_standardize

    total_work = len(draw_dirs) * len(requested_methods)
    outer_pbar = tqdm(draw_dirs, desc=f"Draws ({len(requested_methods)} methods)",
                      unit="draw", total=len(draw_dirs))
    work_pbar = tqdm(total=total_work, desc="Method×Draw", unit="step", leave=False)

    for draw_dir in outer_pbar:
        true = np.load(draw_dir / "true.npy")
        observed = np.load(draw_dir / "observed.npy")
        treatment_mask = np.load(draw_dir / "treatment_mask.npy")
        metadata = json.loads((draw_dir / "metadata.json").read_text(encoding="utf-8"))

        missing_mask = np.isnan(observed)

        if standardize:
            col_mean = np.nanmean(observed, axis=0)
            col_std = np.nanstd(observed, axis=0)
            col_std = np.where((col_std == 0) | np.isnan(col_std), 1.0, col_std)
            col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
            observed_in = (observed - col_mean) / col_std
        else:
            col_mean = np.zeros(observed.shape[1], dtype=observed.dtype)
            col_std = np.ones(observed.shape[1], dtype=observed.dtype)
            observed_in = observed

        for method in requested_methods:
            work_pbar.set_postfix_str(f"{draw_dir.name}:{method}")
            out_path = draw_dir / f"{method}.npy"
            if out_path.exists() and not args.force_rerun:
                completed = np.load(out_path)
                runtime_seconds = np.nan
            else:
                start = time.time()
                try:
                    completed_std = registry[method](observed_in.copy()).astype(np.float32)
                except Exception as exc:
                    failures.append({"draw_name": draw_dir.name, "method": method, "error": repr(exc)})
                    work_pbar.update(1)
                    continue
                runtime_seconds = time.time() - start
                completed = (completed_std * col_std) + col_mean
                completed = completed.astype(np.float32)
                completed[~missing_mask] = observed[~missing_mask]
                np.save(out_path, completed)

            metrics = compute_metrics(true, completed, treatment_mask)
            metrics_rows.append({
                "draw_name": draw_dir.name,
                "draw_id": int(metadata["draw_id"]),
                "method": method,
                "runtime_seconds": runtime_seconds,
                **metrics,
            })
            work_pbar.update(1)

    work_pbar.close()
    outer_pbar.close()

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
        pd.DataFrame(failures).to_csv(results_dir / "failures.csv", index=False)

    print(f"Wrote per-draw metrics to {per_draw_path}")
    if failures:
        print(f"Encountered {len(failures)} method failures; see {results_dir / 'failures.csv'}")


if __name__ == "__main__":
    main()
