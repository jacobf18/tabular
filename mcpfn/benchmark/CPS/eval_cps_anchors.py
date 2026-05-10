#!/usr/bin/env python3
"""Evaluate the three anchor tabimpute checkpoints on CPS causal benchmarks.

Same loop shape as kansas/eval_kansas_anchors.py — metrics are driven off the
per-draw treatment_mask so the CPS multi-unit bottom-right block and the Kansas
single-row layout both work unchanged. Pass --benchmark-dir pointing at any
one combo (e.g. CPS/causal_eval/log_wage_min_wage); loop externally over all
nine combos with --methods defaulting to the anchor trio.

Anchor names (per user request):
    dwip-v2-1  <- dwip_v2_t002 (model_new_stable.py)
    dwip-v2-2  <- dwip_v2_t003 (model_new_stable.py)
    dwip-v1    <- dwip_v1_local14 (model_new.py)
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

CPS_DIR = Path(__file__).resolve().parent
BENCH_DIR = CPS_DIR.parent
for extra in (BENCH_DIR, CPS_DIR):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from eval_cps_causal import (  # noqa: E402
    ROUND2_T2_CONFIG,
    compute_metrics,
    make_mcpfn_imputer,
)


ANCHOR_CKPT_DIR = Path("/root/tabimpute_anchor3_named")
V1_CONFIG = {"config": {**ROUND2_T2_CONFIG["config"], "rope_fraction": 0.75}}

ANCHOR_SPECS: list[dict] = [
    {
        "name": "dwip-v2-1",
        "model_family": "new_stable",
        "checkpoint_path": str(ANCHOR_CKPT_DIR / "dwip_v2_t002_checkpoint_17000.pth"),
        "json_config": ROUND2_T2_CONFIG,
    },
    {
        "name": "dwip-v2-2",
        "model_family": "new_stable",
        "checkpoint_path": str(ANCHOR_CKPT_DIR / "dwip_v2_t003_checkpoint_17000.pth"),
        "json_config": ROUND2_T2_CONFIG,
    },
    {
        "name": "dwip-v1",
        "model_family": "new",
        "checkpoint_path": str(ANCHOR_CKPT_DIR / "dwip_v1_local14_checkpoint_10000.pth"),
        "json_config": V1_CONFIG,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-dir", type=Path, required=True)
    p.add_argument("--methods", type=str, default="dwip-v2-1,dwip-v2-2,dwip-v1",
                   help="Comma-separated subset of {dwip-v2-1, dwip-v2-2, dwip-v1}.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force-rerun", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.benchmark_dir.exists():
        raise SystemExit(f"Benchmark directory does not exist: {args.benchmark_dir}")

    requested = [m.strip() for m in args.methods.split(",") if m.strip()]
    spec_by_name = {s["name"]: s for s in ANCHOR_SPECS}
    unknown = [m for m in requested if m not in spec_by_name]
    if unknown:
        raise SystemExit(f"Unknown anchor methods: {unknown}. Known: {list(spec_by_name)}")

    missing = [s["checkpoint_path"] for s in ANCHOR_SPECS
               if s["name"] in requested and not Path(s["checkpoint_path"]).exists()]
    if missing:
        raise SystemExit(f"Missing anchor checkpoint(s): {missing}")

    print(f"Benchmark: {args.benchmark_dir}")
    print("Loading anchor models:")
    registry = {}
    for name in requested:
        spec = spec_by_name[name]
        print(f"  {name}  <-  {spec['checkpoint_path']} ({spec['model_family']})")
        registry[name] = make_mcpfn_imputer(spec, args.device)

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
    standardize = not args.no_standardize

    total_work = len(draw_dirs) * len(requested)
    outer_pbar = tqdm(draw_dirs, desc=f"{args.benchmark_dir.name} ({len(requested)} anchors)",
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

        for method in requested:
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
        pd.DataFrame(failures).to_csv(results_dir / "failures_anchors.csv", index=False)
        print(f"Encountered {len(failures)} anchor failures; see results/failures_anchors.csv")

    print(f"Wrote {len(metrics_rows)} anchor rows to {per_draw_path}")


if __name__ == "__main__":
    main()
