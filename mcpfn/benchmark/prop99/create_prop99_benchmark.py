#!/usr/bin/env python3
"""Convert prop99 benchmark draws into per-draw directories with the
CPS/kansas-compatible layout (true.npy, observed.npy, treatment_mask.npy,
metadata.json) so the shared eval pipeline can run on them unchanged.

Input is the flat directory produced by paper_1710_10251_benchmark.py's
'build-california' or 'build-stocks' subcommands: manifest.csv plus
draw_XXXX_Y.csv / draw_XXXX_W.csv / draw_XXXX_meta.json per draw.

Output mirrors kansas/causal_eval/<run>/ — one subdirectory per draw.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Directory from build-california or build-stocks.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory where per-draw benchmark folders will be written.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true",
                   help="Overwrite output directory if it already exists.")
    return p.parse_args()


def load_numeric_csv(path: Path) -> np.ndarray:
    return pd.read_csv(path).to_numpy(dtype=np.float32)


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    manifest_path = args.input_dir / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    if args.output_dir.exists():
        if not args.force:
            raise SystemExit(
                f"Output directory already exists: {args.output_dir}. Use --force to overwrite."
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()

    benchmark_rows: list[dict[str, object]] = []

    for row in manifest.to_dict(orient="records"):
        draw_id = int(row["draw_id"])
        draw_name = str(row["draw_name"])
        y_path = args.input_dir / f"{draw_name}_Y.csv"
        w_path = args.input_dir / f"{draw_name}_W.csv"
        meta_path = args.input_dir / f"{draw_name}_meta.json"

        for req in (y_path, w_path):
            if not req.exists():
                raise FileNotFoundError(f"Missing draw file: {req}")

        src_meta: dict = {}
        if meta_path.exists():
            src_meta = json.loads(meta_path.read_text(encoding="utf-8"))

        y = load_numeric_csv(y_path)
        w = load_numeric_csv(w_path).astype(np.int8)
        if y.shape != w.shape:
            raise ValueError(f"Shape mismatch for {draw_name}: Y={y.shape}, W={w.shape}")

        observed = y.copy()
        observed[w == 1] = np.nan

        draw_dir = args.output_dir / draw_name
        draw_dir.mkdir(parents=True, exist_ok=True)
        np.save(draw_dir / "true.npy", y)
        np.save(draw_dir / "observed.npy", observed)
        np.save(draw_dir / "treatment_mask.npy", w)

        # Geometry derived directly from the mask (same logic as kansas
        # create_kansas_benchmark so downstream code never needs to special-case
        # the simultaneous / staggered / stock-block designs).
        treated_rows = np.flatnonzero(w.any(axis=1))
        treated_cols = np.flatnonzero(w.any(axis=0))
        n_rows, n_cols = int(y.shape[0]), int(y.shape[1])

        design = src_meta.get("design") or src_meta.get("assignment") or str(row.get("design", ""))
        outcome = src_meta.get("outcome") or str(row.get("outcome", ""))
        Nt = int(src_meta.get("Nt") or row.get("Nt") or len(treated_rows))
        T0_meta = src_meta.get("T0")
        T0 = int(T0_meta) if T0_meta is not None else (int(treated_cols.min()) if treated_cols.size else n_cols)

        metadata = {
            "draw_id": draw_id,
            "draw_name": draw_name,
            "dataset": src_meta.get("dataset") or str(row.get("dataset", "")),
            "design": design,
            "assignment": design,
            "outcome": outcome,
            "shape": {"n_rows": n_rows, "n_cols": n_cols},
            "n_treated_cells": int(w.sum()),
            "n_observed_cells": int((w == 0).sum()),
            "n_missing_cells": int(np.isnan(observed).sum()),
            "N": n_rows,
            "T": n_cols,
            "Nt": Nt,
            "T0": T0,
            "N0": int(n_rows - Nt),
            "N1_realized": int(len(treated_rows)),
            "T1_realized": int(n_cols - T0),
            "t0_ratio": float(src_meta.get("t0_ratio") or row.get("t0_ratio") or (T0 / max(n_cols, 1))),
            "treated_row_start": int(treated_rows.min()) if treated_rows.size else 0,
            "treated_row_end": int(treated_rows.max() + 1) if treated_rows.size else 0,
            "post_col_start": int(treated_cols.min()) if treated_cols.size else n_cols,
            "post_col_end": int(treated_cols.max() + 1) if treated_cols.size else n_cols,
            "treated_row_idx": int(treated_rows[0]) if treated_rows.size == 1 else None,
            "treated_row_indices": treated_rows.tolist(),
            "post_col_indices": treated_cols.tolist(),
            # Carry through design-specific extras when present — useful for
            # grouped summaries / staggered-aware baselines.
            "adoption_times": src_meta.get("adoption_times"),
            "adoption_time": src_meta.get("adoption_time"),
            "source": src_meta.get("source"),
        }
        with (draw_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        benchmark_rows.append({
            k: v for k, v in metadata.items()
            if k not in ("adoption_times", "treated_row_indices", "post_col_indices")
        })

    pd.DataFrame(benchmark_rows).to_csv(args.output_dir / "benchmark_manifest.csv", index=False)
    print(f"Wrote {len(benchmark_rows)} prop99 benchmark draws to {args.output_dir}")


if __name__ == "__main__":
    main()
