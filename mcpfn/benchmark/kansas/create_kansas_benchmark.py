#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Kansas causal benchmark folders by masking the treated-post "
            "block produced by generate_kansas_semisynth.py."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory with manifest.csv and draw_XXXX_[Y|W].csv.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory where per-draw benchmark folders will be written.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite output directory if it already exists.")
    return parser.parse_args()


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
        draw_name = f"draw_{draw_id:04d}"
        y_path = args.input_dir / f"{draw_name}_Y.csv"
        w_path = args.input_dir / f"{draw_name}_W.csv"

        if not y_path.exists():
            raise FileNotFoundError(f"Missing draw matrix: {y_path}")
        if not w_path.exists():
            raise FileNotFoundError(f"Missing treatment mask: {w_path}")

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

        # Kansas manifest indices are 0-based end-exclusive. The treated row can
        # be anywhere — downstream code must use treatment_mask rather than
        # assume a bottom-right treated block.
        treated_rows = np.flatnonzero(w.any(axis=1))
        treated_cols = np.flatnonzero(w.any(axis=0))
        metadata = {
            "draw_id": draw_id,
            "draw_name": draw_name,
            "shape": {"n_rows": int(y.shape[0]), "n_cols": int(y.shape[1])},
            "n_treated_cells": int(w.sum()),
            "n_observed_cells": int((w == 0).sum()),
            "n_missing_cells": int(np.isnan(observed).sum()),
            "outcome": row.get("outcome"),
            "assignment": row.get("assignment"),
            "N0": int(row["N0"]),
            "T0": int(row["T0"]),
            "N1_realized": int(row["N1_realized"]),
            "T1_realized": int(row["T1_realized"]),
            "treated_row_start": int(row["treated_row_start"]),
            "treated_row_end": int(row["treated_row_end"]),
            "post_col_start": int(row["post_col_start"]),
            "post_col_end": int(row["post_col_end"]),
            "treated_row_idx": int(treated_rows[0]) if treated_rows.size == 1 else None,
            "treated_row_indices": treated_rows.tolist(),
            "post_col_indices": treated_cols.tolist(),
        }
        with (draw_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        benchmark_rows.append(metadata)

    pd.DataFrame(benchmark_rows).to_csv(
        args.output_dir / "benchmark_manifest.csv", index=False
    )
    print(f"Wrote {len(benchmark_rows)} Kansas benchmark draws to {args.output_dir}")


if __name__ == "__main__":
    main()
