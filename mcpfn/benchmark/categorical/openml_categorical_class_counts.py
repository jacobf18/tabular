#!/usr/bin/env python
"""Report the number of distinct classes in each categorical column for OpenML categorical benchmark datasets.

Uses the same categorical-column heuristic as ``plot_accuracy.py`` (object/category dtype or
non-numeric). Class counts use unique **observed** values (NaN excluded).

Run from ``mcpfn/benchmark`` (or anywhere; paths are resolved relative to this file):

    python openml_categorical_class_counts.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _is_categorical_column(df: pd.DataFrame, col_idx: int) -> bool:
    col = df.iloc[:, col_idx]
    return df.dtypes.iloc[col_idx] in ["object", "category"] or not pd.api.types.is_numeric_dtype(
        col
    )


def _n_classes_in_column(df: pd.DataFrame, col_idx: int) -> int:
    s = pd.Series(df.iloc[:, col_idx]).astype(object)
    return int(s.dropna().nunique())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Path to openml_categorical folder (default: <this_dir>/datasets/openml_categorical)",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    root = args.root if args.root is not None else here / "datasets" / "openml_categorical"
    if not root.is_dir():
        raise SystemExit(f"Directory not found: {root}")

    dataset_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not dataset_dirs:
        raise SystemExit(f"No dataset subdirectories under {root}")

    grand_total_cols = 0
    for dpath in dataset_dirs:
        pkl = dpath / "dataframe.pkl"
        if not pkl.is_file():
            print(f"== {dpath.name} ==\n  (missing dataframe.pkl)\n")
            continue

        df = pd.read_pickle(pkl)
        print(f"== {dpath.name} ({df.shape[0]} rows × {df.shape[1]} columns) ==")

        found = False
        for j in range(df.shape[1]):
            if not _is_categorical_column(df, j):
                continue
            found = True
            grand_total_cols += 1
            name = df.columns[j]
            n_cls = _n_classes_in_column(df, j)
            print(f"  [{j:3d}] {name!s}: {n_cls} classes")

        if not found:
            print("  (no categorical columns under this heuristic)")
        print()

    print(f"Total categorical columns (across datasets): {grand_total_cols}")


if __name__ == "__main__":
    main()
