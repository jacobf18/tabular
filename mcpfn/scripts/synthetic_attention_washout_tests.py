#!/usr/bin/env python
"""Synthetic attention washout tests for direct TabImpute models.

The test sweeps matrix sizes and synthetic data mechanisms, then reports
per-layer attention statistics from the row and column axial attentions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch

from tabimpute.interface import ImputePFN


DEFAULT_SIZES = [
    (10, 10),
    (20, 20),
    (30, 30),
    (40, 40),
    (50, 50),
    (100, 100),
    (200, 200),
]

STAT_KEYS = [
    "normalized_entropy_mean",
    "effective_key_fraction_mean",
    "max_prob_mean",
    "top5_mass_mean",
    "kl_to_uniform_mean",
    "logit_std_mean",
    "logit_range_mean",
]


def parse_size(size: str) -> tuple[int, int]:
    if "x" not in size.lower():
        raise argparse.ArgumentTypeError(f"Expected ROWSxCOLS, got {size}")
    rows, cols = size.lower().split("x", maxsplit=1)
    return int(rows), int(cols)


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def mask_values(x: np.ndarray, missing_rate: float, rng: np.random.Generator) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    x[rng.random(size=x.shape) < missing_rate] = np.nan
    return x


def make_low_rank(
    rows: int,
    cols: int,
    rng: np.random.Generator,
    missing_rate: float,
    rank: int,
) -> np.ndarray:
    if rank > min(rows, cols):
        raise ValueError(
            f"rank={rank} must be <= min(rows, cols)={min(rows, cols)} "
            "to keep rank fixed across the size sweep."
        )
    left = rng.normal(size=(rows, rank))
    right = rng.normal(size=(rank, cols))
    x = left @ right
    x += 0.05 * rng.normal(size=x.shape)
    return mask_values(x, missing_rate, rng)


def make_row_col_additive(
    rows: int,
    cols: int,
    rng: np.random.Generator,
    missing_rate: float,
    rank: int,
) -> np.ndarray:
    del rank
    row_signal = rng.normal(size=(rows, 1))
    col_signal = rng.normal(size=(1, cols))
    x = row_signal + col_signal + 0.05 * rng.normal(size=(rows, cols))
    return mask_values(x, missing_rate, rng)


def make_block(
    rows: int,
    cols: int,
    rng: np.random.Generator,
    missing_rate: float,
    rank: int,
) -> np.ndarray:
    del rank
    row_bins = np.array_split(np.arange(rows), min(4, rows))
    col_bins = np.array_split(np.arange(cols), min(4, cols))
    levels = rng.normal(size=(len(row_bins), len(col_bins)))
    x = np.zeros((rows, cols), dtype=np.float32)
    for row_idx, row_bin in enumerate(row_bins):
        for col_idx, col_bin in enumerate(col_bins):
            x[np.ix_(row_bin, col_bin)] = levels[row_idx, col_idx]
    x += 0.05 * rng.normal(size=x.shape)
    return mask_values(x, missing_rate, rng)


def make_iid_noise(
    rows: int,
    cols: int,
    rng: np.random.Generator,
    missing_rate: float,
    rank: int,
) -> np.ndarray:
    del rank
    return mask_values(rng.normal(size=(rows, cols)), missing_rate, rng)


GENERATORS = {
    "low_rank": make_low_rank,
    "row_col_additive": make_row_col_additive,
    "block": make_block,
    "iid_noise": make_iid_noise,
}


def flatten_diagnostics(
    size_label: str,
    pattern: str,
    seed: int,
    diagnostics: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for layer in diagnostics:
        for axis in ("features", "datapoints"):
            stats = layer[axis]["stats"]
            if stats is None:
                continue
            rows.append(
                {
                    "size": size_label,
                    "pattern": pattern,
                    "seed": seed,
                    "layer": layer["layer"],
                    "axis": axis,
                    **stats,
                }
            )
    return rows


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["size"], row["pattern"], row["layer"], row["axis"])
        grouped.setdefault(key, []).append(row)

    out = []
    for (size, pattern, layer, axis), group in grouped.items():
        item = {
            "size": size,
            "pattern": pattern,
            "layer": layer,
            "axis": axis,
            "n_seeds": len(group),
            "seq_len": group[0]["seq_len"],
        }
        for stat_key in STAT_KEYS:
            values = [float(row[stat_key]) for row in group]
            item[f"{stat_key}_mean"] = mean(values)
            item[f"{stat_key}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out.append(item)

    return sorted(out, key=lambda x: (str(x["pattern"]), str(x["size"]), x["layer"], x["axis"]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    columns = [
        "pattern",
        "size",
        "layer",
        "axis",
        "seq_len",
        "normalized_entropy_mean_mean",
        "effective_key_fraction_mean_mean",
        "max_prob_mean_mean",
        "top5_mass_mean_mean",
        "logit_std_mean_mean",
    ]
    print("\t".join(columns))
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        print("\t".join(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--json-config", required=True)
    parser.add_argument("--model-family", default="new_stable")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sizes", nargs="+", type=parse_size, default=DEFAULT_SIZES)
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=sorted(GENERATORS),
        default=["low_rank", "row_col_additive", "block", "iid_noise"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--missing-rate", type=float, default=0.4)
    parser.add_argument("--raw-csv", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument(
        "--store-weights",
        action="store_true",
        help="Also store full attention tensors in memory. The CSV only records stats.",
    )
    args = parser.parse_args()

    min_swept_dim = min(min(rows, cols) for rows, cols in args.sizes)
    if args.rank > min_swept_dim:
        raise ValueError(
            f"--rank {args.rank} is larger than the smallest swept dimension "
            f"({min_swept_dim}). Choose rank <= {min_swept_dim} so rank stays fixed."
        )

    imputer = ImputePFN(
        device=args.device,
        entry_wise_features=False,
        checkpoint_path=args.checkpoint_path,
        json_config=load_config(args.json_config),
        model_family=args.model_family,
    )

    raw_rows = []
    for pattern in args.patterns:
        generator = GENERATORS[pattern]
        for rows, cols in args.sizes:
            size_label = f"{rows}x{cols}"
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                x = generator(rows, cols, rng, args.missing_rate, args.rank)
                diagnostics = imputer.diagnose_attention(
                    x,
                    store_weights=args.store_weights,
                )
                raw_rows.extend(flatten_diagnostics(size_label, pattern, seed, diagnostics))

    summary_rows = aggregate_rows(raw_rows)
    print_summary(summary_rows)

    if args.raw_csv is not None:
        write_csv(args.raw_csv, raw_rows)
    if args.summary_csv is not None:
        write_csv(args.summary_csv, summary_rows)


if __name__ == "__main__":
    main()
