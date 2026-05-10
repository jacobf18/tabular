#!/usr/bin/env python
"""Measure whether TabImpute attention gets close to uniform at larger sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tabimpute.interface import ImputePFN


def parse_size(size: str) -> tuple[int, int]:
    if "x" not in size.lower():
        raise argparse.ArgumentTypeError(f"Expected ROWSxCOLS, got {size}")
    rows, cols = size.lower().split("x", maxsplit=1)
    return int(rows), int(cols)


def make_low_rank_matrix(
    rows: int,
    cols: int,
    rank: int,
    missing_rate: float,
    seed: int,
) -> np.ndarray:
    if rank > min(rows, cols):
        raise ValueError(
            f"rank={rank} must be <= min(rows, cols)={min(rows, cols)} "
            "to keep rank fixed across the size sweep."
        )
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(rows, rank))
    right = rng.normal(size=(rank, cols))
    x = left @ right
    x += 0.05 * rng.normal(size=x.shape)
    mask = rng.random(size=x.shape) < missing_rate
    x[mask] = np.nan
    return x.astype(np.float32)


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def iter_rows(size: str, diagnostics: list[dict[str, object]]):
    for layer in diagnostics:
        for axis in ("features", "datapoints"):
            stats = layer[axis]["stats"]
            if stats is None:
                continue
            yield {
                "size": size,
                "layer": layer["layer"],
                "axis": axis,
                **stats,
            }


def print_table(rows: list[dict[str, object]]) -> None:
    headers = [
        "size",
        "layer",
        "axis",
        "seq_len",
        "norm_entropy",
        "eff_frac",
        "max_prob",
        "top5_mass",
        "kl_uniform",
        "logit_std",
    ]
    print("\t".join(headers))
    for row in rows:
        values = [
            row["size"],
            str(row["layer"]),
            row["axis"],
            f"{row['seq_len']:.0f}",
            f"{row['normalized_entropy_mean']:.4f}",
            f"{row['effective_key_fraction_mean']:.4f}",
            f"{row['max_prob_mean']:.4f}",
            f"{row['top5_mass_mean']:.4f}",
            f"{row['kl_to_uniform_mean']:.4f}",
            f"{row['logit_std_mean']:.4f}",
        ]
        print("\t".join(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--json-config", required=True)
    parser.add_argument("--model-family", default="new_stable")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=parse_size,
        default=[(50, 50), (100, 100), (200, 200)],
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--missing-rate", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--store-weights",
        action="store_true",
        help="Also keep full token attention tensors; can be very memory intensive.",
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

    rows = []
    for idx, (num_rows, num_cols) in enumerate(args.sizes):
        label = f"{num_rows}x{num_cols}"
        x = make_low_rank_matrix(
            num_rows,
            num_cols,
            rank=args.rank,
            missing_rate=args.missing_rate,
            seed=args.seed + idx,
        )
        diagnostics = imputer.diagnose_attention(x, store_weights=args.store_weights)
        rows.extend(iter_rows(label, diagnostics))

    print_table(rows)


if __name__ == "__main__":
    main()
