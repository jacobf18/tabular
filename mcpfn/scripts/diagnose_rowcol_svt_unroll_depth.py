#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from tabimpute.interface import ImputePFN
from tabimpute.model.bar_distribution import FullSupportBarDistribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--family", default="rowcol_svt_hybrid")
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--missing-rate", type=float, default=0.2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", nargs="+", type=int, default=[0, 1, 2, 3, 4, 6, 8, 12, 20])
    parser.add_argument("--max-ranks", nargs="+", type=int, default=[10, 20, 50])
    parser.add_argument("--tau-fractions", nargs="+", type=float, default=[0.0, 0.002, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--dampings", nargs="+", type=float, default=[1.0])
    parser.add_argument("--scale", choices=("rank", "sqrt_rank", "unit_std", "none"), default="rank")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def load_model_config(checkpoint: Path) -> dict:
    with (checkpoint.parent / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "model" in cfg and "embedding_size" in cfg["model"]:
        return dict(cfg["model"])
    if "config" in cfg:
        return dict(cfg["config"])
    if "embedding_size" in cfg:
        return dict(cfg)
    raise ValueError(f"Could not find model config in {checkpoint.parent}")


def make_low_rank(
    size: int,
    rank: int,
    missing_rate: float,
    seed: int,
    scale: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(size, rank))
    right = rng.normal(size=(rank, size))
    x_true = left @ right
    if scale == "sqrt_rank":
        x_true = x_true / np.sqrt(rank)
    elif scale == "rank":
        x_true = x_true / rank
    elif scale == "unit_std":
        x_true = (x_true - x_true.mean()) / (x_true.std() + 1e-16)
    elif scale != "none":
        raise ValueError(scale)
    mask = rng.random(size=x_true.shape) < missing_rate
    x_missing = x_true.astype(np.float32, copy=True)
    x_missing[mask] = np.nan
    return x_true.astype(np.float32), x_missing, mask


def observed_col_stats(x_missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(x_missing, axis=0)
    stds = np.nanstd(x_missing, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    stds = np.where(np.isnan(stds), 1.0, stds)
    return means.astype(np.float32), stds.astype(np.float32)


def rmse_mae(x_true_missing: np.ndarray, pred_missing: np.ndarray) -> tuple[float, float]:
    err = pred_missing - x_true_missing
    return float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err)))


def inverse_sigmoid(value: float) -> float:
    value = min(max(value, 1e-8), 1.0 - 1e-8)
    return math.log(value / (1.0 - value))


def set_threshold_fraction(model: torch.nn.Module, value: float) -> None:
    min_value = float(getattr(model, "svt_threshold_fraction_min", 0.0))
    if value < min_value:
        model.svt_threshold_fraction_min = float(value)
        min_value = float(value)
    unconstrained = inverse_sigmoid(max(value - min_value, 1e-8))
    model.svt_threshold_unconstrained.data.fill_(unconstrained)


def build_imputer(
    checkpoint: Path,
    family: str,
    device: str,
    dtype: torch.dtype,
) -> ImputePFN:
    imputer = ImputePFN(
        device=device,
        entry_wise_features=False,
        checkpoint_path=str(checkpoint),
        json_config={"config": load_model_config(checkpoint)},
        model_family=family,
        max_num_rows=None,
        max_num_cols=None,
        max_num_chunks=1,
        preprocessors=None,
        prediction_stat="median",
    )
    imputer.model.to(device=device, dtype=dtype)
    imputer.model.eval()
    imputer.bar_distribution = FullSupportBarDistribution(
        borders=imputer.borders.to(device)
    )
    return imputer


@torch.no_grad()
def evaluate_grid(
    imputer: ImputePFN,
    x_true: np.ndarray,
    x_missing: np.ndarray,
    mask_np: np.ndarray,
    *,
    steps: int,
    max_rank: int,
    tau_fraction: float,
    damping: float,
    device: str,
    dtype: torch.dtype,
) -> list[dict[str, object]]:
    model = imputer.model
    model.svt_steps = int(steps)
    model.svt_max_rank = int(max_rank)
    model.svt_update_damping = float(damping)
    set_threshold_fraction(model, float(tau_fraction))

    means, stds = observed_col_stats(x_missing)
    x_norm = (x_missing - means) / (stds + 1e-16)
    x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device=device, dtype=dtype)
    missing = torch.isnan(x_tensor)
    batch_idx, row_idx, col_idx = missing.nonzero(as_tuple=True)
    del batch_idx

    row_state, col_state = model.encode_latents(x_tensor)
    initial_grid = model._initial_grid(row_state, col_state)
    point_grid = model._svt_refine(x_tensor, initial_grid)
    point_norm = point_grid[0, row_idx, col_idx].float()
    logits = model._logits_from_point(point_norm)
    median_norm = imputer.bar_distribution.median(logits.float())
    mean_norm = imputer.bar_distribution.mean(logits.float())

    true_missing = x_true[mask_np]
    cols_np = col_idx.detach().cpu().numpy()
    scale = stds[cols_np] + 1e-16
    shift = means[cols_np]
    preds = {
        "point": point_norm.detach().cpu().numpy() * scale + shift,
        "mean": mean_norm.detach().cpu().numpy() * scale + shift,
        "median": median_norm.detach().cpu().numpy() * scale + shift,
    }
    rows = []
    for stat, pred in preds.items():
        rmse, mae = rmse_mae(true_missing, pred)
        rows.append(
            {
                "steps": steps,
                "max_rank": max_rank,
                "tau_fraction": tau_fraction,
                "damping": damping,
                "stat": stat,
                "rmse": rmse,
                "mae": mae,
                "n_missing": int(mask_np.sum()),
            }
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["steps"],
            row["max_rank"],
            row["tau_fraction"],
            row["damping"],
            row["stat"],
        )
        groups.setdefault(key, []).append(row)
    summary = []
    for key, group in sorted(groups.items()):
        rmses = np.array([float(row["rmse"]) for row in group], dtype=np.float64)
        maes = np.array([float(row["mae"]) for row in group], dtype=np.float64)
        summary.append(
            {
                "steps": key[0],
                "max_rank": key[1],
                "tau_fraction": key[2],
                "damping": key[3],
                "stat": key[4],
                "rmse": float(rmses.mean()),
                "rmse_sem": float(rmses.std(ddof=1) / np.sqrt(len(rmses)))
                if len(rmses) > 1
                else 0.0,
                "mae": float(maes.mean()),
                "n": len(group),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    imputer = build_imputer(
        args.checkpoint,
        family=args.family,
        device=args.device,
        dtype=dtype,
    )

    rows = []
    for seed in args.seeds:
        x_true, x_missing, mask = make_low_rank(
            args.size,
            args.rank,
            args.missing_rate,
            seed,
            args.scale,
        )
        for steps in args.steps:
            for max_rank in args.max_ranks:
                for tau_fraction in args.tau_fractions:
                    for damping in args.dampings:
                        for row in evaluate_grid(
                            imputer,
                            x_true,
                            x_missing,
                            mask,
                            steps=steps,
                            max_rank=max_rank,
                            tau_fraction=tau_fraction,
                            damping=damping,
                            device=args.device,
                            dtype=dtype,
                        ):
                            rows.append({"seed": seed, **row})
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary = summarize(rows)
    write_csv(args.output_prefix.with_suffix(".per_seed.csv"), rows)
    write_csv(args.output_prefix.with_suffix(".summary.csv"), summary)

    print("steps\tmax_rank\ttau\tstat\trmse\tsem\tmae\tn")
    for row in sorted(summary, key=lambda x: float(x["rmse"]))[:30]:
        print(
            f"{row['steps']}\t{row['max_rank']}\t{row['tau_fraction']}\t"
            f"{row['stat']}\t{row['rmse']:.8f}\t{row['rmse_sem']:.5g}\t"
            f"{row['mae']:.8f}\t{row['n']}"
        )
    print(f"Wrote {args.output_prefix.with_suffix('.summary.csv')}")


if __name__ == "__main__":
    main()
