#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gc
import importlib.resources
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tabimpute.interface import ImputePFN
from tabimpute.model.bar_distribution import FullSupportBarDistribution


@dataclass(frozen=True)
class Spec:
    name: str
    checkpoint: Path
    family: str


def parse_spec(value: str) -> Spec:
    parts = value.split("|")
    if len(parts) != 3:
        raise ValueError("--spec must be formatted as 'name|checkpoint|family'")
    name, checkpoint, family = (part.strip() for part in parts)
    path = Path(checkpoint).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    return Spec(name=name, checkpoint=path, family=family)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", action="append", required=True, type=parse_spec)
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 50, 75, 100, 150, 200, 300, 400, 500])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    parser.add_argument("--stats", nargs="+", choices=["median", "mean"], default=["median", "mean"])
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--missing-rate", type=float, default=0.2)
    parser.add_argument("--scale", choices=("rank", "sqrt_rank", "unit_std", "none"), default="rank")
    parser.add_argument("--generator", choices=("gaussian_factors",), default="gaussian_factors")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fallback-device", default="cpu")
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


def make_low_rank(size: int, rank: int, missing_rate: float, seed: int, scale: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def score(x_true: np.ndarray, pred_missing: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    err = pred_missing - x_true[mask]
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.std(x_true[mask]))
    return {
        "rmse": rmse,
        "mae": mae,
        "nrmse": rmse / denom if denom > 1e-16 else float("nan"),
    }


def build_imputer(spec: Spec, device: str, dtype: torch.dtype) -> ImputePFN:
    imputer = ImputePFN(
        device=device,
        entry_wise_features=False,
        checkpoint_path=str(spec.checkpoint),
        json_config={"config": load_model_config(spec.checkpoint)},
        model_family=spec.family,
        max_num_rows=None,
        max_num_cols=None,
        max_num_chunks=1,
        preprocessors=None,
        prediction_stat="median",
    )
    imputer.model.to(device=device, dtype=dtype)
    imputer.model.eval()
    borders = imputer.borders.to(device)
    imputer.bar_distribution = FullSupportBarDistribution(borders=borders)
    return imputer


@torch.no_grad()
def evaluate_spec(
    spec: Spec,
    *,
    device: str,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    imputer = build_imputer(spec, device=device, dtype=dtype)
    rows: list[dict[str, object]] = []
    for size in args.sizes:
        for seed in args.seeds:
            x_true, x_missing, mask = make_low_rank(
                size,
                args.rank,
                args.missing_rate,
                seed,
                args.scale,
            )
            means, stds = observed_col_stats(x_missing)
            x_norm = (x_missing - means) / (stds + 1e-16)
            x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device=device, dtype=dtype)
            missing = torch.isnan(x_tensor)
            logits = imputer.model(x_tensor, decode_mask=missing).float()
            for temperature in args.temperatures:
                scaled_logits = logits / float(temperature)
                preds_by_stat = {}
                if "median" in args.stats:
                    preds_by_stat["median"] = imputer.bar_distribution.median(scaled_logits)
                if "mean" in args.stats:
                    preds_by_stat["mean"] = imputer.bar_distribution.mean(scaled_logits)
                for stat, pred_norm in preds_by_stat.items():
                    batch_idx, row_idx, col_idx = missing.nonzero(as_tuple=True)
                    del batch_idx, row_idx
                    pred = pred_norm.detach().cpu().numpy()
                    pred = pred * (stds[col_idx.detach().cpu().numpy()] + 1e-16) + means[col_idx.detach().cpu().numpy()]
                    rows.append(
                        {
                            "method": spec.name,
                            "checkpoint": str(spec.checkpoint),
                            "family": spec.family,
                            "device": device,
                            "size": size,
                            "seed": seed,
                            "temperature": float(temperature),
                            "stat": stat,
                            **score(x_true, pred, mask),
                        }
                    )
            del logits, x_tensor, missing
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    del imputer
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["method"], row["size"], row["temperature"], row["stat"], row["device"])
        groups.setdefault(key, []).append(row)
    out = []
    for (method, size, temperature, stat, device), group in sorted(groups.items()):
        rmses = np.array([float(row["rmse"]) for row in group], dtype=np.float64)
        maes = np.array([float(row["mae"]) for row in group], dtype=np.float64)
        nrmses = np.array([float(row["nrmse"]) for row in group], dtype=np.float64)
        out.append(
            {
                "method": method,
                "size": size,
                "temperature": temperature,
                "stat": stat,
                "device": device,
                "rmse": float(rmses.mean()),
                "rmse_std": float(rmses.std(ddof=1)) if len(rmses) > 1 else 0.0,
                "rmse_sem": float(rmses.std(ddof=1) / np.sqrt(len(rmses))) if len(rmses) > 1 else 0.0,
                "mae": float(maes.mean()),
                "nrmse": float(nrmses.mean()),
                "n": len(group),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    all_rows: list[dict[str, object]] = []
    for spec in args.spec:
        try:
            print(f"Evaluating {spec.name} on {args.device}", flush=True)
            all_rows.extend(evaluate_spec(spec, device=args.device, dtype=dtype, args=args))
        except torch.cuda.OutOfMemoryError:
            if args.fallback_device == args.device:
                raise
            print(
                f"CUDA OOM for {spec.name}; retrying on {args.fallback_device}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            all_rows.extend(evaluate_spec(spec, device=args.fallback_device, dtype=torch.float32, args=args))
    summary_rows = summarize(all_rows)
    write_csv(args.output_prefix.with_suffix(".csv"), all_rows)
    write_csv(args.output_prefix.with_name(args.output_prefix.name + "_summary").with_suffix(".csv"), summary_rows)
    print(f"Wrote {args.output_prefix.with_suffix('.csv')}", flush=True)
    print(f"Wrote {args.output_prefix.with_name(args.output_prefix.name + '_summary').with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    main()
