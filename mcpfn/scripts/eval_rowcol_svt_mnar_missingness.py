#!/usr/bin/env python
"""Evaluate direct row/column SVT models on informative missingness patterns."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from fancyimpute import SoftImpute

from tabimpute.interface import ImputePFN
from tabimpute.prior.training_set_generation import MissingnessPrior

warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'.*",
    category=FutureWarning,
)


DEFAULT_OUT = Path("/root/tabular/mcpfn/runs/curriculum_eval/rowcol_svt_mnar_missingness")

BASE_PRIOR_CONFIG = {
    "p_missing": 0.2,
    "apply_feature_warping_prob": 0.0,
    "apply_quantization_prob": 0.0,
    "latent_rank_low": 10,
    "latent_rank_high": 10,
    "latent_spike_p": 0.3,
    "latent_slab_sigma": 2.0,
    "spatial_block_n_blocks": 16,
    "spatial_block_p_geom": 0.035,
    "censor_quantile": 0.2,
    "cluster_level_n_row_clusters": 8,
    "cluster_level_n_col_clusters": 8,
    "cluster_level_tau_r_std": 1.5,
    "cluster_level_tau_c_std": 1.5,
    "cluster_level_epsilon_std": 0.5,
}


@dataclass(frozen=True)
class DirectSpec:
    name: str
    checkpoint: Path
    chunk_size: int
    family: str
    prior_index: int | None = None


def parse_direct(value: str) -> DirectSpec:
    parts = value.split("|")
    if len(parts) not in {4, 5}:
        raise ValueError(
            "--extra-direct must be formatted as name|checkpoint|chunk_size|family[|prior_index]"
        )
    name, checkpoint, chunk_size, family = (part.strip() for part in parts[:4])
    prior_index = int(parts[4]) if len(parts) == 5 and parts[4].strip() else None
    path = Path(checkpoint).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    return DirectSpec(name, path, int(chunk_size), family, prior_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[300])
    parser.add_argument("--ranks", nargs="+", type=int, default=[10, 80])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--missingness-types",
        nargs="+",
        default=["mnar_latent_factor", "mnar_spatial_block", "mnar_censoring"],
    )
    parser.add_argument("--generator-type", default="latent_factor")
    parser.add_argument("--prior-overrides-json", type=str, default=None)
    parser.add_argument("--softimpute-max-iters", type=int, default=200)
    parser.add_argument("--softimpute-shrinkage", type=float, default=None)
    parser.add_argument("--skip-softimpute", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--extra-direct", action="append", type=parse_direct, default=[])
    parser.add_argument("--direct-prediction-stat", choices=("median", "mean"), default="mean")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def load_model_config(checkpoint: Path) -> dict:
    with (checkpoint.parent / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "model" in cfg and "embedding_size" in cfg["model"]:
        return dict(cfg["model"])
    if "config" in cfg:
        return dict(cfg["config"])
    raise ValueError(f"Could not find model config in {checkpoint.parent}")


def observed_col_stats(x_missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x_missing)
    counts = finite.sum(axis=0)
    safe = np.where(finite, x_missing, 0.0)
    means = safe.sum(axis=0) / np.maximum(counts, 1)
    centered = np.where(finite, x_missing - means, 0.0)
    stds = np.sqrt((centered * centered).sum(axis=0) / np.maximum(counts, 1))
    means = np.where(counts == 0, 0.0, means)
    stds = np.where((counts == 0) | (stds < 1e-12), 1.0, stds)
    return means.astype(np.float32), stds.astype(np.float32)


def score(target: np.ndarray, imputed: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    err = imputed[mask] - target[mask]
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.std(target[mask]))
    return {
        "rmse": rmse,
        "mae": mae,
        "nrmse": rmse / denom if denom > 1e-16 else float("nan"),
    }


def make_case(
    *,
    size: int,
    rank: int,
    seed: int,
    missingness_type: str,
    generator_type: str,
    overrides: dict[str, object] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    config = {
        **BASE_PRIOR_CONFIG,
        "num_rows_low": size,
        "num_rows_high": size,
        "num_cols_low": size,
        "num_cols_high": size,
        "latent_rank_low": rank,
        "latent_rank_high": rank,
    }
    if overrides:
        config.update(overrides)
    last_state = None
    for attempt in range(25):
        attempt_seed = seed + attempt * 100_003
        random.seed(attempt_seed)
        np.random.seed(attempt_seed)
        torch.manual_seed(attempt_seed)
        prior = MissingnessPrior(
            generator_type=generator_type,
            missingness_type=missingness_type,
            config=config,
            batch_size=1,
            device="cpu",
            verbose=False,
            entry_wise_features=False,
        )
        (x_missing_t, target_t, _, _, _), _ = prior.get_batch()
        x_missing = x_missing_t[0].detach().cpu().numpy().astype(np.float32)
        target = target_t[0].detach().cpu().numpy().astype(np.float32)
        mask = np.isnan(x_missing)
        observed_fraction = float(np.isfinite(x_missing).mean())
        missing_fraction = float(mask.mean())
        last_state = (x_missing, target, mask, observed_fraction, missing_fraction, prior)
        if bool(mask.any()) and 0.001 <= observed_fraction <= 0.999:
            break
    else:
        raise ValueError(
            f"{missingness_type} did not produce a usable mask for "
            f"size={size}, rank={rank}, seed={seed}"
        )
    x_missing, target, mask, observed_fraction, missing_fraction, prior = last_state
    meta = {
        "observed_fraction": observed_fraction,
        "missing_fraction": missing_fraction,
    }
    meta.update(
        {
            key: float(value)
            for key, value in getattr(prior, "last_batch_metrics", {}).items()
            if isinstance(value, (int, float, np.integer, np.floating))
            and np.isfinite(float(value))
        }
    )
    return target, x_missing, mask, meta


def softimpute_fill(
    x_missing: np.ndarray,
    *,
    max_iters: int,
    shrinkage: float | None,
) -> np.ndarray:
    means, stds = observed_col_stats(x_missing)
    x_norm = (x_missing - means) / (stds + 1e-16)
    kwargs = {
        "max_iters": max_iters,
        "convergence_threshold": 1e-5,
        "verbose": False,
    }
    if shrinkage is not None:
        kwargs["shrinkage_value"] = shrinkage
    imputed_norm = SoftImpute(**kwargs).fit_transform(x_norm)
    imputed = imputed_norm * (stds + 1e-16) + means
    out = x_missing.copy()
    missing = np.isnan(out)
    out[missing] = imputed[missing]
    return out


def build_imputer(spec: DirectSpec, args: argparse.Namespace) -> ImputePFN:
    return ImputePFN(
        device=args.device,
        entry_wise_features=False,
        checkpoint_path=str(spec.checkpoint),
        json_config={"config": load_model_config(spec.checkpoint)},
        model_family=spec.family,
        max_num_rows=spec.chunk_size,
        max_num_cols=spec.chunk_size,
        max_num_chunks=1,
        preprocessors=None,
        prediction_stat=args.direct_prediction_stat,
        direct_prior_index=spec.prior_index,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "method",
        "missingness_type",
        "rank",
        "size",
        "seed",
        "rmse",
        "rmse_sem",
        "mae",
        "nrmse",
        "observed_fraction",
        "missing_fraction",
        "n",
    ]
    fields = [key for key in preferred if key in fields] + [
        key for key in fields if key not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["method"]),
            str(row["missingness_type"]),
            int(row["rank"]),
            int(row["size"]),
        )
        groups.setdefault(key, []).append(row)
    out = []
    for (method, missingness_type, rank, size), group in sorted(groups.items()):
        rmses = np.asarray([float(row["rmse"]) for row in group], dtype=np.float64)
        maes = np.asarray([float(row["mae"]) for row in group], dtype=np.float64)
        nrmse = np.asarray([float(row["nrmse"]) for row in group], dtype=np.float64)
        observed = np.asarray([float(row["observed_fraction"]) for row in group], dtype=np.float64)
        out.append(
            {
                "method": method,
                "missingness_type": missingness_type,
                "rank": rank,
                "size": size,
                "rmse": float(rmses.mean()),
                "rmse_sem": float(rmses.std(ddof=1) / np.sqrt(len(rmses)))
                if len(rmses) > 1
                else 0.0,
                "mae": float(maes.mean()),
                "nrmse": float(nrmse.mean()),
                "observed_fraction": float(observed.mean()),
                "missing_fraction": float(1.0 - observed.mean()),
                "n": len(group),
            }
        )
    return out


def plot(summary: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    setups = sorted({(str(row["missingness_type"]), int(row["rank"])) for row in summary})
    if not setups:
        return
    fig, axes = plt.subplots(
        len(setups),
        1,
        figsize=(11, max(4, 3.0 * len(setups))),
        dpi=150,
        squeeze=False,
    )
    for ax, (missingness_type, rank) in zip(axes[:, 0], setups):
        subset = [
            row
            for row in summary
            if row["missingness_type"] == missingness_type and int(row["rank"]) == rank
        ]
        methods = sorted({str(row["method"]) for row in subset})
        for method in methods:
            group = sorted(
                [row for row in subset if row["method"] == method],
                key=lambda row: int(row["size"]),
            )
            ax.errorbar(
                [row["size"] for row in group],
                [row["rmse"] for row in group],
                yerr=[row["rmse_sem"] for row in group],
                marker="o",
                linewidth=2.0,
                capsize=3,
                label=method,
            )
        ax.set_title(f"{missingness_type}, rank {rank}")
        ax.set_xlabel("square matrix size")
        ax.set_ylabel("missing-entry RMSE")
        ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    overrides = json.loads(args.prior_overrides_json) if args.prior_overrides_json else None
    cases = {}
    for missingness_type in args.missingness_types:
        for rank in args.ranks:
            for size in args.sizes:
                for seed in args.seeds:
                    cases[(missingness_type, rank, size, seed)] = make_case(
                        size=size,
                        rank=rank,
                        seed=seed,
                        missingness_type=missingness_type,
                        generator_type=args.generator_type,
                        overrides=overrides,
                    )

    rows: list[dict[str, object]] = []
    for (missingness_type, rank, size, seed), (target, x_missing, mask, meta) in cases.items():
        if not args.skip_softimpute:
            imputed = softimpute_fill(
                x_missing,
                max_iters=args.softimpute_max_iters,
                shrinkage=args.softimpute_shrinkage,
            )
            rows.append(
                {
                    "method": "SoftImpute",
                    "missingness_type": missingness_type,
                    "rank": rank,
                    "size": size,
                    "seed": seed,
                    **score(target, imputed, mask),
                    **meta,
                }
            )

    if not args.skip_direct:
        for spec in args.extra_direct:
            imputer = build_imputer(spec, args)
            for (missingness_type, rank, size, seed), (target, x_missing, mask, meta) in cases.items():
                imputer.max_num_rows = spec.chunk_size
                imputer.max_num_cols = spec.chunk_size
                imputed = imputer.impute(x_missing.copy())
                rows.append(
                    {
                        "method": spec.name,
                        "missingness_type": missingness_type,
                        "rank": rank,
                        "size": size,
                        "seed": seed,
                        **score(target, imputed, mask),
                        **meta,
                    }
                )
            del imputer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize(rows)
    write_csv(args.output_prefix.with_suffix(".per_seed.csv"), rows)
    write_csv(args.output_prefix.with_suffix(".summary.csv"), summary)
    plot(summary, args.output_prefix.with_suffix(".png"))
    print("method\tmissingness\trank\tsize\trmse\tsem\tmae\tobserved\tn")
    for row in summary:
        print(
            f"{row['method']}\t{row['missingness_type']}\t{row['rank']}\t"
            f"{row['size']}\t{row['rmse']:.6g}\t{row['rmse_sem']:.3g}\t"
            f"{row['mae']:.6g}\t{row['observed_fraction']:.3f}\t{row['n']}"
        )
    print(f"Wrote {args.output_prefix.with_suffix('.summary.csv')}")


if __name__ == "__main__":
    main()
