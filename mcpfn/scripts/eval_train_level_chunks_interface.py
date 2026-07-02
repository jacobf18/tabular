#!/usr/bin/env python
"""Interface-level train-chunk synthetic evaluation.

This reproduces the older fixed-rank scaling plot more closely than the
training-loss-style evaluators:

* generate clean fixed-rank matrices with no additive noise;
* mask entries once and pass the same incomplete matrix to every method;
* call TabImpute through ImputePFN / CompressedCrossImputePFN only;
* apply the same observed-column normalization to SoftImpute that the
  TabImpute interfaces apply internally.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from fancyimpute import SoftImpute

from tabimpute.interface import CompressedCrossImputePFN, ImputePFN
from tabimpute.prepreocess import (
    RandomColumnPermutation,
    RandomRowColumnPermutation,
    RandomRowPermutation,
)

warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'.*",
    category=FutureWarning,
)


WORKDIR = Path("/root/tabular/mcpfn/src/tabimpute/workdir")
DEFAULT_OUT = Path("/root/tabular/mcpfn/runs/curriculum_eval")


@dataclass(frozen=True)
class DirectSpec:
    name: str
    checkpoint: Path
    chunk_size: int
    model_family: str


@dataclass(frozen=True)
class CompressedSpec:
    name: str
    checkpoint: Path
    chunk_size: int
    model_family: str
    min_size: int | None = None


DIRECT_SPECS = [
    DirectSpec(
        name="TabImpute 50x50 chunks",
        checkpoint=WORKDIR
        / "tabimpute-new-stable_v2-rank_10_10-50x50"
        / "checkpoint_30000.pth",
        chunk_size=50,
        model_family="new_stable",
    ),
    DirectSpec(
        name="TabImpute 100x100 chunks",
        checkpoint=WORKDIR
        / "tabimpute-new-stable_v2-rank_10_10-100x100-from_50x50_step30000"
        / "checkpoint_45000.pth",
        chunk_size=100,
        model_family="new_stable",
    ),
    DirectSpec(
        name="TabImpute 200x200 chunks",
        checkpoint=WORKDIR
        / "tabimpute-new-stable-memory-efficient_v2-rank_10_10-200x200-from_100x100_step45000"
        / "checkpoint_60000.pth",
        chunk_size=200,
        model_family="new_stable_memory_efficient",
    ),
]


COMPRESSED_SPECS = [
    CompressedSpec(
        name="Compressed 200 K20 chunks",
        checkpoint=WORKDIR
        / "tabimpute-comp-rank10-200x200-100x100chunks-random-k20-100k-from-100comp"
        / "checkpoint_100000.pth",
        chunk_size=100,
        model_family="stable",
    ),
    CompressedSpec(
        name="Compressed random50-300 K20 chunks",
        checkpoint=WORKDIR
        / "tabimpute-comp-rank10-random50to300-randomfrac10to50-min10-k20-from-200comp-v4"
        / "checkpoint_100000.pth",
        chunk_size=100,
        model_family="stable",
    ),
    CompressedSpec(
        name="Compressed 200to300 K20 fixed100",
        checkpoint=WORKDIR
        / "tabimpute-comp-rank10-200to300-100x100chunks-random-k20-30k-from-200comp"
        / "checkpoint_30000.pth",
        chunk_size=100,
        model_family="stable",
        min_size=150,
    ),
    CompressedSpec(
        name="Compressed 200to300 K20 fixed100 lr2e-5",
        checkpoint=WORKDIR
        / "tabimpute-comp-rank10-200to300-100x100chunks-random-k20-30k-from-200comp-lr2e5-min5e6"
        / "checkpoint_30000.pth",
        chunk_size=100,
        model_family="stable",
        min_size=150,
    ),
    CompressedSpec(
        name="Compressed pure RaBEL K20 curriculum",
        checkpoint=WORKDIR
        / "tabimpute-compressed-pure-rabel64-rank10-k20-curriculum-160k"
        / "best_checkpoint.pth",
        chunk_size=100,
        model_family="pure_rabel_memory_efficient",
    ),
]


def load_config(checkpoint: Path) -> dict:
    with (checkpoint.parent / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def model_config(checkpoint: Path) -> dict:
    cfg = load_config(checkpoint)
    if "model" in cfg and "embedding_size" in cfg["model"]:
        return dict(cfg["model"])
    if "config" in cfg:
        return dict(cfg["config"])
    raise ValueError(f"Could not find direct model config in {checkpoint.parent}")


def make_low_rank(
    size: int,
    rank: int,
    missing_rate: float,
    seed: int,
    scale: str,
    generator: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rank > size:
        raise ValueError(f"rank={rank} must be <= size={size}")
    rng = np.random.default_rng(seed)
    if generator == "orthonormal_spectrum":
        left, left_r = np.linalg.qr(rng.standard_normal((size, rank)), mode="reduced")
        right, right_r = np.linalg.qr(rng.standard_normal((size, rank)), mode="reduced")
        left_signs = np.sign(np.diag(left_r))
        right_signs = np.sign(np.diag(right_r))
        left_signs[left_signs == 0.0] = 1.0
        right_signs[right_signs == 0.0] = 1.0
        left = left * left_signs
        right = right * right_signs
        singular_values = np.ones(rank, dtype=np.float64)
        x_true = (left * singular_values) @ right.T
    elif generator == "gaussian_factors":
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
            raise ValueError(f"Unknown scale {scale!r}")
    else:
        raise ValueError(f"Unknown generator {generator!r}")

    mask = rng.random(size=x_true.shape) < missing_rate
    x_missing = x_true.astype(np.float32, copy=True)
    x_missing[mask] = np.nan
    return x_true.astype(np.float32), x_missing, mask


def observed_col_stats(x_missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(x_missing, axis=0)
    stds = np.nanstd(x_missing, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    stds = np.where(np.isnan(stds), 1.0, stds)
    return means, stds


def score(x_true: np.ndarray, x_imputed: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    err = x_imputed[mask] - x_true[mask]
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.std(x_true[mask]))
    return {
        "rmse": rmse,
        "mae": mae,
        "nrmse": rmse / denom if denom > 1e-16 else float("nan"),
    }


def softimpute_observed_normalized(
    x_missing: np.ndarray,
    max_iters: int,
    shrinkage_value: float | None,
) -> np.ndarray:
    means, stds = observed_col_stats(x_missing)
    x_norm = (x_missing - means) / (stds + 1e-16)
    kwargs = {
        "max_iters": max_iters,
        "convergence_threshold": 1e-5,
        "verbose": False,
    }
    if shrinkage_value is not None:
        kwargs["shrinkage_value"] = shrinkage_value
    imputed_norm = SoftImpute(**kwargs).fit_transform(x_norm)
    imputed = imputed_norm * (stds + 1e-16) + means
    out = x_missing.copy()
    missing = np.isnan(out)
    out[missing] = imputed[missing]
    return out


def eval_softimpute(
    sizes: list[int],
    seeds: list[int],
    rank: int,
    missing_rate: float,
    scale: str,
    generator: str,
    max_iters: int,
    max_rank: int | None,
    shrinkage_value: float | None,
    normalize_softimpute: bool,
) -> list[dict[str, object]]:
    rows = []
    for size in sizes:
        for seed in seeds:
            x_true, x_missing, mask = make_low_rank(
                size, rank, missing_rate, seed, scale, generator
            )
            if normalize_softimpute:
                x_imputed = softimpute_observed_normalized(
                    x_missing,
                    max_iters=max_iters,
                    shrinkage_value=shrinkage_value,
                )
            else:
                kwargs = {
                    "max_rank": max_rank,
                    "max_iters": max_iters,
                    "convergence_threshold": 1e-5,
                    "init_fill_method": "zero",
                    "verbose": False,
                }
                if shrinkage_value is not None:
                    kwargs["shrinkage_value"] = shrinkage_value
                x_imputed = SoftImpute(**kwargs).fit_transform(
                    x_missing.astype(np.float64, copy=True)
                )
            rows.append(
                {
                    "method": "SoftImpute",
                    "size": size,
                    "seed": seed,
                    **score(x_true, x_imputed, mask),
                }
            )
    return rows


def eval_direct(
    spec: DirectSpec,
    sizes: list[int],
    seeds: list[int],
    rank: int,
    missing_rate: float,
    scale: str,
    generator: str,
    device: str,
    use_old_preprocessors: bool,
    dynamic_half_until_size: int | None = None,
    dynamic_fixed_chunk_size: int | None = None,
) -> list[dict[str, object]]:
    preprocessors = None
    if use_old_preprocessors:
        preprocessors = [
            RandomRowPermutation(),
            RandomColumnPermutation(),
            RandomRowColumnPermutation(),
            RandomRowColumnPermutation(),
        ]
    imputer = ImputePFN(
        device=device,
        entry_wise_features=False,
        checkpoint_path=str(spec.checkpoint),
        json_config={"config": model_config(spec.checkpoint)},
        model_family=spec.model_family,
        max_num_rows=spec.chunk_size,
        max_num_cols=spec.chunk_size,
        max_num_chunks=1,
        preprocessors=preprocessors,
    )
    rows = []
    for size in sizes:
        chunk_size = chunk_size_for_eval(
            size=size,
            default_chunk_size=spec.chunk_size,
            half_until_size=dynamic_half_until_size,
            fixed_chunk_size=dynamic_fixed_chunk_size,
        )
        imputer.max_num_rows = chunk_size
        imputer.max_num_cols = chunk_size
        for seed in seeds:
            x_true, x_missing, mask = make_low_rank(
                size, rank, missing_rate, seed, scale, generator
            )
            x_imputed = imputer.impute(x_missing.copy())
            rows.append(
                {
                    "method": spec.name,
                    "size": size,
                    "seed": seed,
                    "chunk_size": chunk_size,
                    **score(x_true, x_imputed, mask),
                }
            )
    del imputer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def eval_compressed(
    spec: CompressedSpec,
    sizes: list[int],
    seeds: list[int],
    rank: int,
    missing_rate: float,
    scale: str,
    generator: str,
    device: str,
    dynamic_half_chunk_cap: int | None = None,
    dynamic_half_until_size: int | None = None,
    dynamic_fixed_chunk_size: int | None = None,
) -> list[dict[str, object]]:
    cfg = load_config(spec.checkpoint)
    predictor_checkpoint = Path(cfg["predictor_checkpoint"])
    model_cfg = dict(cfg["model"])
    initial_chunk_size = chunk_size_for_eval(
        size=max(sizes),
        default_chunk_size=spec.chunk_size,
        half_until_size=dynamic_half_until_size,
        fixed_chunk_size=dynamic_fixed_chunk_size,
        half_cap=dynamic_half_chunk_cap,
    )
    imputer = CompressedCrossImputePFN(
        predictor_checkpoint_path=str(predictor_checkpoint),
        compressed_checkpoint_path=str(spec.checkpoint),
        predictor_config=dict(model_cfg["predictor"]),
        compressor_config=dict(model_cfg["compressor"]),
        chunk_rows=initial_chunk_size,
        chunk_cols=initial_chunk_size,
        k_row=int(model_cfg["k_row"]),
        k_col=int(model_cfg["k_col"]),
        device=device,
        model_family=spec.model_family,
    )
    rows = []
    for size in sizes:
        for seed in seeds:
            x_true, x_missing, mask = make_low_rank(
                size, rank, missing_rate, seed, scale, generator
            )
            chunk_size = chunk_size_for_eval(
                size=size,
                default_chunk_size=spec.chunk_size,
                half_until_size=dynamic_half_until_size,
                fixed_chunk_size=dynamic_fixed_chunk_size,
                half_cap=dynamic_half_chunk_cap,
            )
            imputer.chunk_rows = chunk_size
            imputer.chunk_cols = chunk_size
            x_imputed = imputer.impute(x_missing.copy())
            rows.append(
                {
                    "method": spec.name,
                    "size": size,
                    "seed": seed,
                    "chunk_size": chunk_size,
                    **score(x_true, x_imputed, mask),
                }
            )
    del imputer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def chunk_size_for_eval(
    *,
    size: int,
    default_chunk_size: int,
    half_until_size: int | None = None,
    fixed_chunk_size: int | None = None,
    half_cap: int | None = None,
) -> int:
    if half_until_size is not None:
        if fixed_chunk_size is None:
            raise ValueError("--dynamic-fixed-chunk-size is required with --dynamic-half-until-size")
        if size < half_until_size:
            return max(1, math.ceil(size / 2))
        return fixed_chunk_size
    if half_cap is not None:
        return min(math.ceil(size / 2), half_cap)
    return default_chunk_size


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "method",
        "size",
        "seed",
        "chunk_size",
        "rmse",
        "rmse_std",
        "rmse_sem",
        "mae",
        "nrmse",
        "n",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    keys = sorted({(str(row["method"]), int(row["size"])) for row in rows})
    for method, size in keys:
        group = [row for row in rows if row["method"] == method and row["size"] == size]
        rmses = np.asarray([float(row["rmse"]) for row in group])
        maes = np.asarray([float(row["mae"]) for row in group])
        nrmse = np.asarray([float(row["nrmse"]) for row in group])
        out.append(
            {
                "method": method,
                "size": size,
                "rmse": float(rmses.mean()),
                "rmse_std": float(rmses.std(ddof=1)) if len(rmses) > 1 else 0.0,
                "rmse_sem": float(rmses.std(ddof=1) / np.sqrt(len(rmses)))
                if len(rmses) > 1
                else 0.0,
                "mae": float(maes.mean()),
                "nrmse": float(nrmse.mean()),
                "n": len(group),
            }
        )
    return out


def plot(summary_rows: list[dict[str, object]], path: Path, include_compressed: bool) -> None:
    methods = [
        "SoftImpute",
        "TabImpute 100x100 chunks",
        "TabImpute 200x200 chunks",
        "TabImpute 50x50 chunks",
    ]
    if include_compressed:
        methods.extend([spec.name for spec in COMPRESSED_SPECS])

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    for method in methods:
        group = [row for row in summary_rows if row["method"] == method]
        if not group:
            continue
        group = sorted(group, key=lambda row: row["size"])
        xs = [row["size"] for row in group]
        ys = [row["rmse"] for row in group]
        yerr = [row["rmse_sem"] for row in group]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=2.5, capsize=4, label=method)

    ax.set_title("Fixed rank 10: TabImpute train-level chunks vs SoftImpute")
    ax.set_xlabel("square matrix size")
    ax.set_ylabel("missing-entry RMSE")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 50, 75, 100, 150, 200, 300, 400, 500])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--missing-rate", type=float, default=0.5)
    parser.add_argument("--generator", choices=["orthonormal_spectrum", "gaussian_factors"], default="orthonormal_spectrum")
    parser.add_argument("--scale", choices=["none", "sqrt_rank", "rank", "unit_std"], default="rank")
    parser.add_argument("--softimpute-max-iters", type=int, default=200)
    parser.add_argument("--softimpute-max-rank", type=int, default=50)
    parser.add_argument("--softimpute-shrinkage", type=float, default=None)
    parser.add_argument("--normalize-softimpute", action="store_true")
    parser.add_argument("--no-old-preprocessors", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include-compressed", action="store_true")
    parser.add_argument("--skip-softimpute", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--direct-names", nargs="*", default=None)
    parser.add_argument("--compressed-names", nargs="*", default=None)
    parser.add_argument(
        "--compressed-dynamic-half-chunk-cap",
        type=int,
        default=None,
        help=(
            "For compressed models, use chunk_size=min(ceil(size/2), cap) "
            "instead of the spec's fixed chunk size."
        ),
    )
    parser.add_argument(
        "--dynamic-half-until-size",
        type=int,
        default=None,
        help=(
            "For chunking models, use ceil(size/2) below this matrix size and "
            "--dynamic-fixed-chunk-size at this size and above."
        ),
    )
    parser.add_argument(
        "--dynamic-fixed-chunk-size",
        type=int,
        default=None,
        help="Fixed chunk size used at/above --dynamic-half-until-size.",
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUT / "interface_train_level_chunks_clean_rank10")
    args = parser.parse_args()

    all_rows: list[dict[str, object]] = []
    if not args.skip_softimpute:
        all_rows.extend(
            eval_softimpute(
                sizes=args.sizes,
                seeds=args.seeds,
                rank=args.rank,
                missing_rate=args.missing_rate,
                scale=args.scale,
                generator=args.generator,
                max_iters=args.softimpute_max_iters,
                max_rank=args.softimpute_max_rank,
                shrinkage_value=args.softimpute_shrinkage,
                normalize_softimpute=args.normalize_softimpute,
            )
        )

    if not args.skip_direct:
        direct_specs = DIRECT_SPECS
        if args.direct_names is not None:
            requested = set(args.direct_names)
            direct_specs = [spec for spec in DIRECT_SPECS if spec.name in requested]
            missing = requested - {spec.name for spec in direct_specs}
            if missing:
                raise ValueError(f"Unknown direct spec name(s): {sorted(missing)}")
        for spec in direct_specs:
            print(f"Evaluating {spec.name} via ImputePFN")
            all_rows.extend(
                eval_direct(
                    spec=spec,
                    sizes=args.sizes,
                    seeds=args.seeds,
                    rank=args.rank,
                    missing_rate=args.missing_rate,
                    scale=args.scale,
                    generator=args.generator,
                    device=args.device,
                    use_old_preprocessors=not args.no_old_preprocessors,
                    dynamic_half_until_size=args.dynamic_half_until_size,
                    dynamic_fixed_chunk_size=args.dynamic_fixed_chunk_size,
                )
            )

    if args.include_compressed:
        compressed_specs = COMPRESSED_SPECS
        if args.compressed_names is not None:
            requested = set(args.compressed_names)
            compressed_specs = [
                spec for spec in COMPRESSED_SPECS if spec.name in requested
            ]
            missing = requested - {spec.name for spec in compressed_specs}
            if missing:
                raise ValueError(f"Unknown compressed spec name(s): {sorted(missing)}")
        for spec in compressed_specs:
            print(f"Evaluating {spec.name} via CompressedCrossImputePFN")
            sizes = args.sizes
            if spec.min_size is not None:
                sizes = [size for size in sizes if size >= spec.min_size]
            all_rows.extend(
                eval_compressed(
                    spec=spec,
                    sizes=sizes,
                    seeds=args.seeds,
                    rank=args.rank,
                    missing_rate=args.missing_rate,
                    scale=args.scale,
                    generator=args.generator,
                    device=args.device,
                    dynamic_half_chunk_cap=args.compressed_dynamic_half_chunk_cap,
                    dynamic_half_until_size=args.dynamic_half_until_size,
                    dynamic_fixed_chunk_size=args.dynamic_fixed_chunk_size,
                )
            )

    summary_rows = summarize(all_rows)
    write_csv(args.output_prefix.with_suffix(".csv"), all_rows)
    write_csv(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summary_rows)
    plot(summary_rows, args.output_prefix.with_suffix(".png"), args.include_compressed)

    print("method\tsize\trmse\tsem\tmae\tnrmse\tn")
    for row in summary_rows:
        print(
            f"{row['method']}\t{row['size']}\t{row['rmse']:.6g}\t"
            f"{row['rmse_sem']:.3g}\t{row['mae']:.6g}\t{row['nrmse']:.6g}\t{row['n']}"
        )
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
