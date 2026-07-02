#!/usr/bin/env python
"""Compare direct TabImpute checkpoints on whole synthetic matrices.

The same clean rank-10 matrix and MCAR mask are passed to every method.
TabImpute models are invoked through ImputePFN with chunking disabled.
SoftImpute is rank-agnostic and uses the same observed-column normalization
as ImputePFN.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from fancyimpute import SoftImpute

from tabimpute.interface import ImputePFN


warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'.*",
    category=FutureWarning,
)

ROOT = Path("/root/tabular/mcpfn")
WORKDIR = ROOT / "src/tabimpute/workdir"
DEFAULT_OUT = (
    ROOT
    / "runs/figures_by_setup/11_p04_gaussian_rank10_whole_matrix_pure_rabel"
    / "direct_whole_matrix_pure_rabel_vs_previous"
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path
    model_family: str


MODEL_SPECS = (
    ModelSpec(
        "TabImpute standard 50",
        WORKDIR
        / "tabimpute-new-stable_v2-rank_10_10-50x50"
        / "checkpoint_30000.pth",
        "new_stable",
    ),
    ModelSpec(
        "TabImpute standard 100",
        WORKDIR
        / "tabimpute-new-stable_v2-rank_10_10-100x100-from_50x50_step30000"
        / "checkpoint_45000.pth",
        "new_stable",
    ),
    ModelSpec(
        "TabImpute standard 200",
        WORKDIR
        / "tabimpute-new-stable-memory-efficient_v2-rank_10_10-200x200-from_100x100_step45000"
        / "checkpoint_60000.pth",
        "new_stable_memory_efficient",
    ),
    ModelSpec(
        "TabImpute pure RaBEL",
        WORKDIR
        / "tabimpute-direct-pure-rabel64-rank10-curriculum-160k"
        / "checkpoint_160000.pth",
        "pure_rabel_memory_efficient",
    ),
)


def model_config(checkpoint: Path) -> dict:
    with (checkpoint.parent / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    if "model" in config:
        return dict(config["model"])
    if "config" in config:
        return dict(config["config"])
    return dict(config)


def make_matrix(
    size: int,
    rank: int,
    missing_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rank > size:
        raise ValueError(f"rank={rank} exceeds matrix size={size}")
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(size, rank))
    right = rng.normal(size=(rank, size))
    target = (left @ right / rank).astype(np.float32)
    missing = rng.random(target.shape) < missing_rate
    observed = target.copy()
    observed[missing] = np.nan
    return target, observed, missing


def score(
    target: np.ndarray,
    prediction: np.ndarray,
    missing: np.ndarray,
) -> dict[str, float]:
    error = prediction[missing] - target[missing]
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "rmse": rmse,
        "mae": float(np.mean(np.abs(error))),
        "nrmse": rmse / float(np.std(target[missing])),
    }


def observed_col_stats(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(observed, axis=0)
    stds = np.nanstd(observed, axis=0)
    return (
        np.where(np.isnan(means), 0.0, means),
        np.where(np.isnan(stds), 1.0, stds),
    )


def softimpute(
    observed: np.ndarray,
    max_iters: int,
) -> np.ndarray:
    means, stds = observed_col_stats(observed)
    normalized = (observed - means) / (stds + 1e-16)
    prediction = SoftImpute(
        max_iters=max_iters,
        convergence_threshold=1e-5,
        verbose=False,
    ).fit_transform(normalized)
    prediction = prediction * (stds + 1e-16) + means
    output = observed.copy()
    missing = np.isnan(output)
    output[missing] = prediction[missing]
    return output


def evaluate_softimpute(
    sizes: list[int],
    seeds: list[int],
    rank: int,
    missing_rate: float,
    max_iters: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for size in sizes:
        for seed in seeds:
            target, observed, missing = make_matrix(
                size, rank, missing_rate, seed
            )
            prediction = softimpute(observed, max_iters)
            rows.append(
                {
                    "method": "SoftImpute",
                    "size": size,
                    "seed": seed,
                    **score(target, prediction, missing),
                }
            )
        print(f"SoftImpute: completed size {size}", flush=True)
    return rows


def evaluate_model(
    spec: ModelSpec,
    sizes: list[int],
    seeds: list[int],
    rank: int,
    missing_rate: float,
    device: str,
) -> list[dict[str, object]]:
    if not spec.checkpoint.exists():
        raise FileNotFoundError(spec.checkpoint)
    imputer = ImputePFN(
        device=device,
        entry_wise_features=False,
        checkpoint_path=str(spec.checkpoint),
        json_config={"config": model_config(spec.checkpoint)},
        model_family=spec.model_family,
        max_num_rows=None,
        max_num_cols=None,
        preprocessors=None,
    )
    rows: list[dict[str, object]] = []
    for size in sizes:
        for seed in seeds:
            target, observed, missing = make_matrix(
                size, rank, missing_rate, seed
            )
            prediction = imputer.impute(observed.copy())
            rows.append(
                {
                    "method": spec.name,
                    "size": size,
                    "seed": seed,
                    **score(target, prediction, missing),
                }
            )
        print(f"{spec.name}: completed size {size}", flush=True)
    del imputer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    groups = sorted({(str(row["method"]), int(row["size"])) for row in rows})
    for method, size in groups:
        group = [
            row
            for row in rows
            if row["method"] == method and row["size"] == size
        ]
        rmses = np.asarray([float(row["rmse"]) for row in group])
        maes = np.asarray([float(row["mae"]) for row in group])
        nrmses = np.asarray([float(row["nrmse"]) for row in group])
        summary.append(
            {
                "method": method,
                "size": size,
                "rmse": float(rmses.mean()),
                "rmse_std": float(rmses.std(ddof=1)) if len(group) > 1 else 0.0,
                "rmse_sem": (
                    float(rmses.std(ddof=1) / np.sqrt(len(group)))
                    if len(group) > 1
                    else 0.0
                ),
                "mae": float(maes.mean()),
                "nrmse": float(nrmses.mean()),
                "n": len(group),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "size",
        "seed",
        "rmse",
        "rmse_std",
        "rmse_sem",
        "mae",
        "nrmse",
        "n",
    ]
    present = {key for row in rows for key in row}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=[field for field in fields if field in present]
        )
        writer.writeheader()
        writer.writerows(rows)


def plot(summary: list[dict[str, object]], path: Path) -> None:
    order = ["SoftImpute", *(spec.name for spec in MODEL_SPECS)]
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    for method in order:
        group = sorted(
            (row for row in summary if row["method"] == method),
            key=lambda row: int(row["size"]),
        )
        if not group:
            continue
        ax.errorbar(
            [row["size"] for row in group],
            [row["rmse"] for row in group],
            yerr=[row["rmse_sem"] for row in group],
            marker="o",
            linewidth=2.4,
            capsize=3,
            label=method,
        )
    ax.set_title("Direct whole-matrix evaluation: fixed rank 10, 40% MCAR")
    ax.set_xlabel("square matrix size")
    ax.set_ylabel("missing-entry RMSE")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[20, 50, 75, 100, 150, 200, 300, 400, 500],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--missing-rate", type=float, default=0.4)
    parser.add_argument("--softimpute-max-iters", type=int, default=200)
    parser.add_argument("--skip-softimpute", action="store_true")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    selected = list(MODEL_SPECS)
    if args.methods is not None:
        requested = set(args.methods)
        selected = [spec for spec in selected if spec.name in requested]
        unknown = requested - {spec.name for spec in selected}
        if unknown:
            raise ValueError(f"Unknown methods: {sorted(unknown)}")

    rows: list[dict[str, object]] = []
    if not args.skip_softimpute:
        rows.extend(
            evaluate_softimpute(
                args.sizes,
                args.seeds,
                args.rank,
                args.missing_rate,
                args.softimpute_max_iters,
            )
        )
    for spec in selected:
        print(f"Loading {spec.name}: {spec.checkpoint}", flush=True)
        rows.extend(
            evaluate_model(
                spec,
                args.sizes,
                args.seeds,
                args.rank,
                args.missing_rate,
                args.device,
            )
        )

    summary = summarize(rows)
    write_csv(args.output_prefix.with_suffix(".csv"), rows)
    write_csv(
        args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"),
        summary,
    )
    plot(summary, args.output_prefix.with_suffix(".png"))

    print("method\tsize\trmse\tsem\tmae\tnrmse\tn")
    for row in summary:
        print(
            f"{row['method']}\t{row['size']}\t{row['rmse']:.6g}\t"
            f"{row['rmse_sem']:.3g}\t{row['mae']:.6g}\t"
            f"{row['nrmse']:.6g}\t{row['n']}"
        )
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
