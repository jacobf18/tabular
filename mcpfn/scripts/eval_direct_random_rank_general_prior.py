#!/usr/bin/env python
from __future__ import annotations

"""Evaluate direct TabImpute-style checkpoints on randomized-rank priors.

This is a training-style validation script: it samples normalized batches from
the same MissingnessPrior family used during training, evaluates held-out
missing-cell distributional loss for neural models, and optionally compares
median/mean imputation RMSE against SoftImpute on the exact same batches.
"""

import argparse
import csv
import gc
import importlib.resources
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModel as TabImputeModelStable
from tabimpute.model.model_new_stable_memory_efficient import (
    TabImputeModel as TabImputeModelMemoryEfficient,
)
from tabimpute.model.model_new_stable_rabel import TabImputeModelPureRaBEL
from tabimpute.model.model_new_stable_rabel_context_decoder import (
    TabImputeModelPureRaBELContextDecoder,
)
from tabimpute.model.model_rowcol_latent import TabImputeModelRowColumnLatent
from tabimpute.prior.training_set_generation import MissingnessPrior


warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'.*",
    category=FutureWarning,
)

try:
    from fancyimpute import SoftImpute
except Exception:  # pragma: no cover - optional dependency on some machines.
    SoftImpute = None


REPO_DIR = Path("/root/tabular/mcpfn")
WORKDIR = REPO_DIR / "src" / "tabimpute" / "workdir"
DEFAULT_OUTPUT_DIR = REPO_DIR / "runs" / "general_prior_random_rank_eval"

DEFAULT_MODEL_SPECS = (
    "Pure RaBEL random-rank 20k|"
    + str(
        WORKDIR
        / "direct-pure-rabel-random-rank-sigma05-20k-3seeds-sigma05_lr2e4_s303-20000steps"
        / "best_checkpoint.pth"
    )
    + "|pure_rabel_memory_efficient",
    "RowCol latent random-rank current best|"
    + str(
        WORKDIR
        / "tabimpute-direct-rowcol-latent512-randomrank-lowrank-100k-scratch"
        / "best_checkpoint.pth"
    )
    + "|rowcol_latent",
)

VALIDATION_SPECS: tuple[tuple[str, int, int, int, int], ...] = (
    ("50_square", 50, 50, 50, 50),
    ("50_to_100", 50, 100, 50, 100),
    ("100_to_200", 100, 200, 100, 200),
    ("200_to_300", 200, 300, 200, 300),
    ("300_to_500", 300, 500, 300, 500),
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path
    model_family: str


def parse_model_spec(value: str) -> ModelSpec:
    parts = value.split("|")
    if len(parts) not in {2, 3}:
        raise ValueError(
            "--model must be formatted as 'name|checkpoint_path|model_family' "
            "or 'name|checkpoint_path'."
        )
    name = parts[0].strip()
    checkpoint = Path(parts[1]).expanduser()
    if not name:
        raise ValueError("Model name cannot be empty.")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if len(parts) == 3 and parts[2].strip():
        model_family = parts[2].strip()
    else:
        model_family = str(load_run_config(checkpoint.parent).get("model_family", ""))
        if not model_family:
            raise ValueError(
                f"No model_family supplied and none found in {checkpoint.parent / 'config.json'}"
            )
    return ModelSpec(name=name, checkpoint=checkpoint, model_family=model_family)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model spec formatted as name|checkpoint_path|model_family. "
            "May be repeated. Defaults to old pure-RaBEL random-rank and the "
            "current row/column latent random-rank best checkpoint."
        ),
    )
    parser.add_argument(
        "--include-softimpute",
        action="store_true",
        help="Also evaluate SoftImpute RMSE/MAE on the same normalized batches.",
    )
    parser.add_argument(
        "--spec",
        action="append",
        choices=[spec[0] for spec in VALIDATION_SPECS],
        help="Restrict to one or more validation specs. Defaults to all <=300.",
    )
    parser.add_argument(
        "--include-extended",
        action="store_true",
        help="Include the 300_to_500 extrapolation spec.",
    )
    parser.add_argument("--latent-rank-low", type=int, default=1)
    parser.add_argument("--latent-rank-high", type=int, default=40)
    parser.add_argument("--p-missing", type=float, default=0.4)
    parser.add_argument("--missingness-type", default="mcar")
    parser.add_argument("--generator-type", default="latent_factor")
    parser.add_argument("--batches-per-spec", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--softimpute-max-iters", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="random_rank_general_prior")
    return parser.parse_args()


def load_run_config(run_dir: Path) -> dict:
    with (run_dir / "config.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def model_config(checkpoint: Path) -> dict:
    config = load_run_config(checkpoint.parent)
    if "model" in config:
        return dict(config["model"])
    if "config" in config:
        return dict(config["config"])
    return dict(config)


def build_model(model_family: str, config: dict) -> torch.nn.Module:
    family = model_family.lower()
    if family in {"new_stable", "stable", "train_new_stable"}:
        return TabImputeModelStable(**config)
    if family in {
        "new_stable_memory_efficient",
        "memory_efficient",
        "train_new_stable_memory_efficient",
    }:
        return TabImputeModelMemoryEfficient(**config)
    if family in {
        "pure_rabel",
        "pure_rabel_memory_efficient",
        "new_stable_pure_rabel",
    }:
        return TabImputeModelPureRaBEL(**config)
    if family in {
        "pure_rabel_context_decoder",
        "pure_rabel_rowcol_decoder",
        "new_stable_pure_rabel_context_decoder",
    }:
        return TabImputeModelPureRaBELContextDecoder(**config)
    if family in {"rowcol_latent", "row_column_latent", "bipartite_latent"}:
        return TabImputeModelRowColumnLatent(**config)
    raise ValueError(f"Unsupported model_family: {model_family}")


def selected_specs(args: argparse.Namespace) -> tuple[tuple[str, int, int, int, int], ...]:
    specs = tuple(
        spec
        for spec in VALIDATION_SPECS
        if args.include_extended or spec[0] != "300_to_500"
    )
    if args.spec is not None:
        requested = set(args.spec)
        specs = tuple(spec for spec in specs if spec[0] in requested)
    if not specs:
        raise ValueError("No validation specs selected.")
    return specs


@torch.no_grad()
def build_validation_batches(
    *,
    specs: tuple[tuple[str, int, int, int, int], ...],
    args: argparse.Namespace,
) -> dict[str, list[tuple[torch.Tensor, torch.Tensor]]]:
    batches_by_spec: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    base_config = {
        "p_missing": args.p_missing,
        "apply_feature_warping_prob": 0.0,
        "apply_quantization_prob": 0.0,
        "latent_rank_low": args.latent_rank_low,
        "latent_rank_high": args.latent_rank_high,
        "latent_spike_p": 0.3,
        "latent_slab_sigma": 2.0,
    }
    for spec_index, (name, rows_low, rows_high, cols_low, cols_high) in enumerate(specs):
        np.random.seed(args.seed + 1000 * spec_index)
        torch.manual_seed(args.seed + 1000 * spec_index)
        prior = MissingnessPrior(
            generator_type=args.generator_type,
            missingness_type=args.missingness_type,
            config={
                **base_config,
                "num_rows_low": rows_low,
                "num_rows_high": rows_high,
                "num_cols_low": cols_low,
                "num_cols_high": cols_high,
            },
            batch_size=args.batch_size,
            device="cpu",
            verbose=False,
            entry_wise_features=False,
        )
        batches = []
        for _ in range(args.batches_per_spec):
            (batch_x, batch_target, _, _, _), _ = prior.get_batch()
            batches.append((batch_x.cpu(), batch_target.cpu()))
        batches_by_spec[name] = batches
    return batches_by_spec


def prediction_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    error = prediction[mask] - target[mask]
    mae = float(error.abs().mean().detach().cpu())
    rmse = float(torch.sqrt(torch.mean(error.square())).detach().cpu())
    return mae, rmse


@torch.no_grad()
def evaluate_checkpoint(
    *,
    spec: ModelSpec,
    batches_by_spec: dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
    criterion: FullSupportBarDistribution,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = build_model(spec.model_family, model_config(spec.checkpoint)).to(
        device=device,
        dtype=compute_dtype,
    )
    checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    metrics_by_spec: dict[str, dict[str, float]] = {}
    for spec_name, batches in batches_by_spec.items():
        losses: list[float] = []
        median_maes: list[float] = []
        median_rmses: list[float] = []
        mean_maes: list[float] = []
        mean_rmses: list[float] = []
        missing_counts: list[int] = []
        for batch_x_cpu, batch_target_cpu in batches:
            batch_x = batch_x_cpu.to(device=device, dtype=compute_dtype)
            batch_target = batch_target_cpu.to(device=device, dtype=compute_dtype)
            missing_mask = torch.isnan(batch_x)
            logits = model(batch_x, decode_mask=missing_mask)
            target_missing = batch_target[missing_mask]
            loss = criterion(logits, target_missing).mean()
            median_prediction = criterion.median(logits)
            mean_prediction = criterion.mean(logits)
            median_error = median_prediction - target_missing
            mean_error = mean_prediction - target_missing
            losses.append(float(loss.detach().cpu()))
            median_maes.append(float(median_error.abs().mean().detach().cpu()))
            median_rmses.append(
                float(torch.sqrt(torch.mean(median_error.square())).detach().cpu())
            )
            mean_maes.append(float(mean_error.abs().mean().detach().cpu()))
            mean_rmses.append(
                float(torch.sqrt(torch.mean(mean_error.square())).detach().cpu())
            )
            missing_counts.append(int(missing_mask.sum().detach().cpu()))
        metrics_by_spec[spec_name] = {
            "loss": float(np.mean(losses)),
            "median_mae": float(np.mean(median_maes)),
            "median_rmse": float(np.mean(median_rmses)),
            "mean_mae": float(np.mean(mean_maes)),
            "mean_rmse": float(np.mean(mean_rmses)),
            "missing_count": float(np.mean(missing_counts)),
        }

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics_by_spec


def softimpute_one(matrix: np.ndarray, max_iters: int) -> np.ndarray:
    if SoftImpute is None:
        raise RuntimeError("fancyimpute SoftImpute is not available.")
    prediction = SoftImpute(
        max_iters=max_iters,
        convergence_threshold=1e-5,
        verbose=False,
    ).fit_transform(matrix)
    output = matrix.copy()
    missing = np.isnan(output)
    output[missing] = prediction[missing]
    return output


def evaluate_softimpute(
    *,
    batches_by_spec: dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
    max_iters: int,
) -> dict[str, dict[str, float]]:
    metrics_by_spec: dict[str, dict[str, float]] = {}
    for spec_name, batches in batches_by_spec.items():
        maes: list[float] = []
        rmses: list[float] = []
        missing_counts: list[int] = []
        for batch_x_cpu, batch_target_cpu in batches:
            for index in range(batch_x_cpu.shape[0]):
                observed = batch_x_cpu[index].numpy().astype(np.float64)
                target = batch_target_cpu[index].numpy().astype(np.float64)
                missing = np.isnan(observed)
                prediction = softimpute_one(observed, max_iters=max_iters)
                error = prediction[missing] - target[missing]
                maes.append(float(np.mean(np.abs(error))))
                rmses.append(float(np.sqrt(np.mean(error**2))))
                missing_counts.append(int(missing.sum()))
        metrics_by_spec[spec_name] = {
            "loss": float("nan"),
            "median_mae": float(np.mean(maes)),
            "median_rmse": float(np.mean(rmses)),
            "mean_mae": float(np.mean(maes)),
            "mean_rmse": float(np.mean(rmses)),
            "missing_count": float(np.mean(missing_counts)),
        }
    return metrics_by_spec


def flatten_results(
    all_metrics: dict[str, dict[str, dict[str, float]]],
    model_paths: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, metrics_by_spec in all_metrics.items():
        losses = [
            metrics["loss"]
            for metrics in metrics_by_spec.values()
            if np.isfinite(metrics["loss"])
        ]
        row: dict[str, object] = {
            "method": method,
            "checkpoint": model_paths.get(method, ""),
            "macro_loss": float(np.mean(losses)) if losses else float("nan"),
            "macro_median_rmse": float(
                np.mean([metrics["median_rmse"] for metrics in metrics_by_spec.values()])
            ),
            "macro_mean_rmse": float(
                np.mean([metrics["mean_rmse"] for metrics in metrics_by_spec.values()])
            ),
            "macro_median_mae": float(
                np.mean([metrics["median_mae"] for metrics in metrics_by_spec.values()])
            ),
            "macro_mean_mae": float(
                np.mean([metrics["mean_mae"] for metrics in metrics_by_spec.values()])
            ),
        }
        for spec_name, metrics in metrics_by_spec.items():
            for metric_name, value in metrics.items():
                row[f"{spec_name}_{metric_name}"] = value
        rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row["macro_loss"])
            if np.isfinite(float(row["macro_loss"]))
            else float("inf"),
            float(row["macro_median_rmse"]),
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    leading = [
        "method",
        "checkpoint",
        "macro_loss",
        "macro_median_rmse",
        "macro_mean_rmse",
        "macro_median_mae",
        "macro_mean_mae",
    ]
    fieldnames = [key for key in leading if key in fieldnames] + [
        key for key in fieldnames if key not in leading
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_long_csv(
    path: Path,
    all_metrics: dict[str, dict[str, dict[str, float]]],
    model_paths: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "checkpoint",
        "spec",
        "loss",
        "median_rmse",
        "mean_rmse",
        "median_mae",
        "mean_mae",
        "missing_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for method, metrics_by_spec in all_metrics.items():
            for spec_name, metrics in metrics_by_spec.items():
                writer.writerow(
                    {
                        "method": method,
                        "checkpoint": model_paths.get(method, ""),
                        "spec": spec_name,
                        **metrics,
                    }
                )


def plot_metric(
    *,
    path: Path,
    all_metrics: dict[str, dict[str, dict[str, float]]],
    metric: str,
    ylabel: str,
) -> None:
    spec_names = [
        spec[0]
        for spec in VALIDATION_SPECS
        if any(spec[0] in metrics_by_spec for metrics_by_spec in all_metrics.values())
    ]
    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    for method, metrics_by_spec in all_metrics.items():
        xs = []
        ys = []
        for idx, spec_name in enumerate(spec_names):
            if spec_name not in metrics_by_spec:
                continue
            value = metrics_by_spec[spec_name][metric]
            if not np.isfinite(value):
                continue
            xs.append(idx)
            ys.append(value)
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linewidth=2.2, label=method)
    ax.set_xticks(range(len(spec_names)))
    ax.set_xticklabels(spec_names, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("validation size range")
    ax.grid(True, alpha=0.25)
    if "rmse" in metric:
        ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            output[key] = str(value)
        else:
            output[key] = value
    return output


def main() -> None:
    args = parse_args()
    model_values = args.model if args.model is not None else list(DEFAULT_MODEL_SPECS)
    model_specs = [parse_model_spec(value) for value in model_values]
    specs = selected_specs(args)

    batches_by_spec = build_validation_batches(specs=specs, args=args)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)
    borders_path = importlib.resources.files("tabimpute") / "data" / "borders.pt"
    with importlib.resources.as_file(borders_path) as path:
        criterion = FullSupportBarDistribution(
            borders=torch.load(path, map_location=device).to(device)
        )

    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    model_paths: dict[str, str] = {}
    if args.include_softimpute:
        print("Evaluating SoftImpute", flush=True)
        all_metrics["SoftImpute"] = evaluate_softimpute(
            batches_by_spec=batches_by_spec,
            max_iters=args.softimpute_max_iters,
        )
    for spec in model_specs:
        print(f"Evaluating {spec.name}: {spec.checkpoint}", flush=True)
        all_metrics[spec.name] = evaluate_checkpoint(
            spec=spec,
            batches_by_spec=batches_by_spec,
            criterion=criterion,
            device=device,
        )
        model_paths[spec.name] = str(spec.checkpoint)

    rows = flatten_results(all_metrics, model_paths)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / f"{args.tag}_summary.csv"
    long_csv = output_dir / f"{args.tag}_by_spec.csv"
    json_path = output_dir / f"{args.tag}.json"
    write_csv(summary_csv, rows)
    write_long_csv(long_csv, all_metrics, model_paths)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "args": jsonable_args(args),
                "models": [spec.__dict__ | {"checkpoint": str(spec.checkpoint)} for spec in model_specs],
                "specs": specs,
                "metrics": all_metrics,
            },
            file,
            indent=2,
            sort_keys=True,
        )
    plot_metric(
        path=output_dir / f"{args.tag}_loss.png",
        all_metrics=all_metrics,
        metric="loss",
        ylabel="missing-entry NLL",
    )
    plot_metric(
        path=output_dir / f"{args.tag}_median_rmse.png",
        all_metrics=all_metrics,
        metric="median_rmse",
        ylabel="missing-entry median RMSE",
    )

    print("method\tmacro_loss\tmacro_median_rmse\tmacro_mean_rmse")
    for row in rows:
        print(
            f"{row['method']}\t{float(row['macro_loss']):.6g}\t"
            f"{float(row['macro_median_rmse']):.6g}\t"
            f"{float(row['macro_mean_rmse']):.6g}",
            flush=True,
        )
    print(f"Wrote {summary_csv}")
    print(f"Wrote {long_csv}")
    print(f"Wrote {output_dir / (args.tag + '_median_rmse.png')}")


if __name__ == "__main__":
    main()
