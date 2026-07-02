#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.resources
import json
from pathlib import Path

import numpy as np
import torch

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable_rabel import TabImputeModelPureRaBEL
from tabimpute.prior.spectrum_training_set_generation import SpectrumMissingnessPrior


DEFAULT_WORKDIR = Path("/root/tabular/mcpfn/src/tabimpute/workdir")
DEFAULT_PREFIX = "tabimpute-direct-pure-rabel64-spectrum-decay-top40-curriculum-160k"
DEFAULT_RUN_SUFFIXES = ("lr2e5", "lr5e5", "lr8e5", "lr1e4")


VALIDATION_SPECS = (
    ("phase1_50_square", 50, 50, 50, 50),
    ("phase2_50_to_100", 50, 100, 50, 100),
    ("phase3_100_to_200", 100, 200, 100, 200),
    ("phase4_200_to_300", 200, 300, 200, 300),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--run-suffixes", nargs="+", default=list(DEFAULT_RUN_SUFFIXES))
    parser.add_argument("--checkpoint-name", default="best_checkpoint.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--batches-per-spec", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--spec",
        action="append",
        choices=[spec[0] for spec in VALIDATION_SPECS],
        help="Restrict to one or more validation specs. Defaults to all specs.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "/root/tabular/mcpfn/runs/spectrum_lr_validation/"
            "direct_rabel_spectrum_lr_best_checkpoint_validation.csv"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "/root/tabular/mcpfn/runs/spectrum_lr_validation/"
            "direct_rabel_spectrum_lr_best_checkpoint_validation.json"
        ),
    )
    return parser.parse_args()


def load_config(run_dir: Path) -> dict:
    with (run_dir / "config.json").open("r", encoding="utf-8") as file:
        return json.load(file)


@torch.no_grad()
def build_validation_batches(
    *,
    base_prior_config: dict,
    specs: tuple[tuple[str, int, int, int, int], ...],
    batches_per_spec: int,
    batch_size: int,
    seed: int,
) -> dict[str, list[tuple[torch.Tensor, torch.Tensor]]]:
    batches_by_spec: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for spec_index, (name, rows_low, rows_high, cols_low, cols_high) in enumerate(specs):
        np.random.seed(seed + 1000 * spec_index)
        torch.manual_seed(seed + 1000 * spec_index)
        prior_config = {
            **base_prior_config,
            "num_rows_low": rows_low,
            "num_rows_high": rows_high,
            "num_cols_low": cols_low,
            "num_cols_high": cols_high,
        }
        prior = SpectrumMissingnessPrior(
            missingness_type="mcar",
            config=prior_config,
            batch_size=batch_size,
            device="cpu",
            verbose=False,
            entry_wise_features=False,
        )
        batches = []
        for _ in range(batches_per_spec):
            (batch_x, batch_target, _, _, _), _ = prior.get_batch()
            batches.append((batch_x.cpu(), batch_target.cpu()))
        batches_by_spec[name] = batches
    return batches_by_spec


@torch.no_grad()
def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    model_config: dict,
    batches_by_spec: dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
    criterion: FullSupportBarDistribution,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = TabImputeModelPureRaBEL(**model_config).to(device=device, dtype=compute_dtype)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    metrics_by_spec: dict[str, dict[str, float]] = {}
    for spec_name, batches in batches_by_spec.items():
        losses = []
        maes = []
        rmses = []
        missing_counts = []
        for batch_x_cpu, batch_target_cpu in batches:
            batch_x = batch_x_cpu.to(device=device, dtype=compute_dtype)
            batch_target = batch_target_cpu.to(device=device, dtype=compute_dtype)
            missing_mask = torch.isnan(batch_x)
            logits = model(batch_x, decode_mask=missing_mask)
            target_missing = batch_target[missing_mask]
            loss = criterion(logits, target_missing).mean()
            medians = criterion.median(logits)
            error = medians - target_missing
            losses.append(float(loss.detach().cpu()))
            maes.append(float(error.abs().mean().detach().cpu()))
            rmses.append(float(torch.sqrt(torch.mean(error.square())).detach().cpu()))
            missing_counts.append(int(missing_mask.sum().detach().cpu()))
        metrics_by_spec[spec_name] = {
            "loss": float(np.mean(losses)),
            "mae": float(np.mean(maes)),
            "rmse": float(np.mean(rmses)),
            "missing_count": float(np.mean(missing_counts)),
        }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics_by_spec


def main() -> None:
    args = parse_args()
    run_dirs = {
        suffix: args.workdir / f"{args.prefix}-{suffix}"
        for suffix in args.run_suffixes
    }
    missing = [
        str(run_dir / args.checkpoint_name)
        for run_dir in run_dirs.values()
        if not (run_dir / args.checkpoint_name).exists()
    ]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s):\n" + "\n".join(missing))

    first_config = load_config(next(iter(run_dirs.values())))
    model_config = first_config["model"]
    base_prior_config = first_config["prior"]
    selected_specs = tuple(
        spec for spec in VALIDATION_SPECS if args.spec is None or spec[0] in args.spec
    )
    if not selected_specs:
        raise ValueError("No validation specs selected.")

    batches_by_spec = build_validation_batches(
        base_prior_config=base_prior_config,
        specs=selected_specs,
        batches_per_spec=args.batches_per_spec,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)
    borders_path = importlib.resources.files("tabimpute") / "data" / "borders.pt"
    with importlib.resources.as_file(borders_path) as path:
        criterion = FullSupportBarDistribution(
            borders=torch.load(path, map_location=device).to(device)
        )

    rows: list[dict[str, float | str | int]] = []
    results: dict[str, dict] = {}
    for suffix, run_dir in run_dirs.items():
        config = load_config(run_dir)
        checkpoint_path = run_dir / args.checkpoint_name
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        metrics_by_spec = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            model_config=config["model"],
            batches_by_spec=batches_by_spec,
            criterion=criterion,
            device=device,
        )
        losses = [metrics["loss"] for metrics in metrics_by_spec.values()]
        maes = [metrics["mae"] for metrics in metrics_by_spec.values()]
        rmses = [metrics["rmse"] for metrics in metrics_by_spec.values()]
        row = {
            "run_suffix": suffix,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "train_best_ema": float(
                checkpoint.get(
                    "best_smoothed_loss",
                    checkpoint.get("metrics", {}).get("best_smoothed_loss", float("nan")),
                )
            ),
            "macro_loss": float(np.mean(losses)),
            "macro_mae": float(np.mean(maes)),
            "macro_rmse": float(np.mean(rmses)),
        }
        for spec_name, metrics in metrics_by_spec.items():
            row[f"{spec_name}_loss"] = metrics["loss"]
            row[f"{spec_name}_mae"] = metrics["mae"]
            row[f"{spec_name}_rmse"] = metrics["rmse"]
        rows.append(row)
        results[suffix] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "train_metrics": checkpoint.get("metrics", {}),
            "validation": metrics_by_spec,
            "summary": {
                "macro_loss": row["macro_loss"],
                "macro_mae": row["macro_mae"],
                "macro_rmse": row["macro_rmse"],
            },
        }

    rows.sort(key=lambda row: float(row["macro_loss"]))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with args.output_json.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "checkpoint_name": args.checkpoint_name,
                "device": args.device,
                "seed": args.seed,
                "batches_per_spec": args.batches_per_spec,
                "batch_size": args.batch_size,
                "results": results,
                "ranking_by_macro_loss": [row["run_suffix"] for row in rows],
            },
            file,
            indent=2,
            sort_keys=True,
        )

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")
    print("ranking_by_macro_loss")
    for row in rows:
        print(
            f"{row['run_suffix']}: macro_loss={row['macro_loss']:.6f}, "
            f"macro_mae={row['macro_mae']:.6f}, "
            f"macro_rmse={row['macro_rmse']:.6f}, "
            f"epoch={row['checkpoint_epoch']}"
        )


if __name__ == "__main__":
    main()
