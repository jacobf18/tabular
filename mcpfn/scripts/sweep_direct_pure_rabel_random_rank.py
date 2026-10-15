#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.resources
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable_rabel import TabImputeModelPureRaBEL
from tabimpute.prior.training_set_generation import MissingnessPrior


REPO_DIR = Path("/root/tabular/mcpfn")
WORKDIR = REPO_DIR / "src" / "tabimpute" / "workdir"
RUNS_DIR = REPO_DIR / "runs"


@dataclass(frozen=True)
class SweepConfig:
    name: str
    lr: float
    min_lr_ratio: float
    weight_decay: float
    grad_clip_norm: float
    model_overrides: dict
    seed: int


CONFIGS: tuple[SweepConfig, ...] = (
    SweepConfig(
        "base_lr2e4_s101",
        2e-4,
        0.10,
        0.03,
        0.5,
        {},
        101,
    ),
    SweepConfig(
        "base_lr2e4_s202",
        2e-4,
        0.10,
        0.03,
        0.5,
        {},
        202,
    ),
    SweepConfig(
        "base_lr1e4_s101",
        1e-4,
        0.10,
        0.03,
        0.5,
        {},
        101,
    ),
    SweepConfig(
        "base_lr1e4_s202",
        1e-4,
        0.10,
        0.03,
        0.5,
        {},
        202,
    ),
    SweepConfig(
        "sigma05_lr2e4_s101",
        2e-4,
        0.10,
        0.03,
        0.5,
        {"rabel_sigma": 0.5},
        101,
    ),
    SweepConfig(
        "sigma05_lr2e4_s202",
        2e-4,
        0.10,
        0.03,
        0.5,
        {"rabel_sigma": 0.5},
        202,
    ),
    SweepConfig(
        "sigma05_lr2e4_s303",
        2e-4,
        0.10,
        0.03,
        0.5,
        {"rabel_sigma": 0.5},
        303,
    ),
    SweepConfig(
        "sigma15_lr2e4_s101",
        2e-4,
        0.10,
        0.03,
        0.5,
        {"rabel_sigma": 1.5},
        101,
    ),
    SweepConfig(
        "kernels128_lr2e4_s101",
        2e-4,
        0.10,
        0.03,
        0.5,
        {"rabel_num_kernels": 128},
        101,
    ),
    SweepConfig(
        "nodrop_lr2e4_s101",
        2e-4,
        0.10,
        0.03,
        0.5,
        {"ffn_dropout": 0.0, "drop_path_rate": 0.0},
        101,
    ),
)


VALIDATION_SPECS: tuple[tuple[str, int, int, int, int], ...] = (
    ("50_square", 50, 50, 50, 50),
    ("75_to_100", 75, 100, 75, 100),
    ("100_to_150", 100, 150, 100, 150),
    ("150_to_200", 150, 200, 150, 200),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-name", default="direct-pure-rabel-random-rank-sweep-8k")
    parser.add_argument("--steps-per-config", type=int, default=8_000)
    parser.add_argument("--latent-rank-low", type=int, default=1)
    parser.add_argument("--latent-rank-high", type=int, default=30)
    parser.add_argument("--p-missing", type=float, default=0.4)
    parser.add_argument(
        "--missingness-type",
        default="mcar",
        help="Missingness pattern used for both training and synthetic validation.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--batches-per-spec", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument("--validation-seed", type=int, default=20260707)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def make_phases(steps: int, batch_size_override: int | None) -> list[dict]:
    if steps < 4:
        raise ValueError("--steps-per-config must be >= 4")
    ends = [
        max(1, round(0.25 * steps)),
        max(2, round(0.50 * steps)),
        max(3, round(0.75 * steps)),
        steps,
    ]
    batches = [32, 16, 8, 8]
    if batch_size_override is not None:
        batches = [batch_size_override] * 4
    return [
        {
            "name": "50_square",
            "end_step": ends[0],
            "rows_low": 50,
            "rows_high": 50,
            "cols_low": 50,
            "cols_high": 50,
            "batch_size": batches[0],
        },
        {
            "name": "50_to_100",
            "end_step": ends[1],
            "rows_low": 50,
            "rows_high": 100,
            "cols_low": 50,
            "cols_high": 100,
            "batch_size": batches[1],
        },
        {
            "name": "100_to_150",
            "end_step": ends[2],
            "rows_low": 100,
            "rows_high": 150,
            "cols_low": 100,
            "cols_high": 150,
            "batch_size": batches[2],
        },
        {
            "name": "150_to_200",
            "end_step": ends[3],
            "rows_low": 150,
            "rows_high": 200,
            "cols_low": 150,
            "cols_high": 200,
            "batch_size": batches[3],
        },
    ]


def scheduler_parts(steps: int) -> tuple[int, int, int]:
    warmup = min(500, max(1, steps // 10))
    decay = min(1_000, max(1, steps // 8))
    stable = steps - warmup - decay
    if stable < 0:
        stable = 0
        decay = steps - warmup
    return warmup, stable, decay


def build_validation_batches(
    *,
    latent_rank_low: int,
    latent_rank_high: int,
    p_missing: float,
    missingness_type: str,
    seed: int,
    batches_per_spec: int,
    batch_size: int,
) -> dict[str, list[tuple[torch.Tensor, torch.Tensor]]]:
    batches_by_spec = {}
    base_config = {
        "p_missing": p_missing,
        "apply_feature_warping_prob": 0.0,
        "apply_quantization_prob": 0.0,
        "latent_rank_low": latent_rank_low,
        "latent_rank_high": latent_rank_high,
        "latent_spike_p": 0.3,
        "latent_slab_sigma": 2.0,
    }
    for spec_index, (name, rows_low, rows_high, cols_low, cols_high) in enumerate(
        VALIDATION_SPECS
    ):
        np.random.seed(seed + 1000 * spec_index)
        torch.manual_seed(seed + 1000 * spec_index)
        prior = MissingnessPrior(
            generator_type="latent_factor",
            missingness_type=missingness_type,
            config={
                **base_config,
                "num_rows_low": rows_low,
                "num_rows_high": rows_high,
                "num_cols_low": cols_low,
                "num_cols_high": cols_high,
            },
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

    out = {}
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
            target = batch_target[missing_mask]
            loss = criterion(logits, target).mean()
            medians = criterion.median(logits)
            error = medians - target
            losses.append(float(loss.detach().cpu()))
            maes.append(float(error.abs().mean().detach().cpu()))
            rmses.append(float(torch.sqrt(torch.mean(error.square())).detach().cpu()))
            missing_counts.append(int(missing_mask.sum().detach().cpu()))
        out[spec_name] = {
            "loss": float(np.mean(losses)),
            "mae": float(np.mean(maes)),
            "rmse": float(np.mean(rmses)),
            "missing_count": float(np.mean(missing_counts)),
        }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def train_one(
    *,
    cfg: SweepConfig,
    args: argparse.Namespace,
    phases_json: str,
    env: dict[str, str],
) -> Path:
    run_name = f"{args.sweep_name}-{cfg.name}-{args.steps_per_config}steps"
    run_dir = WORKDIR / run_name
    warmup, stable, decay = scheduler_parts(args.steps_per_config)
    cmd = [
        sys.executable,
        str(REPO_DIR / "scripts" / "train_direct_pure_rabel_curriculum.py"),
        "--run-name",
        run_name,
        "--device",
        args.device,
        "--prior-kind",
        "latent_factor",
        "--p-missing",
        str(args.p_missing),
        "--missingness-type",
        args.missingness_type,
        "--latent-rank-low",
        str(args.latent_rank_low),
        "--latent-rank-high",
        str(args.latent_rank_high),
        "--lr",
        str(cfg.lr),
        "--weight-decay",
        str(cfg.weight_decay),
        "--grad-clip-norm",
        str(cfg.grad_clip_norm),
        "--warmup-steps",
        str(warmup),
        "--stable-steps",
        str(stable),
        "--decay-steps",
        str(decay),
        "--min-lr-ratio",
        str(cfg.min_lr_ratio),
        "--checkpoint-every",
        str(args.steps_per_config),
        "--seed",
        str(cfg.seed),
        "--phases-json",
        phases_json,
        "--model-overrides-json",
        json.dumps(cfg.model_overrides, sort_keys=True),
    ]
    print("\n=== training", run_name, "===", flush=True)
    subprocess.run(cmd, cwd=REPO_DIR, env=env, check=True)
    checkpoint_path = run_dir / "best_checkpoint.pth"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / f"checkpoint_{args.steps_per_config}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found for {run_name}")
    return checkpoint_path


def main() -> None:
    args = parse_args()
    selected = [cfg for cfg in CONFIGS if args.configs is None or cfg.name in args.configs]
    if not selected:
        raise ValueError("No sweep configs selected.")
    output_dir = args.output_dir or (RUNS_DIR / args.sweep_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sweep_configs.json").open("w", encoding="utf-8") as file:
        json.dump([asdict(cfg) for cfg in selected], file, indent=2, sort_keys=True)

    phases = make_phases(args.steps_per_config, args.batch_size)
    phases_json = json.dumps(phases, sort_keys=True)
    with (output_dir / "phases.json").open("w", encoding="utf-8") as file:
        json.dump(phases, file, indent=2, sort_keys=True)

    env = os.environ.copy()
    env["WANDB_MODE"] = args.wandb_mode
    env["PYTHONPATH"] = (
        f"{REPO_DIR / 'src'}:{REPO_DIR / 'benchmark' / 'CPS'}:"
        + env.get("PYTHONPATH", "")
    )

    batches_by_spec = build_validation_batches(
        latent_rank_low=args.latent_rank_low,
        latent_rank_high=args.latent_rank_high,
        p_missing=args.p_missing,
        missingness_type=args.missingness_type,
        seed=args.validation_seed,
        batches_per_spec=args.batches_per_spec,
        batch_size=args.validation_batch_size,
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)
    borders_path = importlib.resources.files("tabimpute") / "data" / "borders.pt"
    with importlib.resources.as_file(borders_path) as path:
        criterion = FullSupportBarDistribution(
            borders=torch.load(path, map_location=device).to(device)
        )

    all_rows = []
    for cfg in selected:
        checkpoint_path = train_one(cfg=cfg, args=args, phases_json=phases_json, env=env)
        run_config = json.loads((checkpoint_path.parent / "config.json").read_text())
        metrics_by_spec = evaluate_checkpoint(
            checkpoint_path,
            run_config["model"],
            batches_by_spec,
            criterion,
            device,
        )
        losses = [m["loss"] for m in metrics_by_spec.values()]
        maes = [m["mae"] for m in metrics_by_spec.values()]
        rmses = [m["rmse"] for m in metrics_by_spec.values()]
        row = {
            "method": cfg.name,
            "checkpoint": str(checkpoint_path),
            "macro_loss": float(np.mean(losses)),
            "macro_mae": float(np.mean(maes)),
            "macro_rmse": float(np.mean(rmses)),
            "lr": cfg.lr,
            "min_lr_ratio": cfg.min_lr_ratio,
            "weight_decay": cfg.weight_decay,
            "grad_clip_norm": cfg.grad_clip_norm,
            "seed": cfg.seed,
            "model_overrides": json.dumps(cfg.model_overrides, sort_keys=True),
        }
        for spec_name, metrics in metrics_by_spec.items():
            for metric_name, value in metrics.items():
                row[f"{spec_name}_{metric_name}"] = value
        all_rows.append(row)
        pd_path = output_dir / f"{cfg.name}_summary.csv"
        with pd_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

    all_rows.sort(key=lambda row: row["macro_loss"])
    ranking_path = output_dir / "ranking.csv"
    with ranking_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print("\n=== ranking by macro validation loss ===", flush=True)
    for index, row in enumerate(all_rows, start=1):
        print(
            f"{index}. {row['method']} loss={row['macro_loss']:.6f} "
            f"rmse={row['macro_rmse']:.6f} ckpt={row['checkpoint']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
