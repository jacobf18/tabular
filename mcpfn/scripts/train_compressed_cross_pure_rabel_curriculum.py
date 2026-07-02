#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.resources
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_compressed_cross import (
    build_compressor_config_from_predictor,
)
from tabimpute.prior.training_set_generation import MissingnessPrior
from tabimpute.train.callbacks import WandbLoggerCallback
from train_compressed_cross_rank10 import (
    build_compressed_model,
    build_compressed_scheduler,
    build_split_optimizer,
    train_one_random_chunk_step,
)


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    end_step: int
    rows_low: int
    rows_high: int
    cols_low: int
    cols_high: int
    chunk_rows: int
    chunk_cols: int
    batch_size: int


DEFAULT_PHASES = (
    CurriculumPhase(
        "100_square_50_chunks",
        50_000,
        100,
        100,
        100,
        100,
        50,
        50,
        8,
    ),
    CurriculumPhase(
        "100_to_200_100_chunks",
        85_000,
        100,
        200,
        100,
        200,
        100,
        100,
        4,
    ),
    CurriculumPhase(
        "200_to_300_100_chunks",
        125_000,
        200,
        300,
        200,
        300,
        100,
        100,
        4,
    ),
    CurriculumPhase(
        "300_to_500_100_chunks",
        160_000,
        300,
        500,
        300,
        500,
        100,
        100,
        2,
    ),
)

DEFAULT_PREDICTOR_CHECKPOINT = Path(
    "/root/tabular/mcpfn/src/tabimpute/workdir/"
    "tabimpute-direct-pure-rabel64-rank10-curriculum-160k/"
    "checkpoint_160000.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default="tabimpute-compressed-pure-rabel64-rank10-k20-curriculum-160k",
    )
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument(
        "--predictor-checkpoint",
        type=Path,
        default=DEFAULT_PREDICTOR_CHECKPOINT,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k-row", type=int, default=20)
    parser.add_argument("--k-col", type=int, default=20)
    parser.add_argument("--predictor-lr", type=float, default=2e-5)
    parser.add_argument("--compressor-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--grad-clip-norm", type=float, default=2.0)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--stable-steps", type=int, default=125_000)
    parser.add_argument("--decay-steps", type=int, default=30_000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--wandb-project", default="tabimpute")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one optimizer step at the maximum shape of each phase.",
    )
    return parser.parse_args()


def load_predictor_config(checkpoint: Path) -> tuple[dict, dict]:
    config_path = checkpoint.parent / "config.json"
    with config_path.open("r", encoding="utf-8") as file:
        run_config = json.load(file)
    return dict(run_config["model"]), dict(run_config["prior"])


def phase_for_step(
    phases: tuple[CurriculumPhase, ...],
    step: int,
) -> tuple[int, CurriculumPhase]:
    for index, phase in enumerate(phases):
        if step <= phase.end_step:
            return index, phase
    raise ValueError(f"Step {step} exceeds curriculum end {phases[-1].end_step}.")


def build_prior(
    phase: CurriculumPhase,
    prior_config: dict,
    *,
    device: str,
) -> MissingnessPrior:
    config = {
        **prior_config,
        "num_rows_low": phase.rows_low,
        "num_rows_high": phase.rows_high,
        "num_cols_low": phase.cols_low,
        "num_cols_high": phase.cols_high,
        "latent_rank_low": 10,
        "latent_rank_high": 10,
        "p_missing": 0.4,
        "apply_feature_warping_prob": 0.0,
        "apply_quantization_prob": 0.0,
    }
    return MissingnessPrior(
        generator_type="latent_factor",
        missingness_type="mcar",
        config=config,
        batch_size=phase.batch_size,
        device=device,
        verbose=False,
        entry_wise_features=False,
    )


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def model_checkpoint(
    model: torch.nn.Module,
    *,
    step: int,
    metrics: dict,
) -> dict:
    return {
        "epoch": step,
        "model": model.state_dict(),
        "metrics": metrics,
    }


def training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    step: int,
    smoothed_loss: float | None,
    best_loss_ema: float,
    metrics: dict,
) -> dict:
    return {
        **model_checkpoint(model, step=step, metrics=metrics),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "smoothed_loss": smoothed_loss,
        "best_loss_ema": best_loss_ema,
    }


def main() -> None:
    args = parse_args()
    if args.smoke:
        phases = tuple(
            CurriculumPhase(
                phase.name,
                index + 1,
                phase.rows_high,
                phase.rows_high,
                phase.cols_high,
                phase.cols_high,
                phase.chunk_rows,
                phase.chunk_cols,
                phase.batch_size,
            )
            for index, phase in enumerate(DEFAULT_PHASES)
        )
        args.warmup_steps = 1
        args.stable_steps = 2
        args.decay_steps = 1
        args.checkpoint_every = 0
    else:
        phases = DEFAULT_PHASES

    total_steps = phases[-1].end_step
    schedule_steps = (
        args.warmup_steps + args.stable_steps + args.decay_steps
    )
    if schedule_steps != total_steps:
        raise ValueError(
            f"Scheduler covers {schedule_steps} steps but curriculum covers "
            f"{total_steps} steps."
        )
    if not args.predictor_checkpoint.is_file():
        raise FileNotFoundError(args.predictor_checkpoint)

    work_dir = args.work_dir or (
        Path("/root/tabular/mcpfn/src/tabimpute/workdir") / args.run_name
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)

    predictor_config, prior_config = load_predictor_config(
        args.predictor_checkpoint
    )
    compressor_config = build_compressor_config_from_predictor(
        predictor_config,
        k_row=args.k_row,
        k_col=args.k_col,
        num_layers=predictor_config["num_layers"],
        mlp_hidden_size=predictor_config["mlp_hidden_size"],
    )
    for key in (
        "rabel_num_kernels",
        "rabel_sigma",
        "rabel_exponent_digits",
        "rabel_exponent_embedding_size",
        "checkpoint_use_reentrant",
    ):
        if key in predictor_config:
            compressor_config[key] = predictor_config[key]

    model = build_compressed_model(
        predictor_config=predictor_config,
        predictor_checkpoint_path=args.predictor_checkpoint,
        compressor_config=compressor_config,
        k_row=args.k_row,
        k_col=args.k_col,
        model_family="pure_rabel",
        device=args.device,
        init_compressor_embedders_from_predictor=True,
        init_compressor_transformers_from_predictor=True,
    ).to(device=device, dtype=compute_dtype)
    optimizer = build_split_optimizer(
        model,
        optimizer_name="adamw",
        predictor_lr=args.predictor_lr,
        compressor_lr=args.compressor_lr,
        encoder_lr=None,
        weight_decay=args.weight_decay,
        optimizer_betas=(0.9, 0.999),
        optimizer_eps=1e-8,
    )
    scheduler = build_compressed_scheduler(
        optimizer=optimizer,
        scheduler_name="warmup_stable_cosine",
        total_steps=total_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        one_cycle_pct_start=0.1,
        stable_steps=args.stable_steps,
        decay_steps=args.decay_steps,
    )

    start_step = 0
    smoothed_loss = None
    best_loss_ema = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["epoch"])
        smoothed_loss = checkpoint.get("smoothed_loss")
        best_loss_ema = float(checkpoint.get("best_loss_ema", float("inf")))

    borders_path = importlib.resources.files("tabimpute") / "data" / "borders.pt"
    with importlib.resources.as_file(borders_path) as path:
        criterion = FullSupportBarDistribution(
            borders=torch.load(path, map_location=device).to(device)
        )

    run_config = {
        "run_name": args.run_name,
        "predictor_checkpoint": str(args.predictor_checkpoint),
        "model": {
            "predictor": predictor_config,
            "compressor": compressor_config,
            "k_row": args.k_row,
            "k_col": args.k_col,
            "model_family": "pure_rabel",
            "target_chunk_sampling": "random",
            "random_targets_per_step": 1,
            "compressor_transformers_initialized_from_predictor": True,
            "compressor_embedders_initialized_from_predictor": True,
        },
        "prior": {
            **prior_config,
            "latent_rank_low": 10,
            "latent_rank_high": 10,
            "p_missing": 0.4,
        },
        "curriculum": [asdict(phase) for phase in phases],
        "optimizer": {
            "name": "adamw",
            "predictor_lr": args.predictor_lr,
            "compressor_lr": args.compressor_lr,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
        },
        "scheduler": {
            "name": "warmup_stable_cosine",
            "warmup_steps": args.warmup_steps,
            "stable_steps": args.stable_steps,
            "decay_steps": args.decay_steps,
            "min_lr_ratio": args.min_lr_ratio,
        },
        "training": {
            "checkpoint_every": args.checkpoint_every,
            "device": args.device,
            "cudnn_sdpa_enabled": False,
            "starts_from_pretrained_predictor": True,
            "scalar_encoder_residual": False,
        },
    }
    with (work_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2, sort_keys=True)

    callback = WandbLoggerCallback(
        project=args.wandb_project,
        name=args.run_name,
        id=args.run_name,
        api_key=os.environ.get("WANDB_KEY"),
        config=run_config,
        log_dir=str(work_dir / "wandb"),
    )

    current_phase_index = -1
    prior = None
    last_metrics: dict[str, object] = {}
    best_checkpoint_ema = best_loss_ema
    model.train()
    try:
        progress = tqdm(range(start_step + 1, total_steps + 1))
        for step in progress:
            phase_index, phase = phase_for_step(phases, step)
            if phase_index != current_phase_index:
                current_phase_index = phase_index
                prior = build_prior(
                    phase,
                    prior_config,
                    device=args.device,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(
                    f"Starting phase {phase_index + 1}/{len(phases)} "
                    f"{phase.name}: rows={phase.rows_low}-{phase.rows_high}, "
                    f"cols={phase.cols_low}-{phase.cols_high}, "
                    f"chunks={phase.chunk_rows}x{phase.chunk_cols}, "
                    f"batch={phase.batch_size}",
                    flush=True,
                )

            step_start = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            prior_batch, _ = prior.get_batch()
            batch_x = prior_batch[0].to(device=device, dtype=compute_dtype)
            batch_target = prior_batch[1].to(
                device=device,
                dtype=compute_dtype,
            )
            _, metrics = train_one_random_chunk_step(
                model=model,
                criterion=criterion,
                batch_x=batch_x,
                batch_target=batch_target,
                chunk_rows=phase.chunk_rows,
                chunk_cols=phase.chunk_cols,
                num_target_chunks=1,
            )
            loss_value = float(metrics["loss_missing"])
            if not torch.isfinite(torch.tensor(loss_value)):
                raise FloatingPointError(
                    f"Non-finite loss at step {step}: {loss_value}"
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()

            smoothed_loss = (
                loss_value
                if smoothed_loss is None
                else 0.995 * smoothed_loss + 0.005 * loss_value
            )
            best_loss_ema = min(best_loss_ema, smoothed_loss)
            lr_by_name = {
                group.get("name", f"group_{index}"): group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            }
            elapsed = time.time() - step_start
            last_metrics = {
                **metrics,
                "loss_missing_ema": smoothed_loss,
                "best_loss_missing_ema": best_loss_ema,
                "lr_predictor": lr_by_name.get("predictor", 0.0),
                "lr_compressor": lr_by_name.get("compressors", 0.0),
                "grad_norm": float(grad_norm.detach().cpu()),
                "grad_was_clipped": float(
                    float(grad_norm.detach().cpu()) > args.grad_clip_norm
                ),
                "phase": phase_index + 1,
                "phase_name": phase.name,
                "matrix_rows": batch_x.shape[1],
                "matrix_cols": batch_x.shape[2],
                "chunk_rows": phase.chunk_rows,
                "chunk_cols": phase.chunk_cols,
                "batch_size": batch_x.shape[0],
            }
            if device.type == "cuda":
                last_metrics["cuda_peak_allocated_gib"] = (
                    torch.cuda.max_memory_allocated(device) / (1024**3)
                )
                last_metrics["cuda_peak_reserved_gib"] = (
                    torch.cuda.max_memory_reserved(device) / (1024**3)
                )
            callback.on_epoch_end(
                step,
                elapsed,
                loss_value,
                None,
                dist=criterion,
                log_dict=last_metrics,
            )
            progress.set_postfix(
                phase=phase_index + 1,
                loss=f"{loss_value:.3f}",
                ema=f"{smoothed_loss:.3f}",
            )

            if step == phase.end_step:
                atomic_torch_save(
                    model_checkpoint(
                        model,
                        step=step,
                        metrics=last_metrics,
                    ),
                    work_dir
                    / f"checkpoint_phase{phase_index + 1}_{step}.pth",
                )
            if (
                args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
            ):
                atomic_torch_save(
                    training_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        step=step,
                        smoothed_loss=smoothed_loss,
                        best_loss_ema=best_loss_ema,
                        metrics=last_metrics,
                    ),
                    work_dir / "latest_checkpoint.pth",
                )
                if smoothed_loss < best_checkpoint_ema:
                    best_checkpoint_ema = smoothed_loss
                    atomic_torch_save(
                        model_checkpoint(
                            model,
                            step=step,
                            metrics=last_metrics,
                        ),
                        work_dir / "best_checkpoint.pth",
                    )

        atomic_torch_save(
            model_checkpoint(
                model,
                step=total_steps,
                metrics=last_metrics,
            ),
            work_dir / f"checkpoint_{total_steps}.pth",
        )
        with (work_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(last_metrics, file, indent=2, sort_keys=True)
    finally:
        callback.close()


if __name__ == "__main__":
    main()
