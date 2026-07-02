#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.resources
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable_rabel import TabImputeModelPureRaBEL
from tabimpute.prior.spectrum_training_set_generation import SpectrumMissingnessPrior
from tabimpute.prior.training_set_generation import MissingnessPrior
from tabimpute.train.callbacks import WandbLoggerCallback


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    end_step: int
    rows_low: int
    rows_high: int
    cols_low: int
    cols_high: int
    batch_size: int


DEFAULT_PHASES = (
    CurriculumPhase("50_square", 50_000, 50, 50, 50, 50, 32),
    CurriculumPhase("50_to_100", 85_000, 50, 100, 50, 100, 16),
    CurriculumPhase("100_to_200", 125_000, 100, 200, 100, 200, 8),
    CurriculumPhase("200_to_300", 160_000, 200, 300, 200, 300, 4),
)

MODEL_CONFIG = {
    "embedding_size": 768,
    "num_attention_heads": 16,
    "mlp_hidden_size": 1536,
    "num_layers": 8,
    "num_outputs": 5000,
    "num_cls": 12,
    "use_rope": True,
    "rope_base": 10000.0,
    "rope_fraction": 0.5,
    "use_absolute_positional_embeddings": False,
    "positional_damping_factor": 0.1,
    "attention_dropout": 0.0,
    "ffn_dropout": 0.05,
    "drop_path_rate": 0.01,
    "residual_scale_init": 0.5,
    "embedding_dropout": 0.0,
    "rms_norm_eps": 1e-6,
    "use_activation_checkpointing": True,
    "checkpoint_use_reentrant": False,
    "rabel_num_kernels": 64,
    "rabel_sigma": 1.0,
    "rabel_exponent_digits": 1,
    "rabel_exponent_embedding_size": 32,
}

PRIOR_CONFIG = {
    "p_missing": 0.4,
    "apply_feature_warping_prob": 0.0,
    "apply_quantization_prob": 0.0,
    "latent_rank_low": 10,
    "latent_rank_high": 10,
    "latent_spike_p": 0.3,
    "latent_slab_sigma": 2.0,
}

SPECTRUM_DECAY_PRIOR_CONFIG = {
    "p_missing": 0.4,
    "spectrum_rank_low": 80,
    "spectrum_rank_high": 128,
    "spectrum_alpha_low": 0.30,
    "spectrum_alpha_high": 0.50,
    "spectrum_head_weight_low": 3.0,
    "spectrum_head_weight_high": 6.0,
    "spectrum_head_exp_scale_low": 2.5,
    "spectrum_head_exp_scale_high": 4.5,
    "spectrum_head_exp_power_low": 2.2,
    "spectrum_head_exp_power_high": 3.0,
    "spectrum_head_handoff_scale_low": 1.0,
    "spectrum_head_handoff_scale_high": 4.0,
    "spectrum_tail_amplitude_low": 0.25,
    "spectrum_tail_amplitude_high": 0.45,
    "spectrum_taper_strength_low": 1.2,
    "spectrum_taper_strength_high": 1.8,
    "spectrum_taper_power_low": 3.0,
    "spectrum_taper_power_high": 4.5,
    "spectrum_scale_low": 1.0,
    "spectrum_scale_high": 1.0,
    "spectrum_noise_std_low": 0.001,
    "spectrum_noise_std_high": 0.005,
    "spectrum_target_rank": 40,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default="tabimpute-direct-pure-rabel64-rank10-curriculum-160k",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--post-warmup-lr",
        type=float,
        default=None,
        help=(
            "Optional LR to use after warmup. This allows a short high-LR "
            "warmup/early phase followed by a lower stable LR."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--stable-steps", type=int, default=125_000)
    parser.add_argument("--decay-steps", type=int, default=30_000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--wandb-project", default="tabimpute")
    parser.add_argument(
        "--prior-kind",
        choices=("latent_factor", "spectrum_decay"),
        default="latent_factor",
    )
    parser.add_argument("--p-missing", type=float, default=None)
    parser.add_argument("--spectrum-rank-low", type=int, default=None)
    parser.add_argument("--spectrum-rank-high", type=int, default=None)
    parser.add_argument("--spectrum-target-rank", type=int, default=None)
    parser.add_argument("--spectrum-alpha-low", type=float, default=None)
    parser.add_argument("--spectrum-alpha-high", type=float, default=None)
    parser.add_argument("--spectrum-noise-std-low", type=float, default=None)
    parser.add_argument("--spectrum-noise-std-high", type=float, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one optimizer step at each curriculum size.",
    )
    return parser.parse_args()


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float,
    post_warmup_lr_ratio: float = 1.0,
) -> LambdaLR:
    total_steps = warmup_steps + stable_steps + decay_steps
    if post_warmup_lr_ratio <= 0:
        raise ValueError(
            f"post_warmup_lr_ratio must be positive, got {post_warmup_lr_ratio}"
        )

    def lr_lambda(current_step: int) -> float:
        step = current_step + 1
        if step <= warmup_steps:
            return step / max(1, warmup_steps)
        if step <= warmup_steps + stable_steps:
            return post_warmup_lr_ratio
        progress = (step - warmup_steps - stable_steps) / max(1, decay_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        return post_warmup_lr_ratio * decayed

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler.total_steps = total_steps
    return scheduler


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
    *,
    device: str,
    prior_kind: str,
    base_config: dict,
) -> MissingnessPrior | SpectrumMissingnessPrior:
    config = {
        **base_config,
        "num_rows_low": phase.rows_low,
        "num_rows_high": phase.rows_high,
        "num_cols_low": phase.cols_low,
        "num_cols_high": phase.cols_high,
    }
    if prior_kind == "spectrum_decay":
        return SpectrumMissingnessPrior(
            missingness_type="mcar",
            config=config,
            batch_size=phase.batch_size,
            device=device,
            verbose=False,
            entry_wise_features=False,
        )
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
    scheduler: LambdaLR,
    *,
    step: int,
    smoothed_loss: float | None,
    best_smoothed_loss: float,
    metrics: dict,
) -> dict:
    return {
        **model_checkpoint(model, step=step, metrics=metrics),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "smoothed_loss": smoothed_loss,
        "best_smoothed_loss": best_smoothed_loss,
    }


def build_base_prior_config(args: argparse.Namespace) -> dict:
    if args.prior_kind == "spectrum_decay":
        config = dict(SPECTRUM_DECAY_PRIOR_CONFIG)
        spectrum_overrides = {
            "p_missing": args.p_missing,
            "spectrum_rank_low": args.spectrum_rank_low,
            "spectrum_rank_high": args.spectrum_rank_high,
            "spectrum_target_rank": args.spectrum_target_rank,
            "spectrum_alpha_low": args.spectrum_alpha_low,
            "spectrum_alpha_high": args.spectrum_alpha_high,
            "spectrum_noise_std_low": args.spectrum_noise_std_low,
            "spectrum_noise_std_high": args.spectrum_noise_std_high,
        }
        for key, value in spectrum_overrides.items():
            if value is not None:
                config[key] = value
        return config

    config = dict(PRIOR_CONFIG)
    if args.p_missing is not None:
        config["p_missing"] = args.p_missing
    return config


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
                phase.batch_size,
            )
            for index, phase in enumerate(DEFAULT_PHASES)
        )
        args.warmup_steps = 1
        args.stable_steps = len(phases) - 1
        args.decay_steps = 0
        args.checkpoint_every = 0
    else:
        phases = DEFAULT_PHASES

    total_steps = phases[-1].end_step
    scheduler_steps = args.warmup_steps + args.stable_steps + args.decay_steps
    if scheduler_steps != total_steps:
        raise ValueError(
            f"Scheduler covers {scheduler_steps} steps but curriculum covers "
            f"{total_steps} steps."
        )

    work_dir = args.work_dir or (
        Path("/root/tabular/mcpfn/src/tabimpute/workdir") / args.run_name
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base_prior_config = build_base_prior_config(args)
    if device.type == "cuda":
        # cuDNN SDPA can fail during bf16 backward when the sampled matrix
        # dimensions change between steps. Flash/memory-efficient SDPA is
        # stable for this variable-shape curriculum.
        torch.backends.cuda.enable_cudnn_sdp(False)

    model = TabImputeModelPureRaBEL(**MODEL_CONFIG).to(
        device=device,
        dtype=compute_dtype,
    )
    post_warmup_lr_ratio = 1.0
    if args.post_warmup_lr is not None:
        post_warmup_lr_ratio = args.post_warmup_lr / args.lr
        if post_warmup_lr_ratio > 1.0:
            raise ValueError(
                "--post-warmup-lr should be <= --lr for the warm-drop schedule."
            )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=device.type == "cuda",
    )
    scheduler = make_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        stable_steps=args.stable_steps,
        decay_steps=args.decay_steps,
        min_lr_ratio=args.min_lr_ratio,
        post_warmup_lr_ratio=post_warmup_lr_ratio,
    )

    start_step = 0
    smoothed_loss = None
    best_smoothed_loss = float("inf")
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
        best_smoothed_loss = float(
            checkpoint.get("best_smoothed_loss", best_smoothed_loss)
        )

    borders_path = importlib.resources.files("tabimpute") / "data" / "borders.pt"
    with importlib.resources.as_file(borders_path) as path:
        criterion = FullSupportBarDistribution(
            borders=torch.load(path, map_location=device).to(device)
        )

    run_config = {
        "run_name": args.run_name,
        "model_family": "pure_rabel_memory_efficient",
        "model": MODEL_CONFIG,
        "prior_kind": args.prior_kind,
        "prior": base_prior_config,
        "curriculum": [asdict(phase) for phase in phases],
        "optimizer": {
            "name": "adamw",
            "lr": args.lr,
            "post_warmup_lr": args.post_warmup_lr,
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
            "decode_only_missing": True,
            "cudnn_sdpa_enabled": False,
            "device": args.device,
            "starts_from_pretrained_weights": False,
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
    last_metrics = {}
    model.train()
    try:
        progress = tqdm(range(start_step + 1, total_steps + 1))
        for step in progress:
            phase_index, phase = phase_for_step(phases, step)
            if phase_index != current_phase_index:
                current_phase_index = phase_index
                prior = build_prior(
                    phase,
                    device=args.device,
                    prior_kind=args.prior_kind,
                    base_config=base_prior_config,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(
                    f"Starting phase {phase_index + 1}/{len(phases)} "
                    f"{phase.name}: rows={phase.rows_low}-{phase.rows_high}, "
                    f"cols={phase.cols_low}-{phase.cols_high}, "
                    f"batch={phase.batch_size}",
                    flush=True,
                )

            step_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            (batch_x, batch_target, _, _, _), _ = prior.get_batch()
            batch_x = batch_x.to(device=device, dtype=compute_dtype)
            batch_target = batch_target.to(device=device, dtype=compute_dtype)
            missing_mask = torch.isnan(batch_x)
            if not missing_mask.any():
                raise RuntimeError("Generated batch contains no missing entries.")

            logits = model(batch_x, decode_mask=missing_mask)
            target_missing = batch_target[missing_mask]
            losses = criterion(logits, target_missing)
            loss = losses.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss}")

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                medians = criterion.median(logits)
                mae = (medians - target_missing).abs().mean()
            loss_value = float(loss.detach().cpu())
            smoothed_loss = (
                loss_value
                if smoothed_loss is None
                else 0.995 * smoothed_loss + 0.005 * loss_value
            )
            elapsed = time.time() - step_start
            last_metrics = {
                "loss_missing": loss_value,
                "loss_missing_ema": smoothed_loss,
                "mae_missing": float(mae.detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": float(grad_norm.detach().cpu()),
                "grad_was_clipped": float(
                    float(grad_norm.detach().cpu()) > args.grad_clip_norm
                ),
                "phase": phase_index + 1,
                "phase_name": phase.name,
                "matrix_rows": batch_x.shape[1],
                "matrix_cols": batch_x.shape[2],
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

            phase_ended = step == phase.end_step
            if phase_ended:
                phase_path = work_dir / f"checkpoint_phase{phase_index + 1}_{step}.pth"
                atomic_torch_save(
                    model_checkpoint(
                        model,
                        step=step,
                        metrics=last_metrics,
                    ),
                    phase_path,
                )
            if (
                args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
            ):
                best_payload = None
                if smoothed_loss is not None and smoothed_loss < best_smoothed_loss:
                    best_smoothed_loss = smoothed_loss
                    best_payload = model_checkpoint(
                        model,
                        step=step,
                        metrics={
                            **last_metrics,
                            "best_smoothed_loss": best_smoothed_loss,
                        },
                    )
                atomic_torch_save(
                    training_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        step=step,
                        smoothed_loss=smoothed_loss,
                        best_smoothed_loss=best_smoothed_loss,
                        metrics=last_metrics,
                    ),
                    work_dir / "latest_checkpoint.pth",
                )
                if best_payload is not None:
                    atomic_torch_save(best_payload, work_dir / "best_checkpoint.pth")
                    with (work_dir / "best_metrics.json").open(
                        "w",
                        encoding="utf-8",
                    ) as file:
                        json.dump(best_payload["metrics"], file, indent=2, sort_keys=True)

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
