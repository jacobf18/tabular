#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.resources
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.encoders import normalize_data
from tabimpute.model.model_new_stable_rabel import TabImputeModelPureRaBEL
from tabimpute.train.callbacks import WandbLoggerCallback


ROOT = Path("/root/tabular/mcpfn")
WORKDIR = ROOT / "src/tabimpute/workdir"
BASE_RABEL_CHECKPOINT = (
    WORKDIR
    / "tabimpute-direct-pure-rabel64-rank10-curriculum-160k"
    / "checkpoint_160000.pth"
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


class CausalPanelPrior:
    """Synthetic causal-panel prior matched to CPS/Kansas/Prop99 geometries."""

    def __init__(
        self,
        *,
        batch_size: int,
        device: str,
        seed: int | None = None,
        profile_weights: dict[str, float] | None = None,
    ):
        self.batch_size = int(batch_size)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)
        self.profile_weights = profile_weights or {
            "cps_low_rank": 0.22,
            "cps_mid_rank": 0.28,
            "kansas_single": 0.18,
            "prop99_simultaneous": 0.14,
            "prop99_staggered": 0.12,
            "stock_block": 0.06,
        }
        total = sum(self.profile_weights.values())
        self.profiles = list(self.profile_weights)
        self.profile_probs = np.asarray(
            [self.profile_weights[p] / total for p in self.profiles],
            dtype=np.float64,
        )
        self.last_batch_metrics: dict[str, float | str] = {}

    def _smooth_factors(self, rank: int, cols: int) -> np.ndarray:
        t = np.linspace(0.0, 1.0, cols)
        factors = []
        for _ in range(rank):
            kind = self.rng.choice(["poly", "sin", "rw", "ar"])
            if kind == "poly":
                coefs = self.rng.normal(0.0, [1.0, 0.8, 0.4])
                f = coefs[0] * t + coefs[1] * (t - 0.5) ** 2 + coefs[2]
            elif kind == "sin":
                freq = self.rng.integers(1, 5)
                phase = self.rng.uniform(0.0, 2.0 * np.pi)
                f = np.sin(2.0 * np.pi * freq * t + phase)
            elif kind == "rw":
                f = np.cumsum(self.rng.normal(0.0, 0.15, size=cols))
            else:
                f = np.zeros(cols)
                rho = self.rng.uniform(0.75, 0.97)
                eps = self.rng.normal(0.0, 0.25, size=cols)
                for j in range(1, cols):
                    f[j] = rho * f[j - 1] + eps[j]
            f = (f - f.mean()) / (f.std() + 1e-8)
            factors.append(f)
        return np.stack(factors, axis=0)

    def _sample_panel(
        self,
        rows: int,
        cols: int,
        rank_low: int,
        rank_high: int,
        *,
        noise_low: float,
        noise_high: float,
        row_scale: float,
        time_scale: float,
        factor_scale: float,
    ) -> np.ndarray:
        rank = int(self.rng.integers(rank_low, min(rank_high, rows, cols) + 1))
        t = np.linspace(0.0, 1.0, cols)

        row_effect = self.rng.normal(0.0, row_scale, size=(rows, 1))
        linear = self.rng.normal(0.0, time_scale) * t
        quadratic = self.rng.normal(0.0, 0.5 * time_scale) * (t - 0.5) ** 2
        seasonal = (
            self.rng.normal(0.0, 0.35 * time_scale)
            * np.sin(2.0 * np.pi * self.rng.integers(1, 4) * t + self.rng.uniform(0, 2 * np.pi))
        )
        random_walk = np.cumsum(self.rng.normal(0.0, 0.05 * time_scale, size=cols))
        time_effect = (linear + quadratic + seasonal + random_walk)[None, :]

        loadings = self.rng.normal(0.0, 1.0, size=(rows, rank))
        factors = self._smooth_factors(rank, cols)
        interactive = factor_scale * (loadings @ factors) / math.sqrt(rank)

        # Unit-specific slopes are crucial for synthetic-control style panels.
        unit_slopes = self.rng.normal(0.0, 0.35 * factor_scale, size=(rows, 1))
        unit_trends = unit_slopes * (t[None, :] - 0.5)

        if self.rng.random() < 0.45:
            n_groups = int(self.rng.integers(3, 8))
            groups = self.rng.integers(0, n_groups, size=rows)
            group_paths = self.rng.normal(0.0, 0.2, size=(n_groups, rank)) @ factors
            group_paths = group_paths / math.sqrt(rank)
            cluster_component = 0.4 * factor_scale * group_paths[groups]
        else:
            cluster_component = 0.0

        noise = self.rng.uniform(noise_low, noise_high) * self.rng.normal(
            0.0,
            1.0,
            size=(rows, cols),
        )
        y = row_effect + time_effect + interactive + unit_trends + cluster_component + noise
        if self.rng.random() < 0.35:
            y = y + self.rng.normal(0.0, 0.08, size=(rows, 1)) * np.sign(y)
        return y.astype(np.float32)

    def _mask_simultaneous(
        self,
        rows: int,
        cols: int,
        *,
        nt_low: int,
        nt_high: int,
        t0_low: float,
        t0_high: float,
        bottom_rows_prob: float,
    ) -> np.ndarray:
        nt = int(self.rng.integers(nt_low, min(nt_high, rows - 2) + 1))
        t0 = int(round(cols * self.rng.uniform(t0_low, t0_high)))
        t0 = min(max(t0, 1), cols - 1)
        if self.rng.random() < bottom_rows_prob:
            treated_rows = np.arange(rows - nt, rows)
        else:
            treated_rows = self.rng.choice(rows, size=nt, replace=False)
        mask = np.zeros((rows, cols), dtype=bool)
        mask[treated_rows[:, None], np.arange(t0, cols)] = True
        return mask

    def _mask_staggered(self, rows: int, cols: int) -> np.ndarray:
        nt = int(self.rng.integers(max(4, int(0.55 * rows)), max(5, rows - 2) + 1))
        first_t0 = int(round(cols * self.rng.uniform(0.30, 0.55)))
        first_t0 = min(max(first_t0, 1), cols - 2)
        treated_rows = self.rng.choice(rows, size=nt, replace=False)
        mask = np.zeros((rows, cols), dtype=bool)
        for row in treated_rows:
            adopt = int(self.rng.integers(first_t0, cols))
            mask[row, adopt:] = True
        return mask

    def _sample_profile_fixed_shape(
        self,
        profile: str,
        rows: int,
        cols: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if profile == "kansas_single":
            y = self._sample_panel(
                rows,
                cols,
                2,
                6,
                noise_low=0.015,
                noise_high=0.09,
                row_scale=0.45,
                time_scale=0.8,
                factor_scale=0.9,
            )
            post_len = int(self.rng.integers(10, min(26, cols - 1)))
            t0 = cols - post_len
            row = int(self.rng.integers(0, rows))
            mask = np.zeros((rows, cols), dtype=bool)
            mask[row, t0:] = True
        elif profile == "prop99_simultaneous":
            y = self._sample_panel(
                rows,
                cols,
                2,
                6,
                noise_low=0.02,
                noise_high=0.14,
                row_scale=0.9,
                time_scale=1.2,
                factor_scale=1.2,
            )
            mask = self._mask_simultaneous(
                rows,
                cols,
                nt_low=5,
                nt_high=12,
                t0_low=0.35,
                t0_high=0.90,
                bottom_rows_prob=0.25,
            )
        elif profile == "prop99_staggered":
            y = self._sample_panel(
                rows,
                cols,
                2,
                6,
                noise_low=0.02,
                noise_high=0.14,
                row_scale=0.9,
                time_scale=1.2,
                factor_scale=1.2,
            )
            mask = self._mask_staggered(rows, cols)
        elif profile == "stock_block":
            y = self._sample_panel(
                rows,
                cols,
                4,
                10,
                noise_low=0.015,
                noise_high=0.08,
                row_scale=0.3,
                time_scale=0.4,
                factor_scale=1.0,
            )
            treated_rows = self.rng.choice(rows, size=max(1, rows // 2), replace=False)
            mask = np.zeros((rows, cols), dtype=bool)
            mask[treated_rows[:, None], np.arange(cols // 2, cols)] = True
        else:
            if profile == "cps_low_rank":
                rank_low, rank_high = 4, 14
                factor_scale, noise_high = 1.2, 0.18
            else:
                rank_low, rank_high = 10, 30
                factor_scale, noise_high = 1.0, 0.45
            y = self._sample_panel(
                rows,
                cols,
                rank_low,
                rank_high,
                noise_low=0.06,
                noise_high=noise_high,
                row_scale=0.7,
                time_scale=0.7,
                factor_scale=factor_scale,
            )
            post_len_high = min(16, cols - 1)
            nt_high = min(12, rows - 2)
            mask = self._mask_simultaneous(
                rows,
                cols,
                nt_low=6,
                nt_high=nt_high,
                t0_low=max(0.45, 1.0 - post_len_high / cols),
                t0_high=0.82,
                bottom_rows_prob=0.70,
            )
        return y, mask

    def _sample_profile(self) -> tuple[str, np.ndarray, np.ndarray]:
        profile = str(self.rng.choice(self.profiles, p=self.profile_probs))
        if profile == "kansas_single":
            rows = 50
            cols = int(self.rng.integers(85, 121))
        elif profile in {"prop99_simultaneous", "prop99_staggered"}:
            rows = int(self.rng.integers(35, 46))
            cols = int(self.rng.integers(28, 41))
        elif profile == "stock_block":
            rows = int(self.rng.integers(35, 71))
            cols = int(self.rng.integers(35, 99))
        else:
            rows = int(self.rng.integers(45, 61))
            cols = int(self.rng.integers(35, 51))
        y, mask = self._sample_profile_fixed_shape(profile, rows, cols)
        return profile, y, mask

    def get_batch(self):
        profile, y0, mask0 = self._sample_profile()
        rows, cols = y0.shape
        x_list = []
        target_list = []
        mask_fracs = []
        for i in range(self.batch_size):
            if i == 0:
                y, mask = y0, mask0
            else:
                # Same shape/profile within a batch, different latent draw.
                y, mask = self._sample_profile_fixed_shape(profile, rows, cols)
            x = torch.tensor(y, dtype=torch.float32)
            missing = torch.tensor(mask, dtype=torch.bool)
            x_missing = x.clone()
            x_missing[missing] = torch.nan
            x_missing_norm, (mean, std) = normalize_data(
                x_missing.unsqueeze(1),
                return_scaling=True,
            )
            target_norm = normalize_data(
                x.unsqueeze(1),
                mean=mean,
                std=std,
            )
            x_list.append(x_missing_norm.squeeze(1))
            target_list.append(target_norm.squeeze(1))
            mask_fracs.append(float(missing.float().mean()))
        batch_x = torch.stack(x_list, dim=0).to(self.device)
        batch_target = torch.stack(target_list, dim=0).to(self.device)
        d = torch.full((self.batch_size,), cols, dtype=torch.long, device=self.device)
        seq_lens = torch.full((self.batch_size,), rows, dtype=torch.long, device=self.device)
        train_sizes = torch.full((self.batch_size,), rows, dtype=torch.long, device=self.device)
        weights = torch.ones(self.batch_size, dtype=torch.float32, device=self.device)
        self.last_batch_metrics = {
            "prior_profile": profile,
            "prior_rows": float(rows),
            "prior_cols": float(cols),
            "prior_missing_frac": float(np.mean(mask_fracs)),
        }
        return (batch_x, batch_target, d, seq_lens, train_sizes), weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="tabimpute-direct-pure-rabel64-causal-panel-bs16-from-rank10-100k")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--stable-steps", type=int, default=78_000)
    parser.add_argument("--decay-steps", type=int, default=20_000)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--init-checkpoint", type=Path, default=BASE_RABEL_CHECKPOINT)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--wandb-project", default="tabimpute")
    parser.add_argument(
        "--profile-weights-json",
        default=None,
        help="Optional JSON dict overriding causal prior profile weights.",
    )
    parser.add_argument(
        "--model-overrides-json",
        default=None,
        help="Optional JSON dict for checkpoint-compatible model config overrides.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    if warmup_steps + stable_steps + decay_steps != steps:
        raise ValueError("warmup + stable + decay must equal --steps")

    def lr_lambda(current_step: int) -> float:
        step = current_step + 1
        if step <= warmup_steps:
            return step / max(1, warmup_steps)
        if step <= warmup_steps + stable_steps:
            return 1.0
        progress = (step - warmup_steps - stable_steps) / max(1, decay_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def checkpoint_payload(model: torch.nn.Module, step: int, metrics: dict) -> dict:
    return {"epoch": step, "model": model.state_dict(), "metrics": metrics}


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.steps = 8
        args.warmup_steps = 2
        args.stable_steps = 4
        args.decay_steps = 2
        args.checkpoint_every = 0
        args.batch_size = 4

    work_dir = args.work_dir or (WORKDIR / args.run_name)
    work_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_config = dict(MODEL_CONFIG)
    if args.model_overrides_json:
        model_config.update(json.loads(args.model_overrides_json))

    profile_weights = None
    if args.profile_weights_json:
        profile_weights = json.loads(args.profile_weights_json)

    model = TabImputeModelPureRaBEL(**model_config).to(device=device, dtype=compute_dtype)
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
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        stable_steps=args.stable_steps,
        decay_steps=args.decay_steps,
        min_lr_ratio=args.min_lr / args.lr,
    )

    start_step = 0
    smoothed_loss = None
    best_smoothed_loss = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["epoch"])
        smoothed_loss = checkpoint.get("smoothed_loss")
        best_smoothed_loss = float(checkpoint.get("best_smoothed_loss", best_smoothed_loss))
    elif args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)

    prior = CausalPanelPrior(
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        profile_weights=profile_weights,
    )

    borders_path = importlib.resources.files("tabimpute") / "data" / "borders.pt"
    with importlib.resources.as_file(borders_path) as path:
        criterion = FullSupportBarDistribution(
            borders=torch.load(path, map_location=device).to(device)
        )

    run_config = {
        "run_name": args.run_name,
        "model_family": "pure_rabel_memory_efficient",
        "model": model_config,
        "prior_kind": "causal_panel",
        "prior": {
            "profile_weights": prior.profile_weights,
            "normalization": "observed-column",
            "mask_types": [
                "cps_block",
                "kansas_single_treated",
                "prop99_simultaneous",
                "prop99_staggered",
                "stock_block",
            ],
        },
        "optimizer": {
            "name": "adamw",
            "lr": args.lr,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
        },
        "scheduler": {
            "name": "warmup_stable_cosine",
            "warmup_steps": args.warmup_steps,
            "stable_steps": args.stable_steps,
            "decay_steps": args.decay_steps,
        },
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "checkpoint_every": args.checkpoint_every,
            "decode_only_missing": True,
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            "device": args.device,
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

    model.train()
    last_metrics = {}
    try:
        progress = tqdm(range(start_step + 1, args.steps + 1))
        for step in progress:
            step_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            (batch_x, batch_target, _, _, _), _ = prior.get_batch()
            batch_x = batch_x.to(device=device, dtype=compute_dtype)
            batch_target = batch_target.to(device=device, dtype=compute_dtype)
            missing_mask = torch.isnan(batch_x)
            if not missing_mask.any():
                raise RuntimeError("Causal prior produced no hidden cells.")

            logits = model(batch_x, decode_mask=missing_mask)
            target_missing = batch_target[missing_mask]
            loss = criterion(logits, target_missing).mean()
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
            last_metrics = {
                "loss_missing": loss_value,
                "loss_missing_ema": smoothed_loss,
                "mae_missing": float(mae.detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": float(grad_norm.detach().cpu()),
                "grad_was_clipped": float(float(grad_norm.detach().cpu()) > args.grad_clip_norm),
                "batch_size": args.batch_size,
                "matrix_rows": batch_x.shape[1],
                "matrix_cols": batch_x.shape[2],
                **prior.last_batch_metrics,
            }
            if device.type == "cuda":
                last_metrics["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / (1024**3)
                last_metrics["cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / (1024**3)

            callback.on_epoch_end(
                step,
                time.time() - step_start,
                loss_value,
                None,
                dist=criterion,
                log_dict=last_metrics,
            )
            progress.set_postfix(
                profile=str(prior.last_batch_metrics.get("prior_profile", "")),
                loss=f"{loss_value:.3f}",
                ema=f"{smoothed_loss:.3f}",
            )

            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                payload = {
                    **checkpoint_payload(model, step, last_metrics),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "smoothed_loss": smoothed_loss,
                    "best_smoothed_loss": best_smoothed_loss,
                }
                atomic_torch_save(payload, work_dir / "latest_checkpoint.pth")
                atomic_torch_save(checkpoint_payload(model, step, last_metrics), work_dir / f"checkpoint_{step}.pth")
                if smoothed_loss is not None and smoothed_loss < best_smoothed_loss:
                    best_smoothed_loss = smoothed_loss
                    best_payload = checkpoint_payload(
                        model,
                        step,
                        {**last_metrics, "best_smoothed_loss": best_smoothed_loss},
                    )
                    atomic_torch_save(best_payload, work_dir / "best_checkpoint.pth")
                    with (work_dir / "best_metrics.json").open("w", encoding="utf-8") as file:
                        json.dump(best_payload["metrics"], file, indent=2, sort_keys=True)

        atomic_torch_save(
            checkpoint_payload(model, args.steps, last_metrics),
            work_dir / f"checkpoint_{args.steps}.pth",
        )
        with (work_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(last_metrics, file, indent=2, sort_keys=True)
    finally:
        callback.close()


if __name__ == "__main__":
    main()
