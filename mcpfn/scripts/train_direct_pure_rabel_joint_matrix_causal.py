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
from tabimpute.prior.spectrum_training_set_generation import SpectrumMissingnessPrior
from tabimpute.train.callbacks import WandbLoggerCallback


ROOT = Path("/root/tabular/mcpfn")
WORKDIR = ROOT / "src/tabimpute/workdir"
BASE_SPECTRUM_CHECKPOINT = (
    WORKDIR
    / "tabimpute-direct-pure-rabel64-spectrum-decay-top40-curriculum-160k-lr2e5"
    / "best_checkpoint.pth"
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

MODEL_FAMILY = "pure_rabel_memory_efficient"


SPECTRUM_CONFIG = {
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


class TabularCATEBlockPrior:
    """Synthetic Do-PFN-style matrix-completion prior without using benchmark data.

    Each matrix is [treatment, covariates..., outcome]. The covariates are fully
    observed. The outcome is observed for the train block and missing for a
    held-out row block, matching the matrix built by the Do-PFN MC adapter.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        device: str,
        rows_low: int,
        rows_high: int,
        covariates_low: int,
        covariates_high: int,
        heldout_frac_low: float,
        heldout_frac_high: float,
        query_treatment_mode: str,
        tau_scale: float,
        query_noise_scale: float,
        seed: int | None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.device = device
        self.rows_low = int(rows_low)
        self.rows_high = int(rows_high)
        self.covariates_low = int(covariates_low)
        self.covariates_high = int(covariates_high)
        self.heldout_frac_low = float(heldout_frac_low)
        self.heldout_frac_high = float(heldout_frac_high)
        if query_treatment_mode not in {"factual", "random", "paired"}:
            raise ValueError(
                "query_treatment_mode must be one of {'factual', 'random', 'paired'}, "
                f"got {query_treatment_mode!r}."
            )
        self.query_treatment_mode = query_treatment_mode
        self.tau_scale = float(tau_scale)
        self.query_noise_scale = float(query_noise_scale)
        self.rng = np.random.default_rng(seed)
        self.last_batch_metrics: dict[str, float | str] = {}

    def _sample_shape(self) -> tuple[int, int, int]:
        rows = int(self.rng.integers(self.rows_low, self.rows_high + 1))
        covariates = int(self.rng.integers(self.covariates_low, self.covariates_high + 1))
        heldout_frac = float(self.rng.uniform(self.heldout_frac_low, self.heldout_frac_high))
        n_test = int(round(rows * heldout_frac))
        n_test = min(max(n_test, 8), rows - 8)
        return rows, covariates, n_test

    def _sample_one(
        self,
        rows: int,
        covariates: int,
        n_test: int,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        z_rank_low = 1 if covariates == 1 else 2
        z_rank = int(self.rng.integers(z_rank_low, min(8, covariates) + 1))
        z = self.rng.normal(0.0, 1.0, size=(rows, z_rank))
        loadings = self.rng.normal(0.0, 1.0 / math.sqrt(z_rank), size=(z_rank, covariates))
        x = z @ loadings + self.rng.normal(0.0, 0.35, size=(rows, covariates))
        x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)

        prop_w = self.rng.normal(0.0, 0.8 / math.sqrt(covariates), size=covariates)
        logits = x @ prop_w
        if covariates >= 2:
            logits += 0.4 * np.sin(x[:, 0]) - 0.25 * x[:, 1] ** 2
        prop = 1.0 / (1.0 + np.exp(-np.clip(logits, -4.0, 4.0)))
        treatment = self.rng.binomial(1, prop).astype(np.float64)

        base_w = self.rng.normal(0.0, 1.0 / math.sqrt(covariates), size=covariates)
        baseline = x @ base_w
        if covariates >= 4:
            baseline += 0.35 * np.sin(x[:, 0] + 0.5 * x[:, 1])
            baseline += 0.20 * x[:, 2] * x[:, 3]

        tau_w = self.rng.normal(0.0, 0.45 / math.sqrt(covariates), size=covariates)
        tau = self.rng.normal(0.35, 0.25) + x @ tau_w
        if covariates >= 3:
            tau += 0.25 * np.tanh(x[:, 0] - x[:, 2])
        tau *= self.tau_scale

        noise = self.rng.uniform(0.03, 0.20) * self.rng.normal(0.0, 1.0, size=rows)
        y = baseline + treatment * tau + noise

        matrix = np.concatenate(
            [treatment[:, None], x, y[:, None]],
            axis=1,
        ).astype(np.float32)
        missing = np.zeros_like(matrix, dtype=bool)

        if self.rng.random() < 0.85:
            test_rows = np.arange(rows - n_test, rows)
        else:
            test_rows = self.rng.choice(rows, size=n_test, replace=False)

        train_rows_mask = np.ones(rows, dtype=bool)
        train_rows_mask[test_rows] = False
        train_rows = int(train_rows_mask.sum())

        if self.query_treatment_mode == "paired":
            query_noise_std = self.query_noise_scale * self.rng.uniform(0.03, 0.20)
            query_noise_0 = query_noise_std * self.rng.normal(0.0, 1.0, size=n_test)
            query_noise_1 = query_noise_std * self.rng.normal(0.0, 1.0, size=n_test)
            train_matrix = matrix[train_rows_mask]
            query_0 = np.concatenate(
                [
                    np.zeros((n_test, 1), dtype=np.float32),
                    x[test_rows].astype(np.float32),
                    (baseline[test_rows] + query_noise_0).astype(np.float32)[:, None],
                ],
                axis=1,
            )
            query_1 = np.concatenate(
                [
                    np.ones((n_test, 1), dtype=np.float32),
                    x[test_rows].astype(np.float32),
                    (baseline[test_rows] + tau[test_rows] + query_noise_1).astype(np.float32)[:, None],
                ],
                axis=1,
            )
            matrix = np.concatenate([train_matrix, query_0, query_1], axis=0).astype(np.float32)
            missing = np.zeros_like(matrix, dtype=bool)
            missing[train_rows:, -1] = True
            return matrix, missing, train_rows, 2 * n_test

        missing[test_rows, -1] = True

        if self.query_treatment_mode == "random":
            query_treatment = self.rng.binomial(1, 0.5, size=n_test).astype(np.float64)
            query_noise = (
                self.query_noise_scale
                * self.rng.uniform(0.03, 0.20)
                * self.rng.normal(0.0, 1.0, size=n_test)
            )
            matrix[test_rows, 0] = query_treatment.astype(np.float32)
            matrix[test_rows, -1] = (
                baseline[test_rows] + query_treatment * tau[test_rows] + query_noise
            ).astype(np.float32)
        return matrix, missing, train_rows, n_test

    def get_batch(self):
        rows, covariates, n_test = self._sample_shape()
        x_list = []
        target_list = []
        missing_fracs = []
        train_rows_actual = rows - n_test
        test_rows_actual = n_test
        for _ in range(self.batch_size):
            dense, missing, train_rows_actual, test_rows_actual = self._sample_one(rows, covariates, n_test)
            x_missing = torch.tensor(dense)
            missing_t = torch.tensor(missing, dtype=torch.bool)
            x_missing[missing_t] = torch.nan
            x_norm, (mean, std) = normalize_data(x_missing.unsqueeze(1), return_scaling=True)
            target_norm = normalize_data(torch.tensor(dense).unsqueeze(1), mean=mean, std=std)
            x_list.append(x_norm.squeeze(1))
            target_list.append(target_norm.squeeze(1))
            missing_fracs.append(float(missing_t.float().mean()))

        batch_x = torch.stack(x_list, dim=0).to(self.device)
        batch_target = torch.stack(target_list, dim=0).to(self.device)
        d = torch.full((self.batch_size,), covariates + 2, dtype=torch.long, device=self.device)
        seq_lens = torch.full((self.batch_size,), batch_x.shape[1], dtype=torch.long, device=self.device)
        train_sizes = torch.full((self.batch_size,), train_rows_actual, dtype=torch.long, device=self.device)
        weights = torch.ones(self.batch_size, dtype=torch.float32, device=self.device)
        self.last_batch_metrics = {
            "prior_family": "tabular_cate_block",
            "prior_rows": float(batch_x.shape[1]),
            "prior_cols": float(covariates + 2),
            "prior_train_rows": float(train_rows_actual),
            "prior_test_rows": float(test_rows_actual),
            "prior_missing_frac": float(np.mean(missing_fracs)),
            "prior_tau_scale": float(self.tau_scale),
        }
        return (batch_x, batch_target, d, seq_lens, train_sizes), weights


class SkinnyBlockLowRankPrior:
    """Generic rectangular low-rank prior with a held-out row block in one column.

    This matches the out-of-sample matrix-completion geometry used by the Do-PFN
    adapter without introducing a causal-specific loss or potential-outcome
    target. The model sees all columns for train rows and must infer a held-out
    last column from the row covariates/features.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        device: str,
        rows_low: int,
        rows_high: int,
        cols_low: int,
        cols_high: int,
        rank_low: int,
        rank_high: int,
        heldout_frac_low: float,
        heldout_frac_high: float,
        extra_mcar_prob: float,
        seed: int | None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.device = device
        self.rows_low = int(rows_low)
        self.rows_high = int(rows_high)
        self.cols_low = int(cols_low)
        self.cols_high = int(cols_high)
        self.rank_low = int(rank_low)
        self.rank_high = int(rank_high)
        self.heldout_frac_low = float(heldout_frac_low)
        self.heldout_frac_high = float(heldout_frac_high)
        self.extra_mcar_prob = float(extra_mcar_prob)
        self.rng = np.random.default_rng(seed)
        self.last_batch_metrics: dict[str, float | str] = {}

    def _sample_shape(self) -> tuple[int, int, int, int]:
        rows = int(self.rng.integers(self.rows_low, self.rows_high + 1))
        cols = int(self.rng.integers(self.cols_low, self.cols_high + 1))
        rank = int(self.rng.integers(self.rank_low, min(self.rank_high, cols, rows) + 1))
        heldout_frac = float(self.rng.uniform(self.heldout_frac_low, self.heldout_frac_high))
        n_test = int(round(rows * heldout_frac))
        n_test = min(max(n_test, 8), rows - 8)
        return rows, cols, rank, n_test

    def _sample_one(self, rows: int, cols: int, rank: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
        row_factors = self.rng.normal(0.0, 1.0, size=(rows, rank))
        col_factors = self.rng.normal(0.0, 1.0 / math.sqrt(rank), size=(rank, cols))
        singular_decay = self.rng.uniform(0.35, 0.85)
        spectrum = np.power(np.arange(1, rank + 1, dtype=np.float64), -singular_decay)
        low_rank = (row_factors * spectrum[None, :]) @ col_factors

        row_bias = self.rng.normal(0.0, self.rng.uniform(0.05, 0.25), size=(rows, 1))
        col_bias = self.rng.normal(0.0, self.rng.uniform(0.05, 0.25), size=(1, cols))
        noise = self.rng.uniform(0.01, 0.08) * self.rng.normal(0.0, 1.0, size=(rows, cols))
        dense = (low_rank + row_bias + col_bias + noise).astype(np.float32)

        missing = np.zeros_like(dense, dtype=bool)
        if self.rng.random() < 0.85:
            test_rows = np.arange(rows - n_test, rows)
        else:
            test_rows = self.rng.choice(rows, size=n_test, replace=False)
        missing[test_rows, -1] = True

        if self.extra_mcar_prob > 0.0:
            extra = self.rng.random(size=dense.shape) < self.extra_mcar_prob
            extra[test_rows, -1] = False
            missing |= extra
        return dense, missing

    def get_batch(self):
        rows, cols, rank, n_test = self._sample_shape()
        x_list = []
        target_list = []
        missing_fracs = []
        for _ in range(self.batch_size):
            dense, missing = self._sample_one(rows, cols, rank, n_test)
            x_missing = torch.tensor(dense)
            missing_t = torch.tensor(missing, dtype=torch.bool)
            x_missing[missing_t] = torch.nan
            x_norm, (mean, std) = normalize_data(x_missing.unsqueeze(1), return_scaling=True)
            target_norm = normalize_data(torch.tensor(dense).unsqueeze(1), mean=mean, std=std)
            x_list.append(x_norm.squeeze(1))
            target_list.append(target_norm.squeeze(1))
            missing_fracs.append(float(missing_t.float().mean()))

        batch_x = torch.stack(x_list, dim=0).to(self.device)
        batch_target = torch.stack(target_list, dim=0).to(self.device)
        d = torch.full((self.batch_size,), cols, dtype=torch.long, device=self.device)
        seq_lens = torch.full((self.batch_size,), rows, dtype=torch.long, device=self.device)
        train_sizes = torch.full((self.batch_size,), rows - n_test, dtype=torch.long, device=self.device)
        weights = torch.ones(self.batch_size, dtype=torch.float32, device=self.device)
        self.last_batch_metrics = {
            "prior_family": "skinny_block_low_rank",
            "prior_rows": float(rows),
            "prior_cols": float(cols),
            "prior_rank": float(rank),
            "prior_train_rows": float(rows - n_test),
            "prior_test_rows": float(n_test),
            "prior_missing_frac": float(np.mean(missing_fracs)),
        }
        return (batch_x, batch_target, d, seq_lens, train_sizes), weights


class MixedMatrixCausalPrior:
    def __init__(
        self,
        *,
        batch_size: int,
        device: str,
        rows_low: int,
        rows_high: int,
        cols_low: int,
        cols_high: int,
        causal_prob: float,
        cate_covariates_low: int,
        cate_covariates_high: int,
        skinny_block_prob: float,
        skinny_cols_low: int,
        skinny_cols_high: int,
        skinny_rank_low: int,
        skinny_rank_high: int,
        skinny_extra_mcar_prob: float,
        query_treatment_mode: str,
        cate_tau_scale: float,
        cate_query_noise_scale: float,
        seed: int | None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.causal_prob = float(causal_prob)
        self.skinny_block_prob = float(skinny_block_prob)
        if self.causal_prob + self.skinny_block_prob > 1.0:
            raise ValueError("--causal-prob + --skinny-block-prob must be <= 1.0")
        spectrum_config = {
            **SPECTRUM_CONFIG,
            "num_rows_low": int(rows_low),
            "num_rows_high": int(rows_high),
            "num_cols_low": int(cols_low),
            "num_cols_high": int(cols_high),
        }
        self.spectrum = SpectrumMissingnessPrior(
            missingness_type="mcar",
            config=spectrum_config,
            batch_size=batch_size,
            device=device,
            verbose=False,
            entry_wise_features=False,
        )
        causal_rows_low = min(max(32, rows_low), max(8, rows_high - 16))
        self.causal = TabularCATEBlockPrior(
            batch_size=batch_size,
            device=device,
            rows_low=causal_rows_low,
            rows_high=rows_high,
            covariates_low=cate_covariates_low,
            covariates_high=cate_covariates_high,
            heldout_frac_low=0.15,
            heldout_frac_high=0.35,
            query_treatment_mode=query_treatment_mode,
            tau_scale=cate_tau_scale,
            query_noise_scale=cate_query_noise_scale,
            seed=None if seed is None else seed + 1009,
        )
        self.skinny = SkinnyBlockLowRankPrior(
            batch_size=batch_size,
            device=device,
            rows_low=min(max(32, rows_low), max(8, rows_high - 16)),
            rows_high=rows_high,
            cols_low=skinny_cols_low,
            cols_high=skinny_cols_high,
            rank_low=skinny_rank_low,
            rank_high=skinny_rank_high,
            heldout_frac_low=0.15,
            heldout_frac_high=0.35,
            extra_mcar_prob=skinny_extra_mcar_prob,
            seed=None if seed is None else seed + 2027,
        )
        self.last_batch_metrics: dict[str, float | str] = {}

    def get_batch(self):
        draw = self.rng.random()
        if draw < self.skinny_block_prob:
            batch = self.skinny.get_batch()
            self.last_batch_metrics = dict(self.skinny.last_batch_metrics)
            return batch
        if draw < self.skinny_block_prob + self.causal_prob:
            batch = self.causal.get_batch()
            self.last_batch_metrics = dict(self.causal.last_batch_metrics)
            return batch
        batch = self.spectrum.get_batch()
        self.last_batch_metrics = {
            "prior_family": "spectrum_decay",
            "prior_rows": float(batch[0][0].shape[1]),
            "prior_cols": float(batch[0][0].shape[2]),
            "prior_missing_frac": float(torch.isnan(batch[0][0]).float().mean().detach().cpu()),
        }
        return batch


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="tabimpute-direct-pure-rabel64-joint-spectrum-cateblock-60k")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rows-low", type=int, default=150)
    parser.add_argument("--rows-high", type=int, default=300)
    parser.add_argument("--cols-low", type=int, default=50)
    parser.add_argument("--cols-high", type=int, default=300)
    parser.add_argument("--causal-prob", type=float, default=0.35)
    parser.add_argument("--cate-covariates-low", type=int, default=4)
    parser.add_argument("--cate-covariates-high", type=int, default=14)
    parser.add_argument("--skinny-block-prob", type=float, default=0.0)
    parser.add_argument("--skinny-cols-low", type=int, default=6)
    parser.add_argument("--skinny-cols-high", type=int, default=20)
    parser.add_argument("--skinny-rank-low", type=int, default=2)
    parser.add_argument("--skinny-rank-high", type=int, default=8)
    parser.add_argument("--skinny-extra-mcar-prob", type=float, default=0.0)
    parser.add_argument("--cate-tau-scale", type=float, default=1.0)
    parser.add_argument("--cate-query-noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--query-treatment-mode",
        choices=("factual", "random", "paired"),
        default="factual",
        help=(
            "For tabular CATE-block batches, whether hidden target rows keep "
            "their factual treatment or are assigned randomized query "
            "treatments with targets recomputed from the synthetic potential "
            "outcome model."
        ),
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--stable-steps", type=int, default=45_000)
    parser.add_argument("--decay-steps", type=int, default=13_000)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--init-checkpoint", type=Path, default=BASE_SPECTRUM_CHECKPOINT)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--wandb-project", default="tabimpute")
    parser.add_argument("--model-overrides-json", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.steps = 4
        args.warmup_steps = 1
        args.stable_steps = 2
        args.decay_steps = 1
        args.checkpoint_every = 0
        args.batch_size = 2
        args.rows_low = 32
        args.rows_high = 48
        args.cols_low = 16
        args.cols_high = 24
        if args.causal_prob == 0.0 and args.skinny_block_prob == 0.0:
            args.skinny_block_prob = 0.25
        args.skinny_cols_low = 6
        args.skinny_cols_high = 10

    work_dir = args.work_dir or (WORKDIR / args.run_name)
    work_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_config = dict(MODEL_CONFIG)
    if args.model_overrides_json:
        model_config.update(json.loads(args.model_overrides_json))

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
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("Use only one of --resume or --init-checkpoint.")
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

    prior = MixedMatrixCausalPrior(
        batch_size=args.batch_size,
        device=args.device,
        rows_low=args.rows_low,
        rows_high=args.rows_high,
        cols_low=args.cols_low,
        cols_high=args.cols_high,
        causal_prob=args.causal_prob,
        cate_covariates_low=args.cate_covariates_low,
        cate_covariates_high=args.cate_covariates_high,
        skinny_block_prob=args.skinny_block_prob,
        skinny_cols_low=args.skinny_cols_low,
        skinny_cols_high=args.skinny_cols_high,
        skinny_rank_low=args.skinny_rank_low,
        skinny_rank_high=args.skinny_rank_high,
        skinny_extra_mcar_prob=args.skinny_extra_mcar_prob,
        query_treatment_mode=args.query_treatment_mode,
        cate_tau_scale=args.cate_tau_scale,
        cate_query_noise_scale=args.cate_query_noise_scale,
        seed=args.seed,
    )

    borders_path = importlib.resources.files("tabimpute") / "data" / "borders.pt"
    with importlib.resources.as_file(borders_path) as path:
        criterion = FullSupportBarDistribution(
            borders=torch.load(path, map_location=device).to(device)
        )

    run_config = {
        "run_name": args.run_name,
        "model_family": MODEL_FAMILY,
        "model": model_config,
        "prior_kind": "mixed_spectrum_cate_block",
        "prior": {
            "spectrum": SPECTRUM_CONFIG,
            "causal_prob": args.causal_prob,
            "rows_low": args.rows_low,
            "rows_high": args.rows_high,
            "cols_low": args.cols_low,
            "cols_high": args.cols_high,
            "skinny_block_prob": args.skinny_block_prob,
            "skinny_block_low_rank": {
                "cols_low": args.skinny_cols_low,
                "cols_high": args.skinny_cols_high,
                "rank_low": args.skinny_rank_low,
                "rank_high": args.skinny_rank_high,
                "heldout_frac_low": 0.15,
                "heldout_frac_high": 0.35,
                "extra_mcar_prob": args.skinny_extra_mcar_prob,
                "target": "dense_low_rank_last_column_block",
                "observed_input": "same_matrix_with_heldout_last_column_nan",
            },
            "tabular_cate_block": {
                "covariates_low": args.cate_covariates_low,
                "covariates_high": args.cate_covariates_high,
                "heldout_frac_low": 0.15,
                "heldout_frac_high": 0.35,
                "query_treatment_mode": args.query_treatment_mode,
                "tau_scale": args.cate_tau_scale,
                "query_noise_scale": args.cate_query_noise_scale,
                "target": "dense_outcome_column",
                "observed_input": "same_matrix_with_heldout_outcome_nan",
            },
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
            "seed": args.seed,
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
                raise RuntimeError("Mixed prior produced no missing cells.")

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
                family=str(prior.last_batch_metrics.get("prior_family", "")),
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
