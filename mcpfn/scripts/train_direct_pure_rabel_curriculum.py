#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.resources
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from optimizer_continuation import retain_adamw_state

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.prediction_distribution import prediction_distribution, distribution_target
from tabimpute.model.model_new_stable_rabel import TabImputeModelPureRaBEL
from tabimpute.prior.spectrum_training_set_generation import SpectrumMissingnessPrior
from tabimpute.prior.training_set_generation import MissingnessPrior
from tabimpute.train.callbacks import WandbLoggerCallback
from rowcol_fixed_validation import (
    apply_support_mask, evaluate_fixed_validation, load_or_build_validation, preserved_rng,
)


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


def parse_substrings(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def keep_selected_parameters_fp32(
    model: torch.nn.Module,
    substrings: tuple[str, ...],
) -> list[str]:
    if not substrings:
        return []
    matched = []
    for name, parameter in model.named_parameters():
        if any(substring in name for substring in substrings):
            parameter.data = parameter.data.float()
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.float()
            matched.append(name)
    return matched


def apply_parameter_value_overrides(
    model: torch.nn.Module,
    overrides_json: str | None,
) -> list[str]:
    if not overrides_json:
        return []
    overrides = json.loads(overrides_json)
    if not isinstance(overrides, dict):
        raise ValueError("--parameter-value-overrides-json must be a JSON object.")
    named_parameters = dict(model.named_parameters())
    applied = []
    with torch.no_grad():
        for name, value in overrides.items():
            if name not in named_parameters:
                raise KeyError(f"Unknown parameter override name: {name}")
            parameter = named_parameters[name]
            value_tensor = torch.as_tensor(
                value,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            if value_tensor.numel() == 1:
                parameter.fill_(value_tensor.item())
            else:
                if tuple(value_tensor.shape) != tuple(parameter.shape):
                    raise ValueError(
                        f"Override for {name} has shape {tuple(value_tensor.shape)} "
                        f"but parameter shape is {tuple(parameter.shape)}."
                    )
                parameter.copy_(value_tensor)
            applied.append(name)
    return applied


def _sample_uniform(low: float, high: float) -> float:
    if high <= low:
        return float(low)
    return float(low + (high - low) * random.random())


def objective_logit_gradient_stats(logits, terms):
    gradients, stats = {}, {}
    for label, component, weight in terms:
        if weight > 0 and component.requires_grad:
            gradient = torch.autograd.grad(weight * component, logits, retain_graph=True)[0].float()
            gradients[label] = gradient
            stats[f"logit_gradient_norm/{label}"] = float(gradient.norm())
    if "nll" in gradients and "point" in gradients:
        gn, gp = gradients["nll"], gradients["point"]
        stats["logit_gradient_cosine/nll_point"] = float(
            (gn * gp).sum() / (gn.norm() * gp.norm()).clamp_min(1e-30))
    return stats


def apply_benchmark_selection_curriculum(
    batch_target: torch.Tensor,
    *,
    query_fraction_low: float,
    query_fraction_high: float,
    context_observed_prob_low: float,
    context_observed_prob_high: float,
    query_k_low: int,
    query_k_high: int,
    query_selected_observed_prob_low: float,
    query_selected_observed_prob_high: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Rewrite a complete batch into benchmark-selection train/eval shape."""

    if batch_target.ndim != 3:
        raise ValueError(
            f"batch_target must have shape (batch, rows, cols), got {tuple(batch_target.shape)}"
        )
    batch_size, rows, cols = batch_target.shape
    source = torch.full_like(batch_target.float(), float("nan"))
    loss_mask = torch.zeros(
        batch_target.shape,
        device=batch_target.device,
        dtype=torch.bool,
    )
    context_observed_rates = []
    query_observed_rates = []
    query_rows_list = []
    query_k_list = []

    for batch_index in range(batch_size):
        query_fraction = _sample_uniform(query_fraction_low, query_fraction_high)
        query_rows = max(1, min(rows - 1, int(round(rows * query_fraction))))
        permutation = torch.randperm(rows, device=batch_target.device)
        query_idx = permutation[:query_rows]
        context_idx = permutation[query_rows:]
        if context_idx.numel() == 0:
            context_idx = query_idx[:1]
            query_idx = query_idx[1:]
            query_rows = int(query_idx.numel())
        k_low = max(1, min(int(query_k_low), cols))
        k_high = max(k_low, min(int(query_k_high), cols))
        query_k = int(
            torch.randint(k_low, k_high + 1, (1,), device=batch_target.device).item()
        )
        selected_cols = torch.randperm(cols, device=batch_target.device)[:query_k]

        context_observed_prob = _sample_uniform(
            context_observed_prob_low,
            context_observed_prob_high,
        )
        context_obs = (
            torch.rand((context_idx.numel(), cols), device=batch_target.device)
            < context_observed_prob
        )
        source[batch_index, context_idx] = torch.where(
            context_obs,
            batch_target[batch_index, context_idx].float(),
            source[batch_index, context_idx],
        )

        query_selected_observed_prob = _sample_uniform(
            query_selected_observed_prob_low,
            query_selected_observed_prob_high,
        )
        query_selected_obs = (
            torch.rand((query_idx.numel(), query_k), device=batch_target.device)
            < query_selected_observed_prob
        )
        query_source = source[batch_index, query_idx]
        query_source[:, selected_cols] = torch.where(
            query_selected_obs,
            batch_target[batch_index, query_idx[:, None], selected_cols].float(),
            query_source[:, selected_cols],
        )
        source[batch_index, query_idx] = query_source

        query_loss = torch.ones(
            (query_idx.numel(), cols),
            device=batch_target.device,
            dtype=torch.bool,
        )
        query_loss[:, selected_cols] = False
        loss_mask[batch_index, query_idx] = query_loss

        context_observed_rates.append(float(context_obs.float().mean().cpu()))
        query_observed_rates.append(
            float(query_selected_obs.float().mean().cpu())
            * query_k
            / max(float(cols), 1.0)
        )
        query_rows_list.append(float(query_rows))
        query_k_list.append(float(query_k))

    observed = ~torch.isnan(source)
    obs_float = observed.float()
    counts = obs_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    means = torch.nan_to_num(source, nan=0.0).sum(dim=1, keepdim=True) / counts
    centered_observed = torch.where(observed, source - means, torch.zeros_like(source))
    denom = (counts - 1.0).clamp_min(1.0)
    stds = (centered_observed.square().sum(dim=1, keepdim=True) / denom).sqrt()
    stds = torch.where(stds == 0.0, torch.ones_like(stds), stds)

    normalized_source = (source - means) / (stds + 1.0e-16)
    normalized_target = (batch_target.float() - means) / (stds + 1.0e-16)
    normalized_source = normalized_source.clamp(-100.0, 100.0)
    normalized_target = normalized_target.clamp(-100.0, 100.0)

    stats = {
        "benchmark_query_rows_mean": float(np.mean(query_rows_list)),
        "benchmark_query_k_mean": float(np.mean(query_k_list)),
        "benchmark_context_observed_rate": float(np.mean(context_observed_rates)),
        "benchmark_query_observed_rate": float(np.mean(query_observed_rates)),
        "benchmark_loss_fraction": float(loss_mask.float().mean().cpu()),
    }
    return normalized_source, normalized_target, loss_mask, stats


def apply_conditional_denoising_curriculum(
    batch_target: torch.Tensor,
    *,
    row_observed_prob_low: float,
    row_observed_prob_high: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Sample broad row-wise support masks for general conditional inpainting."""

    if batch_target.ndim != 3:
        raise ValueError(
            f"batch_target must have shape (batch, rows, cols), got {tuple(batch_target.shape)}"
        )
    batch_size, rows, cols = batch_target.shape
    source = torch.full_like(batch_target.float(), float("nan"))
    observed_rates = []
    row_min_rates = []
    row_max_rates = []

    log_low = math.log(row_observed_prob_low)
    log_high = math.log(row_observed_prob_high)
    for batch_index in range(batch_size):
        row_probs = torch.empty(rows, device=batch_target.device).uniform_(
            log_low,
            log_high,
        ).exp()
        observed = (
            torch.rand((rows, cols), device=batch_target.device)
            < row_probs[:, None]
        )
        for row_index in range(rows):
            if not bool(observed[row_index].any()):
                col_index = int(torch.randint(cols, (1,), device=batch_target.device))
                observed[row_index, col_index] = True
            if bool(observed[row_index].all()) and cols > 1:
                col_index = int(torch.randint(cols, (1,), device=batch_target.device))
                observed[row_index, col_index] = False
        empty_cols = torch.where(~observed.any(dim=0))[0]
        for col_index in empty_cols.tolist():
            row_index = int(torch.randint(rows, (1,), device=batch_target.device))
            observed[row_index, col_index] = True
            if bool(observed[row_index].all()) and cols > 1:
                hidden_candidates = torch.where(
                    torch.arange(cols, device=batch_target.device) != col_index
                )[0]
                hidden_choice = int(
                    hidden_candidates[
                        torch.randint(
                            hidden_candidates.numel(),
                            (1,),
                            device=batch_target.device,
                        )
                    ]
                )
                observed[row_index, hidden_choice] = False
        source[batch_index] = torch.where(
            observed,
            batch_target[batch_index].float(),
            source[batch_index],
        )
        row_rates = observed.float().mean(dim=1)
        observed_rates.append(float(observed.float().mean().cpu()))
        row_min_rates.append(float(row_rates.min().cpu()))
        row_max_rates.append(float(row_rates.max().cpu()))

    observed = ~torch.isnan(source)
    loss_mask = ~observed
    obs_float = observed.float()
    counts = obs_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    means = torch.nan_to_num(source, nan=0.0).sum(dim=1, keepdim=True) / counts
    centered_observed = torch.where(observed, source - means, torch.zeros_like(source))
    denom = (counts - 1.0).clamp_min(1.0)
    stds = (centered_observed.square().sum(dim=1, keepdim=True) / denom).sqrt()
    stds = torch.where(stds == 0.0, torch.ones_like(stds), stds)

    normalized_source = (source - means) / (stds + 1.0e-16)
    normalized_target = (batch_target.float() - means) / (stds + 1.0e-16)
    normalized_source = normalized_source.clamp(-100.0, 100.0)
    normalized_target = normalized_target.clamp(-100.0, 100.0)

    stats = {
        "conditional_observed_rate": float(np.mean(observed_rates)),
        "conditional_row_min_observed_rate": float(np.mean(row_min_rates)),
        "conditional_row_max_observed_rate": float(np.mean(row_max_rates)),
        "conditional_loss_fraction": float(loss_mask.float().mean().cpu()),
    }
    return normalized_source, normalized_target, loss_mask, stats


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

DEFAULT_MIXED_PRIORS = (
    {
        "name": "latent_rank10_mcar20",
        "kind": "latent_factor",
        "weight": 1.0,
        "config": {
            "p_missing": 0.2,
            "latent_rank_low": 10,
            "latent_rank_high": 10,
            "latent_spike_p": 0.3,
            "latent_slab_sigma": 2.0,
        },
    },
    {
        "name": "latent_rank40_100_mcar20",
        "kind": "latent_factor",
        "weight": 1.0,
        "config": {
            "p_missing": 0.2,
            "latent_rank_low": 40,
            "latent_rank_high": 100,
            "latent_spike_p": 0.0,
            "latent_slab_sigma": 1.0,
        },
    },
    {
        "name": "spectrum_default_hard",
        "kind": "spectrum_decay",
        "weight": 1.0,
        "config": SPECTRUM_DECAY_PRIOR_CONFIG,
    },
    {
        "name": "spectrum_harder_tail",
        "kind": "spectrum_decay",
        "weight": 1.0,
        "config": {
            **SPECTRUM_DECAY_PRIOR_CONFIG,
            "p_missing": 0.2,
            "spectrum_rank_low": 80,
            "spectrum_rank_high": 160,
            "spectrum_alpha_low": 0.25,
            "spectrum_alpha_high": 0.55,
            "spectrum_head_weight_low": 2.0,
            "spectrum_head_weight_high": 6.0,
            "spectrum_head_exp_scale_low": 2.5,
            "spectrum_head_exp_scale_high": 7.0,
            "spectrum_head_exp_power_low": 1.6,
            "spectrum_head_exp_power_high": 3.0,
            "spectrum_head_handoff_scale_low": 1.0,
            "spectrum_head_handoff_scale_high": 8.0,
            "spectrum_tail_amplitude_low": 0.25,
            "spectrum_tail_amplitude_high": 0.60,
            "spectrum_taper_strength_low": 0.6,
            "spectrum_taper_strength_high": 1.8,
            "spectrum_taper_power_low": 2.5,
            "spectrum_taper_power_high": 4.5,
            "spectrum_noise_std_low": 0.001,
            "spectrum_noise_std_high": 0.01,
            "spectrum_target_rank": 60,
        },
    },
)


class MixedMatrixPrior:
    """Sample whole batches from a weighted mixture of matrix-to-matrix priors."""

    def __init__(
        self,
        *,
        phase: CurriculumPhase,
        device: str,
        missingness_type: str,
        mixed_config: dict,
    ) -> None:
        raw_specs = mixed_config.get("mixed_priors", DEFAULT_MIXED_PRIORS)
        if not isinstance(raw_specs, list) and not isinstance(raw_specs, tuple):
            raise ValueError("mixed_priors must be a list of prior specs.")

        self.names: list[str] = []
        self.weights: list[float] = []
        self.children = []
        self.last_prior_name = ""
        self.last_prior_index = -1
        self.selection_mode = str(mixed_config.get("selection_mode", "random"))
        if self.selection_mode not in {"random", "cycle"}:
            raise ValueError(
                "mixed_matrix selection_mode must be 'random' or 'cycle', "
                f"got {self.selection_mode!r}"
            )
        self._cycle_order: list[int] = []
        self._cycle_position = 0
        for index, raw_spec in enumerate(raw_specs):
            if not isinstance(raw_spec, dict):
                raise ValueError(f"mixed_priors[{index}] must be an object.")
            name = str(raw_spec.get("name", f"prior_{index}"))
            kind = str(raw_spec.get("kind", "latent_factor"))
            child_missingness_type = str(raw_spec.get("missingness_type", missingness_type))
            weight = float(raw_spec.get("weight", 1.0))
            if weight <= 0.0:
                continue
            if kind == "spectrum_decay":
                child_config = dict(SPECTRUM_DECAY_PRIOR_CONFIG)
            elif kind == "latent_factor":
                child_config = dict(PRIOR_CONFIG)
            else:
                raise ValueError(f"Unsupported mixed prior kind '{kind}' for {name}.")
            child_config.update(dict(raw_spec.get("config", {})))
            child_config.update(
                {
                    "num_rows_low": phase.rows_low,
                    "num_rows_high": phase.rows_high,
                    "num_cols_low": phase.cols_low,
                    "num_cols_high": phase.cols_high,
                }
            )
            if kind == "spectrum_decay":
                child = SpectrumMissingnessPrior(
                    missingness_type=child_missingness_type,
                    config=child_config,
                    batch_size=phase.batch_size,
                    device=device,
                    verbose=False,
                    entry_wise_features=False,
                )
            else:
                child = MissingnessPrior(
                    generator_type="latent_factor",
                    missingness_type=child_missingness_type,
                    config=child_config,
                    batch_size=phase.batch_size,
                    device=device,
                    verbose=False,
                    entry_wise_features=False,
                )
            self.names.append(name)
            self.weights.append(weight)
            self.children.append(child)
        if not self.children:
            raise ValueError("No positive-weight mixed priors were configured.")
        if self.selection_mode == "cycle":
            # Integer repeat counts approximate weights while guaranteeing that
            # every configured prior is seen in each short training window.
            min_weight = min(self.weights)
            order = []
            for index, weight in enumerate(self.weights):
                repeats = max(1, int(round(weight / min_weight)))
                order.extend([index] * repeats)
            self._cycle_order = order

    def get_batch(self):
        if self.selection_mode == "cycle":
            index = self._cycle_order[self._cycle_position % len(self._cycle_order)]
            self._cycle_position += 1
        else:
            index = random.choices(range(len(self.children)), weights=self.weights, k=1)[0]
        self.last_prior_name = self.names[index]
        self.last_prior_index = index
        return self.children[index].get_batch()


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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None,
                        help="Isolate per-step prior sampling from model/dropout RNG.")
    parser.add_argument("--support-mask-mode", choices=("mixed", "broad_support"), default="mixed")
    parser.add_argument("--fixed-validation-cache", type=Path, default=None)
    parser.add_argument("--validation-every", type=int, default=2000)
    parser.add_argument("--validation-seed", type=int, default=20260906)
    parser.add_argument("--update-diagnostics-every", type=int, default=0)
    parser.add_argument(
        "--fp32-param-substrings",
        default="",
        help=(
            "Comma-separated parameter-name substrings to keep in float32 after "
            "the model is moved to the compute dtype. This is useful for scalar "
            "calibration parameters whose updates can be below BF16 resolution."
        ),
    )
    parser.add_argument(
        "--precision-mode",
        choices=("full_bfloat16", "amp_bfloat16", "amp_float16", "float32"),
        default="full_bfloat16",
        help=(
            "Numerical mode. full_bfloat16 preserves the historical behavior "
            "of casting the model and inputs to BF16. amp_* keeps model weights "
            "in FP32 and uses autocast for selected ops."
        ),
    )
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
    parser.add_argument("--snapshot-steps", type=int, nargs="*", default=[],
                        help="Keep model-only snapshots at these steps without changing the curriculum.")
    parser.add_argument(
        "--metrics-jsonl-every",
        type=int,
        default=100,
        help="Write one local metrics JSONL row every N steps; 0 disables it.",
    )
    parser.add_argument(
        "--nll-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Weight for the bar-distribution NLL on missing entries. Lower "
            "values make training more point-loss driven while preserving a "
            "differentiable distributional decoder."
        ),
    )
    parser.add_argument(
        "--point-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional weight for an auxiliary point-estimate loss on missing "
            "entries. Use this to sharpen RMSE while keeping the NLL objective."
        ),
    )
    parser.add_argument(
        "--point-loss-kind",
        choices=("mse", "huber", "mae"),
        default="mse",
    )
    parser.add_argument(
        "--point-loss-prediction",
        choices=("mean", "median"),
        default="mean",
        help="Distribution statistic used for the auxiliary point loss.",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=0.05,
        help="Delta for --point-loss-kind huber.",
    )
    parser.add_argument(
        "--soft-target-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional distance-aware CE loss. Targets are smoothed over nearby "
            "bar centers with a Gaussian kernel in normalized value space."
        ),
    )
    parser.add_argument(
        "--soft-target-sigma",
        type=float,
        default=0.02,
        help="Gaussian sigma for --soft-target-loss-weight.",
    )
    parser.add_argument(
        "--cdf-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional ordinal/CDF alignment loss between predicted CDF and "
            "the empirical target step function."
        ),
    )
    parser.add_argument(
        "--entropy-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional normalized entropy penalty on missing-entry predictions. "
            "Positive values encourage sharper distributions."
        ),
    )
    parser.add_argument(
        "--svt-mixture-depth-distill-weight",
        type=float,
        default=0.0,
        help=(
            "Optional auxiliary CE loss for adaptive SVT mixtures. The target "
            "depth distribution is computed from per-depth RMSE on the current "
            "training mask, not from prior labels."
        ),
    )
    parser.add_argument(
        "--svt-mixture-depth-distill-temperature",
        type=float,
        default=0.005,
        help="Softmax temperature for the per-depth RMSE distillation target.",
    )
    parser.add_argument(
        "--svt-mixture-cv-regret-weight",
        type=float,
        default=0.0,
        help=(
            "Optional auxiliary controller loss for SVT mixtures. It minimizes "
            "the controller-weighted cross-validation regret relative to the "
            "best held-out SVT depth on observed entries."
        ),
    )
    parser.add_argument(
        "--svt-mixture-entropy-bonus-weight",
        type=float,
        default=0.0,
        help=(
            "Optional positive weight for maximizing normalized entropy of "
            "adaptive SVT mixture weights. This discourages premature one-depth "
            "collapse while keeping the controller differentiable."
        ),
    )
    parser.add_argument(
        "--svt-adaptive-threshold-distill-weight",
        type=float,
        default=0.0,
        help=(
            "Optional auxiliary loss for adaptive SVT threshold heads. A no-grad "
            "oracle evaluates candidate threshold scales on the current training "
            "mask, then the learned scale is regressed toward the soft oracle "
            "expectation in log-scale."
        ),
    )
    parser.add_argument(
        "--svt-adaptive-threshold-distill-temperature",
        type=float,
        default=0.005,
        help="Softmax temperature for threshold-scale oracle RMSE values.",
    )
    parser.add_argument(
        "--svt-adaptive-threshold-distill-scales",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
        help="Candidate threshold multipliers for threshold-scale distillation.",
    )
    parser.add_argument(
        "--distribution-loss-chunk-size",
        type=int,
        default=8192,
        help="Chunk size for dense auxiliary distribution losses.",
    )
    parser.add_argument(
        "--benchmark-selection-curriculum",
        action="store_true",
        help=(
            "Rewrite synthetic batches into benchmark-selection shape: dense "
            "context rows plus sparse query rows with a shared selected-column "
            "panel, and train only on query hidden columns."
        ),
    )
    parser.add_argument("--benchmark-query-fraction-low", type=float, default=0.08)
    parser.add_argument("--benchmark-query-fraction-high", type=float, default=0.18)
    parser.add_argument(
        "--benchmark-context-observed-prob-low",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--benchmark-context-observed-prob-high",
        type=float,
        default=0.90,
    )
    parser.add_argument("--benchmark-query-k-low", type=int, default=1)
    parser.add_argument("--benchmark-query-k-high", type=int, default=15)
    parser.add_argument(
        "--benchmark-query-selected-observed-prob-low",
        type=float,
        default=0.65,
    )
    parser.add_argument(
        "--benchmark-query-selected-observed-prob-high",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--conditional-denoising-curriculum",
        action="store_true",
        help=(
            "Rewrite synthetic batches with broad row-wise observation budgets "
            "and train on the hidden complement. This teaches general "
            "conditional matrix inpainting without matching a specific benchmark."
        ),
    )
    parser.add_argument(
        "--conditional-row-observed-prob-low",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--conditional-row-observed-prob-high",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--per-prior-loss-ema-normalize",
        action="store_true",
        help=(
            "For mixed_matrix training, scale each prior's loss by the ratio of "
            "global prior-loss EMA to that prior's EMA. This balances gradient "
            "scale while keeping logged losses raw."
        ),
    )
    parser.add_argument(
        "--prior-loss-ema-beta",
        type=float,
        default=0.99,
        help="EMA beta for --per-prior-loss-ema-normalize.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--retain-init-optimizer-state", action="store_true",
                        help="With a full init checkpoint, retain AdamW moments/steps but restart the schedule.")
    parser.add_argument(
        "--init-model-checkpoint",
        type=Path,
        default=None,
        help=(
            "Initialize model weights from a checkpoint without restoring "
            "optimizer/scheduler state."
        ),
    )
    parser.add_argument(
        "--allow-partial-init",
        action="store_true",
        help=(
            "With --init-model-checkpoint, load matching tensors with "
            "strict=False so small architecture additions can warm-start."
        ),
    )
    parser.add_argument("--wandb-project", default="tabimpute")
    parser.add_argument(
        "--prior-kind",
        choices=("latent_factor", "spectrum_decay", "mixed_matrix"),
        default="latent_factor",
    )
    parser.add_argument(
        "--generator-type",
        default="latent_factor",
        help=(
            "Generator type for the MissingnessPrior branch. Examples include "
            "latent_factor, nonlinear_factor, robust_pca, generator_mix, and "
            "priormix_v1."
        ),
    )
    parser.add_argument(
        "--missingness-type",
        default="mcar",
        help=(
            "Missingness pattern passed to the prior. Keep this generic "
            "(for example mcar, profile_mixed, mnar_spatial_block)."
        ),
    )
    parser.add_argument("--p-missing", type=float, default=None)
    parser.add_argument("--latent-rank-low", type=int, default=None)
    parser.add_argument("--latent-rank-high", type=int, default=None)
    parser.add_argument(
        "--prior-overrides-json",
        default=None,
        help="JSON object merged into the prior config after CLI scalar overrides.",
    )
    parser.add_argument(
        "--mixed-priors-json",
        default=None,
        help=(
            "JSON list of mixed prior specs for --prior-kind mixed_matrix. "
            "Each spec needs kind, optional name/weight/missingness_type, and config."
        ),
    )
    parser.add_argument(
        "--model-overrides-json",
        default=None,
        help="JSON object merged into the default pure-RaBEL model config.",
    )
    parser.add_argument(
        "--parameter-value-overrides-json",
        default=None,
        help=(
            "Optional JSON object mapping exact parameter names to scalar or "
            "list values after checkpoint init. Useful for resetting learned "
            "gates while warm-starting the rest of the model."
        ),
    )
    parser.add_argument(
        "--trainable-param-substrings",
        default=None,
        help=(
            "Optional comma-separated parameter-name substrings. When set, "
            "only parameters whose names contain one of these substrings are "
            "trainable. Useful for controller-only optimization probes."
        ),
    )
    parser.add_argument(
        "--phases-json",
        default=None,
        help=(
            "Optional JSON list of curriculum phases. Each phase needs name, "
            "end_step, rows_low, rows_high, cols_low, cols_high, batch_size."
        ),
    )
    parser.add_argument("--freeze-param-substrings", default="")
    parser.add_argument("--validation-cases-per-group", type=int, default=2)
    parser.add_argument("--param-lr-multipliers-json", default="{}")
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
    generator_type: str,
    missingness_type: str,
    base_config: dict,
) -> MissingnessPrior | SpectrumMissingnessPrior:
    config = {
        **base_config,
        "num_rows_low": phase.rows_low,
        "num_rows_high": phase.rows_high,
        "num_cols_low": phase.cols_low,
        "num_cols_high": phase.cols_high,
    }
    if prior_kind == "mixed_matrix":
        return MixedMatrixPrior(
            phase=phase,
            device=device,
            missingness_type=missingness_type,
            mixed_config=base_config,
        )
    if prior_kind == "spectrum_decay":
        return SpectrumMissingnessPrior(
            missingness_type=missingness_type,
            config=config,
            batch_size=phase.batch_size,
            device=device,
            verbose=False,
            entry_wise_features=False,
        )
    return MissingnessPrior(
        generator_type=generator_type,
        missingness_type=missingness_type,
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


def soft_target_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: FullSupportBarDistribution,
    *,
    sigma: float,
    chunk_size: int,
) -> torch.Tensor:
    if sigma <= 0.0:
        raise ValueError(f"soft-target sigma must be positive, got {sigma}.")
    centers = criterion.bucket_means.to(device=logits.device, dtype=torch.float32)
    total = logits.new_zeros((), dtype=torch.float32)
    count = 0
    for start in range(0, logits.shape[0], chunk_size):
        end = min(start + chunk_size, logits.shape[0])
        logits_chunk = logits[start:end].float()
        target_chunk = targets[start:end].float()
        distance = centers.unsqueeze(0) - target_chunk.unsqueeze(1)
        target_log_weights = -0.5 * (distance / float(sigma)).square()
        target_probs = torch.softmax(target_log_weights, dim=-1)
        log_probs = torch.log_softmax(logits_chunk, dim=-1)
        total = total + (-(target_probs * log_probs).sum(dim=-1)).sum()
        count += end - start
    return (total / max(count, 1)).to(dtype=logits.dtype)


def cdf_alignment_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: FullSupportBarDistribution,
    *,
    chunk_size: int,
) -> torch.Tensor:
    centers = criterion.bucket_means.to(device=logits.device, dtype=torch.float32)
    total = logits.new_zeros((), dtype=torch.float32)
    count = 0
    for start in range(0, logits.shape[0], chunk_size):
        end = min(start + chunk_size, logits.shape[0])
        logits_chunk = logits[start:end].float()
        target_chunk = targets[start:end].float()
        pred_cdf = torch.softmax(logits_chunk, dim=-1).cumsum(dim=-1)
        target_cdf = (centers.unsqueeze(0) >= target_chunk.unsqueeze(1)).to(
            dtype=pred_cdf.dtype
        )
        total = total + (pred_cdf - target_cdf).square().mean(dim=-1).sum()
        count += end - start
    return (total / max(count, 1)).to(dtype=logits.dtype)


def normalized_entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    entropy = entropy / math.log(float(logits.shape[-1]))
    return entropy.mean().to(dtype=logits.dtype)


def svt_mixture_depth_distill_loss(
    model: torch.nn.Module,
    target_missing: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if temperature <= 0.0:
        raise ValueError("--svt-mixture-depth-distill-temperature must be positive.")
    weights = getattr(model, "last_svt_mixture_weights", None)
    snapshot_points = getattr(model, "last_svt_mixture_snapshot_points", None)
    batch_idx = getattr(model, "last_svt_mixture_snapshot_batch_idx", None)
    if weights is None or snapshot_points is None or batch_idx is None:
        return None, None

    weights_float = weights.float().clamp_min(1e-8)
    snapshot_points = snapshot_points.float().detach()
    target_missing = target_missing.float()
    batch_idx = batch_idx.to(device=weights.device, dtype=torch.long)
    per_depth_rmse = []
    for batch_index in range(weights.shape[0]):
        mask = batch_idx == batch_index
        if not bool(mask.any()):
            per_depth_rmse.append(
                weights_float.new_zeros((weights.shape[1],), dtype=torch.float32)
            )
            continue
        errors = snapshot_points[mask] - target_missing[mask].unsqueeze(1)
        per_depth_rmse.append(errors.square().mean(dim=0).clamp_min(0.0).sqrt())
    depth_rmse = torch.stack(per_depth_rmse, dim=0)
    target_probs = torch.softmax(-depth_rmse / float(temperature), dim=-1).detach()
    log_probs = getattr(model, "last_svt_mixture_log_probs", None)
    if log_probs is None:
        log_probs = weights_float.log()
    distill = -(target_probs * log_probs.float()).sum(dim=-1).mean()
    return distill.to(dtype=weights.dtype), depth_rmse.detach()


def svt_mixture_cv_regret_loss(
    model: torch.nn.Module,
) -> torch.Tensor | None:
    weights = getattr(model, "last_svt_mixture_weights", None)
    cv_rmse = getattr(model, "last_svt_mixture_cv_rmse", None)
    if weights is None or cv_rmse is None:
        return None

    weights_float = weights.float()
    regret = cv_rmse.float().detach()
    regret = regret - regret.min(dim=-1, keepdim=True).values
    scale = regret.mean(dim=-1, keepdim=True).clamp_min(1e-4)
    regret = regret / scale
    loss = (weights_float * regret).sum(dim=-1).mean()
    return loss.to(dtype=weights.dtype)


def svt_mixture_entropy_bonus(
    model: torch.nn.Module,
) -> tuple[torch.Tensor | None, float | None]:
    weights = getattr(model, "last_svt_mixture_weights", None)
    if weights is None:
        return None, None
    weights_float = weights.float().clamp_min(1e-8)
    entropy = -(weights_float * weights_float.log()).sum(dim=-1)
    entropy = entropy / math.log(float(weights.shape[-1]))
    entropy_mean = entropy.mean()
    return -entropy_mean.to(dtype=weights.dtype), float(entropy_mean.detach().cpu())


def svt_adaptive_threshold_distill_loss(
    model: torch.nn.Module,
    batch_x: torch.Tensor,
    batch_target: torch.Tensor,
    missing_mask: torch.Tensor,
    prior_index: int | torch.Tensor | None,
    *,
    scales: list[float],
    temperature: float,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if temperature <= 0.0:
        raise ValueError(
            "--svt-adaptive-threshold-distill-temperature must be positive."
        )
    if not scales:
        return None, {}
    if any(scale <= 0.0 for scale in scales):
        raise ValueError("--svt-adaptive-threshold-distill-scales must be positive.")
    predicted_scale = getattr(model, "last_svt_adaptive_threshold_scale", None)
    steps = getattr(model, "svt_mixture_steps", None)
    required = [
        "_prior_ids",
        "encode_latents",
        "_apply_prior_latent_adapter",
        "_initial_grid",
        "_svt_iterates",
    ]
    if (
        predicted_scale is None
        or steps is None
        or any(not hasattr(model, name) for name in required)
    ):
        return None, {}

    was_training = model.training
    rmse_by_scale = []
    with torch.no_grad():
        model.eval()
        try:
            prior_ids = model._prior_ids(batch_x, prior_index)
            row_state, col_state = model.encode_latents(batch_x)
            row_state, col_state = model._apply_prior_latent_adapter(
                row_state,
                col_state,
                prior_ids,
            )
            initial_grid = model._initial_grid(row_state, col_state, prior_ids)
            target = batch_target.float()
            for scale in scales:
                threshold_scale = torch.full(
                    (batch_x.shape[0], 1),
                    float(scale),
                    device=batch_x.device,
                    dtype=torch.float32,
                )
                snapshots = model._svt_iterates(
                    batch_x,
                    initial_grid,
                    prior_ids,
                    steps,
                    threshold_scale=threshold_scale,
                )
                per_batch_depth_rmse = []
                for batch_index in range(batch_x.shape[0]):
                    mask = missing_mask[batch_index]
                    if not bool(mask.any()):
                        per_batch_depth_rmse.append(
                            snapshots.new_full(
                                (len(steps),),
                                float("inf"),
                                dtype=torch.float32,
                            )
                        )
                        continue
                    errors = (
                        snapshots[batch_index].float()[:, mask]
                        - target[batch_index][mask].view(1, -1)
                    )
                    per_batch_depth_rmse.append(
                        errors.square().mean(dim=1).clamp_min(0.0).sqrt()
                    )
                depth_rmse = torch.stack(per_batch_depth_rmse, dim=0)
                rmse_by_scale.append(depth_rmse.min(dim=1).values)
        finally:
            model.train(was_training)

    scale_rmse = torch.stack(rmse_by_scale, dim=1).to(
        device=predicted_scale.device,
        dtype=torch.float32,
    )
    finite = torch.isfinite(scale_rmse).all(dim=1)
    if not bool(finite.any()):
        return None, {}
    scale_rmse = scale_rmse[finite]
    predicted_scale = predicted_scale.float().view(-1, 1)[finite]
    scale_values = torch.tensor(
        scales,
        device=predicted_scale.device,
        dtype=torch.float32,
    ).view(1, -1)
    target_probs = torch.softmax(-scale_rmse / float(temperature), dim=1).detach()
    target_log_scale = (target_probs * scale_values.log()).sum(dim=1, keepdim=True)
    predicted_log_scale = predicted_scale.clamp_min(1e-8).log()
    loss = torch.nn.functional.mse_loss(predicted_log_scale, target_log_scale)
    oracle_index = scale_rmse.argmin(dim=1)
    oracle_scales = scale_values.expand(scale_rmse.shape[0], -1).gather(
        1,
        oracle_index.view(-1, 1),
    )
    stats = {
        "svt_adaptive_threshold_distill_target_log_scale": float(
            target_log_scale.mean().detach().cpu()
        ),
        "svt_adaptive_threshold_distill_oracle_scale": float(
            oracle_scales.mean().detach().cpu()
        ),
        "svt_adaptive_threshold_distill_oracle_rmse": float(
            scale_rmse.min(dim=1).values.mean().detach().cpu()
        ),
    }
    return loss.to(dtype=predicted_scale.dtype), stats


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
    scaler: torch.cuda.amp.GradScaler,
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
        "scaler": scaler.state_dict(),
        "smoothed_loss": smoothed_loss,
        "best_smoothed_loss": best_smoothed_loss,
    }


def build_base_prior_config(args: argparse.Namespace) -> dict:
    if args.prior_kind == "mixed_matrix":
        config = {
            "mixed_priors": (
                json.loads(args.mixed_priors_json)
                if args.mixed_priors_json
                else list(DEFAULT_MIXED_PRIORS)
            )
        }
        if args.prior_overrides_json:
            config.update(json.loads(args.prior_overrides_json))
        return config

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
        if args.prior_overrides_json:
            config.update(json.loads(args.prior_overrides_json))
        return config

    config = dict(PRIOR_CONFIG)
    if args.p_missing is not None:
        config["p_missing"] = args.p_missing
    if args.latent_rank_low is not None:
        config["latent_rank_low"] = args.latent_rank_low
    if args.latent_rank_high is not None:
        config["latent_rank_high"] = args.latent_rank_high
    if args.prior_overrides_json:
        config.update(json.loads(args.prior_overrides_json))
    if int(config["latent_rank_high"]) < int(config["latent_rank_low"]):
        raise ValueError(
            "latent_rank_high must be >= latent_rank_low, got "
            f"{config['latent_rank_high']} < {config['latent_rank_low']}"
        )
    return config


def parse_phases(args: argparse.Namespace) -> tuple[CurriculumPhase, ...]:
    if args.smoke:
        return tuple(
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
    if args.phases_json is None:
        return DEFAULT_PHASES

    raw_phases = json.loads(args.phases_json)
    if not isinstance(raw_phases, list) or not raw_phases:
        raise ValueError("--phases-json must be a non-empty JSON list.")
    phases = []
    for raw in raw_phases:
        phases.append(
            CurriculumPhase(
                name=str(raw["name"]),
                end_step=int(raw["end_step"]),
                rows_low=int(raw["rows_low"]),
                rows_high=int(raw["rows_high"]),
                cols_low=int(raw["cols_low"]),
                cols_high=int(raw["cols_high"]),
                batch_size=int(raw["batch_size"]),
            )
        )
    previous_end = 0
    for phase in phases:
        if phase.end_step <= previous_end:
            raise ValueError("Curriculum phase end_step values must be increasing.")
        previous_end = phase.end_step
    return tuple(phases)


def main() -> None:
    args = parse_args()
    if args.support_mask_mode != "mixed" and (args.benchmark_selection_curriculum or args.conditional_denoising_curriculum):
        raise ValueError("Use support masks or a legacy mask curriculum, not both")
    if args.validation_every <= 0 or args.update_diagnostics_every < 0:
        raise ValueError("validation-every must be positive; update-diagnostics-every nonnegative")
    if args.benchmark_selection_curriculum and args.conditional_denoising_curriculum:
        raise ValueError(
            "Use only one of --benchmark-selection-curriculum and "
            "--conditional-denoising-curriculum."
        )
    if args.benchmark_selection_curriculum:
        if not 0.0 < args.benchmark_query_fraction_low <= args.benchmark_query_fraction_high < 1.0:
            raise ValueError(
                "benchmark query fraction bounds must satisfy 0 < low <= high < 1."
            )
        if not 0.0 < args.benchmark_context_observed_prob_low <= args.benchmark_context_observed_prob_high <= 1.0:
            raise ValueError(
                "benchmark context observed probability bounds must satisfy 0 < low <= high <= 1."
            )
        if args.benchmark_query_k_low < 1 or args.benchmark_query_k_high < args.benchmark_query_k_low:
            raise ValueError("benchmark query k bounds must satisfy 1 <= low <= high.")
        if not 0.0 < args.benchmark_query_selected_observed_prob_low <= args.benchmark_query_selected_observed_prob_high <= 1.0:
            raise ValueError(
                "benchmark query selected observed probability bounds must satisfy 0 < low <= high <= 1."
            )
    if args.conditional_denoising_curriculum:
        if not 0.0 < args.conditional_row_observed_prob_low <= args.conditional_row_observed_prob_high < 1.0:
            raise ValueError(
                "conditional row observed probability bounds must satisfy "
                "0 < low <= high < 1."
            )
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    phases = parse_phases(args)
    if args.smoke:
        args.warmup_steps = 1
        args.stable_steps = len(phases) - 1
        args.decay_steps = 0
        args.checkpoint_every = 0

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
    if args.precision_mode == "full_bfloat16" and device.type == "cuda":
        model_dtype = torch.bfloat16
        input_dtype = torch.bfloat16
        autocast_enabled = False
        autocast_dtype = torch.bfloat16
    elif args.precision_mode == "amp_bfloat16" and device.type == "cuda":
        model_dtype = torch.float32
        input_dtype = torch.float32
        autocast_enabled = True
        autocast_dtype = torch.bfloat16
    elif args.precision_mode == "amp_float16" and device.type == "cuda":
        model_dtype = torch.float32
        input_dtype = torch.float32
        autocast_enabled = True
        autocast_dtype = torch.float16
    else:
        model_dtype = torch.float32
        input_dtype = torch.float32
        autocast_enabled = False
        autocast_dtype = torch.float32
    base_prior_config = build_base_prior_config(args)
    model_config = dict(MODEL_CONFIG)
    if args.model_overrides_json:
        model_config.update(json.loads(args.model_overrides_json))
    if device.type == "cuda":
        # cuDNN SDPA can fail during bf16 backward when the sampled matrix
        # dimensions change between steps. Flash/memory-efficient SDPA is
        # stable for this variable-shape curriculum.
        torch.backends.cuda.enable_cudnn_sdp(False)

    model = TabImputeModelPureRaBEL(**model_config).to(
        device=device,
        dtype=model_dtype,
    )
    fp32_param_substrings = parse_substrings(args.fp32_param_substrings)
    trainable_param_substrings = None
    if args.trainable_param_substrings:
        trainable_param_substrings = parse_substrings(args.trainable_param_substrings)
        if not trainable_param_substrings:
            raise ValueError("--trainable-param-substrings did not contain any names.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(
                substring in name for substring in trainable_param_substrings
            )
    frozen_substrings = parse_substrings(args.freeze_param_substrings)
    for name, parameter in model.named_parameters():
        if any(part in name for part in frozen_substrings):
            parameter.requires_grad_(False)
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameter_count <= 0:
        raise ValueError("No trainable parameters remain after applying freeze filters.")
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Trainable parameters: {trainable_parameter_count:,}/{total_parameter_count:,}",
        flush=True,
    )
    post_warmup_lr_ratio = 1.0
    if args.post_warmup_lr is not None:
        post_warmup_lr_ratio = args.post_warmup_lr / args.lr
        if post_warmup_lr_ratio > 1.0:
            raise ValueError(
                "--post-warmup-lr should be <= --lr for the warm-drop schedule."
            )
    lr_multipliers = json.loads(args.param_lr_multipliers_json)
    if any(not key or not math.isfinite(float(value)) or float(value) <= 0
           for key, value in lr_multipliers.items()):
        raise ValueError("LR multipliers require nonempty names and positive finite values")
    parameter_groups = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            matches = [key for key in lr_multipliers if key in name]
            if len(matches) > 1:
                raise ValueError(f"Overlapping LR multipliers for {name}: {matches}")
            multiplier = float(lr_multipliers[matches[0]]) if matches else 1.0
            parameter_groups.append((parameter, multiplier))
    optimizer = torch.optim.AdamW(
        [{"params": [p for p, m in parameter_groups if m == multiplier], "lr": args.lr * multiplier}
         for multiplier in sorted({m for _, m in parameter_groups})],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.precision_mode == "amp_float16" and device.type == "cuda",
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
    optimizer_continuation = None
    if args.retain_init_optimizer_state and (
        args.init_model_checkpoint is None or args.resume is not None
        or args.allow_partial_init or args.freeze_param_substrings
        or args.trainable_param_substrings or lr_multipliers
        or json.loads(args.parameter_value_overrides_json or "{}")
    ):
        raise ValueError("Retained AdamW initialization requires strict full-model initialization without parameter filters/overrides")
    smoothed_loss = None
    best_smoothed_loss = float("inf")
    if args.resume is not None and args.init_model_checkpoint is not None:
        raise ValueError("Use only one of --resume or --init-model-checkpoint.")
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["epoch"])
        smoothed_loss = checkpoint.get("smoothed_loss")
        best_smoothed_loss = float(
            checkpoint.get("best_smoothed_loss", best_smoothed_loss)
        )
    elif args.init_model_checkpoint is not None:
        checkpoint = torch.load(
            args.init_model_checkpoint,
            map_location=device,
            weights_only=True,
        )
        checkpoint_model = checkpoint["model"]
        if args.allow_partial_init:
            model_state = model.state_dict()
            skipped_shape_keys = []
            filtered_checkpoint_model = {}
            for key, value in checkpoint_model.items():
                if key in model_state and model_state[key].shape == value.shape:
                    filtered_checkpoint_model[key] = value
                elif key in model_state:
                    skipped_shape_keys.append(
                        f"{key}: checkpoint {tuple(value.shape)} != model {tuple(model_state[key].shape)}"
                    )
            checkpoint_model = filtered_checkpoint_model
        load_result = model.load_state_dict(
            checkpoint_model,
            strict=not args.allow_partial_init,
        )
        if args.allow_partial_init:
            print(
                "Partial init loaded with "
                f"{len(load_result.missing_keys)} missing and "
                f"{len(load_result.unexpected_keys)} unexpected keys.",
                flush=True,
            )
            if skipped_shape_keys:
                preview = "; ".join(skipped_shape_keys[:8])
                suffix = "" if len(skipped_shape_keys) <= 8 else " ..."
                print(
                    f"Skipped {len(skipped_shape_keys)} shape-mismatched init tensors: "
                    f"{preview}{suffix}",
                    flush=True,
                )

    if args.retain_init_optimizer_state:
        optimizer_continuation = retain_adamw_state(optimizer, checkpoint, model)
        print(f"Retained optimizer state: {optimizer_continuation}", flush=True)

    fp32_parameter_names = keep_selected_parameters_fp32(model, fp32_param_substrings)
    parameter_value_override_names = apply_parameter_value_overrides(
        model,
        args.parameter_value_overrides_json,
    )
    if fp32_param_substrings:
        print(
            "Keeping "
            f"{len(fp32_parameter_names)} parameters in float32 for substrings "
            f"{fp32_param_substrings}: {', '.join(fp32_parameter_names[:12])}"
            f"{' ...' if len(fp32_parameter_names) > 12 else ''}",
            flush=True,
        )
    if parameter_value_override_names:
        print(
            "Applied parameter value overrides: "
            f"{', '.join(parameter_value_override_names)}",
            flush=True,
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
        "trainable_param_substrings": trainable_param_substrings,
        "freeze_param_substrings": frozen_substrings,
        "param_lr_multipliers": lr_multipliers,
        "fp32_param_substrings": fp32_param_substrings,
        "fp32_parameter_names": fp32_parameter_names,
        "parameter_value_override_names": parameter_value_override_names,
        "precision_mode": args.precision_mode,
        "fixed_validation": {
            "cache": str(args.fixed_validation_cache) if args.fixed_validation_cache else None,
            "seed": args.validation_seed, "every": args.validation_every,
            "cases_per_group": args.validation_cases_per_group,
            "selection_metric": "equal_group_mean_rmse_excluding_monitor_groups",
        },
        "trainable_parameter_count": trainable_parameter_count,
        "total_parameter_count": total_parameter_count,
        "prior_kind": args.prior_kind,
        "generator_type": args.generator_type,
        "missingness_type": args.missingness_type,
        "prior": base_prior_config,
        "curriculum": [asdict(phase) for phase in phases],
        "optimizer": {
            "name": "adamw",
            "lr": args.lr,
            "post_warmup_lr": args.post_warmup_lr,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
        },
        "auxiliary_losses": {
            "nll_loss_weight": args.nll_loss_weight,
            "point_loss_weight": args.point_loss_weight,
            "point_loss_kind": args.point_loss_kind,
            "point_loss_prediction": args.point_loss_prediction,
            "huber_delta": args.huber_delta,
            "soft_target_loss_weight": args.soft_target_loss_weight,
            "soft_target_sigma": args.soft_target_sigma,
            "cdf_loss_weight": args.cdf_loss_weight,
            "entropy_loss_weight": args.entropy_loss_weight,
            "svt_mixture_depth_distill_weight": args.svt_mixture_depth_distill_weight,
            "svt_mixture_depth_distill_temperature": (
                args.svt_mixture_depth_distill_temperature
            ),
            "svt_mixture_cv_regret_weight": args.svt_mixture_cv_regret_weight,
            "svt_mixture_entropy_bonus_weight": (
                args.svt_mixture_entropy_bonus_weight
            ),
            "svt_adaptive_threshold_distill_weight": (
                args.svt_adaptive_threshold_distill_weight
            ),
            "svt_adaptive_threshold_distill_temperature": (
                args.svt_adaptive_threshold_distill_temperature
            ),
            "svt_adaptive_threshold_distill_scales": (
                args.svt_adaptive_threshold_distill_scales
            ),
            "distribution_loss_chunk_size": args.distribution_loss_chunk_size,
            "per_prior_loss_ema_normalize": args.per_prior_loss_ema_normalize,
            "prior_loss_ema_beta": args.prior_loss_ema_beta,
        },
        "scheduler": {
            "name": "warmup_stable_cosine",
            "warmup_steps": args.warmup_steps,
            "stable_steps": args.stable_steps,
            "decay_steps": args.decay_steps,
            "min_lr_ratio": args.min_lr_ratio,
        },
        "training": {
            "seed": args.seed,
            "data_seed": args.data_seed,
            "support_mask_mode": args.support_mask_mode,
            "update_diagnostics_every": args.update_diagnostics_every,
            "checkpoint_every": args.checkpoint_every,
            "snapshot_steps": args.snapshot_steps,
            "metrics_jsonl_every": args.metrics_jsonl_every,
            "decode_only_missing": True,
            "nll_loss_weight": args.nll_loss_weight,
            "point_loss_weight": args.point_loss_weight,
            "point_loss_kind": args.point_loss_kind,
            "point_loss_prediction": args.point_loss_prediction,
            "huber_delta": args.huber_delta,
            "svt_mixture_depth_distill_weight": args.svt_mixture_depth_distill_weight,
            "svt_mixture_depth_distill_temperature": (
                args.svt_mixture_depth_distill_temperature
            ),
            "svt_mixture_cv_regret_weight": args.svt_mixture_cv_regret_weight,
            "svt_mixture_entropy_bonus_weight": (
                args.svt_mixture_entropy_bonus_weight
            ),
            "svt_adaptive_threshold_distill_weight": (
                args.svt_adaptive_threshold_distill_weight
            ),
            "svt_adaptive_threshold_distill_temperature": (
                args.svt_adaptive_threshold_distill_temperature
            ),
            "svt_adaptive_threshold_distill_scales": (
                args.svt_adaptive_threshold_distill_scales
            ),
            "cudnn_sdpa_enabled": False,
            "device": args.device,
            "precision_mode": args.precision_mode,
            "model_dtype": str(model_dtype).replace("torch.", ""),
            "input_dtype": str(input_dtype).replace("torch.", ""),
            "autocast_enabled": autocast_enabled,
            "autocast_dtype": str(autocast_dtype).replace("torch.", ""),
            "starts_from_pretrained_weights": args.init_model_checkpoint is not None,
            "optimizer_continuation": optimizer_continuation,
            "init_model_checkpoint": (
                str(args.init_model_checkpoint)
                if args.init_model_checkpoint is not None
                else None
            ),
            "scalar_encoder_residual": False,
        },
    }
    with (work_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2, sort_keys=True)

    validation_cases = None
    best_validation_rmse = float("inf")
    if args.fixed_validation_cache:
        validation_cases = load_or_build_validation(
            args.fixed_validation_cache, base_prior_config, seed=args.validation_seed,
            cases_per_group=args.validation_cases_per_group
        )
        saved_validation = work_dir / "best_validation_metrics.json"
        if args.resume and saved_validation.exists():
            best_validation_rmse = json.loads(saved_validation.read_text())["macro_rmse"]

    def validate(step):
        nonlocal best_validation_rmse
        if validation_cases is None:
            return
        metrics = evaluate_fixed_validation(
            model, criterion, validation_cases, device, args.precision_mode,
            prediction_path=work_dir / "validation_predictions_latest.pth",
        )
        metrics["step"] = step
        with (work_dir / "validation_stream.jsonl").open("a") as file:
            file.write(json.dumps(metrics) + "\n")
        print(f"[validation] step={step} macro_rmse={metrics['macro_rmse']:.6f} "
              f"macro_nll={metrics['macro_nll']:.6f}", flush=True)
        if metrics["macro_rmse"] < best_validation_rmse:
            best_validation_rmse = metrics["macro_rmse"]
            atomic_torch_save(model_checkpoint(model, step=step, metrics=metrics),
                              work_dir / "best_validation_checkpoint.pth")
            (work_dir / "best_validation_metrics.json").write_text(json.dumps(metrics, indent=2))
            import shutil
            shutil.copyfile(work_dir / "validation_predictions_latest.pth",
                            work_dir / "validation_predictions_best.pth")

    validate(start_step)

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
    prior_loss_ema: dict[int, float] = {}
    model.train()
    try:
        progress = tqdm(range(start_step + 1, total_steps + 1))
        for step in progress:
            phase_index, phase = phase_for_step(phases, step)
            if phase_index != current_phase_index:
                current_phase_index = phase_index
                with (nullcontext() if args.data_seed is None else preserved_rng(args.data_seed + phase_index)):
                    prior = build_prior(
                        phase,
                        device=args.device,
                        prior_kind=args.prior_kind,
                        generator_type=args.generator_type,
                        missingness_type=args.missingness_type,
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
            if args.data_seed is None:
                (batch_x, batch_target, _, _, _), _ = prior.get_batch()
            else:
                with preserved_rng(args.data_seed + step):
                    (batch_x, batch_target, _, _, _), _ = prior.get_batch()
            batch_x = batch_x.to(device=device, dtype=input_dtype)
            batch_target = batch_target.to(device=device, dtype=input_dtype)
            benchmark_curriculum_stats = {}
            conditional_curriculum_stats = {}
            support_stats = {}
            if args.support_mask_mode != "mixed":
                batch_x, batch_target, missing_mask, support_stats = apply_support_mask(
                    batch_x, batch_target, args.support_mask_mode,
                    seed=(args.data_seed if args.data_seed is not None else (args.seed or 0)) + 10000000 + step,
                )
                batch_x = batch_x.to(input_dtype)
            elif args.benchmark_selection_curriculum:
                (
                    batch_x_float,
                    batch_target_float,
                    missing_mask,
                    benchmark_curriculum_stats,
                ) = apply_benchmark_selection_curriculum(
                    batch_target.float(),
                    query_fraction_low=args.benchmark_query_fraction_low,
                    query_fraction_high=args.benchmark_query_fraction_high,
                    context_observed_prob_low=args.benchmark_context_observed_prob_low,
                    context_observed_prob_high=args.benchmark_context_observed_prob_high,
                    query_k_low=args.benchmark_query_k_low,
                    query_k_high=args.benchmark_query_k_high,
                    query_selected_observed_prob_low=(
                        args.benchmark_query_selected_observed_prob_low
                    ),
                    query_selected_observed_prob_high=(
                        args.benchmark_query_selected_observed_prob_high
                    ),
                )
                batch_x = batch_x_float.to(device=device, dtype=input_dtype)
                batch_target = batch_target_float.to(
                    device=device,
                    dtype=input_dtype,
                )
            elif args.conditional_denoising_curriculum:
                (
                    batch_x_float,
                    batch_target_float,
                    missing_mask,
                    conditional_curriculum_stats,
                ) = apply_conditional_denoising_curriculum(
                    batch_target.float(),
                    row_observed_prob_low=args.conditional_row_observed_prob_low,
                    row_observed_prob_high=args.conditional_row_observed_prob_high,
                )
                batch_x = batch_x_float.to(device=device, dtype=input_dtype)
                batch_target = batch_target_float.to(
                    device=device,
                    dtype=input_dtype,
                )
            else:
                missing_mask = torch.isnan(batch_x)
            if not missing_mask.any():
                raise RuntimeError("Generated batch contains no missing entries.")

            sampled_prior_index = int(getattr(prior, "last_prior_index", -1))
            prior_index = None
            if sampled_prior_index >= 0:
                prior_index = torch.full(
                    (batch_x.shape[0],),
                    sampled_prior_index,
                    device=device,
                    dtype=torch.long,
                )

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                logits = model(
                    batch_x,
                    decode_mask=missing_mask,
                    prior_index=prior_index,
                )
            target_missing = batch_target[missing_mask]
            output_criterion = prediction_distribution(model, criterion)
            losses = output_criterion(logits.float(), target_missing.float())
            nll_loss = losses.mean()
            point_loss = logits.new_zeros((), dtype=torch.float32)
            soft_target_loss = logits.new_zeros((), dtype=torch.float32)
            cdf_loss = logits.new_zeros((), dtype=torch.float32)
            entropy_loss = logits.new_zeros((), dtype=torch.float32)
            svt_depth_distill_loss = logits.new_zeros((), dtype=torch.float32)
            svt_cv_regret_loss = logits.new_zeros((), dtype=torch.float32)
            svt_mixture_entropy_loss = logits.new_zeros((), dtype=torch.float32)
            svt_mixture_entropy_value = None
            svt_depth_rmse = None
            svt_threshold_distill_loss = logits.new_zeros((), dtype=torch.float32)
            threshold_distill_stats: dict[str, float] = {}
            point_predictions = None
            if args.point_loss_weight > 0.0:
                if args.point_loss_prediction == "mean":
                    point_predictions = output_criterion.mean(logits.float())
                else:
                    point_predictions = output_criterion.median(logits.float())
                point_errors = point_predictions - target_missing.float()
                if args.point_loss_kind == "mse":
                    point_loss = point_errors.square().mean()
                elif args.point_loss_kind == "huber":
                    point_loss = torch.nn.functional.huber_loss(
                        point_predictions.float(),
                        target_missing.float(),
                        delta=args.huber_delta,
                        reduction="mean",
                    )
                else:
                    point_loss = point_errors.abs().mean()
            if args.soft_target_loss_weight > 0.0:
                soft_target_loss = soft_target_ce_loss(
                    logits,
                    distribution_target(model, target_missing),
                    criterion,
                    sigma=args.soft_target_sigma,
                    chunk_size=args.distribution_loss_chunk_size,
                )
            if args.cdf_loss_weight > 0.0:
                cdf_loss = cdf_alignment_loss(
                    logits,
                    distribution_target(model, target_missing),
                    criterion,
                    chunk_size=args.distribution_loss_chunk_size,
                )
            if args.entropy_loss_weight > 0.0:
                entropy_loss = normalized_entropy_loss(logits)
            if args.svt_mixture_depth_distill_weight > 0.0:
                depth_distill, depth_rmse = svt_mixture_depth_distill_loss(
                    model,
                    target_missing,
                    temperature=args.svt_mixture_depth_distill_temperature,
                )
                if depth_distill is not None:
                    svt_depth_distill_loss = depth_distill.float()
                    svt_depth_rmse = depth_rmse
            if args.svt_mixture_cv_regret_weight > 0.0:
                cv_regret = svt_mixture_cv_regret_loss(model)
                if cv_regret is not None:
                    svt_cv_regret_loss = cv_regret.float()
            if args.svt_mixture_entropy_bonus_weight > 0.0:
                entropy_bonus, entropy_value = svt_mixture_entropy_bonus(model)
                if entropy_bonus is not None:
                    svt_mixture_entropy_loss = entropy_bonus.float()
                    svt_mixture_entropy_value = entropy_value
            if args.svt_adaptive_threshold_distill_weight > 0.0:
                threshold_distill, threshold_distill_stats = (
                    svt_adaptive_threshold_distill_loss(
                        model,
                        batch_x,
                        batch_target,
                        missing_mask,
                        prior_index,
                        scales=args.svt_adaptive_threshold_distill_scales,
                        temperature=(
                            args.svt_adaptive_threshold_distill_temperature
                        ),
                    )
                )
                if threshold_distill is not None:
                    svt_threshold_distill_loss = threshold_distill.float()
            loss = (
                args.nll_loss_weight * nll_loss
                + args.point_loss_weight * point_loss
                + args.soft_target_loss_weight * soft_target_loss
                + args.cdf_loss_weight * cdf_loss
                + args.entropy_loss_weight * entropy_loss
                + args.svt_mixture_depth_distill_weight * svt_depth_distill_loss
                + args.svt_mixture_cv_regret_weight * svt_cv_regret_loss
                + args.svt_mixture_entropy_bonus_weight
                * svt_mixture_entropy_loss
                + args.svt_adaptive_threshold_distill_weight
                * svt_threshold_distill_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss}")

            loss_for_backward = loss
            loss_optimization_scale = 1.0
            if args.per_prior_loss_ema_normalize and sampled_prior_index >= 0:
                detached_loss = float(loss.detach().cpu())
                beta = float(args.prior_loss_ema_beta)
                previous_ema = prior_loss_ema.get(sampled_prior_index, detached_loss)
                current_ema = beta * previous_ema + (1.0 - beta) * detached_loss
                prior_loss_ema[sampled_prior_index] = current_ema
                global_ema = sum(prior_loss_ema.values()) / max(len(prior_loss_ema), 1)
                loss_optimization_scale = global_ema / max(current_ema, 1e-8)
                loss_for_backward = loss * loss.new_tensor(loss_optimization_scale)

            update_samples = {}
            objective_gradient_stats = {}
            measure_update = args.update_diagnostics_every > 0 and step % args.update_diagnostics_every == 0
            if measure_update:
                objective_gradient_stats = objective_logit_gradient_stats(
                    logits, (("nll", nll_loss, args.nll_loss_weight),
                             ("point", point_loss, args.point_loss_weight)))
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        stride = max(1, parameter.numel() // 4096)
                        update_samples[name] = parameter.detach().view(-1)[::stride][:4096].float().clone()
            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip_norm,
                    error_if_nonfinite=True,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_for_backward.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()
            scheduler.step()
            update_stats = {}
            if measure_update:
                grouped_updates = {}
                for name, parameter in model.named_parameters():
                    if name not in update_samples:
                        continue
                    stride = max(1, parameter.numel() // 4096)
                    before = update_samples[name]
                    after = parameter.detach().view(-1)[::stride][:4096].float()
                    parts = name.split(".")
                    group = ".".join(parts[:2]) if len(parts) > 1 and parts[1].isdigit() else parts[0]
                    values = grouped_updates.setdefault(group, [0.0, 0.0])
                    values[0] += float((after - before).square().sum())
                    values[1] += float(before.square().sum())
                update_stats = {f"sampled_update_ratio/{g}": math.sqrt(d / max(w, 1e-30))
                                for g, (d, w) in grouped_updates.items()}
                for name, parameter in model.named_parameters():
                    if "axis_blend" in name:
                        update_stats[f"axis_blend/{name}"] = float(parameter.detach().tanh())
                    if "axis_gate" in name and parameter.grad is not None:
                        update_stats[f"axis_gate_grad_rms/{name}"] = float(parameter.grad.float().square().mean().sqrt())
                del update_samples

            with torch.no_grad():
                medians = output_criterion.median(logits.float()).to(logits.dtype)
                means = output_criterion.mean(logits.float()).to(logits.dtype)
                median_errors = medians - target_missing
                mean_errors = means - target_missing
                mae = median_errors.abs().mean()
                mean_mae = mean_errors.abs().mean()
                median_rmse = median_errors.float().square().mean().sqrt()
                mean_rmse = mean_errors.float().square().mean().sqrt()
            loss_value = float(loss.detach().cpu())
            nll_loss_value = float(nll_loss.detach().cpu())
            point_loss_value = float(point_loss.detach().cpu())
            soft_target_loss_value = float(soft_target_loss.detach().cpu())
            cdf_loss_value = float(cdf_loss.detach().cpu())
            entropy_loss_value = float(entropy_loss.detach().cpu())
            svt_depth_distill_loss_value = float(
                svt_depth_distill_loss.detach().cpu()
            )
            svt_cv_regret_loss_value = float(svt_cv_regret_loss.detach().cpu())
            svt_mixture_entropy_loss_value = float(
                svt_mixture_entropy_loss.detach().cpu()
            )
            svt_threshold_distill_loss_value = float(
                svt_threshold_distill_loss.detach().cpu()
            )
            smoothed_loss = (
                loss_value
                if smoothed_loss is None
                else 0.995 * smoothed_loss + 0.005 * loss_value
            )
            elapsed = time.time() - step_start
            last_metrics = {
                **support_stats,
                **update_stats,
                **objective_gradient_stats,
                "loss_missing": loss_value,
                "loss_backward": float(loss_for_backward.detach().cpu()),
                "loss_optimization_scale": loss_optimization_scale,
                "nll_missing": nll_loss_value,
                "point_loss_missing": point_loss_value,
                "soft_target_loss_missing": soft_target_loss_value,
                "cdf_loss_missing": cdf_loss_value,
                "entropy_loss_missing": entropy_loss_value,
                "svt_mixture_depth_distill_loss": svt_depth_distill_loss_value,
                "svt_mixture_cv_regret_loss": svt_cv_regret_loss_value,
                "svt_mixture_entropy_loss": svt_mixture_entropy_loss_value,
                "svt_adaptive_threshold_distill_loss": (
                    svt_threshold_distill_loss_value
                ),
                "loss_missing_ema": smoothed_loss,
                "mae_missing": float(mae.detach().cpu()),
                "mae_missing_mean": float(mean_mae.detach().cpu()),
                "rmse_missing_median": float(median_rmse.detach().cpu()),
                "rmse_missing_mean": float(mean_rmse.detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
                **{f"lr_group/{i}": group["lr"] for i, group in enumerate(optimizer.param_groups)},
                "grad_norm": float(grad_norm.detach().cpu()),
                "grad_was_clipped": float(
                    float(grad_norm.detach().cpu()) > args.grad_clip_norm
                ),
                "amp_scaler_scale": float(scaler.get_scale()),
                "phase": phase_index + 1,
                "phase_name": phase.name,
                "matrix_rows": batch_x.shape[1],
                "matrix_cols": batch_x.shape[2],
                "batch_size": batch_x.shape[0],
            }
            if benchmark_curriculum_stats:
                last_metrics.update(benchmark_curriculum_stats)
            if conditional_curriculum_stats:
                last_metrics.update(conditional_curriculum_stats)
            if svt_mixture_entropy_value is not None:
                last_metrics["svt_mixture_entropy"] = svt_mixture_entropy_value
            sampled_prior_name = getattr(prior, "last_prior_name", "")
            if sampled_prior_name:
                last_metrics["sampled_prior_name"] = sampled_prior_name
                last_metrics["sampled_prior_index"] = sampled_prior_index
            last_metrics["svt_adaptive_threshold_distill_stats_count"] = len(
                threshold_distill_stats
            )
            if threshold_distill_stats:
                last_metrics.update(threshold_distill_stats)
            svt_mixture_weights = getattr(model, "last_svt_mixture_weights", None)
            svt_mixture_steps = getattr(model, "svt_mixture_steps", None)
            if svt_mixture_weights is not None and svt_mixture_steps is not None:
                weights_cpu = (
                    svt_mixture_weights.detach().float().mean(dim=0).cpu()
                )
                expected_steps = 0.0
                for index, step_count in enumerate(svt_mixture_steps):
                    value = float(weights_cpu[index])
                    last_metrics[f"svt_mixture_weight_step_{step_count}"] = value
                    expected_steps += value * float(step_count)
                last_metrics["svt_mixture_expected_steps"] = expected_steps
            svt_adaptive_threshold_scale = getattr(
                model,
                "last_svt_adaptive_threshold_scale",
                None,
            )
            if svt_adaptive_threshold_scale is not None:
                scale_cpu = svt_adaptive_threshold_scale.detach().float().cpu()
                last_metrics["svt_adaptive_threshold_scale_mean"] = float(
                    scale_cpu.mean()
                )
                last_metrics["svt_adaptive_threshold_scale_min"] = float(
                    scale_cpu.min()
                )
                last_metrics["svt_adaptive_threshold_scale_max"] = float(
                    scale_cpu.max()
                )
            empirical_bayes_ridge = getattr(model, "last_empirical_bayes_ridge", None)
            if empirical_bayes_ridge is not None:
                last_metrics["empirical_bayes_ridge"] = float(
                    empirical_bayes_ridge.detach().float().cpu()
                )
            empirical_bayes_shrinkage = getattr(
                model,
                "last_empirical_bayes_shrinkage",
                None,
            )
            if empirical_bayes_shrinkage is not None:
                last_metrics["empirical_bayes_shrinkage"] = float(
                    empirical_bayes_shrinkage.detach().float().cpu()
                )
            empirical_row_fraction = getattr(
                model,
                "last_empirical_bayes_row_observed_fraction",
                None,
            )
            if empirical_row_fraction is not None:
                last_metrics["empirical_bayes_row_observed_fraction"] = float(
                    empirical_row_fraction.detach().float().mean().cpu()
                )
            empirical_col_fraction = getattr(
                model,
                "last_empirical_bayes_col_observed_fraction",
                None,
            )
            if empirical_col_fraction is not None:
                last_metrics["empirical_bayes_col_observed_fraction"] = float(
                    empirical_col_fraction.detach().float().mean().cpu()
                )
            extra_decoder_scale = getattr(
                getattr(model, "output_decoder", None),
                "extra_decoder_scale",
                None,
            )
            if extra_decoder_scale is not None:
                last_metrics["extra_decoder_scale"] = float(
                    extra_decoder_scale.detach().float().cpu()
                )
            svt_mixture_cv_rmse = getattr(model, "last_svt_mixture_cv_rmse", None)
            if svt_mixture_cv_rmse is not None and svt_mixture_steps is not None:
                cv_rmse_cpu = svt_mixture_cv_rmse.float().mean(dim=0).cpu()
                for index, step_count in enumerate(svt_mixture_steps):
                    last_metrics[f"svt_mixture_cv_rmse_step_{step_count}"] = float(
                        cv_rmse_cpu[index]
                    )
            if svt_depth_rmse is not None and svt_mixture_steps is not None:
                depth_rmse_cpu = svt_depth_rmse.float().mean(dim=0).cpu()
                oracle_index = int(depth_rmse_cpu.argmin().item())
                last_metrics["svt_mixture_depth_oracle_step"] = float(
                    svt_mixture_steps[oracle_index]
                )
                for index, step_count in enumerate(svt_mixture_steps):
                    last_metrics[f"svt_mixture_depth_rmse_step_{step_count}"] = float(
                        depth_rmse_cpu[index]
                    )
            if device.type == "cuda":
                last_metrics["cuda_peak_allocated_gib"] = (
                    torch.cuda.max_memory_allocated(device) / (1024**3)
                )
                last_metrics["cuda_peak_reserved_gib"] = (
                    torch.cuda.max_memory_reserved(device) / (1024**3)
                )
            if (
                args.metrics_jsonl_every > 0
                and (
                    step == start_step + 1
                    or step % args.metrics_jsonl_every == 0
                    or step == total_steps
                )
            ):
                with (work_dir / "metrics_stream.jsonl").open(
                    "a",
                    encoding="utf-8",
                ) as file:
                    file.write(json.dumps({"step": step, **last_metrics}) + "\n")
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
            if step in args.snapshot_steps:
                atomic_torch_save(model_checkpoint(model, step=step, metrics=last_metrics),
                                  work_dir / f"checkpoint_{step}.pth")
            if validation_cases is not None and (step % args.validation_every == 0 or step == total_steps):
                validate(step)
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
                        scaler,
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
        if smoothed_loss is not None and smoothed_loss < best_smoothed_loss:
            best_smoothed_loss = smoothed_loss
            best_payload = model_checkpoint(
                model,
                step=total_steps,
                metrics={
                    **last_metrics,
                    "best_smoothed_loss": best_smoothed_loss,
                },
            )
            atomic_torch_save(best_payload, work_dir / "best_checkpoint.pth")
            with (work_dir / "best_metrics.json").open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(best_payload["metrics"], file, indent=2, sort_keys=True)
    finally:
        callback.close()


if __name__ == "__main__":
    main()
