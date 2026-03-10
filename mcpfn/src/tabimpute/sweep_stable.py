import argparse
import importlib.resources
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import (
    BEST_TRANSFER_MODEL_CONFIG,
    BEST_TRANSFER_OPTIMIZER_CONFIG,
    TabImputeModel,
)
from tabimpute.prior.training_set_generation import MissingnessPrior
from tabimpute.train.callbacks import WandbLoggerCallback
from tabimpute.train_new_sweep import train


REGULARIZATION_FOCUSED_SPACE = {
    "embedding_size": [BEST_TRANSFER_MODEL_CONFIG["embedding_size"]],
    "num_attention_heads": [BEST_TRANSFER_MODEL_CONFIG["num_attention_heads"]],
    "num_layers": [BEST_TRANSFER_MODEL_CONFIG["num_layers"]],
    "mlp_hidden_size": [BEST_TRANSFER_MODEL_CONFIG["mlp_hidden_size"]],
    "num_cls": [BEST_TRANSFER_MODEL_CONFIG["num_cls"]],
    "rope_fraction": [BEST_TRANSFER_MODEL_CONFIG["rope_fraction"]],
    "attention_dropout": [0.0, 0.01, 0.02, 0.03],
    "ffn_dropout": [0.03, 0.05, 0.08, 0.10],
    "drop_path_rate": [0.0, 0.01, 0.02, 0.03],
    "residual_scale_init": [0.15, 0.20, 0.30, 0.50],
    "embedding_dropout": [0.0, 0.03],
    "lr": [BEST_TRANSFER_OPTIMIZER_CONFIG["lr"]],
    "weight_decay": [BEST_TRANSFER_OPTIMIZER_CONFIG["weight_decay"]],
    "grad_clip_norm": [BEST_TRANSFER_OPTIMIZER_CONFIG["grad_clip_norm"]],
    "optimizer_name": [BEST_TRANSFER_OPTIMIZER_CONFIG["optimizer_name"]],
    "scheduler_name": [BEST_TRANSFER_OPTIMIZER_CONFIG["scheduler_name"]],
    "warmup_ratio": [BEST_TRANSFER_OPTIMIZER_CONFIG["warmup_ratio"]],
    "min_lr_ratio": [BEST_TRANSFER_OPTIMIZER_CONFIG["min_lr_ratio"]],
}

DEPTH_EXPANSION_SPACE = {
    "embedding_size": [BEST_TRANSFER_MODEL_CONFIG["embedding_size"]],
    "num_attention_heads": [BEST_TRANSFER_MODEL_CONFIG["num_attention_heads"]],
    "num_layers": [8, 10, 12],
    "mlp_hidden_size": [1536, 2048],
    "num_cls": [BEST_TRANSFER_MODEL_CONFIG["num_cls"]],
    "rope_fraction": [0.75, 1.0],
    "attention_dropout": [0.01, 0.02, 0.03, 0.05],
    "ffn_dropout": [0.05, 0.08, 0.10, 0.12],
    "drop_path_rate": [0.01, 0.03, 0.05, 0.08],
    "residual_scale_init": [0.10, 0.15, 0.20, 0.30],
    "embedding_dropout": [0.0, 0.03],
    "lr": [BEST_TRANSFER_OPTIMIZER_CONFIG["lr"]],
    "weight_decay": [BEST_TRANSFER_OPTIMIZER_CONFIG["weight_decay"]],
    "grad_clip_norm": [BEST_TRANSFER_OPTIMIZER_CONFIG["grad_clip_norm"]],
    "optimizer_name": [BEST_TRANSFER_OPTIMIZER_CONFIG["optimizer_name"]],
    "scheduler_name": [BEST_TRANSFER_OPTIMIZER_CONFIG["scheduler_name"]],
    "warmup_ratio": [BEST_TRANSFER_OPTIMIZER_CONFIG["warmup_ratio"]],
    "min_lr_ratio": [BEST_TRANSFER_OPTIMIZER_CONFIG["min_lr_ratio"]],
}

ANCHOR_STABLE_TRIALS = [
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "mlp_hidden_size": 1536,
        "num_cls": 12,
        "rope_fraction": 0.75,
        "attention_dropout": 0.01,
        "ffn_dropout": 0.05,
        "drop_path_rate": 0.02,
        "residual_scale_init": 0.20,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.03,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "mlp_hidden_size": 1536,
        "num_cls": 12,
        "rope_fraction": 0.5,
        "attention_dropout": 0.01,
        "ffn_dropout": 0.05,
        "drop_path_rate": 0.02,
        "residual_scale_init": 0.20,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.03,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "mlp_hidden_size": 1536,
        "num_cls": 12,
        "rope_fraction": 0.75,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.05,
        "drop_path_rate": 0.0,
        "residual_scale_init": 0.30,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.03,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "mlp_hidden_size": 1536,
        "num_cls": 12,
        "rope_fraction": 0.5,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.03,
        "drop_path_rate": 0.01,
        "residual_scale_init": 0.50,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.02,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 10,
        "mlp_hidden_size": 1536,
        "num_cls": 12,
        "rope_fraction": 0.75,
        "attention_dropout": 0.02,
        "ffn_dropout": 0.08,
        "drop_path_rate": 0.05,
        "residual_scale_init": 0.15,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.03,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 12,
        "mlp_hidden_size": 2048,
        "num_cls": 12,
        "rope_fraction": 0.5,
        "attention_dropout": 0.03,
        "ffn_dropout": 0.10,
        "drop_path_rate": 0.08,
        "residual_scale_init": 0.15,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.03,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
]

ROUND2_TARGETED_TRIALS = [
    # Shallow anchors centered on trial_004 and trial_003.
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "mlp_hidden_size": 1536,
        "num_cls": 12,
        "rope_fraction": 0.5,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.03,
        "drop_path_rate": 0.01,
        "residual_scale_init": 0.50,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.02,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "mlp_hidden_size": 1536,
        "num_cls": 12,
        "rope_fraction": 0.5,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.03,
        "drop_path_rate": 0.0,
        "residual_scale_init": 0.70,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.02,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "mlp_hidden_size": 1536,
        "num_cls": 12,
        "rope_fraction": 0.5,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.05,
        "drop_path_rate": 0.01,
        "residual_scale_init": 0.50,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.025,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "mlp_hidden_size": 1536,
        "num_cls": 12,
        "rope_fraction": 0.75,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.05,
        "drop_path_rate": 0.0,
        "residual_scale_init": 0.30,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.03,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "mlp_hidden_size": 1536,
        "num_cls": 12,
        "rope_fraction": 0.75,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.03,
        "drop_path_rate": 0.0,
        "residual_scale_init": 0.50,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.025,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "mlp_hidden_size": 1536,
        "num_cls": 12,
        "rope_fraction": 0.75,
        "attention_dropout": 0.005,
        "ffn_dropout": 0.03,
        "drop_path_rate": 0.005,
        "residual_scale_init": 0.50,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.025,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
    # Deep probes centered on trial_006, with lighter regularization variants.
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 12,
        "mlp_hidden_size": 2048,
        "num_cls": 12,
        "rope_fraction": 0.5,
        "attention_dropout": 0.03,
        "ffn_dropout": 0.10,
        "drop_path_rate": 0.08,
        "residual_scale_init": 0.15,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.03,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 12,
        "mlp_hidden_size": 2048,
        "num_cls": 12,
        "rope_fraction": 0.5,
        "attention_dropout": 0.02,
        "ffn_dropout": 0.08,
        "drop_path_rate": 0.06,
        "residual_scale_init": 0.20,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.03,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 12,
        "mlp_hidden_size": 2048,
        "num_cls": 12,
        "rope_fraction": 0.5,
        "attention_dropout": 0.01,
        "ffn_dropout": 0.06,
        "drop_path_rate": 0.05,
        "residual_scale_init": 0.25,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.02,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 12,
        "mlp_hidden_size": 2048,
        "num_cls": 12,
        "rope_fraction": 0.75,
        "attention_dropout": 0.02,
        "ffn_dropout": 0.08,
        "drop_path_rate": 0.05,
        "residual_scale_init": 0.20,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.02,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 10,
        "mlp_hidden_size": 2048,
        "num_cls": 12,
        "rope_fraction": 0.5,
        "attention_dropout": 0.02,
        "ffn_dropout": 0.06,
        "drop_path_rate": 0.04,
        "residual_scale_init": 0.20,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.03,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
]

# Split the round-two follow-up into explicit shallow and deep sweeps so they can
# be run independently or chained sequentially from the shell.
SHALLOW_REFINE_TRIALS = ROUND2_TARGETED_TRIALS[:6]
DEEP_REFINE_TRIALS = ROUND2_TARGETED_TRIALS[6:]

STABLE_SEED_CONFIRM_CONFIGS = [
    # Best overall downstream stable config.
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "mlp_hidden_size": 1536,
        "num_cls": 12,
        "rope_fraction": 0.5,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.03,
        "drop_path_rate": 0.0,
        "residual_scale_init": 0.70,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.02,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
    # Second-best downstream stable config.
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "mlp_hidden_size": 1536,
        "num_cls": 12,
        "rope_fraction": 0.5,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.05,
        "drop_path_rate": 0.01,
        "residual_scale_init": 0.50,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.025,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
    # Best alternate shallow rope=0.75 family.
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "mlp_hidden_size": 1536,
        "num_cls": 12,
        "rope_fraction": 0.75,
        "attention_dropout": 0.005,
        "ffn_dropout": 0.03,
        "drop_path_rate": 0.005,
        "residual_scale_init": 0.50,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.025,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
    # Best downstream deep stable config.
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 12,
        "mlp_hidden_size": 2048,
        "num_cls": 12,
        "rope_fraction": 0.75,
        "attention_dropout": 0.02,
        "ffn_dropout": 0.08,
        "drop_path_rate": 0.05,
        "residual_scale_init": 0.20,
        "embedding_dropout": 0.0,
        "lr": 2e-4,
        "weight_decay": 0.02,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
    },
]

STABLE_SEED_CONFIRM_TRIALS = [
    dict(cfg) for cfg in STABLE_SEED_CONFIRM_CONFIGS for _ in range(3)
]

DEPTH_CONDITIONED_SPACES = {
    "shallow": {
        "embedding_size": [BEST_TRANSFER_MODEL_CONFIG["embedding_size"]],
        "num_attention_heads": [BEST_TRANSFER_MODEL_CONFIG["num_attention_heads"]],
        "num_layers": [8],
        "mlp_hidden_size": [BEST_TRANSFER_MODEL_CONFIG["mlp_hidden_size"]],
        "num_cls": [BEST_TRANSFER_MODEL_CONFIG["num_cls"]],
        "rope_fraction": [0.5, 0.75],
        "attention_dropout": [0.0, 0.01, 0.02],
        "ffn_dropout": [0.03, 0.05, 0.08],
        "drop_path_rate": [0.0, 0.01, 0.02, 0.03],
        "residual_scale_init": [0.15, 0.20, 0.30, 0.50],
        "embedding_dropout": [0.0, 0.02],
        "lr": [BEST_TRANSFER_OPTIMIZER_CONFIG["lr"]],
        "weight_decay": [0.02, 0.03, 0.04],
        "grad_clip_norm": [BEST_TRANSFER_OPTIMIZER_CONFIG["grad_clip_norm"]],
        "optimizer_name": [BEST_TRANSFER_OPTIMIZER_CONFIG["optimizer_name"]],
        "scheduler_name": [BEST_TRANSFER_OPTIMIZER_CONFIG["scheduler_name"]],
        "warmup_ratio": [BEST_TRANSFER_OPTIMIZER_CONFIG["warmup_ratio"]],
        "min_lr_ratio": [BEST_TRANSFER_OPTIMIZER_CONFIG["min_lr_ratio"]],
    },
    "deep": {
        "embedding_size": [BEST_TRANSFER_MODEL_CONFIG["embedding_size"]],
        "num_attention_heads": [BEST_TRANSFER_MODEL_CONFIG["num_attention_heads"]],
        "num_layers": [10, 12],
        "mlp_hidden_size": [BEST_TRANSFER_MODEL_CONFIG["mlp_hidden_size"], 2048],
        "num_cls": [BEST_TRANSFER_MODEL_CONFIG["num_cls"]],
        "rope_fraction": [0.5, 0.75],
        "attention_dropout": [0.02, 0.03, 0.05],
        "ffn_dropout": [0.05, 0.08, 0.10],
        "drop_path_rate": [0.03, 0.05, 0.08],
        "residual_scale_init": [0.10, 0.15, 0.20, 0.30],
        "embedding_dropout": [0.0, 0.02, 0.03],
        "lr": [BEST_TRANSFER_OPTIMIZER_CONFIG["lr"]],
        "weight_decay": [0.02, 0.03, 0.04],
        "grad_clip_norm": [BEST_TRANSFER_OPTIMIZER_CONFIG["grad_clip_norm"]],
        "optimizer_name": [BEST_TRANSFER_OPTIMIZER_CONFIG["optimizer_name"]],
        "scheduler_name": [BEST_TRANSFER_OPTIMIZER_CONFIG["scheduler_name"]],
        "warmup_ratio": [BEST_TRANSFER_OPTIMIZER_CONFIG["warmup_ratio"]],
        "min_lr_ratio": [BEST_TRANSFER_OPTIMIZER_CONFIG["min_lr_ratio"]],
    },
}


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _finalize_trial_cfg(cfg: dict) -> dict:
    if cfg["embedding_size"] % cfg["num_attention_heads"] != 0:
        return {}

    if cfg["optimizer_name"] == "schedulefree_adamw":
        cfg["scheduler_name"] = "none"
        cfg["warmup_ratio"] = 0.0

    if cfg["scheduler_name"] == "none":
        cfg["warmup_ratio"] = 0.0

    if "mlp_hidden_size" not in cfg:
        cfg["mlp_hidden_size"] = int(cfg["embedding_size"] * cfg.pop("mlp_ratio"))
    elif "mlp_ratio" in cfg:
        cfg.pop("mlp_ratio")

    return cfg


def _sample_trial_config(rng: random.Random, search_space: dict) -> dict:
    cfg = {name: rng.choice(values) for name, values in search_space.items()}
    return _finalize_trial_cfg(cfg)


def _build_depth_conditioned_trials(seed: int, max_trials: int) -> list[dict]:
    rng = random.Random(seed)
    trials = []
    seen = set()
    max_attempts = max(100, max_trials * 200)
    attempts = 0

    while len(trials) < max_trials and attempts < max_attempts:
        attempts += 1
        bucket = "shallow" if len(trials) % 2 == 0 else "deep"
        cfg = _sample_trial_config(rng, DEPTH_CONDITIONED_SPACES[bucket])
        if not cfg:
            continue
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        trials.append(cfg)

    return trials


def _build_trials(seed: int, max_trials: int, search_space: dict | list[dict]) -> list[dict]:
    if isinstance(search_space, list):
        fixed_trials = []
        for cfg in search_space:
            finalized = _finalize_trial_cfg(dict(cfg))
            if finalized:
                fixed_trials.append(finalized)
        return fixed_trials[:max_trials]

    rng = random.Random(seed)
    trials = []
    seen = set()
    max_attempts = max(100, max_trials * 100)
    attempts = 0

    while len(trials) < max_trials and attempts < max_attempts:
        attempts += 1
        cfg = _sample_trial_config(rng, search_space=search_space)
        if not cfg:
            continue
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        trials.append(cfg)

    return trials


def _build_prior_config(args) -> dict:
    return {
        "num_rows_low": args.num_rows_low,
        "num_rows_high": args.num_rows_high,
        "num_cols_low": args.num_cols_low,
        "num_cols_high": args.num_cols_high,
        "p_missing": args.p_missing,
        "apply_feature_warping_prob": args.apply_feature_warping_prob,
        "apply_quantization_prob": args.apply_quantization_prob,
        "latent_rank_low": args.latent_rank_low,
        "latent_rank_high": args.latent_rank_high,
        "latent_spike_p": args.latent_spike_p,
        "latent_slab_sigma": args.latent_slab_sigma,
    }


def _ensure_fresh_run_paths(args, output_dir: Path, num_trials: int) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to reuse existing sweep output directory: {output_dir}. "
            "Choose a new --sweep-name."
        )

    trial_root = Path("workdir")
    conflicting_runs = []
    for trial_id in range(1, num_trials + 1):
        run_name = f"{args.sweep_name}-trial_{trial_id:03d}"
        run_dir = trial_root / run_name
        if run_dir.exists():
            conflicting_runs.append(str(run_dir))

    if conflicting_runs:
        preview = ", ".join(conflicting_runs[:3])
        if len(conflicting_runs) > 3:
            preview += ", ..."
        raise FileExistsError(
            "Refusing to reuse existing trial run directories: "
            f"{preview}. Choose a new --sweep-name."
        )


def _run_trial(
    *,
    args,
    trial_id: int,
    trial_cfg: dict,
    device: str,
    borders: torch.Tensor,
) -> dict:
    run_name = f"{args.sweep_name}-trial_{trial_id:03d}"
    _set_seed(args.seed + trial_id)
    model_dtype = torch.bfloat16 if "cuda" in device else torch.float32

    model = TabImputeModel(
        embedding_size=trial_cfg["embedding_size"],
        num_attention_heads=trial_cfg["num_attention_heads"],
        mlp_hidden_size=trial_cfg["mlp_hidden_size"],
        num_layers=trial_cfg["num_layers"],
        num_outputs=args.num_outputs,
        num_cls=trial_cfg["num_cls"],
        use_rope=True,
        rope_base=args.rope_base,
        rope_fraction=trial_cfg["rope_fraction"],
        use_absolute_positional_embeddings=False,
        positional_damping_factor=0.1,
        attention_dropout=trial_cfg["attention_dropout"],
        ffn_dropout=trial_cfg["ffn_dropout"],
        drop_path_rate=trial_cfg["drop_path_rate"],
        residual_scale_init=trial_cfg["residual_scale_init"],
        embedding_dropout=trial_cfg["embedding_dropout"],
    ).to(device=device, dtype=model_dtype)
    num_params = sum(p.numel() for p in model.parameters())

    prior = MissingnessPrior(
        generator_type=args.generator_type,
        missingness_type=args.missingness_type,
        config=_build_prior_config(args),
        batch_size=args.batch_size,
        verbose=False,
        entry_wise_features=False,
    )
    bar_distribution = FullSupportBarDistribution(borders=borders.to(device))

    callbacks = []
    if args.wandb_project:
        callbacks = [
            WandbLoggerCallback(
                project=args.wandb_project,
                name=run_name,
                config={
                    **trial_cfg,
                    "epochs_per_trial": args.epochs_per_trial,
                    "batch_size": args.batch_size,
                    "generator_type": args.generator_type,
                    "missingness_type": args.missingness_type,
                    "p_missing": args.p_missing,
                },
                log_dir=args.wandb_dir,
            )
        ]

    start = time.time()
    _, metrics = train(
        model=model,
        prior=prior,
        bar_distribution=bar_distribution,
        criterion=bar_distribution,
        epochs=args.epochs_per_trial,
        lr=trial_cfg["lr"],
        weight_decay=trial_cfg["weight_decay"],
        grad_clip_norm=trial_cfg["grad_clip_norm"],
        optimizer_name=trial_cfg["optimizer_name"],
        scheduler_name=trial_cfg["scheduler_name"],
        warmup_ratio=trial_cfg["warmup_ratio"],
        min_lr_ratio=trial_cfg["min_lr_ratio"],
        device=device,
        callbacks=callbacks,
        run_name=run_name,
        checkpoint_every=args.checkpoint_every if args.checkpoint_every > 0 else None,
        return_metrics=True,
    )
    elapsed = time.time() - start

    checkpoint_summary = None
    checkpoint_metrics = metrics.get("checkpoint_metrics")
    if checkpoint_metrics:
        ranked_checkpoints = sorted(
            checkpoint_metrics.items(),
            key=lambda kv: kv[1].get(args.metric, float("inf")),
        )
        best_step, best_metrics = ranked_checkpoints[0]
        checkpoint_summary = {
            "metric": args.metric,
            "best_step": int(best_step),
            "best_value": best_metrics.get(args.metric),
            "available_steps": [int(step) for step in checkpoint_metrics.keys()],
        }

    if "cuda" in device:
        torch.cuda.empty_cache()

    result = {
        "trial_id": trial_id,
        "run_name": run_name,
        "num_params": num_params,
        "seconds": round(elapsed, 2),
        "config": trial_cfg,
        "metrics": metrics,
    }
    if checkpoint_summary is not None:
        result["checkpoint_summary"] = checkpoint_summary
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stable-architecture sweep for TabImputeModel with RMSNorm and stochastic depth."
    )
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--epochs-per-trial", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--search-space",
        type=str,
        choices=[
            "regularization_focused",
            "depth_expansion",
            "depth_conditioned",
            "anchor_stable",
            "round2_targeted",
            "shallow_refine",
            "deep_refine",
            "stable_seed_confirm",
        ],
        default="depth_conditioned",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--metric",
        type=str,
        choices=["loss_missing", "loss_total", "mae_missing", "mae_total"],
        default="mae_missing",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sweep-name", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="workdir")
    parser.add_argument("--checkpoint-every", type=int, default=2500)

    parser.add_argument("--generator-type", type=str, default="latent_factor")
    parser.add_argument("--missingness-type", type=str, default="mcar")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-outputs", type=int, default=5000)
    parser.add_argument("--rope-base", type=float, default=10000.0)

    parser.add_argument("--num-rows-low", type=int, default=10)
    parser.add_argument("--num-rows-high", type=int, default=50)
    parser.add_argument("--num-cols-low", type=int, default=5)
    parser.add_argument("--num-cols-high", type=int, default=50)
    parser.add_argument("--p-missing", type=float, default=0.4)
    parser.add_argument("--apply-feature-warping-prob", type=float, default=0.0)
    parser.add_argument("--apply-quantization-prob", type=float, default=0.0)
    parser.add_argument("--latent-rank-low", type=int, default=1)
    parser.add_argument("--latent-rank-high", type=int, default=11)
    parser.add_argument("--latent-spike-p", type=float, default=0.3)
    parser.add_argument("--latent-slab-sigma", type=float, default=2.0)

    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-dir", type=str, default="./wandb")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    device = _choose_device(args.device)
    if args.search_space == "depth_conditioned":
        search_space = None
    elif args.search_space == "regularization_focused":
        search_space = REGULARIZATION_FOCUSED_SPACE
    elif args.search_space == "depth_expansion":
        search_space = DEPTH_EXPANSION_SPACE
    elif args.search_space == "round2_targeted":
        search_space = ROUND2_TARGETED_TRIALS
    elif args.search_space == "shallow_refine":
        search_space = SHALLOW_REFINE_TRIALS
    elif args.search_space == "deep_refine":
        search_space = DEEP_REFINE_TRIALS
    elif args.search_space == "stable_seed_confirm":
        search_space = STABLE_SEED_CONFIRM_TRIALS
    else:
        search_space = ANCHOR_STABLE_TRIALS

    if args.sweep_name is None:
        args.sweep_name = f"tabimpute-stable-sweep-{int(time.time())}"

    output_dir = Path(args.results_dir) / args.sweep_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    if args.search_space == "depth_conditioned":
        trials = _build_depth_conditioned_trials(
            seed=args.seed,
            max_trials=args.max_trials,
        )
    else:
        trials = _build_trials(
            seed=args.seed,
            max_trials=args.max_trials,
            search_space=search_space,
        )
    if len(trials) == 0:
        raise RuntimeError("No valid trial configurations were sampled.")

    _ensure_fresh_run_paths(args, output_dir, len(trials))

    print(f"Using device: {device}")
    print(f"Search space: {args.search_space}")
    print(f"Planned trials: {len(trials)}")
    print(f"Results file: {results_path}")
    print(f"Target metric (lower is better): {args.metric}")

    if args.dry_run:
        for i, trial_cfg in enumerate(trials, start=1):
            print(f"Trial {i:03d}: {trial_cfg}")
        return

    borders_path = importlib.resources.files("tabimpute") / "data" / "borders.pt"
    with importlib.resources.as_file(borders_path) as path:
        borders = torch.load(path)

    all_results = []
    for trial_id, trial_cfg in enumerate(trials, start=1):
        print(f"\nStarting trial {trial_id}/{len(trials)}: {trial_cfg}")
        result = _run_trial(
            args=args,
            trial_id=trial_id,
            trial_cfg=trial_cfg,
            device=device,
            borders=borders,
        )
        all_results.append(result)
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        print(
            f"Finished trial {trial_id}: {args.metric}="
            f"{result['metrics'].get(args.metric, float('nan')):.6f}"
        )

    ranked = sorted(
        all_results,
        key=lambda r: r["metrics"].get(args.metric, float("inf")),
    )
    top_k = ranked[: max(1, args.top_k)]

    summary = {
        "sweep_name": args.sweep_name,
        "device": device,
        "metric": args.metric,
        "num_trials": len(all_results),
        "best": ranked[0] if ranked else None,
        "top_k": top_k,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTop trials:")
    for item in top_k:
        print(
            f"trial={item['trial_id']:03d} "
            f"{args.metric}={item['metrics'].get(args.metric, float('nan')):.6f} "
            f"params={item['num_params']:,}"
        )
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
