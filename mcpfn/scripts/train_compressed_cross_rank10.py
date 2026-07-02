#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.resources
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_compressed_cross import (
    ColumnCompressionModel,
    CompressedCrossTabImputeModel,
    RowCompressionModel,
    build_compressor_config_from_predictor,
    make_chunk_spans,
)
from tabimpute.model.model_compressed_cross_rabel import (
    ColumnCompressionModelRaBEL,
    CompressedCrossTabImputeModelRaBEL,
    RowCompressionModelRaBEL,
    TabImputeModelRaBEL,
)
from tabimpute.model.model_compressed_cross_pure_rabel import (
    ColumnCompressionModelPureRaBEL,
    CompressedCrossTabImputeModelPureRaBEL,
    RowCompressionModelPureRaBEL,
)
from tabimpute.model.model_new_stable import TabImputeModel as StableTabImputeModel
from tabimpute.model.model_new_stable_memory_efficient import (
    TabImputeModel as MemoryEfficientTabImputeModel,
)
from tabimpute.model.model_new_stable_rabel import TabImputeModelPureRaBEL
from tabimpute.prior.rec_sys_spectrum_exposure import RecSysSpectrumExposurePrior
from tabimpute.prior.spectrum_training_set_generation import SpectrumMissingnessPrior
from tabimpute.prior.training_set_generation import (
    MissingnessPrior,
    RecSysBiasResidualExposedHoldoutPrior,
)
from tabimpute.train.callbacks import WandbLoggerCallback
from tabimpute.train_new_sweep import _build_scheduler


SPECTRUM_DECAY_PRIOR_DEFAULTS = {
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


REC_SYS_SPECTRUM_EXPOSURE_PRIOR_DEFAULTS = {
    "recsys_density_low": 0.03,
    "recsys_density_high": 0.06,
    "recsys_spectrum_rank_low": 80,
    "recsys_spectrum_rank_high": 128,
    "recsys_spectrum_alpha_low": 0.30,
    "recsys_spectrum_alpha_high": 0.55,
    "recsys_spectrum_head_weight_low": 3.0,
    "recsys_spectrum_head_weight_high": 6.0,
    "recsys_spectrum_head_exp_scale_low": 2.5,
    "recsys_spectrum_head_exp_scale_high": 5.0,
    "recsys_spectrum_head_exp_power_low": 2.0,
    "recsys_spectrum_head_exp_power_high": 3.0,
    "recsys_spectrum_tail_amplitude_low": 0.20,
    "recsys_spectrum_tail_amplitude_high": 0.50,
    "recsys_spectrum_taper_strength_low": 1.0,
    "recsys_spectrum_taper_strength_high": 2.0,
    "recsys_spectrum_taper_power_low": 3.0,
    "recsys_spectrum_taper_power_high": 5.0,
    "recsys_residual_scale_low": 1.0,
    "recsys_residual_scale_high": 1.0,
    "recsys_user_bias_std_low": 0.30,
    "recsys_user_bias_std_high": 0.30,
    "recsys_item_bias_std_low": 0.50,
    "recsys_item_bias_std_high": 0.50,
    "recsys_noise_std_low": 0.10,
    "recsys_noise_std_high": 0.30,
    "recsys_user_activity_sigma_low": 1.0,
    "recsys_user_activity_sigma_high": 1.5,
    "recsys_item_popularity_sigma_low": 1.5,
    "recsys_item_popularity_sigma_high": 2.5,
    "recsys_affinity_strength_low": 0.25,
    "recsys_affinity_strength_high": 0.75,
    "recsys_cluster_bonus_std_low": 0.50,
    "recsys_cluster_bonus_std_high": 0.50,
    "recsys_n_user_clusters": 8,
    "recsys_n_item_clusters": 12,
    "recsys_rating_quantization": False,
}


REC_SYS_BIAS_RESIDUAL_PRIOR_DEFAULTS = {
    "recsys_bias_global_mean_low": 3.3,
    "recsys_bias_global_mean_high": 3.7,
    "recsys_bias_user_std_low": 0.25,
    "recsys_bias_user_std_high": 0.60,
    "recsys_bias_item_std_low": 0.35,
    "recsys_bias_item_std_high": 0.80,
    "recsys_bias_user_distribution": "normal",
    "recsys_bias_item_distribution": "normal",
    "recsys_bias_residual_rank_low": 8,
    "recsys_bias_residual_rank_high": 40,
    "recsys_bias_residual_scale_low": 0.15,
    "recsys_bias_residual_scale_high": 0.60,
    "recsys_bias_residual_alpha_low": 0.4,
    "recsys_bias_residual_alpha_high": 0.9,
    "recsys_bias_noise_std_low": 0.05,
    "recsys_bias_noise_std_high": 0.25,
    "recsys_bias_rating_min": 1.0,
    "recsys_bias_rating_max": 5.0,
    "recsys_bias_rating_clip_probability": 1.0,
    "recsys_bias_rating_quantization_probability": 0.8,
    "recsys_exposure_density_low": 0.03,
    "recsys_exposure_density_high": 0.06,
    "recsys_exposure_user_activity_sigma_low": 1.0,
    "recsys_exposure_user_activity_sigma_high": 1.5,
    "recsys_exposure_item_popularity_sigma_low": 1.5,
    "recsys_exposure_item_popularity_sigma_high": 2.5,
    "recsys_exposure_preference_weight_low": 0.20,
    "recsys_exposure_preference_weight_high": 0.75,
    "recsys_exposure_cluster_bonus_std": 0.5,
    "recsys_exposure_force_row_col_coverage": True,
}


REC_SYS_BIAS_RESIDUAL_EXPOSED_HOLDOUT_DEFAULTS = {
    **REC_SYS_BIAS_RESIDUAL_PRIOR_DEFAULTS,
    "recsys_holdout_target_fraction_low": 0.10,
    "recsys_holdout_target_fraction_high": 0.30,
    "recsys_holdout_bias_ridge_reg": 2.0,
    "recsys_holdout_bias_ridge_iters": 30,
}


RECSYS_MOVIELENS_PROXY_20K_CURRICULUM = [
    {
        "phase": 1,
        "end_fraction": 0.25,
        "num_rows_low": 150,
        "num_rows_high": 300,
        "num_cols_low": 300,
        "num_cols_high": 600,
        "recsys_exposure_density_low": 0.08,
        "recsys_exposure_density_high": 0.12,
    },
    {
        "phase": 2,
        "end_fraction": 0.50,
        "num_rows_low": 250,
        "num_rows_high": 500,
        "num_cols_low": 500,
        "num_cols_high": 1000,
        "recsys_exposure_density_low": 0.06,
        "recsys_exposure_density_high": 0.09,
    },
    {
        "phase": 3,
        "end_fraction": 0.75,
        "num_rows_low": 500,
        "num_rows_high": 800,
        "num_cols_low": 900,
        "num_cols_high": 1400,
        "recsys_exposure_density_low": 0.045,
        "recsys_exposure_density_high": 0.07,
    },
    {
        "phase": 4,
        "end_fraction": 1.00,
        "num_rows_low": 800,
        "num_rows_high": 1000,
        "num_cols_low": 1400,
        "num_cols_high": 1800,
        "recsys_exposure_density_low": 0.04,
        "recsys_exposure_density_high": 0.06,
    },
]


def load_run_config(checkpoint_path: Path) -> dict:
    with (checkpoint_path.parent / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def build_compressed_scheduler(
    *,
    optimizer,
    scheduler_name: str,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    one_cycle_pct_start: float,
    stable_steps: int | None,
    decay_steps: int | None,
):
    if scheduler_name.lower() != "warmup_stable_cosine":
        return _build_scheduler(
            optimizer=optimizer,
            scheduler_name=scheduler_name,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
            one_cycle_pct_start=one_cycle_pct_start,
        )

    warmup_steps = max(0, warmup_steps)
    if decay_steps is None:
        decay_steps = max(1, total_steps - warmup_steps - max(0, stable_steps or 0))
    decay_steps = max(1, decay_steps)
    if stable_steps is None:
        stable_steps = max(0, total_steps - warmup_steps - decay_steps)
    stable_steps = max(0, stable_steps)
    total_schedule_steps = warmup_steps + stable_steps + decay_steps

    def lr_lambda(current_step: int) -> float:
        step = current_step + 1
        if warmup_steps > 0 and step <= warmup_steps:
            return step / warmup_steps
        if step <= warmup_steps + stable_steps:
            return 1.0
        decay_progress = (
            step - warmup_steps - stable_steps
        ) / max(1, decay_steps)
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    if total_schedule_steps != total_steps:
        print(
            "warmup_stable_cosine schedule length "
            f"{total_schedule_steps} != total_steps {total_steps}; "
            "extra steps will stay at min_lr_ratio after decay.",
            flush=True,
        )
    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)


def build_checkpoint_state(
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    metrics: dict[str, float] | None = None,
    best_step: int | None = None,
    best_loss_missing: float | None = None,
    include_optimizer: bool = True,
) -> dict[str, object]:
    state = {
        "epoch": step,
        "model": model.state_dict(),
    }
    if include_optimizer:
        state["optimizer"] = optimizer.state_dict()
        state["scheduler"] = scheduler.state_dict() if scheduler is not None else None
    if metrics is not None:
        state["metrics"] = dict(metrics)
    if best_step is not None:
        state["best_step"] = best_step
    if best_loss_missing is not None:
        state["best_loss_missing"] = best_loss_missing
    return state


def save_best_checkpoint(
    *,
    work_dir: Path,
    step: int,
    loss_missing: float,
    metrics: dict[str, float],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    include_optimizer: bool = True,
) -> None:
    checkpoint_path = work_dir / "best_checkpoint.pth"
    torch.save(
        build_checkpoint_state(
            step=step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            best_step=step,
            best_loss_missing=loss_missing,
            include_optimizer=include_optimizer,
        ),
        checkpoint_path,
    )
    best_metrics = {
        "best_step": step,
        "best_loss_missing": loss_missing,
        "best_checkpoint": str(checkpoint_path),
        "metrics": dict(metrics),
    }
    with (work_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2, sort_keys=True)


def build_predictor(model_cfg: dict, model_family: str):
    common_kwargs = {
        "embedding_size": model_cfg["embedding_size"],
        "num_attention_heads": model_cfg["num_attention_heads"],
        "mlp_hidden_size": model_cfg["mlp_hidden_size"],
        "num_layers": model_cfg["num_layers"],
        "num_outputs": model_cfg.get("num_outputs", 5000),
        "num_cls": model_cfg["num_cls"],
        "use_rope": model_cfg.get("use_rope", True),
        "rope_base": model_cfg.get("rope_base", 10000.0),
        "rope_fraction": model_cfg.get("rope_fraction", 1.0),
        "use_absolute_positional_embeddings": model_cfg.get(
            "use_absolute_positional_embeddings",
            False,
        ),
        "positional_damping_factor": model_cfg.get(
            "positional_damping_factor",
            0.1,
        ),
        "attention_dropout": model_cfg.get("attention_dropout", 0.0),
        "ffn_dropout": model_cfg.get("ffn_dropout", 0.0),
        "drop_path_rate": model_cfg.get("drop_path_rate", 0.0),
        "residual_scale_init": model_cfg.get("residual_scale_init", 1.0),
        "rms_norm_eps": model_cfg.get("rms_norm_eps", 1e-6),
        "embedding_dropout": model_cfg.get("embedding_dropout", 0.0),
    }
    if model_family == "memory_efficient":
        return MemoryEfficientTabImputeModel(
            **common_kwargs,
            use_activation_checkpointing=model_cfg.get(
                "use_activation_checkpointing",
                True,
            ),
        )
    if model_family == "stable":
        return StableTabImputeModel(**common_kwargs)
    if model_family == "rabel":
        return TabImputeModelRaBEL(
            **common_kwargs,
            rabel_num_kernels=model_cfg.get("rabel_num_kernels", 64),
            rabel_sigma=model_cfg.get("rabel_sigma", 1.0),
            rabel_exponent_digits=model_cfg.get("rabel_exponent_digits", 1),
            rabel_exponent_embedding_size=model_cfg.get(
                "rabel_exponent_embedding_size",
                32,
            ),
            rabel_initial_scale=model_cfg.get("rabel_initial_scale", 0.05),
        )
    if model_family == "pure_rabel":
        return TabImputeModelPureRaBEL(
            **common_kwargs,
            use_activation_checkpointing=model_cfg.get(
                "use_activation_checkpointing",
                True,
            ),
            checkpoint_use_reentrant=model_cfg.get(
                "checkpoint_use_reentrant",
                False,
            ),
            rabel_num_kernels=model_cfg.get("rabel_num_kernels", 64),
            rabel_sigma=model_cfg.get("rabel_sigma", 1.0),
            rabel_exponent_digits=model_cfg.get("rabel_exponent_digits", 1),
            rabel_exponent_embedding_size=model_cfg.get(
                "rabel_exponent_embedding_size",
                32,
            ),
        )
    raise ValueError(f"Unsupported model_family: {model_family}")


def build_compressed_model(
    *,
    predictor_config: dict,
    predictor_checkpoint_path: Path | None,
    compressor_config: dict,
    k_row: int,
    k_col: int,
    model_family: str,
    device: str,
    init_compressor_embedders_from_predictor: bool = True,
    init_compressor_transformers_from_predictor: bool = False,
):
    predictor = build_predictor(predictor_config, model_family=model_family)
    if predictor_checkpoint_path is not None:
        ckpt = torch.load(
            predictor_checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        predictor.load_state_dict(ckpt["model"], strict=True)

    if model_family == "rabel":
        row_compressor_class = RowCompressionModelRaBEL
        column_compressor_class = ColumnCompressionModelRaBEL
        compressed_model_class = CompressedCrossTabImputeModelRaBEL
    elif model_family == "pure_rabel":
        row_compressor_class = RowCompressionModelPureRaBEL
        column_compressor_class = ColumnCompressionModelPureRaBEL
        compressed_model_class = CompressedCrossTabImputeModelPureRaBEL
    else:
        row_compressor_class = RowCompressionModel
        column_compressor_class = ColumnCompressionModel
        compressed_model_class = CompressedCrossTabImputeModel
    rabel_compressor_kwargs = (
        {"rabel_config": compressor_config}
        if model_family in {"rabel", "pure_rabel"}
        else {}
    )
    row_compressor = row_compressor_class(
        embedding_size=compressor_config["embedding_size"],
        num_attention_heads=compressor_config["num_attention_heads"],
        mlp_hidden_size=compressor_config["mlp_hidden_size"],
        num_layers=compressor_config["num_layers"],
        k_row=k_row,
        num_cls=compressor_config.get("num_cls", predictor_config.get("num_cls", 12)),
        attention_dropout=compressor_config.get("attention_dropout", 0.0),
        ffn_dropout=compressor_config.get("ffn_dropout", 0.0),
        drop_path_rate=compressor_config.get("drop_path_rate", 0.0),
        residual_scale_init=compressor_config.get("residual_scale_init", 1.0),
        rms_norm_eps=compressor_config.get("rms_norm_eps", 1e-6),
        use_rope=compressor_config.get("use_rope", True),
        rope_base=compressor_config.get("rope_base", 10000.0),
        rope_fraction=compressor_config.get("rope_fraction", 1.0),
        use_activation_checkpointing=compressor_config.get(
            "use_activation_checkpointing",
            True,
        ),
        **rabel_compressor_kwargs,
    )
    column_compressor = column_compressor_class(
        embedding_size=compressor_config["embedding_size"],
        num_attention_heads=compressor_config["num_attention_heads"],
        mlp_hidden_size=compressor_config["mlp_hidden_size"],
        num_layers=compressor_config["num_layers"],
        k_col=k_col,
        num_cls=compressor_config.get("num_cls", predictor_config.get("num_cls", 12)),
        attention_dropout=compressor_config.get("attention_dropout", 0.0),
        ffn_dropout=compressor_config.get("ffn_dropout", 0.0),
        drop_path_rate=compressor_config.get("drop_path_rate", 0.0),
        residual_scale_init=compressor_config.get("residual_scale_init", 1.0),
        rms_norm_eps=compressor_config.get("rms_norm_eps", 1e-6),
        use_rope=compressor_config.get("use_rope", True),
        rope_base=compressor_config.get("rope_base", 10000.0),
        rope_fraction=compressor_config.get("rope_fraction", 1.0),
        use_activation_checkpointing=compressor_config.get(
            "use_activation_checkpointing",
            True,
        ),
        **rabel_compressor_kwargs,
    )
    model = compressed_model_class(
        predictor=predictor,
        row_compressor=row_compressor,
        column_compressor=column_compressor,
        k_row=k_row,
        k_col=k_col,
    )
    if init_compressor_embedders_from_predictor and predictor_checkpoint_path is not None:
        model.initialize_compressor_embedders_from_predictor()
    if (
        init_compressor_transformers_from_predictor
        and predictor_checkpoint_path is not None
    ):
        model.initialize_compressor_transformers_from_predictor()
    return model


def _resize_compressor_query_tensor(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    query_axis: int,
) -> torch.Tensor:
    if source.ndim != 4 or target.ndim != 4:
        raise ValueError("Compressor query tensors must be four-dimensional.")
    for axis in range(4):
        if axis != query_axis and source.shape[axis] != target.shape[axis]:
            raise ValueError(
                "Only the compressor query count may change when resizing: "
                f"{tuple(source.shape)} -> {tuple(target.shape)}"
            )

    source = source.to(device=target.device, dtype=target.dtype)
    resized = target.clone()
    copied = min(source.shape[query_axis], target.shape[query_axis])
    source_slice = [slice(None)] * 4
    target_slice = [slice(None)] * 4
    source_slice[query_axis] = slice(0, copied)
    target_slice[query_axis] = slice(0, copied)
    resized[tuple(target_slice)] = source[tuple(source_slice)]

    if target.shape[query_axis] > source.shape[query_axis]:
        extra_slice = [slice(None)] * 4
        extra_slice[query_axis] = slice(source.shape[query_axis], None)
        mean_query = source.mean(dim=query_axis, keepdim=True)
        expand_shape = list(target.shape)
        expand_shape[query_axis] = target.shape[query_axis] - source.shape[query_axis]
        resized[tuple(extra_slice)] = mean_query.expand(*expand_shape)
    return resized


def load_compressed_model_checkpoint(
    model: CompressedCrossTabImputeModel,
    checkpoint_path: Path,
    *,
    device: str,
    allow_query_resize: bool = False,
    allow_encoder_extension: bool = False,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    state_dict = checkpoint["model"]
    if allow_query_resize:
        target_state = model.state_dict()
        resize_specs = {
            "row_compressor.query_rows": 1,
            "column_compressor.query_cols": 2,
        }
        for key, query_axis in resize_specs.items():
            if key not in state_dict or key not in target_state:
                continue
            if state_dict[key].shape != target_state[key].shape:
                print(
                    f"Resizing {key}: {tuple(state_dict[key].shape)} -> "
                    f"{tuple(target_state[key].shape)}",
                    flush=True,
                )
                state_dict[key] = _resize_compressor_query_tensor(
                    state_dict[key],
                    target_state[key],
                    query_axis=query_axis,
                )
    if not allow_encoder_extension:
        model.load_state_dict(state_dict, strict=True)
        return

    target_state = model.state_dict()
    unexpected_source_keys = sorted(set(state_dict) - set(target_state))
    if unexpected_source_keys:
        raise RuntimeError(
            "Checkpoint contains keys that are not present in the RaBEL model: "
            f"{unexpected_source_keys}"
        )
    incompatible_shapes = {
        key: (tuple(value.shape), tuple(target_state[key].shape))
        for key, value in state_dict.items()
        if key in target_state and value.shape != target_state[key].shape
    }
    if incompatible_shapes:
        raise ValueError(
            "Checkpoint has non-RaBEL shape mismatches after query resizing: "
            f"{incompatible_shapes}"
        )
    filtered_state = {
        key: value
        for key, value in state_dict.items()
        if key in target_state
    }
    incompatible = model.load_state_dict(filtered_state, strict=False)
    allowed_missing_fragments = (
        ".rabel_encoder.",
        ".rabel_scale",
    )
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not any(fragment in key for fragment in allowed_missing_fragments)
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected incompatibilities while extending checkpoint with "
            f"RaBEL: missing={invalid_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    if hasattr(model, "initialize_compressor_rabel_from_predictor"):
        model.initialize_compressor_rabel_from_predictor()
    print(
        "Loaded existing compressed weights and initialized "
        f"{len(incompatible.missing_keys)} new RaBEL tensors.",
        flush=True,
    )


def build_split_optimizer(
    model: CompressedCrossTabImputeModel,
    *,
    optimizer_name: str,
    predictor_lr: float,
    compressor_lr: float,
    encoder_lr: float | None = None,
    weight_decay: float,
    optimizer_betas: tuple[float, float],
    optimizer_eps: float,
):
    optimizer_name = optimizer_name.lower()
    param_groups = []
    encoder_param_ids = set()
    encoder_params = []
    if encoder_lr is not None:
        for name, param in model.named_parameters():
            if (
                param.requires_grad
                and (
                    ".rabel_encoder." in name
                    or name.endswith(".rabel_scale")
                )
            ):
                encoder_param_ids.add(id(param))
                encoder_params.append(param)
    predictor_params = [
        param
        for param in model.predictor.parameters()
        if param.requires_grad and id(param) not in encoder_param_ids
    ]
    if predictor_params:
        param_groups.append(
            {
                "params": predictor_params,
                "lr": predictor_lr,
                "weight_decay": weight_decay,
                "name": "predictor",
            }
        )
    compressor_params = [
        param
        for param in (
            list(model.row_compressor.parameters())
            + list(model.column_compressor.parameters())
            + [model.off_axis_token]
            + list(model.region_type_embedding.parameters())
        )
        if param.requires_grad and id(param) not in encoder_param_ids
    ]
    if compressor_params:
        param_groups.append(
            {
                "params": compressor_params,
                "lr": compressor_lr,
                "weight_decay": weight_decay,
                "name": "compressors",
            }
        )
    if encoder_params:
        param_groups.append(
            {
                "params": encoder_params,
                "lr": encoder_lr,
                "weight_decay": weight_decay,
                "name": "rabel_encoder",
            }
        )
    if optimizer_name == "adamw":
        use_fused = next(model.parameters()).is_cuda
        try:
            return torch.optim.AdamW(
                param_groups,
                betas=optimizer_betas,
                eps=optimizer_eps,
                fused=use_fused,
            )
        except (TypeError, RuntimeError):
            return torch.optim.AdamW(
                param_groups,
                betas=optimizer_betas,
                eps=optimizer_eps,
            )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            param_groups,
            betas=optimizer_betas,
            eps=optimizer_eps,
        )
    raise ValueError(
        f"Unsupported split optimizer '{optimizer_name}'. Use adamw or adam."
    )


def train_one_step(
    *,
    model: CompressedCrossTabImputeModel,
    criterion: FullSupportBarDistribution,
    batch_x: torch.Tensor,
    batch_target: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    chunk_rows: int,
    chunk_cols: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    row_spans = make_chunk_spans(batch_x.shape[1], chunk_rows)
    col_spans = make_chunk_spans(batch_x.shape[2], chunk_cols)
    targets_with_loss = 0
    loss_sum = 0.0
    mae_sum = 0.0
    rmse_sum = 0.0

    scale = 1.0 / (len(row_spans) * len(col_spans))
    for row_idx, row_span in enumerate(row_spans):
        for col_idx, col_span in enumerate(col_spans):
            logits = model.forward_target_chunk(
                batch_x,
                chunk_rows=chunk_rows,
                chunk_cols=chunk_cols,
                target_row_idx=row_idx,
                target_col_idx=col_idx,
            )
            target = batch_target[
                :,
                row_span.start : row_span.end,
                col_span.start : col_span.end,
            ]
            if loss_mask is None:
                chunk_loss_mask = torch.isnan(
                    batch_x[
                        :,
                        row_span.start : row_span.end,
                        col_span.start : col_span.end,
                    ]
                )
            else:
                chunk_loss_mask = loss_mask[
                    :,
                    row_span.start : row_span.end,
                    col_span.start : col_span.end,
                ]
            if not chunk_loss_mask.any():
                continue
            losses = criterion(logits[chunk_loss_mask], target[chunk_loss_mask])
            loss = losses.mean()
            (loss * scale).backward()
            with torch.no_grad():
                medians = criterion.median(logits[chunk_loss_mask])
                errors = medians - target[chunk_loss_mask]
                mae = errors.abs().mean()
                rmse = torch.sqrt(torch.mean(errors.square()))
            loss_sum += float(loss.detach().cpu())
            mae_sum += float(mae.detach().cpu())
            rmse_sum += float(rmse.detach().cpu())
            targets_with_loss += 1

    if targets_with_loss == 0:
        zero = batch_target.new_zeros(())
        metrics = {"loss_missing": 0.0, "mae_missing": 0.0, "rmse_missing": 0.0}
        if loss_mask is not None:
            metrics["exposed_holdout_rmse"] = 0.0
        return zero, metrics
    metrics = {
        "loss_missing": loss_sum / targets_with_loss,
        "mae_missing": mae_sum / targets_with_loss,
        "rmse_missing": rmse_sum / targets_with_loss,
    }
    if loss_mask is not None:
        metrics["exposed_holdout_rmse"] = metrics["rmse_missing"]
    return batch_target.new_tensor(loss_sum / targets_with_loss), metrics


def train_one_random_chunk_step(
    *,
    model: CompressedCrossTabImputeModel,
    criterion: FullSupportBarDistribution,
    batch_x: torch.Tensor,
    batch_target: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    chunk_rows: int,
    chunk_cols: int,
    num_target_chunks: int = 1,
) -> tuple[torch.Tensor, dict[str, float]]:
    if num_target_chunks <= 0:
        raise ValueError("num_target_chunks must be positive.")
    row_spans = make_chunk_spans(batch_x.shape[1], chunk_rows)
    col_spans = make_chunk_spans(batch_x.shape[2], chunk_cols)
    num_targets = len(row_spans) * len(col_spans)

    valid_targets = []
    for target_idx in range(num_targets):
        row_idx = target_idx // len(col_spans)
        col_idx = target_idx % len(col_spans)
        row_span = row_spans[row_idx]
        col_span = col_spans[col_idx]
        if loss_mask is None:
            chunk_loss_mask = torch.isnan(
                batch_x[
                    :,
                    row_span.start : row_span.end,
                    col_span.start : col_span.end,
                ]
            )
        else:
            chunk_loss_mask = loss_mask[
                :,
                row_span.start : row_span.end,
                col_span.start : col_span.end,
            ]
        if chunk_loss_mask.any():
            valid_targets.append(
                (row_idx, col_idx, row_span, col_span, chunk_loss_mask)
            )
    if not valid_targets:
        zero = batch_target.new_zeros(())
        metrics = {
            "loss_missing": 0.0,
            "mae_missing": 0.0,
            "rmse_missing": 0.0,
            "target_row_idx": -1.0,
            "target_col_idx": -1.0,
        }
        if loss_mask is not None:
            metrics["exposed_holdout_rmse"] = 0.0
        return zero, metrics

    selected_count = min(num_target_chunks, len(valid_targets))
    order = torch.randperm(len(valid_targets), device=batch_x.device)[
        :selected_count
    ].tolist()
    loss_sum = 0.0
    mae_sum = 0.0
    rmse_sum = 0.0
    first_row_idx = -1
    first_col_idx = -1
    for selected_idx in order:
        row_idx, col_idx, row_span, col_span, chunk_loss_mask = valid_targets[
            selected_idx
        ]
        if first_row_idx < 0:
            first_row_idx = row_idx
            first_col_idx = col_idx
        logits = model.forward_target_chunk(
            batch_x,
            chunk_rows=chunk_rows,
            chunk_cols=chunk_cols,
            target_row_idx=row_idx,
            target_col_idx=col_idx,
        )
        target = batch_target[
            :,
            row_span.start : row_span.end,
            col_span.start : col_span.end,
        ]
        losses = criterion(logits[chunk_loss_mask], target[chunk_loss_mask])
        loss = losses.mean()
        (loss / selected_count).backward()
        with torch.no_grad():
            medians = criterion.median(logits[chunk_loss_mask])
            errors = medians - target[chunk_loss_mask]
            mae = errors.abs().mean()
            rmse = torch.sqrt(torch.mean(errors.square()))
        loss_sum += float(loss.detach().cpu())
        mae_sum += float(mae.detach().cpu())
        rmse_sum += float(rmse.detach().cpu())

    metrics = {
        "loss_missing": loss_sum / selected_count,
        "mae_missing": mae_sum / selected_count,
        "rmse_missing": rmse_sum / selected_count,
        "target_row_idx": float(first_row_idx),
        "target_col_idx": float(first_col_idx),
        "targets_sampled": float(selected_count),
    }
    if loss_mask is not None:
        metrics["exposed_holdout_rmse"] = metrics["rmse_missing"]
    return batch_target.new_tensor(loss_sum / selected_count), metrics


def sample_batch_chunk_sizes(
    *,
    num_rows: int,
    num_cols: int,
    fraction_min: float,
    fraction_max: float,
    min_chunk_rows: int = 1,
    min_chunk_cols: int = 1,
) -> tuple[int, int, float, float]:
    if not (0.0 < fraction_min <= fraction_max <= 1.0):
        raise ValueError("Chunk fractions must satisfy 0 < min <= max <= 1.")
    if min_chunk_rows <= 0 or min_chunk_cols <= 0:
        raise ValueError("Minimum chunk sizes must be positive.")
    row_fraction = float(
        torch.empty((), device="cpu").uniform_(fraction_min, fraction_max).item()
    )
    col_fraction = float(
        torch.empty((), device="cpu").uniform_(fraction_min, fraction_max).item()
    )

    def _sample_axis(size: int, fraction: float, min_chunk_size: int) -> int:
        requested_chunk = max(1, min(size, int(round(size * fraction))))
        requested_num_chunks = max(
            1,
            (size + requested_chunk - 1) // requested_chunk,
        )
        max_num_chunks = max(1, size // min(min_chunk_size, size))
        num_chunks = min(requested_num_chunks, max_num_chunks)
        return max(1, (size + num_chunks - 1) // num_chunks)

    chunk_rows = _sample_axis(num_rows, row_fraction, min_chunk_rows)
    chunk_cols = _sample_axis(num_cols, col_fraction, min_chunk_cols)
    return chunk_rows, chunk_cols, row_fraction, col_fraction


def apply_recsys_curriculum(
    *,
    prior,
    curriculum_name: str,
    step: int,
    total_steps: int,
) -> dict[str, float]:
    if curriculum_name == "none":
        return {}
    if curriculum_name != "movielens_proxy_20k":
        raise ValueError(f"Unknown recsys curriculum: {curriculum_name}")

    progress = step / max(1, total_steps)
    phase_cfg = RECSYS_MOVIELENS_PROXY_20K_CURRICULUM[-1]
    for candidate in RECSYS_MOVIELENS_PROXY_20K_CURRICULUM:
        if progress <= candidate["end_fraction"]:
            phase_cfg = candidate
            break

    updates = {
        key: value
        for key, value in phase_cfg.items()
        if key not in {"phase", "end_fraction"}
    }
    prior.config.update(updates)
    return {
        "curriculum_phase": float(phase_cfg["phase"]),
        "curriculum_progress": float(progress),
        "curriculum_rows_low": float(updates["num_rows_low"]),
        "curriculum_rows_high": float(updates["num_rows_high"]),
        "curriculum_cols_low": float(updates["num_cols_low"]),
        "curriculum_cols_high": float(updates["num_cols_high"]),
        "curriculum_exposure_density_low": float(
            updates["recsys_exposure_density_low"]
        ),
        "curriculum_exposure_density_high": float(
            updates["recsys_exposure_density_high"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictor-checkpoint",
        type=Path,
        default=Path(
            "/root/tabular/mcpfn/src/tabimpute/workdir/"
            "tabimpute-new-stable_v2-rank_10_10-100x100-from_50x50_step30000/"
            "checkpoint_45000.pth"
        ),
    )
    parser.add_argument(
        "--config-checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint directory/config source to use for model, optimizer, and "
            "prior hyperparameters. Defaults to --predictor-checkpoint. Useful "
            "for scratch runs that should reuse a known architecture config "
            "without loading predictor weights."
        ),
    )
    parser.add_argument(
        "--train-from-scratch",
        action="store_true",
        help=(
            "Do not load predictor weights and do not initialize compressor "
            "weights from the predictor. The checkpoint still supplies config."
        ),
    )
    parser.add_argument(
        "--compressed-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional full CompressedCrossTabImputeModel checkpoint to continue "
            "from after constructing the model from the predictor checkpoint."
        ),
    )
    parser.add_argument(
        "--allow-compressor-query-resize",
        action="store_true",
        help=(
            "Resize learned row/column query tensors when loading a compressed "
            "checkpoint with a different K. All other parameter shapes must match."
        ),
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--matrix-rows", type=int, default=200)
    parser.add_argument("--matrix-cols", type=int, default=200)
    parser.add_argument("--matrix-rows-min", type=int, default=None)
    parser.add_argument("--matrix-rows-max", type=int, default=None)
    parser.add_argument("--matrix-cols-min", type=int, default=None)
    parser.add_argument("--matrix-cols-max", type=int, default=None)
    parser.add_argument(
        "--randomize-matrix-size",
        action="store_true",
        help=(
            "Sample one rows/cols shape per batch from the provided min/max "
            "ranges. Defaults to fixed --matrix-rows/--matrix-cols when disabled."
        ),
    )
    parser.add_argument("--chunk-rows", type=int, default=50)
    parser.add_argument("--chunk-cols", type=int, default=50)
    parser.add_argument(
        "--randomize-chunk-fractions",
        action="store_true",
        help=(
            "Sample chunk rows/cols once per batch as independent fractions "
            "of that batch's matrix rows/cols."
        ),
    )
    parser.add_argument("--chunk-fraction-min", type=float, default=0.1)
    parser.add_argument("--chunk-fraction-max", type=float, default=0.5)
    parser.add_argument("--min-chunk-rows", type=int, default=1)
    parser.add_argument("--min-chunk-cols", type=int, default=1)
    parser.add_argument(
        "--target-chunk-sampling",
        choices=["all", "random"],
        default="all",
        help=(
            "Use all target chunks before each optimizer step, or sample one "
            "random target chunk and step immediately."
        ),
    )
    parser.add_argument(
        "--random-targets-per-step",
        type=int,
        default=1,
        help=(
            "Number of valid random target chunks to average before each "
            "optimizer step when --target-chunk-sampling=random."
        ),
    )
    parser.add_argument(
        "--prior-kind",
        choices=[
            "latent_factor",
            "spectrum_decay",
            "rec_sys_spectrum_exposure",
            "rec_sys_bias_residual_rating",
            "rec_sys_bias_residual_residual",
            "rec_sys_bias_residual_exposed_rating",
            "rec_sys_bias_residual_exposed_known_bias",
            "rec_sys_bias_residual_exposed_fitted_bias",
        ],
        default="latent_factor",
        help="Synthetic matrix prior used for training batches.",
    )
    parser.add_argument("--spectrum-rank-low", type=int, default=None)
    parser.add_argument("--spectrum-rank-high", type=int, default=None)
    parser.add_argument("--spectrum-target-rank", type=int, default=None)
    parser.add_argument("--spectrum-alpha-low", type=float, default=None)
    parser.add_argument("--spectrum-alpha-high", type=float, default=None)
    parser.add_argument("--spectrum-noise-std-low", type=float, default=None)
    parser.add_argument("--spectrum-noise-std-high", type=float, default=None)
    parser.add_argument("--recsys-density-low", type=float, default=None)
    parser.add_argument("--recsys-density-high", type=float, default=None)
    parser.add_argument("--recsys-spectrum-rank-low", type=int, default=None)
    parser.add_argument("--recsys-spectrum-rank-high", type=int, default=None)
    parser.add_argument("--recsys-noise-std-low", type=float, default=None)
    parser.add_argument("--recsys-noise-std-high", type=float, default=None)
    parser.add_argument("--recsys-user-activity-sigma-low", type=float, default=None)
    parser.add_argument("--recsys-user-activity-sigma-high", type=float, default=None)
    parser.add_argument("--recsys-item-popularity-sigma-low", type=float, default=None)
    parser.add_argument("--recsys-item-popularity-sigma-high", type=float, default=None)
    parser.add_argument("--recsys-affinity-strength-low", type=float, default=None)
    parser.add_argument("--recsys-affinity-strength-high", type=float, default=None)
    parser.add_argument("--recsys-cluster-bonus-std", type=float, default=None)
    parser.add_argument("--recsys-rating-quantization", action="store_true")
    parser.add_argument("--recsys-global-mean", type=float, default=None)
    parser.add_argument("--recsys-rating-min", type=float, default=None)
    parser.add_argument("--recsys-rating-max", type=float, default=None)
    parser.add_argument("--recsys-rating-clip-probability", type=float, default=None)
    parser.add_argument("--recsys-rating-quantization-probability", type=float, default=None)
    parser.add_argument(
        "--recsys-exposure-calibration",
        choices=["global", "marginal"],
        default=None,
        help=(
            "Exposure calibration mode. 'global' matches total density only; "
            "'marginal' solves row/column exposure biases to match sampled "
            "heavy-tailed degree targets."
        ),
    )
    parser.add_argument(
        "--recsys-exposure-uses-full-preference",
        action="store_true",
        help="Couple exposure to residual plus user/item bias signals.",
    )
    parser.add_argument(
        "--recsys-user-activity-distribution",
        choices=["normal", "lognormal", "pareto"],
        default=None,
    )
    parser.add_argument(
        "--recsys-item-popularity-distribution",
        choices=["normal", "lognormal", "pareto"],
        default=None,
    )
    parser.add_argument("--recsys-popularity-quality-corr-low", type=float, default=None)
    parser.add_argument("--recsys-popularity-quality-corr-high", type=float, default=None)
    parser.add_argument("--recsys-exposure-item-quality-weight", type=float, default=None)
    parser.add_argument("--recsys-exposure-user-bias-weight", type=float, default=None)
    parser.add_argument("--recsys-preference-temperature", type=float, default=None)
    parser.add_argument("--recsys-marginal-calibration-steps", type=int, default=None)
    parser.add_argument("--recsys-marginal-binary-steps", type=int, default=None)
    parser.add_argument("--recsys-bias-global-mean-low", type=float, default=None)
    parser.add_argument("--recsys-bias-global-mean-high", type=float, default=None)
    parser.add_argument("--recsys-bias-user-std-low", type=float, default=None)
    parser.add_argument("--recsys-bias-user-std-high", type=float, default=None)
    parser.add_argument("--recsys-bias-item-std-low", type=float, default=None)
    parser.add_argument("--recsys-bias-item-std-high", type=float, default=None)
    parser.add_argument(
        "--recsys-bias-user-distribution",
        choices=["normal", "lognormal", "pareto"],
        default=None,
    )
    parser.add_argument(
        "--recsys-bias-item-distribution",
        choices=["normal", "lognormal", "pareto"],
        default=None,
    )
    parser.add_argument("--recsys-bias-residual-rank-low", type=int, default=None)
    parser.add_argument("--recsys-bias-residual-rank-high", type=int, default=None)
    parser.add_argument("--recsys-bias-residual-scale-low", type=float, default=None)
    parser.add_argument("--recsys-bias-residual-scale-high", type=float, default=None)
    parser.add_argument("--recsys-bias-noise-std-low", type=float, default=None)
    parser.add_argument("--recsys-bias-noise-std-high", type=float, default=None)
    parser.add_argument("--recsys-bias-rating-quantization-probability", type=float, default=None)
    parser.add_argument("--recsys-exposure-density-low", type=float, default=None)
    parser.add_argument("--recsys-exposure-density-high", type=float, default=None)
    parser.add_argument("--recsys-exposure-preference-weight-low", type=float, default=None)
    parser.add_argument("--recsys-exposure-preference-weight-high", type=float, default=None)
    parser.add_argument("--recsys-exposure-cluster-bonus-std", type=float, default=None)
    parser.add_argument("--recsys-holdout-target-fraction-low", type=float, default=None)
    parser.add_argument("--recsys-holdout-target-fraction-high", type=float, default=None)
    parser.add_argument("--recsys-holdout-bias-ridge-reg", type=float, default=None)
    parser.add_argument("--recsys-holdout-bias-ridge-iters", type=int, default=None)
    parser.add_argument(
        "--recsys-curriculum",
        choices=["none", "movielens_proxy_20k"],
        default="none",
        help=(
            "Optional rec-sys curriculum. movielens_proxy_20k increases "
            "matrix size and sparsity in four phases."
        ),
    )
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--k-row", type=int, default=4)
    parser.add_argument("--k-col", type=int, default=4)
    parser.add_argument("--compressor-layers", type=int, default=2)
    parser.add_argument("--compressor-mlp-hidden-size", type=int, default=None)
    parser.add_argument("--predictor-lr", type=float, default=None)
    parser.add_argument("--compressor-lr", type=float, default=None)
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=None,
        help=(
            "Optional separate LR for RaBEL encoder parameters. When omitted, "
            "they remain in their predictor/compressor optimizer groups."
        ),
    )
    parser.add_argument("--rabel-num-kernels", type=int, default=64)
    parser.add_argument("--rabel-sigma", type=float, default=1.0)
    parser.add_argument("--rabel-exponent-digits", type=int, default=1)
    parser.add_argument(
        "--rabel-exponent-embedding-size",
        type=int,
        default=32,
    )
    parser.add_argument("--rabel-initial-scale", type=float, default=0.05)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Override optimizer weight decay from the config checkpoint.",
    )
    parser.add_argument("--predictor-lr-multiplier", type=float, default=0.1)
    parser.add_argument(
        "--scheduler-name",
        default=None,
        help="Override scheduler from the config checkpoint, e.g. constant.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=None,
        help="Override warmup ratio from the config checkpoint.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Override warmup steps directly; takes precedence over warmup ratio.",
    )
    parser.add_argument(
        "--stable-steps",
        type=int,
        default=None,
        help="Stable/flat LR steps for warmup_stable_cosine.",
    )
    parser.add_argument(
        "--decay-steps",
        type=int,
        default=None,
        help="Cosine decay steps for warmup_stable_cosine.",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=None,
        help="Override minimum LR ratio from the config checkpoint.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="Override grad clip norm from the config checkpoint.",
    )
    parser.add_argument(
        "--disable-cudnn-sdpa",
        action="store_true",
        help=(
            "Disable cuDNN scaled dot product attention. This is more stable for "
            "bf16 backward passes on variable rectangular chunk shapes."
        ),
    )
    parser.add_argument("--freeze-predictor", action="store_true")
    parser.add_argument(
        "--predictor-eval-during-training",
        action="store_true",
        help=(
            "Keep predictor modules in eval mode while still allowing predictor "
            "gradients. This disables predictor dropout/drop-path during tiny-LR "
            "fine-tuning."
        ),
    )
    parser.add_argument(
        "--no-init-compressor-embedders-from-predictor",
        action="store_true",
        help=(
            "Leave compressor scalar embedders randomly initialized instead of "
            "copying the predictor's scalar encoder."
        ),
    )
    parser.add_argument(
        "--init-compressor-transformers-from-predictor",
        action="store_true",
        help=(
            "Copy pretrained predictor transformer blocks into row and column "
            "compressor transformer blocks."
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=2500)
    parser.add_argument(
        "--lightweight-checkpoints",
        action="store_true",
        help="Save model and metrics without optimizer/scheduler state.",
    )
    parser.add_argument(
        "--disable-best-checkpoint",
        action="store_true",
        help=(
            "Track the best sampled metric but do not write best_checkpoint.pth. "
            "Useful for short searches where external validation selects models."
        ),
    )
    parser.add_argument(
        "--skip-final-checkpoint",
        action="store_true",
        help="Do not write the final checkpoint. Intended for memory smoke tests.",
    )
    parser.add_argument(
        "--best-metric",
        default="loss_missing",
        help=(
            "Metric used for best_checkpoint.pth selection. Exposed-holdout "
            "runs should usually use rmse_missing or exposed_holdout_rmse."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-family",
        choices=["memory_efficient", "stable", "rabel", "pure_rabel"],
        default="memory_efficient",
    )
    parser.add_argument("--wandb-project", default="tabimpute")
    args = parser.parse_args()
    if args.disable_cudnn_sdpa and torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)

    config_checkpoint = args.config_checkpoint or args.predictor_checkpoint
    run_config = load_run_config(config_checkpoint)
    predictor_config = dict(run_config["model"])
    if args.model_family in {"rabel", "pure_rabel"}:
        predictor_config.update(
            {
                "rabel_num_kernels": args.rabel_num_kernels,
                "rabel_sigma": args.rabel_sigma,
                "rabel_exponent_digits": args.rabel_exponent_digits,
                "rabel_exponent_embedding_size": (
                    args.rabel_exponent_embedding_size
                ),
                "rabel_initial_scale": args.rabel_initial_scale,
            }
        )
    optimizer_config = dict(run_config["optimizer"])
    prior_config = dict(run_config["prior"])
    matrix_rows_min = args.matrix_rows_min or args.matrix_rows
    matrix_rows_max = args.matrix_rows_max or args.matrix_rows
    matrix_cols_min = args.matrix_cols_min or args.matrix_cols
    matrix_cols_max = args.matrix_cols_max or args.matrix_cols
    if args.randomize_matrix_size:
        if matrix_rows_min > matrix_rows_max or matrix_cols_min > matrix_cols_max:
            raise ValueError("Matrix min dimensions must be <= max dimensions.")
    else:
        matrix_rows_min = matrix_rows_max = args.matrix_rows
        matrix_cols_min = matrix_cols_max = args.matrix_cols
    prior_config.update(
        {
            "num_rows_low": matrix_rows_min,
            "num_rows_high": matrix_rows_max,
            "num_cols_low": matrix_cols_min,
            "num_cols_high": matrix_cols_max,
            "latent_rank_low": args.rank,
            "latent_rank_high": args.rank,
        }
    )
    if args.prior_kind == "spectrum_decay":
        prior_config.update(SPECTRUM_DECAY_PRIOR_DEFAULTS)
        if args.spectrum_rank_low is not None:
            prior_config["spectrum_rank_low"] = args.spectrum_rank_low
        if args.spectrum_rank_high is not None:
            prior_config["spectrum_rank_high"] = args.spectrum_rank_high
        if args.spectrum_target_rank is not None:
            prior_config["spectrum_target_rank"] = args.spectrum_target_rank
        if args.spectrum_alpha_low is not None:
            prior_config["spectrum_alpha_low"] = args.spectrum_alpha_low
        if args.spectrum_alpha_high is not None:
            prior_config["spectrum_alpha_high"] = args.spectrum_alpha_high
        if args.spectrum_noise_std_low is not None:
            prior_config["spectrum_noise_std_low"] = args.spectrum_noise_std_low
        if args.spectrum_noise_std_high is not None:
            prior_config["spectrum_noise_std_high"] = args.spectrum_noise_std_high
    if args.prior_kind == "rec_sys_spectrum_exposure":
        prior_config.update(REC_SYS_SPECTRUM_EXPOSURE_PRIOR_DEFAULTS)
        recsys_overrides = {
            "recsys_density_low": args.recsys_density_low,
            "recsys_density_high": args.recsys_density_high,
            "recsys_spectrum_rank_low": args.recsys_spectrum_rank_low,
            "recsys_spectrum_rank_high": args.recsys_spectrum_rank_high,
            "recsys_noise_std_low": args.recsys_noise_std_low,
            "recsys_noise_std_high": args.recsys_noise_std_high,
            "recsys_user_activity_sigma_low": args.recsys_user_activity_sigma_low,
            "recsys_user_activity_sigma_high": args.recsys_user_activity_sigma_high,
            "recsys_item_popularity_sigma_low": args.recsys_item_popularity_sigma_low,
            "recsys_item_popularity_sigma_high": args.recsys_item_popularity_sigma_high,
            "recsys_affinity_strength_low": args.recsys_affinity_strength_low,
            "recsys_affinity_strength_high": args.recsys_affinity_strength_high,
            "recsys_exposure_calibration": args.recsys_exposure_calibration,
            "recsys_user_activity_distribution": args.recsys_user_activity_distribution,
            "recsys_item_popularity_distribution": args.recsys_item_popularity_distribution,
            "recsys_popularity_quality_corr_low": args.recsys_popularity_quality_corr_low,
            "recsys_popularity_quality_corr_high": args.recsys_popularity_quality_corr_high,
            "recsys_exposure_item_quality_weight": args.recsys_exposure_item_quality_weight,
            "recsys_exposure_user_bias_weight": args.recsys_exposure_user_bias_weight,
            "recsys_preference_temperature": args.recsys_preference_temperature,
            "recsys_marginal_calibration_steps": args.recsys_marginal_calibration_steps,
            "recsys_marginal_binary_steps": args.recsys_marginal_binary_steps,
            "recsys_global_mean": args.recsys_global_mean,
            "recsys_rating_min": args.recsys_rating_min,
            "recsys_rating_max": args.recsys_rating_max,
            "recsys_rating_clip_probability": args.recsys_rating_clip_probability,
            "recsys_rating_quantization_probability": args.recsys_rating_quantization_probability,
        }
        for key, value in recsys_overrides.items():
            if value is not None:
                prior_config[key] = value
        if args.recsys_cluster_bonus_std is not None:
            prior_config["recsys_cluster_bonus_std_low"] = args.recsys_cluster_bonus_std
            prior_config["recsys_cluster_bonus_std_high"] = args.recsys_cluster_bonus_std
        if args.recsys_rating_quantization:
            prior_config["recsys_rating_quantization"] = True
        if args.recsys_exposure_uses_full_preference:
            prior_config["recsys_exposure_uses_full_preference"] = True
    if args.prior_kind in {
        "rec_sys_bias_residual_rating",
        "rec_sys_bias_residual_residual",
        "rec_sys_bias_residual_exposed_rating",
        "rec_sys_bias_residual_exposed_known_bias",
        "rec_sys_bias_residual_exposed_fitted_bias",
    }:
        if args.prior_kind.startswith("rec_sys_bias_residual_exposed_"):
            prior_config.update(REC_SYS_BIAS_RESIDUAL_EXPOSED_HOLDOUT_DEFAULTS)
            target_mode_by_kind = {
                "rec_sys_bias_residual_exposed_rating": "rating",
                "rec_sys_bias_residual_exposed_known_bias": "residual_known_bias",
                "rec_sys_bias_residual_exposed_fitted_bias": "residual_fitted_bias",
            }
            prior_config["recsys_holdout_target_mode"] = target_mode_by_kind[
                args.prior_kind
            ]
        else:
            prior_config.update(REC_SYS_BIAS_RESIDUAL_PRIOR_DEFAULTS)
            prior_config["recsys_bias_output_mode"] = (
                "residual"
                if args.prior_kind == "rec_sys_bias_residual_residual"
                else "rating"
            )
        recsys_bias_overrides = {
            "recsys_bias_global_mean_low": args.recsys_bias_global_mean_low,
            "recsys_bias_global_mean_high": args.recsys_bias_global_mean_high,
            "recsys_bias_user_std_low": args.recsys_bias_user_std_low,
            "recsys_bias_user_std_high": args.recsys_bias_user_std_high,
            "recsys_bias_item_std_low": args.recsys_bias_item_std_low,
            "recsys_bias_item_std_high": args.recsys_bias_item_std_high,
            "recsys_bias_user_distribution": args.recsys_bias_user_distribution,
            "recsys_bias_item_distribution": args.recsys_bias_item_distribution,
            "recsys_bias_residual_rank_low": args.recsys_bias_residual_rank_low,
            "recsys_bias_residual_rank_high": args.recsys_bias_residual_rank_high,
            "recsys_bias_residual_scale_low": args.recsys_bias_residual_scale_low,
            "recsys_bias_residual_scale_high": args.recsys_bias_residual_scale_high,
            "recsys_bias_noise_std_low": args.recsys_bias_noise_std_low,
            "recsys_bias_noise_std_high": args.recsys_bias_noise_std_high,
            "recsys_bias_rating_quantization_probability": args.recsys_bias_rating_quantization_probability,
            "recsys_exposure_density_low": args.recsys_exposure_density_low
            if args.recsys_exposure_density_low is not None
            else args.recsys_density_low,
            "recsys_exposure_density_high": args.recsys_exposure_density_high
            if args.recsys_exposure_density_high is not None
            else args.recsys_density_high,
            "recsys_exposure_preference_weight_low": args.recsys_exposure_preference_weight_low,
            "recsys_exposure_preference_weight_high": args.recsys_exposure_preference_weight_high,
            "recsys_exposure_cluster_bonus_std": args.recsys_exposure_cluster_bonus_std,
            "recsys_holdout_target_fraction_low": args.recsys_holdout_target_fraction_low,
            "recsys_holdout_target_fraction_high": args.recsys_holdout_target_fraction_high,
            "recsys_holdout_bias_ridge_reg": args.recsys_holdout_bias_ridge_reg,
            "recsys_holdout_bias_ridge_iters": args.recsys_holdout_bias_ridge_iters,
        }
        for key, value in recsys_bias_overrides.items():
            if value is not None:
                prior_config[key] = value
    compressor_config = build_compressor_config_from_predictor(
        predictor_config,
        k_row=args.k_row,
        k_col=args.k_col,
        num_layers=args.compressor_layers,
        mlp_hidden_size=args.compressor_mlp_hidden_size,
    )
    if args.model_family in {"rabel", "pure_rabel"}:
        compressor_config.update(
            {
                "rabel_num_kernels": args.rabel_num_kernels,
                "rabel_sigma": args.rabel_sigma,
                "rabel_exponent_digits": args.rabel_exponent_digits,
                "rabel_exponent_embedding_size": (
                    args.rabel_exponent_embedding_size
                ),
                "rabel_initial_scale": args.rabel_initial_scale,
            }
        )

    device = args.device
    compute_dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32
    model = build_compressed_model(
        predictor_config=predictor_config,
        predictor_checkpoint_path=None
        if args.train_from_scratch
        else args.predictor_checkpoint,
        compressor_config=compressor_config,
        k_row=args.k_row,
        k_col=args.k_col,
        model_family=args.model_family,
        device=device,
        init_compressor_embedders_from_predictor=(
            not args.no_init_compressor_embedders_from_predictor
            and not args.train_from_scratch
        ),
        init_compressor_transformers_from_predictor=(
            args.init_compressor_transformers_from_predictor
            and not args.train_from_scratch
        ),
    ).to(device=device, dtype=compute_dtype)
    if args.compressed_checkpoint is not None:
        load_compressed_model_checkpoint(
            model,
            args.compressed_checkpoint,
            device=device,
            allow_query_resize=args.allow_compressor_query_resize,
            allow_encoder_extension=args.model_family == "rabel",
        )
    if args.freeze_predictor:
        model.predictor.requires_grad_(False)
        model.predictor.eval()
    elif args.predictor_eval_during_training:
        model.predictor.eval()

    if args.prior_kind == "spectrum_decay":
        prior = SpectrumMissingnessPrior(
            missingness_type="mcar",
            config=prior_config,
            batch_size=args.batch_size,
            device=device,
            verbose=False,
            entry_wise_features=False,
        )
    elif args.prior_kind == "rec_sys_spectrum_exposure":
        prior = RecSysSpectrumExposurePrior(
            config=prior_config,
            batch_size=args.batch_size,
            device=device,
            verbose=False,
            entry_wise_features=False,
        )
    elif args.prior_kind in {
        "rec_sys_bias_residual_rating",
        "rec_sys_bias_residual_residual",
    }:
        prior = MissingnessPrior(
            generator_type="rec_sys_bias_residual",
            missingness_type="mnar_rec_sys_exposure",
            config=prior_config,
            batch_size=args.batch_size,
            device=device,
            verbose=False,
            entry_wise_features=False,
        )
    elif args.prior_kind in {
        "rec_sys_bias_residual_exposed_rating",
        "rec_sys_bias_residual_exposed_known_bias",
        "rec_sys_bias_residual_exposed_fitted_bias",
    }:
        prior = RecSysBiasResidualExposedHoldoutPrior(
            config=prior_config,
            batch_size=args.batch_size,
            device=device,
            verbose=False,
            entry_wise_features=False,
        )
    else:
        prior = MissingnessPrior(
            generator_type="latent_factor",
            missingness_type="mcar",
            config=prior_config,
            batch_size=args.batch_size,
            device=device,
            verbose=False,
            entry_wise_features=False,
        )
    borders_path = importlib.resources.files("tabimpute") / "data" / "borders.pt"
    with importlib.resources.as_file(borders_path) as path:
        criterion = FullSupportBarDistribution(
            borders=torch.load(path).to(torch.device(device))
        )

    base_lr = optimizer_config.get("lr", 2e-4)
    predictor_lr = (
        args.predictor_lr
        if args.predictor_lr is not None
        else base_lr * args.predictor_lr_multiplier
    )
    compressor_lr = args.compressor_lr if args.compressor_lr is not None else base_lr
    scheduler_name = args.scheduler_name or optimizer_config.get(
        "scheduler_name",
        "none",
    )
    warmup_ratio = (
        args.warmup_ratio
        if args.warmup_ratio is not None
        else optimizer_config.get("warmup_ratio", 0.05)
    )
    min_lr_ratio = (
        args.min_lr_ratio
        if args.min_lr_ratio is not None
        else optimizer_config.get("min_lr_ratio", 0.1)
    )
    grad_clip_norm = (
        args.grad_clip_norm
        if args.grad_clip_norm is not None
        else optimizer_config.get("grad_clip_norm")
    )
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else optimizer_config.get("weight_decay", 0.03)
    )
    optimizer = build_split_optimizer(
        model=model,
        optimizer_name=optimizer_config.get("optimizer_name", "adamw"),
        predictor_lr=predictor_lr,
        compressor_lr=compressor_lr,
        encoder_lr=args.encoder_lr,
        weight_decay=weight_decay,
        optimizer_betas=tuple(optimizer_config.get("optimizer_betas", (0.9, 0.999))),
        optimizer_eps=optimizer_config.get("optimizer_eps", 1e-8),
    )
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps is not None
        else int(args.steps * warmup_ratio)
    )
    scheduler = build_compressed_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        total_steps=args.steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
        one_cycle_pct_start=optimizer_config.get("one_cycle_pct_start", 0.1),
        stable_steps=args.stable_steps,
        decay_steps=args.decay_steps,
    )

    args.work_dir.mkdir(parents=True, exist_ok=True)
    full_config = {
        "run_name": args.run_name,
        "predictor_checkpoint": str(args.predictor_checkpoint),
        "config_checkpoint": str(config_checkpoint),
        "train_from_scratch": args.train_from_scratch,
        "compressed_checkpoint": str(args.compressed_checkpoint)
        if args.compressed_checkpoint is not None
        else None,
        "model": {
            "predictor": predictor_config,
            "compressor": compressor_config,
            "k_row": args.k_row,
            "k_col": args.k_col,
            "target_chunk_sampling": args.target_chunk_sampling,
            "model_family": args.model_family,
            "init_compressor_embedders_from_predictor": (
                not args.no_init_compressor_embedders_from_predictor
                and not args.train_from_scratch
            ),
            "init_compressor_transformers_from_predictor": (
                args.init_compressor_transformers_from_predictor
                and not args.train_from_scratch
            ),
        },
        "optimizer": optimizer_config,
        "optimizer_parameter_groups": {
            "predictor_lr": predictor_lr,
            "compressor_lr": compressor_lr,
            "encoder_lr": args.encoder_lr,
            "predictor_lr_multiplier": args.predictor_lr_multiplier,
            "scheduler_name": scheduler_name,
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "stable_steps": args.stable_steps,
            "decay_steps": args.decay_steps,
            "min_lr_ratio": min_lr_ratio,
            "grad_clip_norm": grad_clip_norm,
            "weight_decay": weight_decay,
            "freeze_predictor": args.freeze_predictor,
            "predictor_eval_during_training": args.predictor_eval_during_training,
        },
        "prior": prior_config,
        "prior_kind": args.prior_kind,
        "recsys_curriculum_schedule": RECSYS_MOVIELENS_PROXY_20K_CURRICULUM
        if args.recsys_curriculum == "movielens_proxy_20k"
        else None,
        "training": vars(args),
    }
    with (args.work_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=2, sort_keys=True, default=str)

    callbacks = [
        WandbLoggerCallback(
            project=args.wandb_project,
            name=args.run_name,
            id=args.run_name,
            api_key=os.environ.get("WANDB_KEY"),
            config=full_config,
            log_dir=str(args.work_dir / "wandb"),
        )
    ]

    model.train()
    if args.freeze_predictor:
        model.predictor.eval()
    elif args.predictor_eval_during_training:
        model.predictor.eval()
    last_metrics = {}
    best_loss_missing = float("inf")
    best_step = None
    try:
        for step in tqdm(range(1, args.steps + 1)):
            start = time.time()
            if torch.cuda.is_available() and "cuda" in str(device):
                torch.cuda.reset_peak_memory_stats(torch.device(device))
            optimizer.zero_grad(set_to_none=True)
            curriculum_metrics = apply_recsys_curriculum(
                prior=prior,
                curriculum_name=args.recsys_curriculum,
                step=step,
                total_steps=args.steps,
            )
            prior_batch, _ = prior.get_batch()
            batch_x, batch_target = prior_batch[0], prior_batch[1]
            loss_mask = None
            if (
                len(prior_batch) >= 3
                and isinstance(prior_batch[2], torch.Tensor)
                and prior_batch[2].shape == batch_x.shape
                and prior_batch[2].dtype == torch.bool
            ):
                loss_mask = prior_batch[2]
            batch_x = batch_x.to(device=device, dtype=compute_dtype)
            batch_target = batch_target.to(device=device, dtype=compute_dtype)
            if loss_mask is not None:
                loss_mask = loss_mask.to(device=device, dtype=torch.bool)
            step_chunk_rows = args.chunk_rows
            step_chunk_cols = args.chunk_cols
            sampled_row_fraction = step_chunk_rows / batch_x.shape[1]
            sampled_col_fraction = step_chunk_cols / batch_x.shape[2]
            if args.randomize_chunk_fractions:
                (
                    step_chunk_rows,
                    step_chunk_cols,
                    sampled_row_fraction,
                    sampled_col_fraction,
                ) = sample_batch_chunk_sizes(
                    num_rows=batch_x.shape[1],
                    num_cols=batch_x.shape[2],
                    fraction_min=args.chunk_fraction_min,
                    fraction_max=args.chunk_fraction_max,
                    min_chunk_rows=args.min_chunk_rows,
                    min_chunk_cols=args.min_chunk_cols,
                )

            if args.target_chunk_sampling == "random":
                loss_tensor, metrics = train_one_random_chunk_step(
                    model=model,
                    criterion=criterion,
                    batch_x=batch_x,
                    batch_target=batch_target,
                    loss_mask=loss_mask,
                    chunk_rows=step_chunk_rows,
                    chunk_cols=step_chunk_cols,
                    num_target_chunks=args.random_targets_per_step,
                )
            else:
                loss_tensor, metrics = train_one_step(
                    model=model,
                    criterion=criterion,
                    batch_x=batch_x,
                    batch_target=batch_target,
                    loss_mask=loss_mask,
                    chunk_rows=step_chunk_rows,
                    chunk_cols=step_chunk_cols,
                )
            if grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm,
                    error_if_nonfinite=True,
                )
                metrics["grad_norm"] = float(grad_norm.detach().cpu())
                metrics["grad_was_clipped"] = float(
                    metrics["grad_norm"] > grad_clip_norm
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            lr_by_name = {
                group.get("name", f"group_{idx}"): group["lr"]
                for idx, group in enumerate(optimizer.param_groups)
            }
            metrics["lr"] = lr_by_name["compressors"]
            metrics["lr_predictor"] = lr_by_name.get("predictor", 0.0)
            metrics["lr_compressor"] = lr_by_name["compressors"]
            metrics["lr_encoder"] = lr_by_name.get("rabel_encoder", 0.0)
            metrics["loss_total"] = metrics["loss_missing"]
            metrics["mae_total"] = metrics["mae_missing"]
            metrics["matrix_rows"] = float(batch_x.shape[1])
            metrics["matrix_cols"] = float(batch_x.shape[2])
            metrics["chunk_rows"] = float(step_chunk_rows)
            metrics["chunk_cols"] = float(step_chunk_cols)
            metrics["chunk_row_fraction"] = sampled_row_fraction
            metrics["chunk_col_fraction"] = sampled_col_fraction
            metrics.update(curriculum_metrics)
            metrics.update(getattr(prior, "last_batch_metrics", {}))
            if torch.cuda.is_available() and "cuda" in str(device):
                metrics["cuda_peak_allocated_gib"] = (
                    torch.cuda.max_memory_allocated(torch.device(device))
                    / (1024**3)
                )
                metrics["cuda_peak_reserved_gib"] = (
                    torch.cuda.max_memory_reserved(torch.device(device))
                    / (1024**3)
                )
            last_metrics = metrics
            if args.best_metric not in metrics:
                raise KeyError(
                    f"Best metric '{args.best_metric}' was not logged. "
                    f"Available metrics: {sorted(metrics.keys())}"
                )
            best_metric_value = float(metrics[args.best_metric])
            metrics["best_metric_name"] = args.best_metric
            metrics["best_metric_value"] = best_metric_value
            if best_metric_value < best_loss_missing:
                best_loss_missing = best_metric_value
                best_step = step
                if not args.disable_best_checkpoint:
                    save_best_checkpoint(
                        work_dir=args.work_dir,
                        step=step,
                        loss_missing=best_loss_missing,
                        metrics=metrics,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        include_optimizer=not args.lightweight_checkpoints,
                    )
            for callback in callbacks:
                callback.on_epoch_end(
                    step,
                    time.time() - start,
                    metrics["loss_missing"],
                    None,
                    dist=criterion,
                    log_dict=metrics,
                )

            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                torch.save(
                    build_checkpoint_state(
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        metrics=metrics,
                        best_step=best_step,
                        best_loss_missing=best_loss_missing,
                        include_optimizer=not args.lightweight_checkpoints,
                    ),
                    args.work_dir / f"checkpoint_{step}.pth",
                )
    finally:
        for callback in callbacks:
            callback.close()

    if not args.skip_final_checkpoint:
        torch.save(
            build_checkpoint_state(
                step=args.steps,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=last_metrics,
                best_step=best_step,
                best_loss_missing=best_loss_missing
                if best_step is not None
                else None,
                include_optimizer=not args.lightweight_checkpoints,
            ),
            args.work_dir / f"checkpoint_{args.steps}.pth",
        )
    with (args.work_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(last_metrics, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
