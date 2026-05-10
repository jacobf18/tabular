#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.resources
import json
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_compressed_cross import (
    ColumnCompressionModel,
    CompressedCrossTabImputeModel,
    RowCompressionModel,
    build_compressor_config_from_predictor,
    make_chunk_spans,
)
from tabimpute.model.model_new_stable import TabImputeModel as StableTabImputeModel
from tabimpute.model.model_new_stable_memory_efficient import (
    TabImputeModel as MemoryEfficientTabImputeModel,
)
from tabimpute.prior.training_set_generation import MissingnessPrior
from tabimpute.train.callbacks import WandbLoggerCallback
from tabimpute.train_new_sweep import _build_scheduler


def load_run_config(checkpoint_path: Path) -> dict:
    with (checkpoint_path.parent / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


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
    raise ValueError(f"Unsupported model_family: {model_family}")


def build_compressed_model(
    *,
    predictor_config: dict,
    predictor_checkpoint_path: Path,
    compressor_config: dict,
    k_row: int,
    k_col: int,
    model_family: str,
    device: str,
):
    predictor = build_predictor(predictor_config, model_family=model_family)
    ckpt = torch.load(predictor_checkpoint_path, map_location=device, weights_only=True)
    predictor.load_state_dict(ckpt["model"], strict=True)

    row_compressor = RowCompressionModel(
        embedding_size=compressor_config["embedding_size"],
        num_attention_heads=compressor_config["num_attention_heads"],
        mlp_hidden_size=compressor_config["mlp_hidden_size"],
        num_layers=compressor_config["num_layers"],
        k_row=k_row,
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
    )
    column_compressor = ColumnCompressionModel(
        embedding_size=compressor_config["embedding_size"],
        num_attention_heads=compressor_config["num_attention_heads"],
        mlp_hidden_size=compressor_config["mlp_hidden_size"],
        num_layers=compressor_config["num_layers"],
        k_col=k_col,
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
    )
    return CompressedCrossTabImputeModel(
        predictor=predictor,
        row_compressor=row_compressor,
        column_compressor=column_compressor,
        k_row=k_row,
        k_col=k_col,
    )


def build_split_optimizer(
    model: CompressedCrossTabImputeModel,
    *,
    optimizer_name: str,
    predictor_lr: float,
    compressor_lr: float,
    weight_decay: float,
    optimizer_betas: tuple[float, float],
    optimizer_eps: float,
):
    optimizer_name = optimizer_name.lower()
    param_groups = []
    predictor_params = [
        param for param in model.predictor.parameters() if param.requires_grad
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
    param_groups.append(
        {
            "params": list(model.row_compressor.parameters())
            + list(model.column_compressor.parameters())
            + [model.off_axis_token],
            "lr": compressor_lr,
            "weight_decay": weight_decay,
            "name": "compressors",
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
    chunk_rows: int,
    chunk_cols: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    row_spans = make_chunk_spans(batch_x.shape[1], chunk_rows)
    col_spans = make_chunk_spans(batch_x.shape[2], chunk_cols)
    targets_with_loss = 0
    loss_sum = 0.0
    mae_sum = 0.0

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
            missing_mask = torch.isnan(
                batch_x[
                    :,
                    row_span.start : row_span.end,
                    col_span.start : col_span.end,
                ]
            )
            if not missing_mask.any():
                continue
            losses = criterion(logits[missing_mask], target[missing_mask])
            loss = losses.mean()
            (loss * scale).backward()
            with torch.no_grad():
                medians = criterion.median(logits[missing_mask])
                mae = (medians - target[missing_mask]).abs().mean()
            loss_sum += float(loss.detach().cpu())
            mae_sum += float(mae.detach().cpu())
            targets_with_loss += 1

    if targets_with_loss == 0:
        zero = batch_target.new_zeros(())
        return zero, {"loss_missing": 0.0, "mae_missing": 0.0}
    return batch_target.new_tensor(loss_sum / targets_with_loss), {
        "loss_missing": loss_sum / targets_with_loss,
        "mae_missing": mae_sum / targets_with_loss,
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
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--matrix-rows", type=int, default=200)
    parser.add_argument("--matrix-cols", type=int, default=200)
    parser.add_argument("--chunk-rows", type=int, default=50)
    parser.add_argument("--chunk-cols", type=int, default=50)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--k-row", type=int, default=4)
    parser.add_argument("--k-col", type=int, default=4)
    parser.add_argument("--compressor-layers", type=int, default=2)
    parser.add_argument("--compressor-mlp-hidden-size", type=int, default=None)
    parser.add_argument("--predictor-lr", type=float, default=None)
    parser.add_argument("--compressor-lr", type=float, default=None)
    parser.add_argument("--predictor-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--freeze-predictor", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=2500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-family",
        choices=["memory_efficient", "stable"],
        default="memory_efficient",
    )
    parser.add_argument("--wandb-project", default="tabimpute")
    args = parser.parse_args()

    run_config = load_run_config(args.predictor_checkpoint)
    predictor_config = dict(run_config["model"])
    optimizer_config = dict(run_config["optimizer"])
    prior_config = dict(run_config["prior"])
    prior_config.update(
        {
            "num_rows_low": args.matrix_rows,
            "num_rows_high": args.matrix_rows,
            "num_cols_low": args.matrix_cols,
            "num_cols_high": args.matrix_cols,
            "latent_rank_low": args.rank,
            "latent_rank_high": args.rank,
        }
    )
    compressor_config = build_compressor_config_from_predictor(
        predictor_config,
        k_row=args.k_row,
        k_col=args.k_col,
        num_layers=args.compressor_layers,
        mlp_hidden_size=args.compressor_mlp_hidden_size,
    )

    device = args.device
    compute_dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32
    model = build_compressed_model(
        predictor_config=predictor_config,
        predictor_checkpoint_path=args.predictor_checkpoint,
        compressor_config=compressor_config,
        k_row=args.k_row,
        k_col=args.k_col,
        model_family=args.model_family,
        device=device,
    ).to(device=device, dtype=compute_dtype)
    if args.freeze_predictor:
        model.predictor.requires_grad_(False)
        model.predictor.eval()

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
    optimizer = build_split_optimizer(
        model=model,
        optimizer_name=optimizer_config.get("optimizer_name", "adamw"),
        predictor_lr=predictor_lr,
        compressor_lr=compressor_lr,
        weight_decay=optimizer_config.get("weight_decay", 0.03),
        optimizer_betas=tuple(optimizer_config.get("optimizer_betas", (0.9, 0.999))),
        optimizer_eps=optimizer_config.get("optimizer_eps", 1e-8),
    )
    scheduler = _build_scheduler(
        optimizer=optimizer,
        scheduler_name=optimizer_config.get("scheduler_name", "none"),
        total_steps=args.steps,
        warmup_steps=int(args.steps * optimizer_config.get("warmup_ratio", 0.05)),
        min_lr_ratio=optimizer_config.get("min_lr_ratio", 0.1),
        one_cycle_pct_start=optimizer_config.get("one_cycle_pct_start", 0.1),
    )

    args.work_dir.mkdir(parents=True, exist_ok=True)
    full_config = {
        "run_name": args.run_name,
        "predictor_checkpoint": str(args.predictor_checkpoint),
        "model": {
            "predictor": predictor_config,
            "compressor": compressor_config,
            "k_row": args.k_row,
            "k_col": args.k_col,
            "model_family": args.model_family,
        },
        "optimizer": optimizer_config,
        "optimizer_parameter_groups": {
            "predictor_lr": predictor_lr,
            "compressor_lr": compressor_lr,
            "predictor_lr_multiplier": args.predictor_lr_multiplier,
            "freeze_predictor": args.freeze_predictor,
        },
        "prior": prior_config,
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
    last_metrics = {}
    try:
        for step in tqdm(range(1, args.steps + 1)):
            start = time.time()
            optimizer.zero_grad(set_to_none=True)
            (batch_x, batch_target, _, _, _), _ = prior.get_batch()
            batch_x = batch_x.to(device=device, dtype=compute_dtype)
            batch_target = batch_target.to(device=device, dtype=compute_dtype)

            loss_tensor, metrics = train_one_step(
                model=model,
                criterion=criterion,
                batch_x=batch_x,
                batch_target=batch_target,
                chunk_rows=args.chunk_rows,
                chunk_cols=args.chunk_cols,
            )
            grad_clip_norm = optimizer_config.get("grad_clip_norm")
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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
            metrics["loss_total"] = float("nan")
            metrics["mae_total"] = float("nan")
            last_metrics = metrics
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
                    {
                        "epoch": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()
                        if scheduler is not None
                        else None,
                    },
                    args.work_dir / f"checkpoint_{step}.pth",
                )
    finally:
        for callback in callbacks:
            callback.close()

    torch.save(
        {
            "epoch": args.steps,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        },
        args.work_dir / f"checkpoint_{args.steps}.pth",
    )
    with (args.work_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(last_metrics, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
