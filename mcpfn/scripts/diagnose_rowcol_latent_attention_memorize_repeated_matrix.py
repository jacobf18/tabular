#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.resources
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train_direct_rowcol_latent_attention_random_rank_curriculum as train_cfg
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModelStable
from tabimpute.model.model_rowcol_latent_attention import (
    TabImputeModelRowColumnLatentAttention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="rowcol-attn-memorize-repeated-matrix")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/root/tabular/mcpfn/src/tabimpute/workdir"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--cols", type=int, default=50)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--missing-rate", type=float, default=0.2)
    parser.add_argument("--mask-mode", choices=("fixed", "random_each_step"), default="fixed")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--point-loss-weight", type=float, default=0.0)
    parser.add_argument("--point-loss-prediction", choices=("mean", "median"), default="mean")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--init-model-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--model-family",
        choices=("rowcol_latent_attention", "new_stable"),
        default="rowcol_latent_attention",
    )
    parser.add_argument(
        "--config-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint whose sibling config.json supplies the model config.",
    )
    parser.add_argument("--model-overrides-json", default=None)
    return parser.parse_args()


def make_low_rank(rows: int, cols: int, rank: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(rows, rank))
    right = rng.normal(size=(rank, cols))
    matrix = (left @ right) / rank
    return torch.from_numpy(matrix.astype(np.float32))


def sample_mask(
    shape: tuple[int, int] | tuple[int, int, int],
    missing_rate: float,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.rand(shape, device=device) < missing_rate
    if len(shape) == 2:
        mask = ensure_observed_support(mask)
    else:
        mask = torch.stack([ensure_observed_support(mask_i) for mask_i in mask], dim=0)
    return mask


def ensure_observed_support(missing_mask: torch.Tensor) -> torch.Tensor:
    missing_mask = missing_mask.clone()
    rows, cols = missing_mask.shape
    for row in range(rows):
        if missing_mask[row].all():
            missing_mask[row, torch.randint(cols, (1,), device=missing_mask.device)] = False
    for col in range(cols):
        if missing_mask[:, col].all():
            missing_mask[torch.randint(rows, (1,), device=missing_mask.device), col] = False
    if not missing_mask.any():
        missing_mask[
            torch.randint(rows, (1,), device=missing_mask.device),
            torch.randint(cols, (1,), device=missing_mask.device),
        ] = True
    return missing_mask


def point_predictions(
    criterion: FullSupportBarDistribution,
    logits: torch.Tensor,
    stat: str,
) -> torch.Tensor:
    logits = logits.float()
    if stat == "mean":
        return criterion.mean(logits)
    return criterion.median(logits)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: FullSupportBarDistribution,
    target: torch.Tensor,
    fixed_missing_mask: torch.Tensor,
    prediction_stat: str,
) -> dict[str, float]:
    model.eval()
    model_dtype = next(model.parameters()).dtype
    x_missing = target.clone().to(dtype=model_dtype)
    x_missing[fixed_missing_mask] = torch.nan
    logits = model(x_missing.unsqueeze(0))
    mean_pred = criterion.mean(logits.float()).squeeze(0)
    median_pred = criterion.median(logits.float()).squeeze(0)
    mean_err = mean_pred[fixed_missing_mask] - target[fixed_missing_mask]
    median_err = median_pred[fixed_missing_mask] - target[fixed_missing_mask]
    chosen = mean_pred if prediction_stat == "mean" else median_pred
    chosen_err = chosen[fixed_missing_mask] - target[fixed_missing_mask]
    missing_logits = logits.squeeze(0)[fixed_missing_mask]
    nll = criterion(missing_logits, target[fixed_missing_mask]).mean()
    model.train()
    return {
        "eval_nll_fixed_missing": float(nll.detach().cpu()),
        "eval_rmse_fixed_missing_mean": float(mean_err.square().mean().sqrt().cpu()),
        "eval_rmse_fixed_missing_median": float(
            median_err.square().mean().sqrt().cpu()
        ),
        "eval_mae_fixed_missing_mean": float(mean_err.abs().mean().cpu()),
        "eval_mae_fixed_missing_median": float(median_err.abs().mean().cpu()),
        "eval_rmse_fixed_missing_chosen": float(
            chosen_err.square().mean().sqrt().cpu()
        ),
    }


def atomic_save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def model_config_from_checkpoint(checkpoint: Path) -> dict:
    with (checkpoint.parent / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "model" in cfg:
        return dict(cfg["model"])
    if "config" in cfg:
        return dict(cfg["config"])
    return dict(cfg)


def build_model_config(args: argparse.Namespace) -> dict:
    if args.config_checkpoint is not None:
        model_config = model_config_from_checkpoint(args.config_checkpoint)
    elif args.model_family == "new_stable":
        default_checkpoint = Path(
            "/root/tabular/mcpfn/src/tabimpute/workdir/"
            "tabimpute-new-stable_v2-rank_10_10-50x50/checkpoint_30000.pth"
        )
        model_config = model_config_from_checkpoint(default_checkpoint)
    else:
        model_config = dict(train_cfg.base.MODEL_CONFIG)
    if args.model_overrides_json:
        model_config.update(json.loads(args.model_overrides_json))
    model_config["use_activation_checkpointing"] = False
    return model_config


def build_model(args: argparse.Namespace, model_config: dict) -> torch.nn.Module:
    if args.model_family == "new_stable":
        allowed = {
            "embedding_size",
            "num_attention_heads",
            "mlp_hidden_size",
            "num_layers",
            "num_outputs",
            "num_cls",
            "use_rope",
            "rope_base",
            "rope_fraction",
            "use_absolute_positional_embeddings",
            "positional_damping_factor",
            "attention_dropout",
            "ffn_dropout",
            "drop_path_rate",
            "residual_scale_init",
            "rms_norm_eps",
            "embedding_dropout",
            "attention_batch_chunk_size",
        }
        return TabImputeModelStable(
            **{key: value for key, value in model_config.items() if key in allowed}
        )
    return TabImputeModelRowColumnLatentAttention(**model_config)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    run_dir = args.work_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_config = build_model_config(args)
    model = build_model(args, model_config).to(
        device=device,
        dtype=dtype,
    )
    if args.init_model_checkpoint is not None:
        checkpoint = torch.load(args.init_model_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)

    borders_path = importlib.resources.files("tabimpute") / "data" / "borders.pt"
    with importlib.resources.as_file(borders_path) as path:
        criterion = FullSupportBarDistribution(
            borders=torch.load(path, map_location=device).to(device)
        )

    target = make_low_rank(args.rows, args.cols, args.rank, args.seed).to(device)
    fixed_missing_mask = sample_mask(
        (args.rows, args.cols),
        args.missing_rate,
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )

    config_args = dict(vars(args))
    for key, value in list(config_args.items()):
        if isinstance(value, Path):
            config_args[key] = str(value)
    config = {
        **config_args,
        "work_dir": str(args.work_dir),
        "init_model_checkpoint": (
            str(args.init_model_checkpoint) if args.init_model_checkpoint else None
        ),
        "model": model_config,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    metrics_path = run_dir / "metrics.jsonl"
    start = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        if args.mask_mode == "fixed":
            missing_mask = fixed_missing_mask.unsqueeze(0).expand(args.batch_size, -1, -1)
        else:
            missing_mask = sample_mask(
                (args.batch_size, args.rows, args.cols),
                args.missing_rate,
                device,
            )
        batch_target = target.unsqueeze(0).expand(args.batch_size, -1, -1)
        batch_x = batch_target.clone().to(dtype=dtype)
        batch_x[missing_mask] = torch.nan
        logits = model(batch_x, decode_mask=missing_mask)
        target_missing = batch_target[missing_mask].to(dtype=dtype)
        nll = criterion(logits, target_missing).mean()
        point_loss = logits.new_zeros(())
        if args.point_loss_weight > 0:
            preds = point_predictions(criterion, logits, args.point_loss_prediction)
            point_loss = F.mse_loss(preds.float(), target_missing.float())
        loss = nll + args.point_loss_weight * point_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            eval_metrics = evaluate(
                model,
                criterion,
                target,
                fixed_missing_mask,
                args.point_loss_prediction,
            )
            row = {
                "step": step,
                "elapsed_sec": time.time() - start,
                "loss": float(loss.detach().cpu()),
                "nll": float(nll.detach().cpu()),
                "point_loss": float(point_loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                **eval_metrics,
            }
            if device.type == "cuda":
                row["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / (1024**3)
                row["cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / (1024**3)
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            atomic_save(
                {"epoch": step, "model": model.state_dict(), "metrics": row},
                run_dir / f"checkpoint_{step}.pth",
            )

    atomic_save(
        {"epoch": args.steps, "model": model.state_dict(), "metrics": row},
        run_dir / f"checkpoint_{args.steps}.pth",
    )


if __name__ == "__main__":
    main()
