#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.resources
import json
from pathlib import Path

import torch

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_compressed_cross import (
    build_compressor_config_from_predictor,
    make_chunk_spans,
)
from tabimpute.prior.training_set_generation import MissingnessPrior

from train_compressed_cross_rank10 import build_compressed_model, build_predictor


DEFAULT_NATIVE_CHECKPOINT = Path(
    "/root/tabular/mcpfn/src/tabimpute/workdir/"
    "tabimpute-new-stable_v2-rank_10_10-100x100-from_50x50_step30000/"
    "checkpoint_45000.pth"
)


def load_config(checkpoint_path: Path) -> dict:
    with (checkpoint_path.parent / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_native_model(
    checkpoint_path: Path,
    model_family: str,
    device: str,
    dtype: torch.dtype,
):
    cfg = load_config(checkpoint_path)
    model_cfg = dict(cfg["model"])
    model = build_predictor(model_cfg, model_family=model_family)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device=device, dtype=dtype).eval()


def load_compressed_model(
    checkpoint_path: Path,
    model_family: str,
    device: str,
    dtype: torch.dtype,
):
    cfg = load_config(checkpoint_path)
    predictor_cfg = dict(cfg["model"]["predictor"])
    compressor_cfg = dict(
        cfg["model"].get(
            "compressor",
            build_compressor_config_from_predictor(
                predictor_cfg,
                k_row=cfg["model"]["k_row"],
                k_col=cfg["model"]["k_col"],
                num_layers=2,
            ),
        )
    )
    model = build_compressed_model(
        predictor_config=predictor_cfg,
        predictor_checkpoint_path=None,
        compressor_config=compressor_cfg,
        k_row=cfg["model"]["k_row"],
        k_col=cfg["model"]["k_col"],
        model_family=model_family or cfg["model"].get("model_family", "stable"),
        device=device,
        init_compressor_embedders_from_predictor=False,
        init_compressor_transformers_from_predictor=False,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device=device, dtype=dtype).eval(), cfg


def make_prior_config(
    native_checkpoint: Path,
    rows: int,
    cols: int,
    rank: int,
) -> dict:
    cfg = load_config(native_checkpoint)
    prior_cfg = dict(cfg["prior"])
    prior_cfg.update(
        {
            "num_rows_low": rows,
            "num_rows_high": rows,
            "num_cols_low": cols,
            "num_cols_high": cols,
            "latent_rank_low": rank,
            "latent_rank_high": rank,
        }
    )
    return prior_cfg


@torch.no_grad()
def score_native(
    model,
    criterion: FullSupportBarDistribution,
    batch_x: torch.Tensor,
    batch_target: torch.Tensor,
) -> dict[str, float]:
    missing_mask = torch.isnan(batch_x)
    logits = model(batch_x, decode_mask=missing_mask)
    target_missing = batch_target[missing_mask]
    losses = criterion(logits, target_missing)
    medians = criterion.median(logits)
    return {
        "loss_missing": float(losses.mean().detach().cpu()),
        "mae_missing": float((medians - target_missing).abs().mean().detach().cpu()),
    }


@torch.no_grad()
def score_compressed(
    model,
    criterion: FullSupportBarDistribution,
    batch_x: torch.Tensor,
    batch_target: torch.Tensor,
    chunk_rows: int,
    chunk_cols: int,
) -> dict[str, float]:
    row_spans = make_chunk_spans(batch_x.shape[1], chunk_rows)
    col_spans = make_chunk_spans(batch_x.shape[2], chunk_cols)
    loss_sum = 0.0
    mae_sum = 0.0
    targets_with_loss = 0

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
            medians = criterion.median(logits[missing_mask])
            loss_sum += float(losses.mean().detach().cpu())
            mae_sum += float(
                (medians - target[missing_mask]).abs().mean().detach().cpu()
            )
            targets_with_loss += 1

    return {
        "loss_missing": loss_sum / targets_with_loss,
        "mae_missing": mae_sum / targets_with_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compressed-checkpoint", type=Path, required=True)
    parser.add_argument("--native-checkpoint", type=Path, default=DEFAULT_NATIVE_CHECKPOINT)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--cols", type=int, default=100)
    parser.add_argument("--chunk-rows", type=int, default=50)
    parser.add_argument("--chunk-cols", type=int, default=50)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--native-model-family", default="stable")
    parser.add_argument("--compressed-model-family", default="")
    args = parser.parse_args()

    dtype = torch.bfloat16 if "cuda" in args.device else torch.float32
    native_model = load_native_model(
        args.native_checkpoint,
        model_family=args.native_model_family,
        device=args.device,
        dtype=dtype,
    )
    compressed_model, compressed_cfg = load_compressed_model(
        args.compressed_checkpoint,
        model_family=args.compressed_model_family,
        device=args.device,
        dtype=dtype,
    )

    prior = MissingnessPrior(
        generator_type="latent_factor",
        missingness_type="mcar",
        config=make_prior_config(args.native_checkpoint, args.rows, args.cols, args.rank),
        batch_size=args.batch_size,
        device=args.device,
        verbose=False,
        entry_wise_features=False,
    )
    borders_path = importlib.resources.files("tabimpute") / "data" / "borders.pt"
    with importlib.resources.as_file(borders_path) as path:
        criterion = FullSupportBarDistribution(
            borders=torch.load(path).to(torch.device(args.device))
        )

    totals = {
        "native_loss_missing": 0.0,
        "native_mae_missing": 0.0,
        "compressed_loss_missing": 0.0,
        "compressed_mae_missing": 0.0,
    }
    for _ in range(args.num_batches):
        (batch_x, batch_target, _, _, _), _ = prior.get_batch()
        batch_x = batch_x.to(device=args.device, dtype=dtype)
        batch_target = batch_target.to(device=args.device, dtype=dtype)
        native = score_native(native_model, criterion, batch_x, batch_target)
        compressed = score_compressed(
            compressed_model,
            criterion,
            batch_x,
            batch_target,
            chunk_rows=args.chunk_rows,
            chunk_cols=args.chunk_cols,
        )
        totals["native_loss_missing"] += native["loss_missing"]
        totals["native_mae_missing"] += native["mae_missing"]
        totals["compressed_loss_missing"] += compressed["loss_missing"]
        totals["compressed_mae_missing"] += compressed["mae_missing"]

    summary = {key: value / args.num_batches for key, value in totals.items()}
    summary["loss_gap_compressed_minus_native"] = (
        summary["compressed_loss_missing"] - summary["native_loss_missing"]
    )
    summary["mae_gap_compressed_minus_native"] = (
        summary["compressed_mae_missing"] - summary["native_mae_missing"]
    )
    summary["compressed_checkpoint"] = str(args.compressed_checkpoint)
    summary["native_checkpoint"] = str(args.native_checkpoint)
    summary["compressed_run_name"] = compressed_cfg.get("run_name")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
