#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.resources
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModelStable
from tabimpute.model.model_rowcol_latent_attention import (
    LatentSelfAttentionLayer,
    TabImputeModelRowColumnLatentAttention,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path
    family: str


def parse_spec(value: str) -> ModelSpec:
    parts = value.split("|")
    if len(parts) != 3:
        raise ValueError("--spec must be formatted as 'name|checkpoint|family'")
    name, checkpoint, family = (part.strip() for part in parts)
    path = Path(checkpoint).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    return ModelSpec(name=name, checkpoint=path, family=family)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", action="append", required=True, type=parse_spec)
    parser.add_argument("--sizes", nargs="+", type=int, default=[50, 100, 300, 500])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--flow-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--missing-rate", type=float, default=0.2)
    parser.add_argument("--scale", choices=("rank", "sqrt_rank", "unit_std", "none"), default="rank")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("/root/tabular/mcpfn/runs/diagnostics/rowcol_information_flow"),
    )
    return parser.parse_args()


def load_config(checkpoint: Path) -> dict:
    with (checkpoint.parent / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "model" in cfg:
        return dict(cfg["model"])
    if "config" in cfg:
        return dict(cfg["config"])
    return dict(cfg)


def build_model(spec: ModelSpec, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    config = load_config(spec.checkpoint)
    family = spec.family.lower()
    if family in {"new_stable", "stable"}:
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
        model = TabImputeModelStable(
            **{key: value for key, value in config.items() if key in allowed}
        )
    elif family in {
        "rowcol_latent_attention",
        "row_column_latent_attention",
        "bipartite_latent_attention",
    }:
        model = TabImputeModelRowColumnLatentAttention(**config)
    else:
        raise ValueError(f"Unsupported family for diagnostics: {spec.family}")

    checkpoint = torch.load(spec.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def make_low_rank(size: int, rank: int, missing_rate: float, seed: int, scale: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(size, rank))
    right = rng.normal(size=(rank, size))
    x_true = left @ right
    if scale == "sqrt_rank":
        x_true = x_true / np.sqrt(rank)
    elif scale == "rank":
        x_true = x_true / rank
    elif scale == "unit_std":
        x_true = (x_true - x_true.mean()) / (x_true.std() + 1e-16)
    elif scale != "none":
        raise ValueError(scale)
    mask = rng.random(size=x_true.shape) < missing_rate
    x_missing = x_true.astype(np.float32, copy=True)
    x_missing[mask] = np.nan
    return x_true.astype(np.float32), x_missing, mask


def normalize_like_interface(x_true: np.ndarray, x_missing: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.nanmean(x_missing, axis=0)
    stds = np.nanstd(x_missing, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    stds = np.where(np.isnan(stds), 1.0, stds)
    return (x_true - means) / (stds + 1e-16), (x_missing - means) / (stds + 1e-16), means, stds


def tensor_stats(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
    x = tensor.detach().float()
    norm = x.norm(dim=-1)
    return {
        f"{prefix}_mean": float(x.mean().cpu()),
        f"{prefix}_std": float(x.std().cpu()),
        f"{prefix}_token_norm_mean": float(norm.mean().cpu()),
        f"{prefix}_token_norm_std": float(norm.std().cpu()),
    }


def cosine_stats(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
    x = torch.nn.functional.normalize(tensor.detach().float().squeeze(0), dim=-1)
    if x.shape[0] <= 1:
        return {f"{prefix}_cos_offdiag_mean": 0.0, f"{prefix}_cos_offdiag_abs_mean": 0.0}
    sim = x @ x.T
    offdiag = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
    return {
        f"{prefix}_cos_offdiag_mean": float(offdiag.mean().cpu()),
        f"{prefix}_cos_offdiag_abs_mean": float(offdiag.abs().mean().cpu()),
        f"{prefix}_cos_offdiag_std": float(offdiag.std().cpu()),
    }


@torch.no_grad()
def latent_attention_softmax_stats(layer: LatentSelfAttentionLayer, state: torch.Tensor, prefix: str) -> dict[str, float]:
    batch_size, num_tokens, embedding_size = state.shape
    qkv = layer.qkv(layer.attn_norm(state))
    qkv = qkv.view(
        batch_size,
        num_tokens,
        3,
        layer.num_attention_heads,
        layer.head_dim,
    ).permute(2, 0, 3, 1, 4)
    query, key, _ = qkv.unbind(dim=0)
    logits = (query.float() @ key.float().transpose(-2, -1)) / math.sqrt(layer.head_dim)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(dim=-1)
    top5 = probs.topk(k=min(5, num_tokens), dim=-1).values.sum(dim=-1)
    diag = probs.diagonal(dim1=-2, dim2=-1)
    log_n = math.log(max(num_tokens, 2))
    return {
        f"{prefix}_attn_entropy_norm": float((entropy / log_n).mean().cpu()),
        f"{prefix}_attn_effective_frac": float((entropy.exp() / num_tokens).mean().cpu()),
        f"{prefix}_attn_max_prob": float(probs.max(dim=-1).values.mean().cpu()),
        f"{prefix}_attn_top5_mass": float(top5.mean().cpu()),
        f"{prefix}_attn_diag_mass": float(diag.mean().cpu()),
        f"{prefix}_attn_logit_std": float(logits.std(dim=-1).mean().cpu()),
    }


@torch.no_grad()
def output_softmax_stats(
    model: torch.nn.Module,
    criterion: FullSupportBarDistribution,
    x_true_norm: np.ndarray,
    x_missing_norm: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    src = torch.from_numpy(x_missing_norm).unsqueeze(0).to(device=device, dtype=dtype)
    decode_mask = torch.from_numpy(mask).unsqueeze(0).to(device=device)
    target = torch.from_numpy(x_true_norm).to(device=device, dtype=torch.float32)[mask]
    logits = model(src, decode_mask=decode_mask)
    logits_f = logits.float()
    probs = torch.softmax(logits_f, dim=-1)
    entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(dim=-1)
    num_bins = probs.shape[-1]
    top_values, top_indices = probs.topk(k=min(20, num_bins), dim=-1)
    target_bucket = criterion.map_to_bucket_idx(target).clamp(0, criterion.num_bars - 1)
    target_prob = probs.gather(-1, target_bucket[:, None]).squeeze(-1)
    mode_bucket = probs.argmax(dim=-1)
    mean_pred = criterion.mean(logits_f)
    median_pred = criterion.median(logits_f)
    mean_err = mean_pred - target
    median_err = median_pred - target
    in_top5 = (top_indices[:, : min(5, num_bins)] == target_bucket[:, None]).any(dim=-1).float()
    in_top20 = (top_indices == target_bucket[:, None]).any(dim=-1).float()
    return {
        "out_entropy_norm": float((entropy / math.log(num_bins)).mean().cpu()),
        "out_effective_bins": float(entropy.exp().mean().cpu()),
        "out_max_prob": float(probs.max(dim=-1).values.mean().cpu()),
        "out_top5_mass": float(top_values[:, : min(5, num_bins)].sum(dim=-1).mean().cpu()),
        "out_top20_mass": float(top_values.sum(dim=-1).mean().cpu()),
        "target_prob_mean": float(target_prob.mean().cpu()),
        "target_prob_median": float(target_prob.median().cpu()),
        "target_in_top5_frac": float(in_top5.mean().cpu()),
        "target_in_top20_frac": float(in_top20.mean().cpu()),
        "mode_target_bucket_abs_dist": float((mode_bucket - target_bucket).abs().float().mean().cpu()),
        "median_rmse_norm": float(median_err.square().mean().sqrt().cpu()),
        "mean_rmse_norm": float(mean_err.square().mean().sqrt().cpu()),
        "median_mae_norm": float(median_err.abs().mean().cpu()),
        "mean_mae_norm": float(mean_err.abs().mean().cpu()),
    }


@torch.no_grad()
def flow_stats(
    model: torch.nn.Module,
    x_missing_norm: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, float | int | str]]:
    if not isinstance(model, TabImputeModelRowColumnLatentAttention):
        return []

    src = torch.from_numpy(x_missing_norm).unsqueeze(0).to(device=device, dtype=dtype)
    rows = []
    value_emb, observed, row_counts, col_counts, row_summary, col_summary = model._encode_values(src)
    row_state, col_state = model._initial_states(
        row_summary,
        col_summary,
        row_counts,
        col_counts,
        src.shape[1],
        src.shape[2],
    )
    base = {
        "layer": -1,
        "stage": "initial",
        "row_count_mean": float(row_counts.float().mean().cpu()),
        "col_count_mean": float(col_counts.float().mean().cpu()),
        "missing_frac": float(torch.from_numpy(mask).float().mean().cpu()),
        **tensor_stats("row", row_state),
        **tensor_stats("col", col_state),
        **cosine_stats("row", row_state),
        **cosine_stats("col", col_state),
    }
    rows.append(base)

    for layer_idx, (message_layer, row_attn, col_attn) in enumerate(
        zip(model.message_layers, model.row_attention_layers, model.col_attention_layers)
    ):
        row_before, col_before = row_state, col_state
        row_after_msg, col_after_msg = message_layer(
            row_state,
            col_state,
            value_emb,
            observed,
            row_counts,
            col_counts,
        )
        row_msg_delta = (row_after_msg - row_before).float().norm(dim=-1).mean()
        col_msg_delta = (col_after_msg - col_before).float().norm(dim=-1).mean()
        rows.append(
            {
                "layer": layer_idx,
                "stage": "after_message",
                "row_delta_norm_mean": float(row_msg_delta.cpu()),
                "col_delta_norm_mean": float(col_msg_delta.cpu()),
                **tensor_stats("row", row_after_msg),
                **tensor_stats("col", col_after_msg),
                **cosine_stats("row", row_after_msg),
                **cosine_stats("col", col_after_msg),
            }
        )
        row_after_attn_stats = latent_attention_softmax_stats(row_attn, row_after_msg, "row")
        col_after_attn_stats = latent_attention_softmax_stats(col_attn, col_after_msg, "col")
        row_state = row_attn(row_after_msg)
        col_state = col_attn(col_after_msg)
        rows.append(
            {
                "layer": layer_idx,
                "stage": "after_attention",
                "row_delta_norm_mean": float((row_state - row_after_msg).float().norm(dim=-1).mean().cpu()),
                "col_delta_norm_mean": float((col_state - col_after_msg).float().norm(dim=-1).mean().cpu()),
                **row_after_attn_stats,
                **col_after_attn_stats,
                **tensor_stats("row", row_state),
                **tensor_stats("col", col_state),
                **cosine_stats("row", row_state),
                **cosine_stats("col", col_state),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], keys: list[str]) -> list[dict[str, object]]:
    out = []
    groups = sorted({(row["model"], row["size"]) for row in rows})
    for model, size in groups:
        group = [row for row in rows if row["model"] == model and row["size"] == size]
        item: dict[str, object] = {"model": model, "size": size, "n": len(group)}
        for key in keys:
            vals = np.asarray([float(row[key]) for row in group], dtype=np.float64)
            item[f"{key}_mean"] = float(vals.mean())
            item[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    borders_path = importlib.resources.files("tabimpute") / "data" / "borders.pt"
    with importlib.resources.as_file(borders_path) as path:
        criterion = FullSupportBarDistribution(
            borders=torch.load(path, map_location=device).to(device)
        )

    softmax_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []
    for spec in args.spec:
        model = build_model(spec, device, dtype)
        for size in args.sizes:
            for seed in args.seeds:
                x_true, x_missing, mask = make_low_rank(
                    size=size,
                    rank=args.rank,
                    missing_rate=args.missing_rate,
                    seed=seed,
                    scale=args.scale,
                )
                x_true_norm, x_missing_norm, _, _ = normalize_like_interface(x_true, x_missing)
                softmax_rows.append(
                    {
                        "model": spec.name,
                        "family": spec.family,
                        "size": size,
                        "seed": seed,
                        **output_softmax_stats(
                            model,
                            criterion,
                            x_true_norm.astype(np.float32),
                            x_missing_norm.astype(np.float32),
                            mask,
                            device,
                            dtype,
                        ),
                    }
                )
                if seed in set(args.flow_seeds):
                    for row in flow_stats(
                        model,
                        x_missing_norm.astype(np.float32),
                        mask,
                        device,
                        dtype,
                    ):
                        flow_rows.append(
                            {
                                "model": spec.name,
                                "family": spec.family,
                                "size": size,
                                "seed": seed,
                                **row,
                            }
                        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    softmax_path = args.output_prefix.with_name(args.output_prefix.name + "_softmax.csv")
    softmax_summary_path = args.output_prefix.with_name(args.output_prefix.name + "_softmax_summary.csv")
    flow_path = args.output_prefix.with_name(args.output_prefix.name + "_flow.csv")
    write_csv(softmax_path, softmax_rows)
    write_csv(
        softmax_summary_path,
        summarize(
            softmax_rows,
            [
                "median_rmse_norm",
                "mean_rmse_norm",
                "out_entropy_norm",
                "out_effective_bins",
                "out_max_prob",
                "out_top5_mass",
                "target_prob_mean",
                "target_in_top5_frac",
                "mode_target_bucket_abs_dist",
            ],
        ),
    )
    write_csv(flow_path, flow_rows)
    print(f"Wrote {softmax_path}")
    print(f"Wrote {softmax_summary_path}")
    print(f"Wrote {flow_path}")


if __name__ == "__main__":
    main()
