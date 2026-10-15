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
from tabimpute.model.model_rowcol_als_hybrid import (
    TabImputeModelRowColumnALSHybrid,
)
from tabimpute.model.model_rowcol_latent_attention import LatentSelfAttentionLayer


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
    parser.add_argument("--sizes", nargs="+", type=int, default=[300])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--flow-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--missing-rate", type=float, default=0.2)
    parser.add_argument(
        "--scale",
        choices=("rank", "sqrt_rank", "unit_std", "none"),
        default="rank",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("/root/tabular/mcpfn/runs/diagnostics/rowcol_als_hybrid_flow"),
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


def build_model(spec: ModelSpec, device: torch.device, dtype: torch.dtype):
    if spec.family.lower() != "rowcol_als_hybrid":
        raise ValueError(f"Unsupported family: {spec.family}")
    model = TabImputeModelRowColumnALSHybrid(**load_config(spec.checkpoint))
    checkpoint = torch.load(spec.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def make_low_rank(
    size: int,
    rank: int,
    missing_rate: float,
    seed: int,
    scale: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def normalize_like_interface(
    x_true: np.ndarray,
    x_missing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        f"{prefix}_norm_mean": float(norm.mean().cpu()),
        f"{prefix}_norm_p99": float(torch.quantile(norm.flatten(), 0.99).cpu()),
        f"{prefix}_norm_max": float(norm.max().cpu()),
    }


def cosine_stats(prefix: str, tensor: torch.Tensor) -> dict[str, float]:
    x = torch.nn.functional.normalize(tensor.detach().float().squeeze(0), dim=-1)
    if x.shape[0] <= 1:
        return {}
    sim = x @ x.T
    offdiag = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
    return {
        f"{prefix}_cos_mean": float(offdiag.mean().cpu()),
        f"{prefix}_cos_abs_mean": float(offdiag.abs().mean().cpu()),
        f"{prefix}_cos_std": float(offdiag.std().cpu()),
    }


@torch.no_grad()
def latent_attention_softmax_stats(
    layer: LatentSelfAttentionLayer,
    state: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
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


def summarize_vector(prefix: str, x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().flatten()
    return {
        f"{prefix}_mean": float(x.mean().cpu()),
        f"{prefix}_std": float(x.std().cpu()),
        f"{prefix}_p50": float(torch.quantile(x, 0.50).cpu()),
        f"{prefix}_p95": float(torch.quantile(x, 0.95).cpu()),
        f"{prefix}_p99": float(torch.quantile(x, 0.99).cpu()),
        f"{prefix}_max": float(x.max().cpu()),
    }


@torch.no_grad()
def output_stats(
    model: TabImputeModelRowColumnALSHybrid,
    criterion: FullSupportBarDistribution,
    src: torch.Tensor,
    target: torch.Tensor,
    decode_mask: torch.Tensor,
) -> dict[str, float]:
    target_missing = target[decode_mask.squeeze(0)]
    logits = model(src, decode_mask=decode_mask).float()
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(dim=-1)
    num_bins = probs.shape[-1]
    top_values, top_indices = probs.topk(k=min(20, num_bins), dim=-1)
    target_bucket = criterion.map_to_bucket_idx(target_missing).clamp(0, criterion.num_bars - 1)
    target_prob = probs.gather(-1, target_bucket[:, None]).squeeze(-1)
    mode_bucket = probs.argmax(dim=-1)
    mean_pred = criterion.mean(logits)
    median_pred = criterion.median(logits)
    mean_err = mean_pred - target_missing
    median_err = median_pred - target_missing
    in_top5 = (top_indices[:, : min(5, num_bins)] == target_bucket[:, None]).any(dim=-1).float()
    in_top20 = (top_indices == target_bucket[:, None]).any(dim=-1).float()
    max_prob = probs.max(dim=-1).values
    return {
        "out_entropy_norm": float((entropy / math.log(num_bins)).mean().cpu()),
        "out_effective_bins": float(entropy.exp().mean().cpu()),
        "out_max_prob_mean": float(max_prob.mean().cpu()),
        "out_max_prob_p99": float(torch.quantile(max_prob, 0.99).cpu()),
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
        **summarize_vector("target_prob", target_prob),
    }


@torch.no_grad()
def als_trace_stats(
    model: TabImputeModelRowColumnALSHybrid,
    src: torch.Tensor,
    target: torch.Tensor,
    decode_mask: torch.Tensor,
) -> list[dict[str, float | int | str]]:
    clean = torch.where(torch.isnan(src), torch.zeros_like(src), src).float()
    observed = (~torch.isnan(src)).float()
    row_state, col_state = model.encode_latents(src)
    row_factors, col_factors, row_bias, col_bias = model._initial_factors(row_state, col_state)
    target_missing = target[decode_mask.squeeze(0)].float()
    eye = torch.eye(model.als_rank, device=src.device, dtype=torch.float32)
    rows: list[dict[str, float | int | str]] = []

    def point_stats(step: int, stage: str, row_f: torch.Tensor, col_f: torch.Tensor) -> None:
        point = torch.einsum("brk,bck->brc", row_f, col_f)
        point = model.prediction_scale.float() * point + model.prediction_bias.float()
        point = point + model.bias_scale.float() * (row_bias.unsqueeze(2) + col_bias.unsqueeze(1))
        pred_missing = point[decode_mask].float()
        err = pred_missing - target_missing
        rows.append(
            {
                "stage": stage,
                "als_step": step,
                "point_rmse_norm": float(err.square().mean().sqrt().cpu()),
                "point_mae_norm": float(err.abs().mean().cpu()),
                "prediction_scale": float(model.prediction_scale.detach().float().cpu()),
                "prediction_bias": float(model.prediction_bias.detach().float().cpu()),
                "bias_scale": float(model.bias_scale.detach().float().cpu()),
                **tensor_stats("row_factor", row_f),
                **tensor_stats("col_factor", col_f),
            }
        )

    def gram_stats(step: int, stage: str, row_f: torch.Tensor, col_f: torch.Tensor) -> None:
        row_system = torch.einsum("brc,bck,bcl->brkl", observed, col_f.float(), col_f.float())
        col_system = torch.einsum("brc,brk,brl->bckl", observed, row_f.float(), row_f.float())
        row_obs = observed.sum(dim=2).clamp_min(1.0)
        col_obs = observed.sum(dim=1).clamp_min(1.0)
        row_ridge = model.als_ridge * (
            row_obs if model.als_ridge_scale_by_obs else torch.ones_like(row_obs)
        )
        col_ridge = model.als_ridge * (
            col_obs if model.als_ridge_scale_by_obs else torch.ones_like(col_obs)
        )
        row_system = row_system + row_ridge.view(*row_ridge.shape, 1, 1) * eye.view(1, 1, model.als_rank, model.als_rank)
        col_system = col_system + col_ridge.view(*col_ridge.shape, 1, 1) * eye.view(1, 1, model.als_rank, model.als_rank)
        if model.als_jitter > 0.0:
            row_system = row_system + model.als_jitter * eye.view(1, 1, model.als_rank, model.als_rank)
            col_system = col_system + model.als_jitter * eye.view(1, 1, model.als_rank, model.als_rank)
        row_eigs = torch.linalg.eigvalsh(0.5 * (row_system + row_system.transpose(-1, -2))).clamp_min(1e-12)
        col_eigs = torch.linalg.eigvalsh(0.5 * (col_system + col_system.transpose(-1, -2))).clamp_min(1e-12)
        row_cond = row_eigs[..., -1] / row_eigs[..., 0]
        col_cond = col_eigs[..., -1] / col_eigs[..., 0]
        rows.append(
            {
                "stage": stage,
                "als_step": step,
                **summarize_vector("row_cond", row_cond),
                **summarize_vector("col_cond", col_cond),
                **summarize_vector("row_min_eig", row_eigs[..., 0]),
                **summarize_vector("col_min_eig", col_eigs[..., 0]),
            }
        )

    point_stats(0, "initial", row_factors, col_factors)
    gram_stats(0, "pre_row_update", row_factors, col_factors)
    for step in range(1, model.als_steps + 1):
        row_system = torch.einsum("brc,bck,bcl->brkl", observed, col_factors.float(), col_factors.float())
        row_rhs = torch.einsum("brc,brc,bck->brk", observed, clean, col_factors.float())
        row_obs = observed.sum(dim=2).clamp_min(1.0)
        row_ridge = model.als_ridge * (
            row_obs if model.als_ridge_scale_by_obs else torch.ones_like(row_obs)
        )
        row_system = row_system + row_ridge.view(*row_ridge.shape, 1, 1) * eye.view(1, 1, model.als_rank, model.als_rank)
        row_solved = model._solve_regularized_system(row_system, row_rhs)
        row_factors = (1.0 - model.als_update_damping) * row_factors.float() + model.als_update_damping * row_solved
        row_factors = model._apply_factor_norm_clip(row_factors)

        col_system = torch.einsum("brc,brk,brl->bckl", observed, row_factors.float(), row_factors.float())
        col_rhs = torch.einsum("brc,brc,brk->bck", observed, clean, row_factors.float())
        col_obs = observed.sum(dim=1).clamp_min(1.0)
        col_ridge = model.als_ridge * (
            col_obs if model.als_ridge_scale_by_obs else torch.ones_like(col_obs)
        )
        col_system = col_system + col_ridge.view(*col_ridge.shape, 1, 1) * eye.view(1, 1, model.als_rank, model.als_rank)
        col_solved = model._solve_regularized_system(col_system, col_rhs)
        col_factors = (1.0 - model.als_update_damping) * col_factors.float() + model.als_update_damping * col_solved
        col_factors = model._apply_factor_norm_clip(col_factors)

        point_stats(step, "after_als_step", row_factors, col_factors)
        gram_stats(step, "pre_next_update", row_factors, col_factors)
    return rows


@torch.no_grad()
def encoder_flow_stats(
    model: TabImputeModelRowColumnALSHybrid,
    src: torch.Tensor,
    missing_frac: float,
) -> list[dict[str, float | int | str]]:
    value_emb, observed, row_counts, col_counts, row_summary, col_summary = model._encode_values(src)
    clean_values = torch.where(torch.isnan(src), torch.zeros_like(src), src)
    row_state, col_state = model._initial_states(
        row_summary,
        col_summary,
        row_counts,
        col_counts,
        src.shape[1],
        src.shape[2],
    )
    rows: list[dict[str, float | int | str]] = [
        {
            "layer": -1,
            "stage": "initial",
            "missing_frac": missing_frac,
            "row_count_mean": float(row_counts.float().mean().cpu()),
            "col_count_mean": float(col_counts.float().mean().cpu()),
            **tensor_stats("row", row_state),
            **tensor_stats("col", col_state),
            **cosine_stats("row", row_state),
            **cosine_stats("col", col_state),
        }
    ]
    for layer_idx, (message_layer, row_attn, col_attn) in enumerate(
        zip(model.message_layers, model.row_attention_layers, model.col_attention_layers)
    ):
        row_before, col_before = row_state, col_state
        row_after_msg, col_after_msg = model._run_layer(
            message_layer,
            row_state,
            col_state,
            value_emb,
            observed,
            row_counts,
            col_counts,
            clean_values,
        )
        rows.append(
            {
                "layer": layer_idx,
                "stage": "after_message",
                "row_delta_norm_mean": float((row_after_msg - row_before).float().norm(dim=-1).mean().cpu()),
                "col_delta_norm_mean": float((col_after_msg - col_before).float().norm(dim=-1).mean().cpu()),
                **tensor_stats("row", row_after_msg),
                **tensor_stats("col", col_after_msg),
                **cosine_stats("row", row_after_msg),
                **cosine_stats("col", col_after_msg),
            }
        )
        row_attn_stats = latent_attention_softmax_stats(row_attn, row_after_msg, "row")
        col_attn_stats = latent_attention_softmax_stats(col_attn, col_after_msg, "col")
        row_state = model._run_latent_attention(row_attn, row_after_msg)
        col_state = model._run_latent_attention(col_attn, col_after_msg)
        rows.append(
            {
                "layer": layer_idx,
                "stage": "after_attention",
                "row_delta_norm_mean": float((row_state - row_after_msg).float().norm(dim=-1).mean().cpu()),
                "col_delta_norm_mean": float((col_state - col_after_msg).float().norm(dim=-1).mean().cpu()),
                **row_attn_stats,
                **col_attn_stats,
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
            vals = [float(row[key]) for row in group if key in row]
            if not vals:
                continue
            vals_arr = np.asarray(vals, dtype=np.float64)
            item[f"{key}_mean"] = float(vals_arr.mean())
            item[f"{key}_std"] = float(vals_arr.std(ddof=1)) if len(vals_arr) > 1 else 0.0
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
    als_rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []
    sigma_rows: list[dict[str, object]] = []
    flow_seed_set = set(args.flow_seeds)
    for spec in args.spec:
        model = build_model(spec, device, dtype)
        sigma = torch.nn.functional.softplus(model.logit_sigma_unconstrained.float())
        sigma = float((sigma + model.logit_sigma_min).detach().cpu())
        sigma_rows.append(
            {
                "model": spec.name,
                "sigma": sigma,
                "als_steps": model.als_steps,
                "als_ridge": model.als_ridge,
                "als_damping": model.als_update_damping,
                "prediction_scale": float(model.prediction_scale.detach().float().cpu()),
                "prediction_bias": float(model.prediction_bias.detach().float().cpu()),
                "bias_scale": float(model.bias_scale.detach().float().cpu()),
                "use_neural_residual": model.use_neural_residual,
                "neural_residual_scale": (
                    None if model.neural_residual_scale is None else float(model.neural_residual_scale.detach().float().cpu()[0])
                ),
            }
        )
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
                src = torch.from_numpy(x_missing_norm.astype(np.float32)).unsqueeze(0).to(device=device, dtype=dtype)
                target = torch.from_numpy(x_true_norm.astype(np.float32)).to(device=device, dtype=torch.float32)
                decode_mask = torch.from_numpy(mask).unsqueeze(0).to(device=device)
                softmax_rows.append(
                    {
                        "model": spec.name,
                        "family": spec.family,
                        "size": size,
                        "seed": seed,
                        **output_stats(model, criterion, src, target, decode_mask),
                    }
                )
                if seed in flow_seed_set:
                    for row in als_trace_stats(model, src, target, decode_mask):
                        als_rows.append(
                            {
                                "model": spec.name,
                                "family": spec.family,
                                "size": size,
                                "seed": seed,
                                **row,
                            }
                        )
                    for row in encoder_flow_stats(model, src, float(mask.mean())):
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

    paths = {
        "softmax": args.output_prefix.with_name(args.output_prefix.name + "_softmax.csv"),
        "softmax_summary": args.output_prefix.with_name(args.output_prefix.name + "_softmax_summary.csv"),
        "als": args.output_prefix.with_name(args.output_prefix.name + "_als.csv"),
        "flow": args.output_prefix.with_name(args.output_prefix.name + "_flow.csv"),
        "sigma": args.output_prefix.with_name(args.output_prefix.name + "_sigma.csv"),
    }
    write_csv(paths["softmax"], softmax_rows)
    write_csv(
        paths["softmax_summary"],
        summarize(
            softmax_rows,
            [
                "median_rmse_norm",
                "mean_rmse_norm",
                "out_entropy_norm",
                "out_effective_bins",
                "out_max_prob_mean",
                "out_top5_mass",
                "target_prob_mean",
                "target_in_top5_frac",
                "target_in_top20_frac",
                "mode_target_bucket_abs_dist",
            ],
        ),
    )
    write_csv(paths["als"], als_rows)
    write_csv(paths["flow"], flow_rows)
    write_csv(paths["sigma"], sigma_rows)
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")
    print("\nSoftmax summary")
    for row in summarize(
        softmax_rows,
        [
            "median_rmse_norm",
            "mean_rmse_norm",
            "out_entropy_norm",
            "out_effective_bins",
            "out_max_prob_mean",
            "target_prob_mean",
            "target_in_top5_frac",
            "target_in_top20_frac",
        ],
    ):
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
