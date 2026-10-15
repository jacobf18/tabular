#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import importlib.resources
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
BENCH_CODE = REPO.parent / "benchmark-selection" / "code"
if str(BENCH_CODE) not in sys.path:
    sys.path.insert(0, str(BENCH_CODE))

from cv_split import cv_folds, cv_train_val  # noqa: E402
from greedy_select import greedy_mi  # noqa: E402
from tabimpute.model.bar_distribution import FullSupportBarDistribution  # noqa: E402
from tabimpute.model.model_new_stable import (  # noqa: E402
    DecoderStable,
    RMSNorm,
    TabImputeModelStable,
)
from tabimpute.model.model_rowcol_latent import (  # noqa: E402
    BipartiteMessagePassingLayer,
    FeedForward,
)
from tabimpute.model.model_rowcol_latent_attention import (  # noqa: E402
    LatentSelfAttentionLayer,
    TabImputeModelRowColumnLatentAttention,
)
from tabimpute.model.model_rowcol_svt_hybrid import (  # noqa: E402
    TabImputeModelRowColumnSVTHybrid,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path
    family: str
    config: Path | None = None


def parse_spec(value: str) -> ModelSpec:
    parts = [part.strip() for part in value.split("|")]
    if len(parts) not in {3, 4}:
        raise ValueError(
            "--spec must be 'name|checkpoint|family' or "
            "'name|checkpoint|family|config_json'"
        )
    checkpoint = Path(parts[1]).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    config = None
    if len(parts) == 4 and parts[3]:
        config = Path(parts[3]).expanduser()
        if not config.exists():
            raise FileNotFoundError(config)
    return ModelSpec(parts[0], checkpoint, parts[2], config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", action="append", required=True, type=parse_spec)
    parser.add_argument("--dataset", default="mteb")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--k-selected", type=int, default=5)
    parser.add_argument("--holdout-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-targets", type=int, default=1536)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision-mode", choices=("full_bfloat16", "amp_bfloat16", "amp_float16", "float32"), default="full_bfloat16")
    parser.add_argument("--run-label", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "runs" / "information_flow_diagnostics",
    )
    return parser.parse_args()


def load_run_config(spec: ModelSpec) -> dict[str, Any]:
    cfg_path = spec.config or (spec.checkpoint.parent / "config.json")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "model" in cfg:
        model = dict(cfg["model"])
    elif "config" in cfg:
        model = dict(cfg["config"])
    else:
        model = dict(cfg)
    return {"raw": cfg, "model": model, "config_path": str(cfg_path)}


def build_model(spec: ModelSpec, device: torch.device, dtype: torch.dtype) -> nn.Module:
    run_cfg = load_run_config(spec)
    config = run_cfg["model"]
    family = spec.family.lower()
    if family in {"new_stable", "stable", "dwip"}:
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
    elif family in {"rowcol_latent_attention", "row_column_latent_attention"}:
        model = TabImputeModelRowColumnLatentAttention(**config)
    elif family in {"rowcol_svt_hybrid", "row_column_svt_hybrid"}:
        model = TabImputeModelRowColumnSVTHybrid(**config)
    else:
        raise ValueError(f"Unsupported model family: {spec.family}")

    checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device=device, dtype=dtype)

    raw_cfg = run_cfg["raw"]
    fp32_substrings = tuple(raw_cfg.get("fp32_param_substrings") or ())
    for name, parameter in model.named_parameters():
        if any(substring in name for substring in fp32_substrings):
            parameter.data = parameter.data.float()

    model.eval()
    return model


def estimate_corr_pairwise(B_train: np.ndarray, O_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    col_count = np.maximum(O_train.sum(axis=0), 1.0)
    col_mean = (B_train * O_train).sum(axis=0) / col_count
    B_c = O_train * (B_train - col_mean[None, :])
    col_var = (B_c**2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    col_std = np.sqrt(np.maximum(col_var, 1e-12))
    B_std = O_train * (B_train - col_mean[None, :]) / col_std[None, :]
    gram = B_std.T @ B_std
    pair_counts = O_train.T @ O_train
    R = gram / np.maximum(pair_counts - 1.0, 1.0)
    R = np.nan_to_num(0.5 * (R + R.T), nan=0.0, posinf=0.0, neginf=0.0)
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-6)
    R = (eigvecs * eigvals[None, :]) @ eigvecs.T
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(R), 1e-12))
    return R * np.outer(d_inv, d_inv), col_mean, col_std


def build_mteb_probe(
    dataset: str,
    fold_idx: int,
    k_selected: int,
    holdout_frac: float,
    seed: int,
    max_targets: int,
) -> dict[str, Any]:
    path = REPO.parent / "benchmark-selection" / "data" / f"{dataset}.matrix.npz"
    d = np.load(path, allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    names = d["benchmark_names"]
    folds = cv_folds(B.shape[0], 10, seed)
    train_idx, val_idx = cv_train_val(folds, fold_idx, holdout_frac, B.shape[0], seed)
    B_train, O_train = B[train_idx], O[train_idx]
    B_test, O_test = B[val_idx], O[val_idx]
    R, col_mean, col_std = estimate_corr_pairwise(B_train, O_train)
    selected = list(greedy_mi(R, max(k_selected, 1), benchmark_names=names)["selected"][:k_selected])

    X_ctx = np.where(O_train == 1, B_train, np.nan)
    X_test = np.full_like(B_test, np.nan, dtype=np.float64)
    for col in selected:
        X_test[O_test[:, col] == 1, col] = B_test[O_test[:, col] == 1, col]
    X = np.vstack([X_ctx, X_test])

    eval_mask = np.zeros_like(X, dtype=bool)
    test_offset = X_ctx.shape[0]
    for local_row in range(B_test.shape[0]):
        for col in range(B.shape[1]):
            if col not in selected and O_test[local_row, col] == 1:
                eval_mask[test_offset + local_row, col] = True

    target_positions = np.argwhere(eval_mask)
    if len(target_positions) > max_targets:
        rng = np.random.default_rng(seed + 1009 * (fold_idx + 1))
        keep = rng.choice(len(target_positions), size=max_targets, replace=False)
        eval_mask[...] = False
        target_positions = target_positions[keep]
        eval_mask[target_positions[:, 0], target_positions[:, 1]] = True

    col_std = np.where(np.isnan(col_std) | (col_std < 1e-8), 1.0, col_std)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    X_norm = (X - col_mean[None, :]) / (col_std[None, :] + 1e-16)
    target_full = np.vstack([B_train, B_test])
    target_norm = (target_full - col_mean[None, :]) / (col_std[None, :] + 1e-16)

    return {
        "dataset": dataset,
        "path": str(path),
        "shape": list(X.shape),
        "train_rows": int(X_ctx.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "selected": [int(x) for x in selected],
        "selected_names": [str(names[x]) for x in selected],
        "num_targets": int(eval_mask.sum()),
        "observed_fraction_input": float(np.isfinite(X).mean()),
        "X_norm": X_norm.astype(np.float32),
        "target_norm": target_norm.astype(np.float32),
        "eval_mask": eval_mask,
    }


def tensor_summary(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    x = tensor.detach()
    xf = x.float()
    finite = torch.isfinite(xf)
    if bool(finite.any()):
        vals = xf[finite]
        abs_vals = vals.abs()
        return {
            "name": name,
            "shape": "x".join(str(v) for v in x.shape),
            "dtype": str(x.dtype).replace("torch.", ""),
            "finite_frac": float(finite.float().mean().cpu()),
            "mean": float(vals.mean().cpu()),
            "std": float(vals.std(unbiased=False).cpu()),
            "rms": float(vals.square().mean().sqrt().cpu()),
            "abs_mean": float(abs_vals.mean().cpu()),
            "abs_max": float(abs_vals.max().cpu()),
            "near_zero_frac": float((abs_vals < 1e-7).float().mean().cpu()),
            "abs_gt_10_frac": float((abs_vals > 10.0).float().mean().cpu()),
            "abs_gt_50_frac": float((abs_vals > 50.0).float().mean().cpu()),
        }
    return {
        "name": name,
        "shape": "x".join(str(v) for v in x.shape),
        "dtype": str(x.dtype).replace("torch.", ""),
        "finite_frac": 0.0,
    }


def summarize_output(name: str, output: Any) -> list[dict[str, Any]]:
    if torch.is_tensor(output):
        return [tensor_summary(name, output)]
    if isinstance(output, (tuple, list)):
        rows = []
        for idx, item in enumerate(output):
            if torch.is_tensor(item):
                rows.append(tensor_summary(f"{name}[{idx}]", item))
        return rows
    if isinstance(output, dict):
        return [
            tensor_summary(f"{name}.{key}", value)
            for key, value in output.items()
            if torch.is_tensor(value)
        ]
    return []


def wants_hook(name: str, module: nn.Module) -> bool:
    interesting_types = (
        nn.Linear,
        RMSNorm,
        DecoderStable,
        FeedForward,
        BipartiteMessagePassingLayer,
        LatentSelfAttentionLayer,
        TabImputeModelStable,
        TabImputeModelRowColumnLatentAttention,
        TabImputeModelRowColumnSVTHybrid,
    )
    if isinstance(module, interesting_types):
        return True
    return name in {
        "feature_encoder",
        "output_decoder",
        "svt_feature_decoder_proj",
        "observed_attention_scalar_proj",
        "observed_attention_context_proj",
        "empirical_bayes_feature_proj",
        "svt_mixture_controller",
        "svt_threshold_controller",
    }


def collect_activations(model: nn.Module) -> tuple[list[dict[str, Any]], list[Any]]:
    rows: list[dict[str, Any]] = []
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if getattr(model, "_in_cv_encoding", False):
                return
            for row in summarize_output(name, output):
                row["module"] = name
                rows.append(row)

        return hook

    for name, module in model.named_modules():
        if name and wants_hook(name, module):
            handles.append(module.register_forward_hook(make_hook(name)))
    return rows, handles


def parameter_dtype_rows(model_name: str, model: nn.Module) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name, parameter in model.named_parameters():
        group = name.split(".", 1)[0]
        grouped[group][str(parameter.dtype).replace("torch.", "")] += parameter.numel()
    return [
        {"model": model_name, "group": group, "dtype": dtype, "num_parameters": count}
        for group, dtypes in grouped.items()
        for dtype, count in dtypes.items()
    ]


def grad_group(name: str) -> str:
    parts = name.split(".")
    if parts[0] in {"transformer_blocks", "message_layers", "row_attention_layers", "col_attention_layers"} and len(parts) > 1:
        return f"{parts[0]}.{parts[1]}"
    if parts[0].startswith("svt") or parts[0].startswith("observed") or parts[0].startswith("empirical"):
        return parts[0]
    if parts[0] == "output_decoder" and len(parts) > 1:
        return ".".join(parts[:2])
    return parts[0]


def gradient_rows(model_name: str, model: nn.Module) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, float]] = defaultdict(lambda: {"sq": 0.0, "max": 0.0, "n": 0.0, "params": 0.0})
    rows = []
    for name, parameter in model.named_parameters():
        group = grad_group(name)
        n_params = float(parameter.numel())
        if parameter.grad is None:
            rows.append(
                {
                    "model": model_name,
                    "name": name,
                    "group": group,
                    "grad_l2": 0.0,
                    "grad_rms": 0.0,
                    "grad_abs_max": 0.0,
                    "has_grad": False,
                    "num_parameters": int(n_params),
                }
            )
            groups[group]["params"] += n_params
            continue
        grad = parameter.grad.detach().float()
        grad_l2 = float(grad.norm().cpu())
        grad_abs_max = float(grad.abs().max().cpu())
        rows.append(
            {
                "model": model_name,
                "name": name,
                "group": group,
                "grad_l2": grad_l2,
                "grad_rms": float(grad.square().mean().sqrt().cpu()),
                "grad_abs_max": grad_abs_max,
                "has_grad": True,
                "num_parameters": int(n_params),
            }
        )
        groups[group]["sq"] += grad_l2**2
        groups[group]["max"] = max(groups[group]["max"], grad_abs_max)
        groups[group]["n"] += 1.0
        groups[group]["params"] += n_params
    rows.extend(
        {
            "model": model_name,
            "name": f"GROUP::{group}",
            "group": group,
            "grad_l2": math.sqrt(values["sq"]),
            "grad_rms": math.sqrt(values["sq"] / max(values["params"], 1.0)),
            "grad_abs_max": values["max"],
            "has_grad": values["n"] > 0,
            "num_parameters": int(values["params"]),
        }
        for group, values in groups.items()
    )
    return rows


def output_distribution_stats(
    logits: torch.Tensor,
    target: torch.Tensor,
    criterion: FullSupportBarDistribution,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    logits_f = logits.float()
    probs = torch.softmax(logits_f, dim=-1)
    entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(dim=-1)
    top_values, top_idx = probs.topk(k=20, dim=-1)
    target_bucket = criterion.map_to_bucket_idx(target).clamp(0, criterion.num_bars - 1)
    target_prob = probs.gather(-1, target_bucket[:, None]).squeeze(-1)
    mean_pred = criterion.mean(logits_f)
    median_pred = criterion.median(logits_f)
    mode_bucket = probs.argmax(dim=-1)
    mean_err = mean_pred - target
    median_err = median_pred - target
    in_top5 = (top_idx[:, :5] == target_bucket[:, None]).any(dim=-1).float()
    in_top20 = (top_idx == target_bucket[:, None]).any(dim=-1).float()
    stats = {
        "nll": float(criterion(logits_f, target).mean().detach().cpu()),
        "mean_rmse_norm": float(mean_err.square().mean().sqrt().detach().cpu()),
        "median_rmse_norm": float(median_err.square().mean().sqrt().detach().cpu()),
        "mean_mae_norm": float(mean_err.abs().mean().detach().cpu()),
        "median_mae_norm": float(median_err.abs().mean().detach().cpu()),
        "logit_std": float(logits_f.std(dim=-1).mean().detach().cpu()),
        "logit_abs_max": float(logits_f.abs().max().detach().cpu()),
        "prob_entropy_norm": float((entropy / math.log(probs.shape[-1])).mean().detach().cpu()),
        "effective_bins": float(entropy.exp().mean().detach().cpu()),
        "max_prob": float(probs.max(dim=-1).values.mean().detach().cpu()),
        "top5_mass": float(top_values[:, :5].sum(dim=-1).mean().detach().cpu()),
        "top20_mass": float(top_values.sum(dim=-1).mean().detach().cpu()),
        "target_prob_mean": float(target_prob.mean().detach().cpu()),
        "target_prob_median": float(target_prob.median().detach().cpu()),
        "target_in_top5_frac": float(in_top5.mean().detach().cpu()),
        "target_in_top20_frac": float(in_top20.mean().detach().cpu()),
        "mode_target_bucket_abs_dist": float((mode_bucket - target_bucket).abs().float().mean().detach().cpu()),
    }
    arrays = {
        "target_prob": target_prob.detach().cpu().numpy(),
        "max_prob": probs.max(dim=-1).values.detach().cpu().numpy(),
        "entropy_norm": (entropy / math.log(probs.shape[-1])).detach().cpu().numpy(),
        "mean_error": mean_err.detach().cpu().numpy(),
        "median_error": median_err.detach().cpu().numpy(),
    }
    return stats, arrays


def attention_rows(model_name: str, diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in diagnostics:
        layer = item.get("layer")
        for axis in ("features", "datapoints", "rows", "cols"):
            if axis not in item:
                continue
            stats = item[axis].get("stats")
            if not stats:
                continue
            rows.append({"model": model_name, "layer": layer, "axis": axis, **stats})
    return rows


def svt_state_rows(model_name: str, model: nn.Module) -> list[dict[str, Any]]:
    if not isinstance(model, TabImputeModelRowColumnSVTHybrid):
        return []
    rows = []
    attrs = {
        "svt_mixture_weights": model.last_svt_mixture_weights,
        "svt_mixture_cv_rmse": model.last_svt_mixture_cv_rmse,
        "svt_mixture_controller_logits": model.last_svt_mixture_controller_logits,
        "svt_adaptive_threshold_scale": model.last_svt_adaptive_threshold_scale,
        "observed_attention_row_support": model.last_observed_attention_row_support,
        "observed_attention_col_support": model.last_observed_attention_col_support,
        "observed_attention_temperature": model.last_observed_attention_temperature,
        "empirical_bayes_ridge": model.last_empirical_bayes_ridge,
        "empirical_bayes_shrinkage": model.last_empirical_bayes_shrinkage,
        "empirical_bayes_row_observed_fraction": model.last_empirical_bayes_row_observed_fraction,
        "empirical_bayes_col_observed_fraction": model.last_empirical_bayes_col_observed_fraction,
    }
    for name, value in attrs.items():
        if torch.is_tensor(value):
            row = tensor_summary(name, value)
            row["model"] = model_name
            rows.append(row)
    snapshots = getattr(model, "last_svt_mixture_snapshot_points", None)
    if torch.is_tensor(snapshots) and snapshots.numel() > 0:
        row = tensor_summary("svt_mixture_snapshot_points", snapshots)
        row["model"] = model_name
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_attention(out_dir: Path, model_name: str, diagnostics: list[dict[str, Any]]) -> None:
    rows = attention_rows(model_name, diagnostics)
    if rows:
        by_axis = defaultdict(list)
        for row in rows:
            by_axis[row["axis"]].append(row)
        fig, axes = plt.subplots(1, len(by_axis), figsize=(5 * len(by_axis), 4), squeeze=False)
        for ax, (axis, axis_rows) in zip(axes[0], by_axis.items()):
            axis_rows = sorted(axis_rows, key=lambda x: x["layer"])
            ax.plot([r["layer"] for r in axis_rows], [r["normalized_entropy_mean"] for r in axis_rows], label="entropy")
            ax.plot([r["layer"] for r in axis_rows], [r["max_prob_mean"] for r in axis_rows], label="max prob")
            ax.plot([r["layer"] for r in axis_rows], [r["top5_mass_mean"] for r in axis_rows], label="top5")
            ax.set_title(axis)
            ax.set_xlabel("layer")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"{model_name}: attention concentration")
        fig.tight_layout()
        fig.savefig(out_dir / f"{model_name}_attention_by_layer.png", dpi=180)
        plt.close(fig)

    for item in diagnostics:
        layer = int(item.get("layer", 0))
        if layer not in {0, len(diagnostics) // 2, len(diagnostics) - 1}:
            continue
        for axis in ("features", "datapoints", "rows", "cols"):
            if axis not in item:
                continue
            weights = item[axis].get("weights")
            if not torch.is_tensor(weights):
                continue
            w = weights.float()
            if w.ndim == 4:
                matrix = w.mean(dim=(0, 1)).numpy()
            elif w.ndim == 5:
                matrix = w.mean(dim=(0, 1, 2)).numpy()
            else:
                continue
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(matrix, vmin=0.0, vmax=np.quantile(matrix, 0.995), cmap="magma")
            ax.set_title(f"{model_name} layer {layer} {axis}")
            ax.set_xlabel("key")
            ax.set_ylabel("query")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(out_dir / f"{model_name}_layer{layer:02d}_{axis}_attention_heatmap.png", dpi=180)
            plt.close(fig)


def plot_bars(out_dir: Path, rows: list[dict[str, Any]], metric: str, filename: str, title: str, top_n: int = 40) -> None:
    data = [r for r in rows if str(r.get("name", "")).startswith("GROUP::")]
    if not data:
        data = rows
    data = sorted(data, key=lambda r: float(r.get(metric, 0.0)), reverse=True)[:top_n]
    if not data:
        return
    labels = [str(r["name"]).replace("GROUP::", "") for r in data]
    values = [float(r.get(metric, 0.0)) for r in data]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(data))))
    ax.barh(range(len(data)), values)
    ax.set_yticks(range(len(data)), labels)
    ax.invert_yaxis()
    ax.set_xscale("log" if max(values) > 0 else "linear")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def plot_output_hists(out_dir: Path, arrays_by_model: dict[str, dict[str, np.ndarray]]) -> None:
    for key in ("target_prob", "max_prob", "entropy_norm", "mean_error"):
        fig, ax = plt.subplots(figsize=(7, 4))
        for model_name, arrays in arrays_by_model.items():
            values = arrays[key]
            ax.hist(values, bins=40, alpha=0.45, density=True, label=model_name)
        ax.set_title(key)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"output_{key}_hist.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir / f"{args.dataset}_fold{args.fold}_k{args.k_selected}"
    if args.run_label:
        out_dir = out_dir / args.run_label
        if out_dir.exists():
            raise FileExistsError(f"Diagnostic output already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.precision_mode == "full_bfloat16" and device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)

    borders_path = importlib.resources.files("tabimpute") / "data" / "borders.pt"
    with importlib.resources.as_file(borders_path) as path:
        criterion = FullSupportBarDistribution(torch.load(path, map_location=device).to(device))

    probe = build_mteb_probe(
        dataset=args.dataset,
        fold_idx=args.fold,
        k_selected=args.k_selected,
        holdout_frac=args.holdout_frac,
        seed=args.seed,
        max_targets=args.max_targets,
    )
    border_min = float(criterion.borders[0].detach().cpu()) + 1e-5
    border_max = float(criterion.borders[-1].detach().cpu()) - 1e-5
    target_values = probe["target_norm"][probe["eval_mask"]]
    target_values = np.clip(target_values, border_min, border_max)

    src = torch.from_numpy(probe["X_norm"]).unsqueeze(0).to(device=device, dtype=dtype)
    decode_mask = torch.from_numpy(probe["eval_mask"]).unsqueeze(0).to(device=device)
    target = torch.from_numpy(target_values.astype(np.float32)).to(device=device)

    summary: list[dict[str, Any]] = []
    all_attention_rows: list[dict[str, Any]] = []
    all_activation_rows: list[dict[str, Any]] = []
    all_grad_rows: list[dict[str, Any]] = []
    all_dtype_rows: list[dict[str, Any]] = []
    all_svt_rows: list[dict[str, Any]] = []
    output_arrays: dict[str, dict[str, np.ndarray]] = {}

    manifest = {
        "precision_mode": args.precision_mode,
        "probe": {k: v for k, v in probe.items() if k not in {"X_norm", "target_norm", "eval_mask"}},
        "specs": [
            {
                "name": spec.name,
                "checkpoint": str(spec.checkpoint),
                "family": spec.family,
                "config": str(spec.config) if spec.config else str(spec.checkpoint.parent / "config.json"),
            }
            for spec in args.spec
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for spec in args.spec:
        print(f"[diagnostic] loading {spec.name}", flush=True)
        model = build_model(spec, device, dtype)
        all_dtype_rows.extend(parameter_dtype_rows(spec.name, model))
        activation_rows, handles = collect_activations(model)
        if hasattr(model, "set_attention_capture"):
            model.set_attention_capture(True, store_weights=True)
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,
                            dtype=torch.float16 if args.precision_mode == "amp_float16" else torch.bfloat16,
                            enabled=args.precision_mode.startswith("amp_")):
            logits = model(src, decode_mask=decode_mask)
        from tabimpute.model.prediction_distribution import prediction_distribution
        output_criterion = prediction_distribution(model, criterion)
        out_stats, arrays = output_distribution_stats(logits, target, output_criterion)
        loss = output_criterion(logits.float(), target).mean()
        loss.backward()
        for handle in handles:
            handle.remove()
        diagnostics = model.get_attention_diagnostics() if hasattr(model, "get_attention_diagnostics") else []
        if hasattr(model, "set_attention_capture"):
            model.set_attention_capture(False)

        model_activation_rows = [
            {"model": spec.name, "order": idx, **row}
            for idx, row in enumerate(activation_rows)
        ]
        all_activation_rows.extend(model_activation_rows)
        grad_rows = gradient_rows(spec.name, model)
        all_grad_rows.extend(grad_rows)
        attn_rows = attention_rows(spec.name, diagnostics)
        all_attention_rows.extend(attn_rows)
        all_svt_rows.extend(svt_state_rows(spec.name, model))
        output_arrays[spec.name] = arrays

        summary.append(
            {
                "model": spec.name,
                "family": spec.family,
                "checkpoint": str(spec.checkpoint),
                "num_parameters": sum(p.numel() for p in model.parameters()),
                **out_stats,
            }
        )
        plot_attention(out_dir, spec.name, diagnostics)
        plot_bars(out_dir, grad_rows, "grad_rms", f"{spec.name}_gradient_rms_by_group.png", f"{spec.name}: gradient RMS")
        plot_bars(out_dir, model_activation_rows, "rms", f"{spec.name}_activation_rms_top.png", f"{spec.name}: largest activation RMS")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "attention_stats.csv", all_attention_rows)
    write_csv(out_dir / "activation_stats.csv", all_activation_rows)
    write_csv(out_dir / "gradient_stats.csv", all_grad_rows)
    write_csv(out_dir / "parameter_dtypes.csv", all_dtype_rows)
    write_csv(out_dir / "svt_internal_state.csv", all_svt_rows)
    plot_output_hists(out_dir, output_arrays)
    print(f"[diagnostic] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
