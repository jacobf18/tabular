#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from eval_rowcol_svt_spectral_tails import SPECTRAL_PRESETS, make_spectral_case
from eval_train_level_chunks_interface import make_low_rank
from tabimpute.interface import ImputePFN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-family", default="rowcol_svt_hybrid")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--harder-spectral-overrides-json", default=None)
    parser.add_argument("--oracle-threshold-scales", nargs="+", type=float, default=[])
    return parser.parse_args()


def observed_col_stats(x_missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(x_missing, axis=0)
    stds = np.nanstd(x_missing, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    stds = np.where(np.isnan(stds), 1.0, stds)
    return means.astype(np.float32), stds.astype(np.float32)


def normalized_tensor(
    x_missing: np.ndarray,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    means, stds = observed_col_stats(x_missing)
    x_norm = (x_missing - means) / (stds + 1e-16)
    x_tensor = (
        torch.from_numpy(x_norm)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )
    missing = torch.isnan(x_tensor)
    return x_tensor, missing, means, stds


def model_input_dtype(model: torch.nn.Module) -> torch.dtype:
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        return torch.float32
    return dtype if dtype.is_floating_point else torch.float32


def threshold_scale_oracle(
    *,
    model: torch.nn.Module,
    x_tensor: torch.Tensor,
    missing: torch.Tensor,
    target_norm: np.ndarray,
    x_missing: np.ndarray,
    scales: list[float],
) -> dict[str, float | int]:
    if not scales:
        return {}
    required = [
        "_prior_ids",
        "encode_latents",
        "_apply_prior_latent_adapter",
        "_initial_grid",
        "_svt_iterates",
    ]
    if any(not hasattr(model, name) for name in required):
        return {}
    steps = getattr(model, "svt_mixture_steps", None)
    if steps is None:
        return {}

    mask_np = np.asarray(np.isnan(x_missing), dtype=bool)
    target_missing = torch.from_numpy(
        target_norm[mask_np].astype(np.float32)
    ).to(device=x_tensor.device)
    best_scale = float("nan")
    best_step = -1
    best_rmse = float("inf")
    with torch.no_grad():
        prior_ids = model._prior_ids(x_tensor, None)
        row_state, col_state = model.encode_latents(x_tensor)
        row_state, col_state = model._apply_prior_latent_adapter(
            row_state,
            col_state,
            prior_ids,
        )
        initial_grid = model._initial_grid(row_state, col_state, prior_ids)
        for scale in scales:
            threshold_scale = torch.full(
                (x_tensor.shape[0], 1),
                float(scale),
                device=x_tensor.device,
                dtype=torch.float32,
            )
            snapshots = model._svt_iterates(
                x_tensor,
                initial_grid,
                prior_ids,
                steps,
                threshold_scale=threshold_scale,
            )
            snapshot_points = snapshots.float().squeeze(0)[:, missing.squeeze(0)]
            rmse = (
                (snapshot_points - target_missing.view(1, -1))
                .square()
                .mean(dim=1)
                .clamp_min(0.0)
                .sqrt()
            )
            oracle_index = int(rmse.argmin().item())
            oracle_rmse = float(rmse[oracle_index].cpu())
            if oracle_rmse < best_rmse:
                best_rmse = oracle_rmse
                best_scale = float(scale)
                best_step = int(steps[oracle_index])
    return {
        "tau_oracle_scale": best_scale,
        "tau_oracle_step": best_step,
        "tau_oracle_rmse_norm": best_rmse,
    }


def record_case(
    *,
    rows: list[dict[str, object]],
    imputer: ImputePFN,
    case_name: str,
    target: np.ndarray,
    x_missing: np.ndarray,
    seed: int,
    oracle_threshold_scales: list[float],
) -> None:
    x_tensor, missing, means, stds = normalized_tensor(
        x_missing,
        imputer.device,
        model_input_dtype(imputer.model),
    )
    with torch.no_grad():
        _ = imputer.model(x_tensor, decode_mask=missing)
    weights = getattr(imputer.model, "last_svt_mixture_weights", None)
    steps = getattr(imputer.model, "svt_mixture_steps", None)
    snapshots = getattr(imputer.model, "last_svt_mixture_snapshots", None)
    controller_logits = getattr(
        imputer.model,
        "last_svt_mixture_controller_logits",
        None,
    )
    controller_features = getattr(
        imputer.model,
        "last_svt_controller_features",
        None,
    )
    matrix_stats = getattr(
        imputer.model,
        "last_svt_controller_matrix_stats",
        None,
    )
    missingness_stats = getattr(
        imputer.model,
        "last_svt_controller_missingness_stats",
        None,
    )
    cv_features = getattr(
        imputer.model,
        "last_svt_controller_cv_features",
        None,
    )
    spectral_stats = getattr(
        imputer.model,
        "last_svt_controller_spectral_stats",
        None,
    )
    spectrum_encoder_stats = getattr(
        imputer.model,
        "last_svt_controller_spectrum_encoder_stats",
        None,
    )
    threshold_scale = getattr(
        imputer.model,
        "last_svt_adaptive_threshold_scale",
        None,
    )
    if weights is None or steps is None:
        raise RuntimeError("Model did not expose svt_mixture weights.")
    weights_np = weights.float().squeeze(0).cpu().numpy()
    row = {
        "case": case_name,
        "seed": seed,
        "expected_steps": float(np.sum(weights_np * np.asarray(steps))),
        "mixture_entropy": float(
            -np.sum(weights_np * np.log(np.clip(weights_np, 1e-12, None)))
        ),
        "mixture_top_weight": float(np.max(weights_np)),
    }
    if weights_np.size > 1:
        row["mixture_weight_margin"] = float(
            np.max(weights_np) - np.partition(weights_np, -2)[-2]
        )
    if threshold_scale is not None:
        row["threshold_scale"] = float(threshold_scale.float().mean().cpu())
    if controller_logits is not None:
        logits_np = controller_logits.float().squeeze(0).cpu().numpy()
        row["controller_logit_margin"] = float(
            np.max(logits_np) - np.partition(logits_np, -2)[-2]
        ) if logits_np.size > 1 else 0.0
        for step, value in zip(steps, logits_np):
            row[f"controller_logit_step_{step}"] = float(value)
    if controller_features is not None:
        features_np = controller_features.float().squeeze(0).cpu().numpy()
        if features_np.size:
            row["controller_feature_l2"] = float(np.linalg.norm(features_np))
            row["controller_feature_abs_mean"] = float(np.mean(np.abs(features_np)))
    if matrix_stats is not None:
        stats_np = matrix_stats.float().squeeze(0).cpu().numpy()
        for index, value in enumerate(stats_np):
            row[f"matrix_stat_{index}"] = float(value)
    if missingness_stats is not None:
        stats_np = missingness_stats.float().squeeze(0).cpu().numpy()
        if stats_np.size:
            row["missingness_feature_l2"] = float(np.linalg.norm(stats_np))
            row["missingness_feature_abs_mean"] = float(np.mean(np.abs(stats_np)))
        for index, value in enumerate(stats_np):
            row[f"missingness_stat_{index}"] = float(value)
    if cv_features is not None:
        stats_np = cv_features.float().squeeze(0).cpu().numpy()
        if stats_np.size:
            row["cv_feature_l2"] = float(np.linalg.norm(stats_np))
            row["cv_feature_abs_mean"] = float(np.mean(np.abs(stats_np)))
        for index, value in enumerate(stats_np):
            row[f"cv_feature_{index}"] = float(value)
    if spectral_stats is not None:
        stats_np = spectral_stats.float().squeeze(0).cpu().numpy()
        for index, value in enumerate(stats_np):
            row[f"spectral_stat_{index}"] = float(value)
    if spectrum_encoder_stats is not None:
        stats_np = spectrum_encoder_stats.float().squeeze(0).cpu().numpy()
        if stats_np.size:
            row["spectrum_encoder_feature_l2"] = float(np.linalg.norm(stats_np))
            row["spectrum_encoder_feature_abs_mean"] = float(np.mean(np.abs(stats_np)))
    if snapshots is not None:
        mask_np = np.asarray(np.isnan(x_missing), dtype=bool)
        target_norm = (target - means) / (stds + 1e-16)
        target_missing = torch.from_numpy(
            target_norm[mask_np].astype(np.float32)
        ).to(device=snapshots.device)
        snapshot_points = snapshots.float().squeeze(0)[:, missing.squeeze(0)]
        errors = snapshot_points - target_missing.view(1, -1)
        rmse = errors.square().mean(dim=1).clamp_min(0.0).sqrt()
        rmse_np = rmse.cpu().numpy()
        oracle_index = int(np.argmin(rmse_np))
        mixture_points = (
            snapshot_points * weights.float().squeeze(0).view(-1, 1)
        ).sum(dim=0)
        mixture_rmse = (
            (mixture_points - target_missing)
            .square()
            .mean()
            .clamp_min(0.0)
            .sqrt()
        )
        row["oracle_step"] = int(steps[oracle_index])
        row["oracle_rmse_norm"] = float(rmse_np[oracle_index])
        row["mixture_rmse_norm"] = float(mixture_rmse.cpu())
        row["mixture_oracle_gap_norm"] = (
            row["mixture_rmse_norm"] - row["oracle_rmse_norm"]
        )
        for step, value in zip(steps, rmse_np):
            row[f"rmse_norm_step_{step}"] = float(value)
        tau_oracle = threshold_scale_oracle(
            model=imputer.model,
            x_tensor=x_tensor,
            missing=missing,
            target_norm=target_norm,
            x_missing=x_missing,
            scales=oracle_threshold_scales,
        )
        if tau_oracle:
            row.update(tau_oracle)
            row["mixture_tau_oracle_gap_norm"] = (
                row["mixture_rmse_norm"] - row["tau_oracle_rmse_norm"]
            )
    for step, weight in zip(steps, weights_np):
        row[f"weight_step_{step}"] = float(weight)
    rows.append(row)


def main() -> None:
    args = parse_args()
    imputer = ImputePFN(
        device=args.device,
        entry_wise_features=False,
        checkpoint_path=str(args.checkpoint),
        json_config={"config": load_model_config(args.checkpoint)},
        model_family=args.model_family,
        max_num_rows=args.chunk_size,
        max_num_cols=args.chunk_size,
        max_num_chunks=1,
        preprocessors=None,
        prediction_stat="mean",
    )
    harder_overrides = (
        json.loads(args.harder_spectral_overrides_json)
        if args.harder_spectral_overrides_json
        else None
    )
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for rank in (10, 80):
            target, x_missing, _ = make_low_rank(
                size=300,
                rank=rank,
                missing_rate=0.2,
                seed=seed,
                scale="rank",
                generator="gaussian_factors",
            )
            record_case(
                rows=rows,
                imputer=imputer,
                case_name=f"gaussian_rank{rank}",
                target=target,
                x_missing=x_missing,
                seed=seed,
                oracle_threshold_scales=args.oracle_threshold_scales,
            )
        target, x_missing, _, _ = make_spectral_case(
            size=300,
            seed=seed,
            preset="hard_tail",
            preset_overrides=None,
            missing_rate=0.2,
        )
        record_case(
            rows=rows,
            imputer=imputer,
            case_name="spectral_default",
            target=target,
            x_missing=x_missing,
            seed=seed,
            oracle_threshold_scales=args.oracle_threshold_scales,
        )
        target, x_missing, _, _ = make_spectral_case(
            size=300,
            seed=seed,
            preset="hard_tail",
            preset_overrides=harder_overrides,
            missing_rate=0.2,
        )
        record_case(
            rows=rows,
            imputer=imputer,
            case_name="spectral_harder",
            target=target,
            x_missing=x_missing,
            seed=seed,
            oracle_threshold_scales=args.oracle_threshold_scales,
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = ["case", "seed", "expected_steps"]
    fields = [key for key in preferred if key in fields] + [
        key for key in fields if key not in preferred
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(
        "case\texpected_steps\tthreshold_scale\toracle_step\toracle_rmse\t"
        "mixture_rmse\tweights"
    )
    for case in sorted({str(row["case"]) for row in rows}):
        group = [row for row in rows if row["case"] == case]
        expected = np.mean([float(row["expected_steps"]) for row in group])
        threshold_scales = [
            float(row["threshold_scale"])
            for row in group
            if "threshold_scale" in row
        ]
        oracle_steps = [
            float(row["oracle_step"]) for row in group if "oracle_step" in row
        ]
        oracle_rmses = [
            float(row["oracle_rmse_norm"])
            for row in group
            if "oracle_rmse_norm" in row
        ]
        mixture_rmses = [
            float(row["mixture_rmse_norm"])
            for row in group
            if "mixture_rmse_norm" in row
        ]
        tau_oracle_scales = [
            float(row["tau_oracle_scale"])
            for row in group
            if "tau_oracle_scale" in row
        ]
        tau_oracle_steps = [
            float(row["tau_oracle_step"])
            for row in group
            if "tau_oracle_step" in row
        ]
        tau_oracle_rmses = [
            float(row["tau_oracle_rmse_norm"])
            for row in group
            if "tau_oracle_rmse_norm" in row
        ]
        weight_items = [
            key for key in fields if key.startswith("weight_step_")
        ]
        weights = ", ".join(
            f"{key.removeprefix('weight_step_')}={np.mean([float(row[key]) for row in group]):.3f}"
            for key in weight_items
        )
        oracle_step = np.mean(oracle_steps) if oracle_steps else float("nan")
        oracle_rmse = np.mean(oracle_rmses) if oracle_rmses else float("nan")
        mixture_rmse = np.mean(mixture_rmses) if mixture_rmses else float("nan")
        threshold_scale = (
            np.mean(threshold_scales) if threshold_scales else float("nan")
        )
        tau_suffix = ""
        if tau_oracle_rmses:
            tau_suffix = (
                f"\ttau_oracle_scale={np.mean(tau_oracle_scales):.3f}"
                f"\ttau_oracle_step={np.mean(tau_oracle_steps):.1f}"
                f"\ttau_oracle_rmse={np.mean(tau_oracle_rmses):.5f}"
            )
        print(
            f"{case}\t{expected:.3f}\t{threshold_scale:.3f}\t"
            f"{oracle_step:.1f}\t"
            f"{oracle_rmse:.5f}\t{mixture_rmse:.5f}\t{weights}{tau_suffix}"
        )


def load_model_config(checkpoint: Path) -> dict:
    with (checkpoint.parent / "config.json").open("r", encoding="utf-8") as file:
        config = json.load(file)
    if "model" in config and isinstance(config["model"], dict):
        return dict(config["model"])
    if "config" in config and isinstance(config["config"], dict):
        return dict(config["config"])
    raise ValueError(f"Could not find model config in {checkpoint.parent}")


if __name__ == "__main__":
    main()
