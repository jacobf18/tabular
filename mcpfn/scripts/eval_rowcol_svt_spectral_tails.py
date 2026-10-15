#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from fancyimpute import SoftImpute

from tabimpute.interface import ImputePFN
from tabimpute.prior.spectrum_training_set_generation import DecayingSpectrumGenerator


SPECTRAL_PRESETS = {
    "moderate_tail": {
        "spectrum_rank_low": 40,
        "spectrum_rank_high": 80,
        "spectrum_alpha_low": 0.55,
        "spectrum_alpha_high": 0.85,
        "spectrum_tail_amplitude_low": 0.08,
        "spectrum_tail_amplitude_high": 0.18,
        "spectrum_noise_std_low": 0.001,
        "spectrum_noise_std_high": 0.005,
        "spectrum_target_rank": 20,
    },
    "hard_tail": {
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
        "spectrum_noise_std_low": 0.001,
        "spectrum_noise_std_high": 0.005,
        "spectrum_target_rank": 40,
    },
    "very_hard_tail": {
        "spectrum_rank_low": 128,
        "spectrum_rank_high": 192,
        "spectrum_alpha_low": 0.20,
        "spectrum_alpha_high": 0.38,
        "spectrum_head_weight_low": 2.0,
        "spectrum_head_weight_high": 4.0,
        "spectrum_head_exp_scale_low": 3.5,
        "spectrum_head_exp_scale_high": 7.0,
        "spectrum_head_exp_power_low": 1.6,
        "spectrum_head_exp_power_high": 2.4,
        "spectrum_head_handoff_scale_low": 2.0,
        "spectrum_head_handoff_scale_high": 8.0,
        "spectrum_tail_amplitude_low": 0.35,
        "spectrum_tail_amplitude_high": 0.65,
        "spectrum_taper_strength_low": 0.6,
        "spectrum_taper_strength_high": 1.2,
        "spectrum_taper_power_low": 2.5,
        "spectrum_taper_power_high": 4.0,
        "spectrum_noise_std_low": 0.002,
        "spectrum_noise_std_high": 0.01,
        "spectrum_target_rank": 60,
    },
}


@dataclass(frozen=True)
class DirectSpec:
    name: str
    checkpoint: Path
    chunk_size: int
    family: str
    prior_index: int | None = None


def parse_direct(value: str) -> DirectSpec:
    parts = value.split("|")
    if len(parts) not in {4, 5}:
        raise ValueError(
            "--extra-direct must be formatted as name|checkpoint|chunk_size|family[|prior_index]"
        )
    name, checkpoint, chunk_size, family = (part.strip() for part in parts[:4])
    prior_index = int(parts[4]) if len(parts) == 5 and parts[4].strip() else None
    path = Path(checkpoint).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    return DirectSpec(name, path, int(chunk_size), family, prior_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(SPECTRAL_PRESETS), default="hard_tail")
    parser.add_argument("--preset-overrides-json", type=str, default=None)
    parser.add_argument("--sizes", nargs="+", type=int, default=[300, 500, 750, 1000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--missing-rate", type=float, default=0.2)
    parser.add_argument("--softimpute-max-iters", type=int, default=200)
    parser.add_argument("--softimpute-shrinkage", type=float, default=None)
    parser.add_argument("--skip-softimpute", action="store_true")
    parser.add_argument("--skip-svt", action="store_true")
    parser.add_argument("--svt-steps", nargs="+", type=int, default=[32])
    parser.add_argument("--svt-max-rank", type=int, default=80)
    parser.add_argument("--svt-tau-fractions", nargs="+", type=float, default=[0.004, 0.006, 0.008, 0.01])
    parser.add_argument("--adaptive-svt", action="store_true")
    parser.add_argument("--adaptive-scale", type=float, default=1.2)
    parser.add_argument("--adaptive-min", type=float, default=0.004)
    parser.add_argument("--adaptive-max", type=float, default=0.02)
    parser.add_argument("--extra-direct", action="append", type=parse_direct, default=[])
    parser.add_argument("--direct-prediction-stat", choices=("median", "mean"), default="median")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def load_model_config(checkpoint: Path) -> dict:
    with (checkpoint.parent / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "model" in cfg and "embedding_size" in cfg["model"]:
        return dict(cfg["model"])
    if "config" in cfg:
        return dict(cfg["config"])
    raise ValueError(f"Could not find model config in {checkpoint.parent}")


def make_spectral_case(
    *,
    size: int,
    seed: int,
    preset: str,
    preset_overrides: dict[str, object] | None = None,
    missing_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    np.random.seed(seed)
    config = {
        "num_rows_low": size,
        "num_rows_high": size,
        "num_cols_low": size,
        "num_cols_high": size,
        "p_missing": missing_rate,
        "spectrum_scale_low": 1.0,
        "spectrum_scale_high": 1.0,
        **SPECTRAL_PRESETS[preset],
    }
    if preset_overrides:
        config.update(preset_overrides)
    bundle = DecayingSpectrumGenerator(config).generate_matrix_bundle()
    x_observed_full = bundle["full_matrix"].detach().cpu().numpy().astype(np.float32)
    target = bundle["target_matrix"].detach().cpu().numpy().astype(np.float32)
    rng = np.random.default_rng(seed + 10_000)
    mask = rng.random(size=x_observed_full.shape) < missing_rate
    x_missing = x_observed_full.copy()
    x_missing[mask] = np.nan
    meta = {
        key: float(value)
        for key, value in bundle.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }
    return target, x_missing, mask, meta


def observed_col_stats(x_missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(x_missing, axis=0)
    stds = np.nanstd(x_missing, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    stds = np.where(np.isnan(stds), 1.0, stds)
    return means.astype(np.float32), stds.astype(np.float32)


def score(target: np.ndarray, imputed: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    err = imputed[mask] - target[mask]
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.std(target[mask]))
    return {
        "rmse": rmse,
        "mae": mae,
        "nrmse": rmse / denom if denom > 1e-16 else float("nan"),
    }


def softimpute_fill(x_missing: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    means, stds = observed_col_stats(x_missing)
    x_norm = (x_missing - means) / (stds + 1e-16)
    kwargs = {
        "max_iters": args.softimpute_max_iters,
        "convergence_threshold": 1e-5,
        "verbose": False,
    }
    if args.softimpute_shrinkage is not None:
        kwargs["shrinkage_value"] = args.softimpute_shrinkage
    imputed_norm = SoftImpute(**kwargs).fit_transform(x_norm)
    imputed = imputed_norm * (stds + 1e-16) + means
    out = x_missing.copy()
    missing = np.isnan(out)
    out[missing] = imputed[missing]
    return out


def algorithmic_svt_fill(
    x_missing: np.ndarray,
    *,
    steps: int,
    max_rank: int,
    tau_fraction: float,
    device: str,
) -> np.ndarray:
    means, stds = observed_col_stats(x_missing)
    x_norm = (x_missing - means) / (stds + 1e-16)
    clean = np.nan_to_num(x_norm, nan=0.0).astype(np.float32)
    observed = ~np.isnan(x_norm)
    x = torch.from_numpy(clean).to(device=device)
    clean_t = torch.from_numpy(clean).to(device=device)
    observed_t = torch.from_numpy(observed).to(device=device)
    for _ in range(steps):
        u, s, vh = torch.linalg.svd(x.float(), full_matrices=False)
        rank = min(max_rank, s.shape[-1])
        tau = float(tau_fraction) * s[:1].detach()
        low_rank = (u[:, :rank] * torch.relu(s[:rank] - tau).unsqueeze(0)) @ vh[:rank, :]
        x = torch.where(observed_t, clean_t, low_rank)
    out = x.detach().cpu().numpy() * (stds + 1e-16) + means
    full = x_missing.copy()
    full[np.isnan(full)] = out[np.isnan(full)]
    return full


def build_imputer(spec: DirectSpec, args: argparse.Namespace) -> ImputePFN:
    return ImputePFN(
        device=args.device,
        entry_wise_features=False,
        checkpoint_path=str(spec.checkpoint),
        json_config={"config": load_model_config(spec.checkpoint)},
        model_family=spec.family,
        max_num_rows=spec.chunk_size,
        max_num_cols=spec.chunk_size,
        max_num_chunks=1,
        preprocessors=None,
        prediction_stat=args.direct_prediction_stat,
        direct_prior_index=spec.prior_index,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    preferred = ["method", "preset", "size", "seed", "rmse", "rmse_sem", "mae", "nrmse", "n"]
    fields = [key for key in preferred if key in fields] + [key for key in fields if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), int(row["size"])), []).append(row)
    out = []
    for (method, size), group in sorted(groups.items()):
        rmses = np.array([float(row["rmse"]) for row in group], dtype=np.float64)
        maes = np.array([float(row["mae"]) for row in group], dtype=np.float64)
        nrmse = np.array([float(row["nrmse"]) for row in group], dtype=np.float64)
        out.append(
            {
                "method": method,
                "preset": group[0]["preset"],
                "size": size,
                "rmse": float(rmses.mean()),
                "rmse_sem": float(rmses.std(ddof=1) / np.sqrt(len(rmses))) if len(rmses) > 1 else 0.0,
                "mae": float(maes.mean()),
                "nrmse": float(nrmse.mean()),
                "n": len(group),
            }
        )
    return out


def plot(summary: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    methods = sorted({row["method"] for row in summary})
    for method in methods:
        group = sorted([row for row in summary if row["method"] == method], key=lambda x: x["size"])
        ax.errorbar(
            [row["size"] for row in group],
            [row["rmse"] for row in group],
            yerr=[row["rmse_sem"] for row in group],
            marker="o",
            linewidth=2.2,
            capsize=3,
            label=method,
        )
    ax.set_title("Spectral-tail missing-entry RMSE")
    ax.set_xlabel("square matrix size")
    ax.set_ylabel("RMSE against truncated recoverable signal")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    preset_overrides = (
        json.loads(args.preset_overrides_json)
        if args.preset_overrides_json
        else None
    )
    cases = {}
    for size in args.sizes:
        for seed in args.seeds:
            cases[(size, seed)] = make_spectral_case(
                size=size,
                seed=seed,
                preset=args.preset,
                preset_overrides=preset_overrides,
                missing_rate=args.missing_rate,
            )

    rows: list[dict[str, object]] = []
    for size in args.sizes:
        for seed in args.seeds:
            target, x_missing, mask, meta = cases[(size, seed)]
            if not args.skip_softimpute:
                imputed = softimpute_fill(x_missing, args)
                rows.append(
                    {
                        "method": "SoftImpute",
                        "preset": args.preset,
                        "size": size,
                        "seed": seed,
                        **score(target, imputed, mask),
                        **meta,
                    }
                )
            if not args.skip_svt:
                for steps in args.svt_steps:
                    tau_values = []
                    if args.adaptive_svt:
                        tau_values.append(
                            min(
                                args.adaptive_max,
                                max(args.adaptive_min, args.adaptive_scale / size),
                            )
                        )
                    tau_values.extend(args.svt_tau_fractions)
                    for tau in sorted(set(float(x) for x in tau_values)):
                        imputed = algorithmic_svt_fill(
                            x_missing,
                            steps=steps,
                            max_rank=args.svt_max_rank,
                            tau_fraction=tau,
                            device=args.device,
                        )
                        rows.append(
                            {
                                "method": f"SVT zero s{steps} r{args.svt_max_rank} tau{tau:g}",
                                "preset": args.preset,
                                "size": size,
                                "seed": seed,
                                **score(target, imputed, mask),
                                **meta,
                            }
                        )
    for spec in args.extra_direct:
        imputer = build_imputer(spec, args)
        for size in args.sizes:
            imputer.max_num_rows = spec.chunk_size
            imputer.max_num_cols = spec.chunk_size
            for seed in args.seeds:
                target, x_missing, mask, meta = cases[(size, seed)]
                imputed = imputer.impute(x_missing.copy())
                rows.append(
                    {
                        "method": spec.name,
                        "preset": args.preset,
                        "size": size,
                        "seed": seed,
                        **score(target, imputed, mask),
                        **meta,
                    }
                )
        del imputer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(rows)
    write_csv(args.output_prefix.with_suffix(".per_seed.csv"), rows)
    write_csv(args.output_prefix.with_suffix(".summary.csv"), summary)
    plot(summary, args.output_prefix.with_suffix(".png"))
    print("method\tsize\trmse\tsem\tmae\tn")
    for row in summary:
        print(
            f"{row['method']}\t{row['size']}\t{row['rmse']:.6g}\t"
            f"{row['rmse_sem']:.3g}\t{row['mae']:.6g}\t{row['n']}"
        )
    print(f"Wrote {args.output_prefix.with_suffix('.summary.csv')}")


if __name__ == "__main__":
    main()
