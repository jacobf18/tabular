#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path("/root/tabular/mcpfn")
BENCHMARK_DIR = ROOT / "benchmark"
WORKDIR = ROOT / "src/tabimpute/workdir"
TRAIN_SCRIPT = ROOT / "scripts/train_direct_pure_rabel_causal_panel.py"
BASE_CHECKPOINT = (
    WORKDIR
    / "tabimpute-direct-pure-rabel64-causal-panel-bs16-from-rank10-100k"
    / "best_checkpoint.pth"
)

SRC_DIR = ROOT / "src"
CPS_DIR = BENCHMARK_DIR / "CPS"
for extra in (SRC_DIR, CPS_DIR):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from eval_cps_causal import compute_metrics  # noqa: E402
from tabimpute.interface import ImputePFN  # noqa: E402


V1_WEIGHTS = {
    "cps_low_rank": 0.22,
    "cps_mid_rank": 0.28,
    "kansas_single": 0.18,
    "prop99_simultaneous": 0.14,
    "prop99_staggered": 0.12,
    "stock_block": 0.06,
}

GENTLE_WEIGHTS = {
    "cps_low_rank": 0.22,
    "cps_mid_rank": 0.30,
    "kansas_single": 0.12,
    "prop99_simultaneous": 0.14,
    "prop99_staggered": 0.18,
    "stock_block": 0.04,
}

CPS_HEAVY_WEIGHTS = {
    "cps_low_rank": 0.28,
    "cps_mid_rank": 0.34,
    "kansas_single": 0.10,
    "prop99_simultaneous": 0.12,
    "prop99_staggered": 0.12,
    "stock_block": 0.04,
}

STAGGERED_WEIGHTS = {
    "cps_low_rank": 0.22,
    "cps_mid_rank": 0.28,
    "kansas_single": 0.10,
    "prop99_simultaneous": 0.14,
    "prop99_staggered": 0.22,
    "stock_block": 0.04,
}

BENCHMARKS = [
    BENCHMARK_DIR / "kansas/causal_eval/smoke",
    BENCHMARK_DIR / "prop99/causal_eval/smoke",
    BENCHMARK_DIR / "CPS/causal_eval/log_wage_min_wage",
    BENCHMARK_DIR / "CPS/causal_eval/hours_min_wage",
]


@dataclass(frozen=True)
class SweepConfig:
    name: str
    lr: float
    min_lr: float
    profile_weights: dict[str, float]
    grad_clip_norm: float = 1.0
    model_overrides: dict[str, float | int | bool] | None = None


SWEEP_CONFIGS = [
    SweepConfig("v1_lr5e6", 5e-6, 2.5e-6, V1_WEIGHTS),
    SweepConfig("v1_lr1e5", 1e-5, 5e-6, V1_WEIGHTS),
    SweepConfig("gentle_lr5e6", 5e-6, 2.5e-6, GENTLE_WEIGHTS),
    SweepConfig("gentle_lr1e5", 1e-5, 5e-6, GENTLE_WEIGHTS),
    SweepConfig("gentle_lr2e5", 2e-5, 7.5e-6, GENTLE_WEIGHTS),
    SweepConfig("cps_heavy_lr1e5", 1e-5, 5e-6, CPS_HEAVY_WEIGHTS),
    SweepConfig("staggered_lr1e5", 1e-5, 5e-6, STAGGERED_WEIGHTS),
    SweepConfig(
        "gentle_sigma08_lr1e5",
        1e-5,
        5e-6,
        GENTLE_WEIGHTS,
        model_overrides={"rabel_sigma": 0.8},
    ),
    SweepConfig(
        "gentle_nodrop_lr1e5",
        1e-5,
        5e-6,
        GENTLE_WEIGHTS,
        model_overrides={"ffn_dropout": 0.0, "drop_path_rate": 0.0},
    ),
]


def standardize_columns(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore"):
        col_mean = np.nanmean(x, axis=0)
        col_std = np.nanstd(x, axis=0)
    col_std = np.where((col_std == 0) | np.isnan(col_std), 1.0, col_std)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    return (x - col_mean) / col_std, col_mean, col_std


def load_model_config(checkpoint: Path) -> dict:
    with (checkpoint.parent / "config.json").open("r", encoding="utf-8") as file:
        config = json.load(file)
    return dict(config["model"] if "model" in config else config["config"])


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    device: str,
    limit: int,
    method_name: str,
) -> pd.DataFrame:
    imputer = ImputePFN(
        device=device,
        entry_wise_features=False,
        checkpoint_path=str(checkpoint),
        json_config={"config": load_model_config(checkpoint)},
        model_family="pure_rabel_memory_efficient",
        max_num_rows=None,
        max_num_cols=None,
        preprocessors=None,
    )
    rows = []
    for benchmark_dir in BENCHMARKS:
        draw_dirs = sorted(
            p for p in benchmark_dir.iterdir() if p.is_dir() and p.name.startswith("draw_")
        )[:limit]
        for draw_dir in draw_dirs:
            true = np.load(draw_dir / "true.npy")
            observed = np.load(draw_dir / "observed.npy")
            treatment_mask = np.load(draw_dir / "treatment_mask.npy")
            missing_mask = np.isnan(observed)
            observed_in, col_mean, col_std = standardize_columns(observed)
            completed_std = imputer.impute(observed_in.copy()).astype(np.float32)
            completed = (completed_std * col_std) + col_mean
            completed = completed.astype(np.float32)
            completed[~missing_mask] = observed[~missing_mask]
            metrics = compute_metrics(true, completed, treatment_mask)
            rows.append(
                {
                    "method": method_name,
                    "benchmark": str(benchmark_dir).replace(str(BENCHMARK_DIR) + "/", ""),
                    "draw_name": draw_dir.name,
                    **metrics,
                }
            )
    del imputer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def summarize_eval(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["method", "benchmark"], as_index=False)
        .agg(
            n=("treated_block_rmse", "size"),
            rmse=("treated_block_rmse", "mean"),
            tau_abs=("tau_abs_error", "mean"),
        )
    )


def train_config(
    cfg: SweepConfig,
    *,
    sweep_name: str,
    steps: int,
    batch_size: int,
    device: str,
    wandb_mode: str,
) -> Path:
    warmup = min(500, max(1, steps // 10))
    decay = min(2000, max(1, steps // 4))
    stable = steps - warmup - decay
    if stable < 1:
        raise ValueError(f"steps too small: {steps}")

    run_name = f"{sweep_name}-{cfg.name}-{steps}steps"
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--run-name",
        run_name,
        "--device",
        device,
        "--steps",
        str(steps),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(cfg.lr),
        "--min-lr",
        str(cfg.min_lr),
        "--weight-decay",
        "0.03",
        "--grad-clip-norm",
        str(cfg.grad_clip_norm),
        "--warmup-steps",
        str(warmup),
        "--stable-steps",
        str(stable),
        "--decay-steps",
        str(decay),
        "--checkpoint-every",
        str(steps),
        "--init-checkpoint",
        str(BASE_CHECKPOINT),
        "--profile-weights-json",
        json.dumps(cfg.profile_weights, sort_keys=True),
        "--wandb-project",
        "tabimpute",
    ]
    if cfg.model_overrides:
        command.extend(
            [
                "--model-overrides-json",
                json.dumps(cfg.model_overrides, sort_keys=True),
            ]
        )

    env = os.environ.copy()
    env["REPO_DIR"] = str(ROOT)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    env["WANDB_MODE"] = wandb_mode
    print(f"\n=== training {run_name} ===", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    return WORKDIR / run_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-name", default="causal-panel-sweep-v1compat")
    parser.add_argument("--steps-per-config", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--configs", default=None, help="Comma-separated config names.")
    args = parser.parse_args()

    selected = SWEEP_CONFIGS
    if args.configs:
        requested = {name.strip() for name in args.configs.split(",") if name.strip()}
        selected = [cfg for cfg in SWEEP_CONFIGS if cfg.name in requested]
        missing = requested - {cfg.name for cfg in selected}
        if missing:
            raise SystemExit(f"Unknown configs: {sorted(missing)}")

    output_dir = BENCHMARK_DIR / "runs" / args.sweep_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sweep_configs.json").open("w", encoding="utf-8") as file:
        json.dump([cfg.__dict__ for cfg in selected], file, indent=2, sort_keys=True)

    baseline_rows = evaluate_checkpoint(
        BASE_CHECKPOINT,
        device=args.device,
        limit=args.limit,
        method_name="v1_best_base",
    )
    baseline_summary = summarize_eval(baseline_rows)
    baseline_summary.to_csv(output_dir / "baseline_summary.csv", index=False)

    all_summaries = []
    all_rows = []
    for cfg in selected:
        started = time.time()
        run_dir = train_config(
            cfg,
            sweep_name=args.sweep_name,
            steps=args.steps_per_config,
            batch_size=args.batch_size,
            device=args.device,
            wandb_mode=args.wandb_mode,
        )
        checkpoint = run_dir / "best_checkpoint.pth"
        if not checkpoint.is_file():
            checkpoint = run_dir / f"checkpoint_{args.steps_per_config}.pth"
        eval_rows = evaluate_checkpoint(
            checkpoint,
            device=args.device,
            limit=args.limit,
            method_name=cfg.name,
        )
        eval_rows["checkpoint"] = str(checkpoint)
        eval_rows["train_seconds"] = time.time() - started
        eval_rows.to_csv(output_dir / f"{cfg.name}_per_draw.csv", index=False)
        summary = summarize_eval(eval_rows)
        summary["checkpoint"] = str(checkpoint)
        summary["lr"] = cfg.lr
        summary["min_lr"] = cfg.min_lr
        summary["profile_weights"] = json.dumps(cfg.profile_weights, sort_keys=True)
        summary["model_overrides"] = json.dumps(cfg.model_overrides or {}, sort_keys=True)
        summary["train_seconds"] = time.time() - started
        summary.to_csv(output_dir / f"{cfg.name}_summary.csv", index=False)
        all_summaries.append(summary)
        all_rows.append(eval_rows)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(output_dir / "combined_summary.csv", index=False)
    pd.concat(all_rows, ignore_index=True).to_csv(output_dir / "combined_per_draw.csv", index=False)

    baseline = baseline_summary[["benchmark", "rmse"]].rename(columns={"rmse": "baseline_rmse"})
    ranked = combined.merge(baseline, on="benchmark", how="left")
    ranked["relative_rmse"] = ranked["rmse"] / ranked["baseline_rmse"]
    ranking = (
        ranked.groupby("method", as_index=False)
        .agg(
            mean_relative_rmse=("relative_rmse", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_tau_abs=("tau_abs", "mean"),
            checkpoint=("checkpoint", "first"),
            lr=("lr", "first"),
            min_lr=("min_lr", "first"),
            profile_weights=("profile_weights", "first"),
            model_overrides=("model_overrides", "first"),
            train_seconds=("train_seconds", "first"),
        )
        .sort_values("mean_relative_rmse")
    )
    ranking.to_csv(output_dir / "ranking.csv", index=False)
    print("\n=== ranking ===")
    print(ranking.to_string(index=False))
    print(f"\nwrote {output_dir}")


if __name__ == "__main__":
    main()
