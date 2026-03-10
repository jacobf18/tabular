import argparse
import importlib.resources
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new import TabImputeModel
from tabimpute.prior.training_set_generation import MissingnessPrior
from tabimpute.train.callbacks import WandbLoggerCallback
from tabimpute.train_new_sweep import train


HIGH_IMPACT_SPACE = {
    "embedding_size": [512, 768, 1024],
    "num_attention_heads": [16, 24, 32],
    "num_layers": [8, 10, 12],
    "mlp_ratio": [1.0, 1.5, 2.0],
    "num_cls": [8, 12, 16],
    "rope_fraction": [0.5, 1.0],
    "lr": [1e-4, 2e-4, 3e-4],
    "weight_decay": [0.0, 1e-2, 5e-2],
    "grad_clip_norm": [0.5, 1.0],
    "optimizer_name": ["adamw", "schedulefree_adamw"],
    "scheduler_name": ["warmup_cosine", "cosine", "none"],
    "warmup_ratio": [0.03, 0.06, 0.1],
    "min_lr_ratio": [0.05, 0.1, 0.2],
}

BEST_LOCAL_SPACE = {
    # Centered around top performers from the previous sweep (trials 007/006/004).
    "embedding_size": [768, 1024],
    "num_attention_heads": [16, 24],
    "num_layers": [8, 10],
    "mlp_ratio": [1.25, 1.5, 2.0],
    "num_cls": [12, 16],
    "rope_fraction": [0.5, 0.75, 1.0],
    "lr": [7e-5, 1e-4, 1.5e-4, 2e-4],
    "weight_decay": [0.01, 0.03, 0.05, 0.08],
    "grad_clip_norm": [0.3, 0.5, 0.8, 1.0],
    "optimizer_name": ["adamw"],
    "scheduler_name": ["none", "warmup_cosine"],
    "warmup_ratio": [0.03, 0.06, 0.1],
    "min_lr_ratio": [0.03, 0.05, 0.1],
}

ANCHOR_PAIR_TRIALS = [
    # Anchor A: old best final (trial_007 from first sweep)
    {
        "embedding_size": 1024,
        "num_attention_heads": 16,
        "num_layers": 8,
        "num_cls": 16,
        "rope_fraction": 0.5,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "none",
        "warmup_ratio": 0.0,
        "min_lr_ratio": 0.05,
        "mlp_hidden_size": 1536,
    },
    # Anchor B: new best checkpoint peak (trial_014 @ 15k from second sweep)
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "num_cls": 12,
        "rope_fraction": 0.75,
        "lr": 2e-4,
        "weight_decay": 0.03,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
        "mlp_hidden_size": 1536,
    },
    # Around Anchor A: lr / wd / schedule / rope refinements
    {
        "embedding_size": 1024,
        "num_attention_heads": 16,
        "num_layers": 8,
        "num_cls": 16,
        "rope_fraction": 0.5,
        "lr": 7e-5,
        "weight_decay": 0.05,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "none",
        "warmup_ratio": 0.0,
        "min_lr_ratio": 0.05,
        "mlp_hidden_size": 1536,
    },
    {
        "embedding_size": 1024,
        "num_attention_heads": 16,
        "num_layers": 8,
        "num_cls": 16,
        "rope_fraction": 0.5,
        "lr": 1.5e-4,
        "weight_decay": 0.05,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "none",
        "warmup_ratio": 0.0,
        "min_lr_ratio": 0.05,
        "mlp_hidden_size": 1536,
    },
    {
        "embedding_size": 1024,
        "num_attention_heads": 16,
        "num_layers": 8,
        "num_cls": 16,
        "rope_fraction": 0.5,
        "lr": 1e-4,
        "weight_decay": 0.03,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "none",
        "warmup_ratio": 0.0,
        "min_lr_ratio": 0.05,
        "mlp_hidden_size": 1536,
    },
    {
        "embedding_size": 1024,
        "num_attention_heads": 16,
        "num_layers": 8,
        "num_cls": 16,
        "rope_fraction": 0.5,
        "lr": 1e-4,
        "weight_decay": 0.08,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "none",
        "warmup_ratio": 0.0,
        "min_lr_ratio": 0.05,
        "mlp_hidden_size": 1536,
    },
    {
        "embedding_size": 1024,
        "num_attention_heads": 16,
        "num_layers": 8,
        "num_cls": 16,
        "rope_fraction": 0.5,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.03,
        "min_lr_ratio": 0.05,
        "mlp_hidden_size": 1536,
    },
    {
        "embedding_size": 1024,
        "num_attention_heads": 16,
        "num_layers": 8,
        "num_cls": 16,
        "rope_fraction": 0.75,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "none",
        "warmup_ratio": 0.0,
        "min_lr_ratio": 0.05,
        "mlp_hidden_size": 1536,
    },
    # Around Anchor B: lr / wd / rope / cls / schedule refinements
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "num_cls": 12,
        "rope_fraction": 0.75,
        "lr": 1.5e-4,
        "weight_decay": 0.03,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
        "mlp_hidden_size": 1536,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "num_cls": 12,
        "rope_fraction": 0.75,
        "lr": 2e-4,
        "weight_decay": 0.05,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
        "mlp_hidden_size": 1536,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "num_cls": 12,
        "rope_fraction": 0.75,
        "lr": 2e-4,
        "weight_decay": 0.01,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
        "mlp_hidden_size": 1536,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "num_cls": 12,
        "rope_fraction": 0.5,
        "lr": 2e-4,
        "weight_decay": 0.03,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
        "mlp_hidden_size": 1536,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "num_cls": 16,
        "rope_fraction": 0.75,
        "lr": 2e-4,
        "weight_decay": 0.03,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
        "mlp_hidden_size": 1536,
    },
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "num_cls": 12,
        "rope_fraction": 0.75,
        "lr": 2e-4,
        "weight_decay": 0.03,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "none",
        "warmup_ratio": 0.0,
        "min_lr_ratio": 0.05,
        "mlp_hidden_size": 1536,
    },
]

OLD_TOP_SEED_CONFIRM_CONFIGS = [
    # local14
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "num_cls": 12,
        "rope_fraction": 0.75,
        "lr": 2e-4,
        "weight_decay": 0.03,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
        "mlp_hidden_size": 1536,
    },
    # anchor12
    {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "num_layers": 8,
        "num_cls": 12,
        "rope_fraction": 0.5,
        "lr": 2e-4,
        "weight_decay": 0.03,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "warmup_cosine",
        "warmup_ratio": 0.06,
        "min_lr_ratio": 0.05,
        "mlp_hidden_size": 1536,
    },
    # old7
    {
        "embedding_size": 1024,
        "num_attention_heads": 16,
        "num_layers": 8,
        "num_cls": 16,
        "rope_fraction": 0.5,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "grad_clip_norm": 0.5,
        "optimizer_name": "adamw",
        "scheduler_name": "none",
        "warmup_ratio": 0.0,
        "min_lr_ratio": 0.05,
        "mlp_hidden_size": 1536,
    },
]

OLD_TOP_SEED_CONFIRM_TRIALS = [
    dict(cfg) for cfg in OLD_TOP_SEED_CONFIRM_CONFIGS for _ in range(3)
]


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _finalize_trial_cfg(cfg: dict) -> dict:
    # Keep attention config valid.
    if cfg["embedding_size"] % cfg["num_attention_heads"] != 0:
        return {}

    # Avoid stacking an external scheduler on schedulefree optimizer.
    if cfg["optimizer_name"] == "schedulefree_adamw":
        cfg["scheduler_name"] = "none"
        cfg["warmup_ratio"] = 0.0

    if cfg["scheduler_name"] == "none":
        cfg["warmup_ratio"] = 0.0

    if "mlp_hidden_size" not in cfg:
        cfg["mlp_hidden_size"] = int(cfg["embedding_size"] * cfg.pop("mlp_ratio"))
    elif "mlp_ratio" in cfg:
        cfg.pop("mlp_ratio")

    return cfg


def _sample_trial_config(rng: random.Random, search_space: dict) -> dict:
    cfg = {name: rng.choice(values) for name, values in search_space.items()}
    return _finalize_trial_cfg(cfg)


def _build_trials(seed: int, max_trials: int, search_space: dict | list[dict]) -> list[dict]:
    if isinstance(search_space, list):
        fixed_trials = []
        for cfg in search_space:
            finalized = _finalize_trial_cfg(dict(cfg))
            if finalized:
                fixed_trials.append(finalized)
        return fixed_trials[:max_trials]

    rng = random.Random(seed)
    trials = []
    seen = set()
    max_attempts = max(100, max_trials * 100)
    attempts = 0

    while len(trials) < max_trials and attempts < max_attempts:
        attempts += 1
        cfg = _sample_trial_config(rng, search_space=search_space)
        if not cfg:
            continue
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        trials.append(cfg)

    return trials


def _build_prior_config(args) -> dict:
    return {
        "num_rows_low": args.num_rows_low,
        "num_rows_high": args.num_rows_high,
        "num_cols_low": args.num_cols_low,
        "num_cols_high": args.num_cols_high,
        "p_missing": args.p_missing,
        "apply_feature_warping_prob": args.apply_feature_warping_prob,
        "apply_quantization_prob": args.apply_quantization_prob,
        "latent_rank_low": args.latent_rank_low,
        "latent_rank_high": args.latent_rank_high,
        "latent_spike_p": args.latent_spike_p,
        "latent_slab_sigma": args.latent_slab_sigma,
    }


def _ensure_fresh_run_paths(args, output_dir: Path, num_trials: int) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to reuse existing sweep output directory: {output_dir}. "
            "Choose a new --sweep-name."
        )

    trial_root = Path("workdir")
    conflicting_runs = []
    for trial_id in range(1, num_trials + 1):
        run_name = f"{args.sweep_name}-trial_{trial_id:03d}"
        run_dir = trial_root / run_name
        if run_dir.exists():
            conflicting_runs.append(str(run_dir))

    if conflicting_runs:
        preview = ", ".join(conflicting_runs[:3])
        if len(conflicting_runs) > 3:
            preview += ", ..."
        raise FileExistsError(
            "Refusing to reuse existing trial run directories: "
            f"{preview}. Choose a new --sweep-name."
        )


def _run_trial(
    *,
    args,
    trial_id: int,
    trial_cfg: dict,
    device: str,
    borders: torch.Tensor,
) -> dict:
    run_name = f"{args.sweep_name}-trial_{trial_id:03d}"
    _set_seed(args.seed + trial_id)
    model_dtype = torch.bfloat16 if "cuda" in device else torch.float32

    model = TabImputeModel(
        embedding_size=trial_cfg["embedding_size"],
        num_attention_heads=trial_cfg["num_attention_heads"],
        mlp_hidden_size=trial_cfg["mlp_hidden_size"],
        num_layers=trial_cfg["num_layers"],
        num_outputs=args.num_outputs,
        num_cls=trial_cfg["num_cls"],
        use_rope=True,
        rope_base=args.rope_base,
        rope_fraction=trial_cfg["rope_fraction"],
        use_absolute_positional_embeddings=False,
        positional_damping_factor=0.1,
    ).to(device=device, dtype=model_dtype)
    num_params = sum(p.numel() for p in model.parameters())

    prior = MissingnessPrior(
        generator_type=args.generator_type,
        missingness_type=args.missingness_type,
        config=_build_prior_config(args),
        batch_size=args.batch_size,
        verbose=False,
        entry_wise_features=False,
    )
    bar_distribution = FullSupportBarDistribution(borders=borders.to(device))

    callbacks = []
    if args.wandb_project:
        callbacks = [
            WandbLoggerCallback(
                project=args.wandb_project,
                name=run_name,
                config={
                    **trial_cfg,
                    "epochs_per_trial": args.epochs_per_trial,
                    "batch_size": args.batch_size,
                    "generator_type": args.generator_type,
                    "missingness_type": args.missingness_type,
                    "p_missing": args.p_missing,
                },
                log_dir=args.wandb_dir,
            )
        ]

    start = time.time()
    _, metrics = train(
        model=model,
        prior=prior,
        bar_distribution=bar_distribution,
        criterion=bar_distribution,
        epochs=args.epochs_per_trial,
        lr=trial_cfg["lr"],
        weight_decay=trial_cfg["weight_decay"],
        grad_clip_norm=trial_cfg["grad_clip_norm"],
        optimizer_name=trial_cfg["optimizer_name"],
        scheduler_name=trial_cfg["scheduler_name"],
        warmup_ratio=trial_cfg["warmup_ratio"],
        min_lr_ratio=trial_cfg["min_lr_ratio"],
        device=device,
        callbacks=callbacks,
        run_name=run_name,
        checkpoint_every=args.checkpoint_every if args.checkpoint_every > 0 else None,
        return_metrics=True,
    )
    elapsed = time.time() - start

    checkpoint_summary = None
    checkpoint_metrics = metrics.get("checkpoint_metrics")
    if checkpoint_metrics:
        ranked_checkpoints = sorted(
            checkpoint_metrics.items(),
            key=lambda kv: kv[1].get(args.metric, float("inf")),
        )
        best_step, best_metrics = ranked_checkpoints[0]
        checkpoint_summary = {
            "metric": args.metric,
            "best_step": int(best_step),
            "best_value": best_metrics.get(args.metric),
            "available_steps": [int(step) for step in checkpoint_metrics.keys()],
        }

    if "cuda" in device:
        torch.cuda.empty_cache()

    result = {
        "trial_id": trial_id,
        "run_name": run_name,
        "num_params": num_params,
        "seconds": round(elapsed, 2),
        "config": trial_cfg,
        "metrics": metrics,
    }
    if checkpoint_summary is not None:
        result["checkpoint_summary"] = checkpoint_summary
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Random high-impact hyperparameter sweep for TabImputeModelNew."
    )
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--epochs-per-trial", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--search-space",
        type=str,
        choices=["high_impact", "best_local", "anchor_pair_fixed", "old_top_seed_confirm"],
        default="high_impact",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--metric",
        type=str,
        choices=["loss_missing", "loss_total", "mae_missing", "mae_total"],
        default="mae_missing",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sweep-name", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="workdir")
    parser.add_argument("--checkpoint-every", type=int, default=0)

    parser.add_argument("--generator-type", type=str, default="latent_factor")
    parser.add_argument("--missingness-type", type=str, default="mcar")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-outputs", type=int, default=5000)
    parser.add_argument("--rope-base", type=float, default=10000.0)

    parser.add_argument("--num-rows-low", type=int, default=10)
    parser.add_argument("--num-rows-high", type=int, default=50)
    parser.add_argument("--num-cols-low", type=int, default=5)
    parser.add_argument("--num-cols-high", type=int, default=50)
    parser.add_argument("--p-missing", type=float, default=0.4)
    parser.add_argument("--apply-feature-warping-prob", type=float, default=0.0)
    parser.add_argument("--apply-quantization-prob", type=float, default=0.0)
    parser.add_argument("--latent-rank-low", type=int, default=1)
    parser.add_argument("--latent-rank-high", type=int, default=11)
    parser.add_argument("--latent-spike-p", type=float, default=0.3)
    parser.add_argument("--latent-slab-sigma", type=float, default=2.0)

    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-dir", type=str, default="./wandb")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    device = _choose_device(args.device)
    if args.search_space == "high_impact":
        search_space = HIGH_IMPACT_SPACE
    elif args.search_space == "best_local":
        search_space = BEST_LOCAL_SPACE
    elif args.search_space == "old_top_seed_confirm":
        search_space = OLD_TOP_SEED_CONFIRM_TRIALS
    else:
        search_space = ANCHOR_PAIR_TRIALS

    if args.sweep_name is None:
        args.sweep_name = f"tabimpute-new-sweep-{int(time.time())}"

    output_dir = Path(args.results_dir) / args.sweep_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    trials = _build_trials(
        seed=args.seed,
        max_trials=args.max_trials,
        search_space=search_space,
    )
    if len(trials) == 0:
        raise RuntimeError("No valid trial configurations were sampled.")

    _ensure_fresh_run_paths(args, output_dir, len(trials))

    print(f"Using device: {device}")
    print(f"Search space: {args.search_space}")
    print(f"Planned trials: {len(trials)}")
    print(f"Results file: {results_path}")
    print(f"Target metric (lower is better): {args.metric}")

    if args.dry_run:
        for i, trial_cfg in enumerate(trials, start=1):
            print(f"Trial {i:03d}: {trial_cfg}")
        return

    borders_path = importlib.resources.files("tabimpute") / "data" / "borders.pt"
    with importlib.resources.as_file(borders_path) as path:
        borders = torch.load(path)

    all_results = []
    for trial_id, trial_cfg in enumerate(trials, start=1):
        print(f"\nStarting trial {trial_id}/{len(trials)}: {trial_cfg}")
        result = _run_trial(
            args=args,
            trial_id=trial_id,
            trial_cfg=trial_cfg,
            device=device,
            borders=borders,
        )
        all_results.append(result)
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        print(
            f"Finished trial {trial_id}: {args.metric}="
            f"{result['metrics'].get(args.metric, float('nan')):.6f}"
        )

    ranked = sorted(
        all_results,
        key=lambda r: r["metrics"].get(args.metric, float("inf")),
    )
    top_k = ranked[: max(1, args.top_k)]

    summary = {
        "sweep_name": args.sweep_name,
        "device": device,
        "metric": args.metric,
        "num_trials": len(all_results),
        "best": ranked[0] if ranked else None,
        "top_k": top_k,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTop trials:")
    for item in top_k:
        print(
            f"trial={item['trial_id']:03d} "
            f"{args.metric}={item['metrics'].get(args.metric, float('nan')):.6f} "
            f"params={item['num_params']:,}"
        )
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
