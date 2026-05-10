import importlib.resources
import json
import os
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import (
    BEST_TRANSFER_OPTIMIZER_CONFIG,
    build_best_transfer_model,
)
from tabimpute.prior.spectrum_training_set_generation import SpectrumMissingnessPrior
from tabimpute.train.callbacks import WandbLoggerCallback
from tabimpute.train_new_sweep import train


DEFAULT_EPOCHS = 30000
DEFAULT_BATCH_SIZE = 16
DEFAULT_P_MISSING = 0.4
DEFAULT_CHECKPOINT_EVERY = 2500
DEFAULT_WANDB_PROJECT = "tabimpute"

MOVIELENS_SURROGATE_PRIOR_CONFIG = {
    # Preserve the MovieLens 100K aspect ratio on a much cheaper canvas.
    "num_rows_low": 128,
    "num_rows_high": 128,
    "num_cols_low": 224,
    "num_cols_high": 224,
    # MovieLens 100K is about 6.3% observed entries.
    # "p_missing": 0.94,
    "p_missing": 0.4,
    # Keep a long, nontrivial spectrum instead of the previous rank<=30 regime.
    "spectrum_rank_low": 80,
    "spectrum_rank_high": 128,
    # Flatter tail to mimic the long MovieLens decay.
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
    "spectrum_scale_low": 1.0,
    "spectrum_scale_high": 1.0,
    "spectrum_noise_std_low": 0.001,
    "spectrum_noise_std_high": 0.005,
    # Roughly matches a scaled-down 90-95% power cutoff from MovieLens.
    "spectrum_target_rank": 40,
}

ROUND2_T003_MODEL_CONFIG = {
    "embedding_size": 768,
    "num_attention_heads": 16,
    "mlp_hidden_size": 1536,
    "num_layers": 8,
    "num_outputs": 5000,
    "num_cls": 12,
    "use_rope": True,
    "rope_base": 10000.0,
    "rope_fraction": 0.5,
    "use_absolute_positional_embeddings": False,
    "positional_damping_factor": 0.1,
    "attention_dropout": 0.0,
    "ffn_dropout": 0.05,
    "drop_path_rate": 0.01,
    "residual_scale_init": 0.5,
    "embedding_dropout": 0.0,
    "rms_norm_eps": 1e-6,
}

# Update parameters to make much larger model

ROUND2_T003_MODEL_CONFIG_LARGE = {
    "embedding_size": 32*48,
    "num_attention_heads": 32,
    "mlp_hidden_size": 3072,
    "num_layers": 8,
    "num_outputs": 5000,
    "num_cls": 16,
    "use_rope": True,
    "rope_base": 10000.0,
    "rope_fraction": 0.5,
    "use_absolute_positional_embeddings": False,
    "positional_damping_factor": 0.1,
    "attention_dropout": 0.0,
    "ffn_dropout": 0.05,
    "drop_path_rate": 0.01,
    "residual_scale_init": 0.5,
    "embedding_dropout": 0.0,
    "rms_norm_eps": 1e-6,
}



if __name__ == "__main__":
    model_cfg = dict(ROUND2_T003_MODEL_CONFIG)
    optimizer_cfg = dict(BEST_TRANSFER_OPTIMIZER_CONFIG)
    wandb_api_key = os.environ.get("WANDB_KEY")

    epochs = 20000
    batch_size = 4

    model = build_best_transfer_model(**model_cfg).to("cuda")
    model = model.to(torch.bfloat16)

    config = dict(MOVIELENS_SURROGATE_PRIOR_CONFIG)

    prior = SpectrumMissingnessPrior(
        missingness_type="mcar",
        config=config,
        batch_size=batch_size,
        verbose=False,
        entry_wise_features=False,
    )

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    borders_path = importlib.resources.files("tabimpute") / "data" / "borders.pt"
    with importlib.resources.as_file(borders_path) as path:
        bar_distribution = FullSupportBarDistribution(
            borders=torch.load(path).to(torch.device("cuda"))
        )

    model.train()

    name = f"tabimpute-spectrum-decay-movielens-surrogate_p_{config['p_missing']:.2f}-{config['num_rows_low']}x{config['num_cols_low']}"

    # ckpt = torch.load(
    #     "/root/tabular/mcpfn/src/tabimpute/workdir/tabimpute-new-stable_v2-rank_10_10-50x50/checkpoint_30000.pth"
    # )
    ckpt = None

    id_name = None

    callbacks = [
        WandbLoggerCallback(
            project=DEFAULT_WANDB_PROJECT,
            name=name,
            id=id_name,
            api_key=wandb_api_key,
            config={
                **model_cfg,
                **optimizer_cfg,
                **config,
                "batch_size": batch_size,
                "epochs": epochs,
                "checkpoint_every": DEFAULT_CHECKPOINT_EVERY,
            },
            log_dir="./wandb",
        )
    ]

    work_dir = os.path.join("workdir", name)
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": name,
                "model": model_cfg,
                "optimizer": optimizer_cfg,
                "training": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "device": "cuda",
                    "checkpoint_every": DEFAULT_CHECKPOINT_EVERY,
                    "wandb_project": DEFAULT_WANDB_PROJECT,
                },
                "prior": config,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    _, metrics = train(
        model,
        prior,
        bar_distribution,
        bar_distribution,
        epochs=epochs,
        device="cuda",
        callbacks=callbacks,
        run_name=name,
        ckpt=ckpt,
        checkpoint_every=DEFAULT_CHECKPOINT_EVERY,
        return_metrics=True,
        **optimizer_cfg,
    )
    with open(os.path.join(work_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print("Training complete")
