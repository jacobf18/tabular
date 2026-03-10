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
    BEST_TRANSFER_MODEL_CONFIG,
    BEST_TRANSFER_OPTIMIZER_CONFIG,
    build_best_transfer_model,
)
from tabimpute.prior.training_set_generation import MissingnessPrior
from tabimpute.train.callbacks import WandbLoggerCallback
from tabimpute.train_new_sweep import train


DEFAULT_EPOCHS = 30000
DEFAULT_BATCH_SIZE = 16
DEFAULT_P_MISSING = 0.4
DEFAULT_CHECKPOINT_EVERY = 2500
DEFAULT_WANDB_PROJECT = "tabimpute"


if __name__ == "__main__":
    model_cfg = dict(BEST_TRANSFER_MODEL_CONFIG)
    optimizer_cfg = dict(BEST_TRANSFER_OPTIMIZER_CONFIG)

    epochs = DEFAULT_EPOCHS
    batch_size = DEFAULT_BATCH_SIZE
    p_missing = DEFAULT_P_MISSING

    model = build_best_transfer_model(num_outputs=5000).to("cuda")
    model = model.to(torch.bfloat16)

    config = {
        "num_rows_low": 10,
        "num_rows_high": 50,
        "num_cols_low": 5,
        "num_cols_high": 50,
        "p_missing": p_missing,
        "apply_feature_warping_prob": 0.0,
        "apply_quantization_prob": 0.0,
        "latent_rank_low": 1,
        "latent_rank_high": 11,
        "latent_spike_p": 0.3,
        "latent_slab_sigma": 2.0,
    }

    prior = MissingnessPrior(
        generator_type="latent_factor",
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

    name = (
        "tabimpute-new-stable_v2-"
        f"mcar_p{p_missing}-num_cls_{model_cfg['num_cls']}-rank_1_11"
    )

    ckpt = None
    id_name = None

    callbacks = [
        WandbLoggerCallback(
            project=DEFAULT_WANDB_PROJECT,
            name=name,
            id=id_name,
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
