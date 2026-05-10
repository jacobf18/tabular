#!/usr/bin/env bash
set -euo pipefail

# Memory-efficient version of train_curriculum_rank10_rectangular_axes.sh.
# It uses the checkpoint-compatible model_new_stable_memory_efficient module,
# which activation-checkpoints transformer blocks during training.

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export REPO_DIR

BASE_RUN_DIR="${BASE_RUN_DIR:-${REPO_DIR}/src/tabimpute/workdir/tabimpute-new-stable_v2-rank_10_10-50x50}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${BASE_RUN_DIR}/checkpoint_30000.pth}"
export BASE_RUN_DIR CHECKPOINT_PATH

SHAPE_SCHEDULE="${SHAPE_SCHEDULE:-50x50,20x100,100x20}"
RUN_NAME="${RUN_NAME:-tabimpute-new-stable-memory-efficient_v2-rank_10_10-rectangular-axes-from_50x50_step30000}"
WORK_DIR="${WORK_DIR:-${REPO_DIR}/src/tabimpute/workdir/${RUN_NAME}}"
export SHAPE_SCHEDULE RUN_NAME WORK_DIR

CURRICULUM_STEPS="${CURRICULUM_STEPS:-15000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2500}"
DEVICE="${DEVICE:-cuda}"
WANDB_PROJECT="${WANDB_PROJECT:-tabimpute}"
WANDB_MODE="${WANDB_MODE:-offline}"
DECODE_ONLY_MISSING="${DECODE_ONLY_MISSING:-1}"
export CURRICULUM_STEPS BATCH_SIZE CHECKPOINT_EVERY DEVICE WANDB_PROJECT WANDB_MODE DECODE_ONLY_MISSING

python - <<'PY'
import importlib.resources
import json
import os
from pathlib import Path

import torch

from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable_memory_efficient import TabImputeModel
from tabimpute.prior.training_set_generation import MissingnessPrior
from tabimpute.train.callbacks import WandbLoggerCallback
from tabimpute.train_new_sweep import train


def parse_shape_schedule(raw: str) -> list[tuple[int, int]]:
    shapes = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Bad shape '{item}'. Expected ROWSxCOLS.")
        rows_raw, cols_raw = item.split("x", 1)
        rows = int(rows_raw)
        cols = int(cols_raw)
        if rows < 10 or cols < 10:
            raise ValueError(
                f"Shape {rows}x{cols} is too small for fixed rank 10."
            )
        shapes.append((rows, cols))
    if not shapes:
        raise ValueError("SHAPE_SCHEDULE did not contain any shapes.")
    return shapes


class AlternatingShapePrior:
    def __init__(
        self,
        *,
        base_prior_config: dict,
        shapes: list[tuple[int, int]],
        batch_size: int,
        device: str,
    ):
        self.batch_size = batch_size
        self.device = device
        self.shapes = shapes
        self._next_shape_index = 0
        self.priors = []
        for rows, cols in shapes:
            shape_config = dict(base_prior_config)
            shape_config.update(
                {
                    "num_rows_low": rows,
                    "num_rows_high": rows,
                    "num_cols_low": cols,
                    "num_cols_high": cols,
                    "latent_rank_low": 10,
                    "latent_rank_high": 10,
                }
            )
            self.priors.append(
                MissingnessPrior(
                    generator_type="latent_factor",
                    missingness_type="mcar",
                    config=shape_config,
                    batch_size=batch_size,
                    device=device,
                    verbose=False,
                    entry_wise_features=False,
                )
            )

    def get_batch(self, batch_size=None):
        if batch_size is not None and batch_size != self.batch_size:
            raise ValueError(
                "AlternatingShapePrior expects the configured batch size "
                f"{self.batch_size}, got {batch_size}."
            )
        prior = self.priors[self._next_shape_index]
        self._next_shape_index = (self._next_shape_index + 1) % len(self.priors)
        return prior.get_batch()


checkpoint_path = Path(os.environ["CHECKPOINT_PATH"])
work_dir = Path(os.environ["WORK_DIR"])
run_name = os.environ["RUN_NAME"]
shape_schedule = parse_shape_schedule(os.environ["SHAPE_SCHEDULE"])
device = os.environ["DEVICE"]
batch_size = int(os.environ["BATCH_SIZE"])
curriculum_steps = int(os.environ["CURRICULUM_STEPS"])
checkpoint_every = int(os.environ["CHECKPOINT_EVERY"])
wandb_project = os.environ["WANDB_PROJECT"]
decode_only_missing = os.environ["DECODE_ONLY_MISSING"].lower() in {
    "1",
    "true",
    "yes",
    "on",
}

if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

base_config_path = checkpoint_path.parent / "config.json"
with base_config_path.open("r", encoding="utf-8") as f:
    base_config = json.load(f)

model_cfg = dict(base_config["model"])
optimizer_cfg = dict(base_config["optimizer"])
prior_cfg = dict(base_config["prior"])

ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
start_epoch = int(ckpt["epoch"])
target_epoch = start_epoch + curriculum_steps

model = TabImputeModel(**model_cfg).to(device=device, dtype=torch.bfloat16)
model.load_state_dict(ckpt["model"], strict=True)

prior = AlternatingShapePrior(
    base_prior_config=prior_cfg,
    shapes=shape_schedule,
    batch_size=batch_size,
    device=device,
)

borders_path = importlib.resources.files("tabimpute") / "data" / "borders.pt"
with importlib.resources.as_file(borders_path) as path:
    bar_distribution = FullSupportBarDistribution(
        borders=torch.load(path).to(torch.device(device))
    )

work_dir.mkdir(parents=True, exist_ok=True)
run_config = {
    "run_name": run_name,
    "base_checkpoint_path": str(checkpoint_path),
    "base_config_path": str(base_config_path),
    "model": model_cfg,
    "optimizer": optimizer_cfg,
    "prior": prior_cfg,
    "memory_efficient_model": {
        "module": "tabimpute.model.model_new_stable_memory_efficient",
        "activation_checkpointing": True,
    },
    "rectangular_curriculum": {
        "shape_schedule": [f"{rows}x{cols}" for rows, cols in shape_schedule],
        "latent_rank": 10,
        "alternation": "batch",
    },
    "training": {
        "batch_size": batch_size,
        "checkpoint_every": checkpoint_every,
        "curriculum_steps": curriculum_steps,
        "decode_only_missing": decode_only_missing,
        "device": device,
        "start_epoch": start_epoch,
        "target_epoch": target_epoch,
        "wandb_project": wandb_project,
    },
}
with (work_dir / "config.json").open("w", encoding="utf-8") as f:
    json.dump(run_config, f, indent=2, sort_keys=True)

callbacks = [
    WandbLoggerCallback(
        project=wandb_project,
        name=run_name,
        id=run_name,
        api_key=os.environ.get("WANDB_KEY"),
        config=run_config,
        log_dir=str(work_dir / "wandb"),
    )
]

print(f"Loading checkpoint: {checkpoint_path}")
print(f"Continuing from step {start_epoch} to {target_epoch}")
print("Using memory-efficient activation-checkpointed model")
print(
    "Alternating fixed-rank-10 shapes: "
    + ", ".join(f"{rows}x{cols}" for rows, cols in shape_schedule)
)
print(f"p_missing={prior_cfg['p_missing']}, batch_size={batch_size}")
print(f"Decode only missing entries: {decode_only_missing}")
print(f"Work dir: {work_dir}")

_, metrics = train(
    model,
    prior,
    bar_distribution,
    bar_distribution,
    epochs=target_epoch,
    device=device,
    callbacks=callbacks,
    run_name=run_name,
    ckpt=ckpt,
    checkpoint_every=checkpoint_every,
    return_metrics=True,
    work_dir=str(work_dir),
    decode_only_missing=decode_only_missing,
    **optimizer_cfg,
)

with (work_dir / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, sort_keys=True)

print("Memory-efficient rectangular-axis curriculum phase complete")
PY
