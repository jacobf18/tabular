#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tabimpute.model.model_rowcol_svt_hybrid import (
    TabImputeModelRowColumnSVTHybrid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_checkpoint.pth")
    parser.add_argument("--model-family", default=None)
    parser.add_argument("--model-overrides-json", required=True)
    parser.add_argument("--note", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source_config_path = source_dir / "config.json"
    source_checkpoint_path = source_dir / args.checkpoint
    if not source_config_path.exists():
        raise FileNotFoundError(source_config_path)
    if not source_checkpoint_path.exists():
        raise FileNotFoundError(source_checkpoint_path)

    with source_config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    model_config = dict(config.get("model", config.get("config", {})))
    if not model_config:
        raise ValueError(f"No model/config object found in {source_config_path}")
    model_config.update(json.loads(args.model_overrides_json))
    model_family = args.model_family or config.get("model_family", "rowcol_svt_hybrid")

    source_checkpoint = torch.load(
        source_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if model_family not in {
        "rowcol_svt_hybrid",
        "rowcol_latent_svt_hybrid",
        "row_column_svt_hybrid",
    }:
        raise ValueError(
            "This helper currently materializes rowcol_svt_hybrid checkpoints, "
            f"got {model_family!r}."
        )
    model = TabImputeModelRowColumnSVTHybrid(**model_config)
    model_state = model.state_dict()
    checkpoint_state = source_checkpoint["model"]
    compatible_state = {}
    skipped_shape_keys = []
    for key, value in checkpoint_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible_state[key] = value
        elif key in model_state:
            skipped_shape_keys.append(
                f"{key}: checkpoint {tuple(value.shape)} != model {tuple(model_state[key].shape)}"
            )
    load_result = model.load_state_dict(compatible_state, strict=False)

    output_config = dict(config)
    if "model" in output_config and isinstance(output_config["model"], dict):
        output_config["model"] = model_config
    else:
        output_config["config"] = model_config
    output_config["model_family"] = model_family
    output_config["partial_init_source_dir"] = str(source_dir)
    output_config["partial_init_source_checkpoint"] = args.checkpoint
    if args.note:
        output_config["partial_init_note"] = args.note

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(output_config, file, indent=2, sort_keys=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": source_checkpoint.get("epoch", 0),
            "metrics": source_checkpoint.get("metrics", {}),
            "partial_init_missing_keys": load_result.missing_keys,
            "partial_init_unexpected_keys": load_result.unexpected_keys,
            "partial_init_skipped_shape_keys": skipped_shape_keys,
        },
        output_dir / args.checkpoint,
    )
    print(
        "Materialized checkpoint with "
        f"{len(load_result.missing_keys)} missing, "
        f"{len(load_result.unexpected_keys)} unexpected, and "
        f"{len(skipped_shape_keys)} skipped shape keys."
    )
    if skipped_shape_keys:
        print("; ".join(skipped_shape_keys[:8]))


if __name__ == "__main__":
    main()
