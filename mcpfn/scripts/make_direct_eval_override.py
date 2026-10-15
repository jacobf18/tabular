#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best_checkpoint.pth")
    parser.add_argument("--model-overrides-json", required=True)
    parser.add_argument("--note", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source_config = source_dir / "config.json"
    source_checkpoint = source_dir / args.checkpoint
    if not source_config.exists():
        raise FileNotFoundError(source_config)
    if not source_checkpoint.exists():
        raise FileNotFoundError(source_checkpoint)

    with source_config.open("r", encoding="utf-8") as file:
        config = json.load(file)
    overrides = json.loads(args.model_overrides_json)
    if not isinstance(overrides, dict):
        raise ValueError("--model-overrides-json must be a JSON object.")
    if "model" in config and isinstance(config["model"], dict):
        config["model"].update(overrides)
    elif "config" in config and isinstance(config["config"], dict):
        config["config"].update(overrides)
    else:
        raise ValueError(f"No model/config object found in {source_config}")
    if args.note:
        config["eval_override_note"] = args.note
    config["eval_override_source_dir"] = str(source_dir)
    config["eval_override_source_checkpoint"] = args.checkpoint

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)
    shutil.copy2(source_checkpoint, output_dir / args.checkpoint)


if __name__ == "__main__":
    main()
