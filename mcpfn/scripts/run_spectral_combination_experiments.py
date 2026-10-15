"""Combine observed-only variance shrinkage with higher learning rates."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import run_spectral_shrinkage_experiments as runner
from run_rowcol_controlled_experiments import REPO, write_json
from run_rowcol_output_remedies import replace_arg


original_build = runner.build_commands


def build_commands(root):
    _, base, prior, anchor = original_build(root)
    trials = [dict(tag="df1_lr6e6", lr=6e-6, variance_df=1),
              dict(tag="df1_lr1e5", lr=1e-5, variance_df=1)]
    commands = {}
    for trial in trials:
        command = list(base["shrinkage_df1"])
        replace_arg(command, "--run-name", f"tabimpute-{root.name}-{trial['tag']}")
        replace_arg(command, "--lr", trial["lr"])
        commands[trial["tag"]] = command
    return trials, commands, prior, anchor


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--queue-name", default="spectral_combination_20260913")
    parser.add_argument("--wait-for", default="spectral_lr_20260912")
    args, rest = parser.parse_known_args()
    root = REPO / "runs/objective_queues" / args.queue_name
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "normalization_choice": "df1: lower independent spectral RMSE and NLL than df4 at 24k",
        "controls": ["spectral_shrinkage_20260910/shrinkage_df1",
                     "spectral_query_20260910/joint_reference", "spectral_lr_20260912/lr1e5"],
        "design": "Same anchor, seeds, prior, masking, 24k cosine and objective; combine df1 with 6e-6/1e-5",
        "selection": "Unmodified synthetic validation only; broad and MTEB reporting only"}
    path = root / "combination_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Combination experiment changed; use a fresh queue name")
    write_json(path, manifest)
    runner.build_commands = build_commands
    sys.argv = [sys.argv[0], "--queue-name", args.queue_name, "--wait-for", args.wait_for, *rest]
    runner.main()


if __name__ == "__main__":
    main()
