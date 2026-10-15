"""Paired normalization follow-ups after the unmodified spectral baseline queue."""
import argparse
import hashlib
import json
import sys

import run_spectral_query_experiments as queue
from run_rowcol_controlled_experiments import REPO, write_json
from run_rowcol_output_remedies import replace_arg


original_build_commands = queue.build_commands


def build_commands(root):
    _, original, prior, anchor = original_build_commands(root)
    trials = [dict(tag=f"shrinkage_df{df}", lr=2e-6, variance_df=df) for df in (1, 4)]
    commands = {}
    for trial in trials:
        command = list(original["joint_low"])
        command[1] = "scripts/train_spectral_support_shrinkage.py"
        replace_arg(command, "--run-name", f"tabimpute-{root.name}-{trial['tag']}")
        command += ["--support-variance-df", str(trial["variance_df"])]
        commands[trial["tag"]] = command
    return trials, commands, prior, anchor


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--queue-name", default="spectral_shrinkage_20260910")
    parser.add_argument("--wait-for", default="spectral_query_20260910")
    args, rest = parser.parse_known_args()
    root = REPO / "runs/objective_queues" / args.queue_name
    root.mkdir(parents=True, exist_ok=True)
    sources = ["scripts/run_spectral_shrinkage_experiments.py", "scripts/train_spectral_support_shrinkage.py",
               "scripts/spectral_support_normalization.py"]
    manifest = {"source_sha256": {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in sources},
        "control": "spectral_query_20260910/joint_low",
        "unchanged": "Anchor, batches, masks, objective, LR, schedule, validation, public interface",
        "changed": "Training sparse-support scale estimation only; variance pseudo-count 1 or 4",
        "selection": "Unmodified synthetic validation macro RMSE; external MTEB report only"}
    path = root / "normalization_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Normalization experiment source changed; use a new queue name")
    write_json(path, manifest)
    queue.build_commands = build_commands
    sys.argv = [sys.argv[0], "--queue-name", args.queue_name, "--wait-for", args.wait_for, *rest]
    queue.main()


if __name__ == "__main__":
    main()
