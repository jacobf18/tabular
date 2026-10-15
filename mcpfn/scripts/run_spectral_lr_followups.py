"""Additional matched LR trials prompted by independent spectral improvement."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import run_spectral_query_experiments as queue
from run_rowcol_controlled_experiments import REPO, write_json
from run_rowcol_output_remedies import replace_arg


original_build_commands = queue.build_commands


def build_commands(root):
    _, original, prior, anchor = original_build_commands(root)
    trials = [dict(tag="lr1e5", lr=1e-5), dict(tag="lr2e5", lr=2e-5)]
    commands = {}
    for trial in trials:
        command = list(original["joint_low"])
        replace_arg(command, "--run-name", f"tabimpute-{root.name}-{trial['tag']}")
        replace_arg(command, "--lr", trial["lr"])
        commands[trial["tag"]] = command
    return trials, commands, prior, anchor


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--queue-name", default="spectral_lr_20260912")
    parser.add_argument("--wait-for", default="spectral_shrinkage_20260910")
    args, rest = parser.parse_known_args()
    root = REPO / "runs/objective_queues" / args.queue_name
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "controls": ["spectral_query_20260910/joint_low", "spectral_query_20260910/joint_reference"],
        "reason": "6e-6 improved independent spectral RMSE versus 2e-6: 1.04167 to .99295",
        "changed": "Peak LR only; 1e-5 and 2e-5, same anchor, seed, spectral prior, objective and 24k cosine",
        "selection": "Synthetic spectral validation only; broad holdout and MTEB report only",
        "risk": "Higher LR may worsen broad NLL or destabilize training; inspect saved curves and gradients"}
    path = root / "lr_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("LR follow-up source changed; use a new queue name")
    write_json(path, manifest)
    queue.build_commands = build_commands
    sys.argv = [sys.argv[0], "--queue-name", args.queue_name, "--wait-for", args.wait_for, *rest]
    queue.main()


if __name__ == "__main__":
    main()
