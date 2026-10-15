"""Matched further training of the synthetic-selected geometry checkpoint."""
import argparse
import hashlib
import json
import sys

import run_spectral_query_experiments as queue
from run_spectral_geometry_experiments import build_commands as geometry_commands
from run_rowcol_controlled_experiments import REPO, WORK, write_json
from run_rowcol_output_remedies import replace_arg


def build_commands(root):
    _, originals, prior, _ = geometry_commands(root)
    anchor = WORK / "tabimpute-spectral_geometry_20260914-geometry_lr1e5/best_validation_checkpoint.pth"
    trials = [dict(tag="continue6e6", lr=6e-6), dict(tag="continue1e5", lr=1e-5)]
    commands = {}
    for trial in trials:
        command = list(originals["geometry_lr1e5"])
        for flag, value in {"--run-name": f"tabimpute-{root.name}-{trial['tag']}",
                "--init-model-checkpoint": str(anchor), "--lr": trial["lr"],
                "--data-seed": 1150924}.items():
            replace_arg(command, flag, value)
        commands[trial["tag"]] = command
    return trials, commands, prior, anchor


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--queue-name", default="spectral_geometry_continue_20260915")
    parser.add_argument("--wait-for", default="spectral_geometry_20260914")
    args, rest = parser.parse_known_args()
    root = REPO / "runs/objective_queues" / args.queue_name
    root.mkdir(parents=True, exist_ok=True)
    sources = ["scripts/run_spectral_geometry_continuations.py", "scripts/run_spectral_geometry_experiments.py",
        "scripts/train_general_spectral_geometry.py", "scripts/general_spectral_geometry.py",
        "scripts/train_spectral_support_shrinkage.py", "scripts/spectral_support_normalization.py"]
    manifest = {"source_sha256": {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in sources},
        "reason": "Geometry independent spectral RMSE improved 1.04268 to 1.00506 from 12k to 24k",
        "initialization": "Same synthetic-selected geometry 24k weights, fresh AdamW for both arms",
        "budget": "24k additional steps per arm, 48k total geometry exposure, about 8-9 hours per run",
        "schedule": "600 warmup then cosine to 5% floor; peaks 6e-6 and 1e-5",
        "data": "Same general spectral geometry distribution, fresh seed 1150924; no benchmark data",
        "selection": "Unchanged synthetic spectral macro RMSE, external broad and MTEB reporting",
        "caution": "This resets optimizer state; not equivalent to uninterrupted 48k training"}
    path = root / "continuation_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Continuation configuration changed; use a new queue name")
    write_json(path, manifest)
    queue.build_commands = build_commands
    sys.argv = [sys.argv[0], "--queue-name", args.queue_name, "--wait-for", args.wait_for, *rest]
    queue.main()


if __name__ == "__main__":
    main()
