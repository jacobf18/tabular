"""Isolate optimizer-history reset using the completed matched fresh-AdamW control."""
import argparse
import hashlib
import json
import sys

import torch

import run_spectral_query_experiments as queue
from run_spectral_geometry_continuations import build_commands as previous_commands
from run_rowcol_controlled_experiments import REPO, write_json
from run_rowcol_output_remedies import replace_arg


def build_commands(root):
    _, originals, prior, selected = previous_commands(root)
    anchor = selected.with_name("latest_checkpoint.pth")
    command = list(originals["continue1e5"])
    replace_arg(command, "--init-model-checkpoint", str(anchor))
    replace_arg(command, "--run-name", f"tabimpute-{root.name}-retained1e5")
    command.append("--retain-init-optimizer-state")
    return [dict(tag="retained1e5", lr=1e-5)], {"retained1e5": command}, prior, anchor


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--queue-name", default="spectral_optimizer_20260915")
    parser.add_argument("--wait-for", default="spectral_geometry_continue_20260915")
    args, rest = parser.parse_known_args()
    root = REPO / "runs/objective_queues" / args.queue_name
    root.mkdir(parents=True, exist_ok=True)
    _, _, _, anchor = build_commands(root)
    source = torch.load(anchor, weights_only=True, map_location="cpu", mmap=True)
    selected = torch.load(anchor.with_name("best_validation_checkpoint.pth"),
                          weights_only=True, map_location="cpu", mmap=True)
    if source["epoch"] != 24000 or source["model"].keys() != selected["model"].keys():
        raise ValueError("Source checkpoint does not match the completed control initialization")
    if not all(torch.equal(value, selected["model"][name]) for name, value in source["model"].items()):
        raise ValueError("Optimizer snapshot weights differ from the control initialization")
    if not source.get("optimizer", {}).get("state"):
        raise ValueError("Missing optimizer history")
    del source, selected
    paths = ["scripts/run_spectral_optimizer_continuation.py", "scripts/optimizer_continuation.py",
             "scripts/run_spectral_geometry_continuations.py", "scripts/general_spectral_geometry.py",
             "scripts/train_general_spectral_geometry.py", "scripts/spectral_support_normalization.py"]
    manifest = dict(
        source_sha256={p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in paths},
        control="spectral_geometry_continue_20260915/continue1e5",
        changed="Retain AdamW first/second moments and parameter update counts from geometry 24k",
        preserved="Exact model weights (verified), generic spectral prior, seed 1150924, masks, objective, 24k cosine with 600 warmup and LR 1e-5",
        selection="Unchanged synthetic selector; cached independent and MTEB evaluation report-only",
        reason="Fresh-AdamW continuations first degrade, then recover; test whether history reset explains loss of transfer",
        caution="Restarted LR schedule and data stream, not uninterrupted training; control is cached, single seed",
        budget="24k additional steps, approximately 10h, total geometry exposure 48k")
    path = root / "optimizer_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Configuration changed; use a new queue")
    write_json(path, manifest)
    queue.build_commands = build_commands
    sys.argv = [sys.argv[0], "--queue-name", args.queue_name, "--wait-for", args.wait_for, *rest]
    queue.main()


if __name__ == "__main__":
    main()
