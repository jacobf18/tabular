"""Matched 48k cosine horizon from original DWIP weights, not a warm restart."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import run_spectral_query_experiments as queue
from run_dwip_spectral_geometry import build_commands as original_commands, SOURCES
from run_rowcol_controlled_experiments import REPO, write_json
from run_rowcol_output_remedies import replace_arg
from tabimpute.model.model_new_stable import TabImputeModelStable

CONTROL = "dwip_spectral_20260915"
CACHED_BASELINE_FILES = ("baseline_spectral.json", "baseline_general.json", "baseline_mteb.json",
    "baseline_spectral_predictions.pth", "baseline_general_predictions.pth", "holdout.pth", "validation.pth")


def build_commands(root):
    trials, commands, prior, anchor = original_commands(root)
    command = commands["dwip1e5"]
    phases = json.loads(command[command.index("--phases-json") + 1])
    phases[-1]["end_step"] = 48000
    replace_arg(command, "--phases-json", json.dumps(phases))
    replace_arg(command, "--decay-steps", 47400)
    position = command.index("--snapshot-steps") + 1
    while position < len(command) and not command[position].startswith("--"):
        position += 1
    command[position:position] = ["24000", "36000"]
    return trials, commands, prior, anchor


def link_baseline(root, control):
    for name in CACHED_BASELINE_FILES:
        source, destination = control / name, root / name
        if not source.exists():
            raise FileNotFoundError(f"Missing matched baseline cache: {source}")
        if destination.is_symlink() and destination.resolve() == source.resolve():
            continue
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"Refusing to replace an existing baseline artifact: {destination}")
        destination.symlink_to(source.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-name", default="dwip_spectral_horizon48k_20260916")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = REPO / "runs/objective_queues" / args.queue_name
    control = REPO / "runs/objective_queues" / CONTROL
    root.mkdir(parents=True, exist_ok=True)
    if json.loads((control / "status.json").read_text())["state"] != "complete":
        raise RuntimeError("The 24k DWIP control must finish before registering its matched comparison")
    link_baseline(root, control)
    sources = [*SOURCES, "scripts/run_dwip_spectral_horizon.py"]
    manifest = {
        "source_sha256": {path: hashlib.sha256((REPO / path).read_bytes()).hexdigest() for path in sources},
        "control": CONTROL, "baseline_cache_source": str(control),
        "changed": "Cosine horizon: 24k to 48k; same 600-step warmup and 5% floor",
        "preserved": "Original DWIP initialization, architecture, fresh AdamW, LR1e-5, data seed1150924, priors, shapes, losses and selector",
        "reason": "Independent spectral RMSE improved from 1.095142 at 12k to 1.068364 at 24k; final synthetic-selected step was24k at the LR floor",
        "design": "Train from the original checkpoint, not the 24k weights. First24k training data match the control; only the LR trajectory changes.",
        "selection": "Unchanged synthetic selector; broad synthetic and all real-world results report-only",
        "dependency": f"{CONTROL}/transfer_evaluation", "budget": "48k steps, expected about2h; external24h timeout",
    }
    path = root / "horizon_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Horizon comparison changed; use a fresh queue name")
    write_json(path, manifest)
    queue.build_commands = build_commands
    queue.MODEL_CLASS = TabImputeModelStable
    queue.ADDITIONAL_SOURCES = sources
    sys.argv = [sys.argv[0], "--queue-name", args.queue_name, "--wait-for",
                f"{CONTROL}/transfer_evaluation", "--wait-for-idle-gpu"]
    if args.dry_run:
        sys.argv.append("--dry-run")
    queue.main()


if __name__ == "__main__":
    main()
