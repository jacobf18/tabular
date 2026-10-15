"""Matched 24k DWIP control with only the mean-MSE coefficient increased."""
import argparse
import json
from pathlib import Path
import sys

import run_spectral_query_experiments as queue
from run_dwip_spectral_geometry import build_commands as control_commands, SOURCES
from run_dwip_spectral_horizon import CONTROL, link_baseline
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, write_json
from run_rowcol_output_remedies import replace_arg
from tabimpute.model.model_new_stable import TabImputeModelStable

DIAGNOSTIC = "dwip_parameter_gradients_20260916"
DEPENDENCY = "dwip_spectral_horizon48k_20260916/transfer_evaluation"


def build_commands(root):
    trials, commands, prior, anchor = control_commands(root)
    replace_arg(commands["dwip1e5"], "--point-loss-weight", 2.)
    return trials, commands, prior, anchor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-name", default="dwip_spectral_point2_20260916")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = REPO / "runs/objective_queues" / args.queue_name
    control = REPO / "runs/objective_queues" / CONTROL
    diagnostic = REPO / "runs/objective_queues" / DIAGNOSTIC
    if json.loads((control / "status.json").read_text())["state"] != "complete":
        raise RuntimeError("The unchanged 24k control must be complete")
    if json.loads((diagnostic / "status.json").read_text())["state"] != "complete":
        raise RuntimeError("Parameter-gradient diagnostic must finish first")
    root.mkdir(parents=True, exist_ok=True)
    link_baseline(root, control)
    sources = [*SOURCES, "scripts/run_dwip_point_objective.py"]
    manifest = {"source_sha256": {path: sha256(REPO / path) for path in sources},
        "control": CONTROL, "diagnostic": DIAGNOSTIC,
        "diagnostic_sha256": {name: sha256(diagnostic / name) for name in ("manifest.json", "results.json")},
        "changed": "Mean-MSE coefficient 0.2 to2.0; no other training factor changes",
        "preserved": "Original DWIP weights and architecture; fresh FP32 AdamW; BF16 autocast; data seed1150924; LR1e-5; 24k cosine schedule with600 warmup and5% floor; priors, missingness, shapes, clipping, NLL1 and CDF0.05",
        "reason": "On24 generic cases, median weighted-point/NLL parameter gradient ratio is0.032259 at initialization and0.096627 after48k. Decoder medians are about0.011. This motivates a bounded hypothesis, not a predicted improvement.",
        "risk": "Sparse-case ratios can exceed0.6 already; 10x may increase clipping, worsen calibration or harm generalization. No parameter-space equalization claim; Adam/CDF interactions unmeasured.",
        "selection": "Unchanged synthetic macro-RMSE selector only; independent synthetic and all real-world scores are report-only",
        "budget": "24k steps; expected about1h training; external24h timeout",
        "dependency": DEPENDENCY}
    path = root / "point_objective_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Point-objective experiment changed; use a new queue")
    write_json(path, manifest)
    queue.build_commands = build_commands
    queue.MODEL_CLASS = TabImputeModelStable
    queue.ADDITIONAL_SOURCES = sources
    sys.argv = [sys.argv[0], "--queue-name", args.queue_name, "--wait-for", DEPENDENCY, "--wait-for-idle-gpu"]
    if args.dry_run:
        sys.argv.append("--dry-run")
    queue.main()


if __name__ == "__main__":
    main()
