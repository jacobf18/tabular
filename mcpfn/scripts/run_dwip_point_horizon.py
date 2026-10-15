"""Complete the original-seed loss-weight by cosine-horizon controlled comparison."""
import argparse
import json
import sys

import run_spectral_query_experiments as queue
from run_dwip_spectral_geometry import SOURCES
from run_dwip_spectral_horizon import build_commands as control_commands, link_baseline
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, write_json
from run_rowcol_output_remedies import replace_arg
from tabimpute.model.model_new_stable import TabImputeModelStable

CONTROL = "dwip_spectral_horizon48k_20260916"
SHORT_POINT_CONTROL = "dwip_spectral_point2_20260916"
AVERAGE = "dwip_spectral_cross_seed_average_20260916"
DEPENDENCY = "dwip_spectral_seed2_48k_20260916/transfer_evaluation"


def build_commands(root):
    trials, commands, prior, anchor = control_commands(root)
    replace_arg(commands["dwip1e5"], "--point-loss-weight", 2.)
    return trials, commands, prior, anchor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-name", default="dwip_spectral_point2_horizon48k_20260916")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = REPO / "runs/objective_queues" / args.queue_name
    for name in (CONTROL, SHORT_POINT_CONTROL):
        if json.loads((root.parent / name / "transfer_evaluation/status.json").read_text())["state"] != "complete":
            raise RuntimeError(f"Matched control must be complete: {name}")
    average = root.parent / AVERAGE
    decision = json.loads((average / "selection.json").read_text())
    if decision["average_accepted"] or json.loads((average / "status.json").read_text())["state"] != "synthetic_evaluation_complete":
        raise RuntimeError("Review the cross-seed average before starting another training trial")
    root.mkdir(parents=True, exist_ok=True)
    link_baseline(root, root.parent / CONTROL)
    sources = [*SOURCES, "scripts/run_dwip_spectral_horizon.py", "scripts/run_dwip_point_horizon.py"]
    manifest = {"source_sha256": {path: sha256(REPO / path) for path in sources},
        "control": CONTROL, "short_point_control": SHORT_POINT_CONTROL,
        "average_decision_sha256": sha256(average / "selection.json"),
        "changed_vs_48k_control": "Only mean-MSE coefficient0.2 to2.0; original training/data seed1150924",
        "changed_vs_24k_point2": "Only cosine horizon24k to48k with identical600-step warmup and5% floor; prespecified extra snapshots",
        "design": "Fourth arm of a2x2 loss-weight(0.2,2.0) by horizon(24k,48k) comparison, all from original weights and seed1150924. Do not combine seed2 or continuous-prior changes.",
        "reason": "The24k point2 arm improves independent broad synthetic RMSE on11/12 groups but barely changes fixed spectral RMSE; the48k original-loss arm improves independent spectral RMSE. Measure their interaction rather than infer that it is additive.",
        "risk": "Point2 increases pre-clip gradient norms and may trade calibration or spectral accuracy for mean error. More training may amplify, not cure, these effects.",
        "preserved": "Unchanged DWIP architecture and original checkpoint, original general priors/missingness/shapes, fresh FP32 AdamW/BF16 autocast, LR1e-5, NLL1/CDF0.05, clipping, identical fixed synthetic cases",
        "selection": "Fixed synthetic macro RMSE only; all independent and real-world scores report-only; no benchmark-specific priors or benchmark-driven checkpoint choice",
        "budget": "48k steps; expected about2h training; external24h timeout", "dependency": DEPENDENCY}
    path = root / "point_horizon_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Loss-horizon experiment changed; use a new queue")
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
