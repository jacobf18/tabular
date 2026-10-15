"""Replicate the point-loss factor on the already-established second data seed."""
import argparse
import json
import sys

import run_spectral_query_experiments as queue
from run_dwip_seed_replication import build_commands as control_commands
from run_dwip_spectral_geometry import SOURCES
from run_dwip_spectral_horizon import link_baseline
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, write_json
from run_rowcol_output_remedies import replace_arg
from tabimpute.model.model_new_stable import TabImputeModelStable

CONTROL = "dwip_spectral_seed2_48k_20260916"
SEED1_CONTROL = "dwip_spectral_point2_horizon48k_20260916"
DEPENDENCY = "dwip_spectral_continuous_horizon48k_20260916/transfer_evaluation"


def build_commands(root):
    trials, commands, prior, anchor = control_commands(root)
    replace_arg(commands["dwip1e5"], "--point-loss-weight", 2.)
    return trials, commands, prior, anchor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-name", default="dwip_spectral_point2_seed2_48k_20260916")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = REPO / "runs/objective_queues" / args.queue_name
    for name in (CONTROL, SEED1_CONTROL):
        if json.loads((root.parent / name / "transfer_evaluation/status.json").read_text())["state"] != "complete":
            raise RuntimeError(f"Matched control must be complete: {name}")
    if json.loads((root.parent / DEPENDENCY / "status.json").read_text())["state"] != "complete":
        raise RuntimeError("Complete the continuous-prior panel before registering another run")
    root.mkdir(parents=True, exist_ok=True)
    link_baseline(root, root.parent / CONTROL)
    sources = [*SOURCES, "scripts/run_dwip_spectral_horizon.py", "scripts/run_dwip_seed_replication.py",
               "scripts/run_dwip_point_seed_replication.py"]
    manifest = {"source_sha256": {path: sha256(REPO / path) for path in sources},
        "control": CONTROL, "seed1_point_control": SEED1_CONTROL,
        "changed_vs_seed2_original_loss": "Only mean-MSE coefficient0.2 to2.0",
        "changed_vs_seed1_point2": "Only training/data seed1150924 to the already-established1150925; model/global seed stays1201",
        "design": "Complete the existing48k loss-weight(0.2,2) by data-seed(1150924,1150925) comparison. This is a fixed replication, not a search over new seeds.",
        "reason": "Point2 improves independent broad synthetic RMSE on11/12 groups at24k and7/12 at48k for seed1; its reproducibility under the existing seed2 is unknown. Original-loss seed2 already has an independent spectral control. The continuous48k prior wins the fixed selector but worsens independent spectral RMSE versus original48k, so do not bundle its prior change with this replication.",
        "risk": "Point2 increases pre-clip gradient norms; one additional seed cannot establish an optimization variability distribution or guarantee a generalization gain. Report both seeds and every regression.",
        "preserved": "Original DWIP checkpoint and unchanged architecture; original discontinuous spectral/geometry priors, noise, missingness, df1 support, shapes;48k cosine with600 warmup and5% floor; LR1e-5; NLL1/CDF0.05; fresh FP32 AdamW/BF16 autocast; identical fixed synthetic cases",
        "selection": "Per-run checkpoint selection remains fixed synthetic macro RMSE only. Independent and all real-world results are report-only, with no benchmark-specific prior or benchmark-driven seed/checkpoint choice.",
        "budget": "48k steps, about2h expected, external24h timeout", "dependency": DEPENDENCY}
    path = root / "point_seed_replication_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Point-seed replication changed; use a fresh queue")
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
