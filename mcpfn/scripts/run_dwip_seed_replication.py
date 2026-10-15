"""One fixed second-seed replication of the original-prior 48k DWIP schedule."""
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
DEPENDENCY = "dwip_spectral_continuous_20260916/transfer_evaluation"
DATA_SEED = 1150925


def build_commands(root):
    trials, commands, prior, anchor = control_commands(root)
    replace_arg(commands["dwip1e5"], "--data-seed", DATA_SEED)
    return trials, commands, prior, anchor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-name", default="dwip_spectral_seed2_48k_20260916")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = REPO / "runs/objective_queues" / args.queue_name
    control = root.parent / CONTROL
    if json.loads((control / "transfer_evaluation/status.json").read_text())["state"] != "complete":
        raise RuntimeError("The original 48k comparison must be complete")
    root.mkdir(parents=True, exist_ok=True)
    link_baseline(root, control)
    sources = [*SOURCES, "scripts/run_dwip_spectral_horizon.py", "scripts/run_dwip_seed_replication.py"]
    manifest = {"source_sha256": {path: sha256(REPO / path) for path in sources},
        "control": CONTROL, "changed": "Training/data seed1150924 to1150925; one fixed replicate, not a seed sweep",
        "preserved": "Original DWIP checkpoint/architecture, original discontinuous spectral priors, FP32 AdamW/BF16 autocast, NLL1/MSE0.2/CDF0.05, LR1e-5, 600 warmup, 48k cosine schedule with5% floor, all shapes/missingness and fixed evaluation caches",
        "reason": "The original48k trial remains strongest on fixed and independent spectral RMSE. Neither point2 nor continuous-prior24k improves independent spectral RMSE. Quantify training/data-seed variability before adding further model/prior/loss factors.",
        "interpretation": "Training matrices and random draws change together; this is not a pure GPU numerical-determinism test. Two seeds do not establish a variability distribution or statistical significance.",
        "selection": "Per-run checkpoint selection uses identical fixed synthetic macro RMSE. Report both seeds and all real-world outcomes; never choose a seed/checkpoint by benchmark scores.",
        "budget": "48k steps; expected about2h training; external24h timeout", "dependency": DEPENDENCY}
    path = root / "seed_replication_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Seed replication changed; use a new queue")
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
