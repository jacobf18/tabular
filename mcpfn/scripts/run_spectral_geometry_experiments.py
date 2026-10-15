"""Matched spectral geometry priors, keeping the established evaluation transforms."""
import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import run_spectral_combination_experiments as runner
from run_rowcol_controlled_experiments import REPO, write_json
from run_rowcol_output_remedies import replace_arg


original_build = runner.build_commands


def build_commands(root):
    _, originals, prior, anchor = original_build(root)
    prior = deepcopy(prior)
    for spec in prior["mixed_priors"]:
        spec["config"]["spectrum_basis_kind"] = "mixed"
    trials = [dict(tag="geometry_lr6e6", lr=6e-6, variance_df=1),
              dict(tag="geometry_lr1e5", lr=1e-5, variance_df=1)]
    commands = {}
    for trial, baseline in zip(trials, ("df1_lr6e6", "df1_lr1e5")):
        command = list(originals[baseline])
        command[1] = "scripts/train_general_spectral_geometry.py"
        replace_arg(command, "--run-name", f"tabimpute-{root.name}-{trial['tag']}")
        replace_arg(command, "--prior-overrides-json", json.dumps(prior))
        commands[trial["tag"]] = command
    return trials, commands, prior, anchor


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--queue-name", default="spectral_geometry_20260914")
    parser.add_argument("--wait-for", default="spectral_combination_20260913")
    args, rest = parser.parse_known_args()
    root = REPO / "runs/objective_queues" / args.queue_name
    root.mkdir(parents=True, exist_ok=True)
    sources = ["scripts/run_spectral_geometry_experiments.py", "scripts/train_general_spectral_geometry.py",
               "scripts/general_spectral_geometry.py"]
    manifest = {"source_sha256": {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in sources},
        "controls": ["spectral_combination_20260913/df1_lr6e6", "spectral_combination_20260913/df1_lr1e5"],
        "changed": "Singular-vector sampling only: independent Haar/coherent/smooth/offset choice per axis",
        "preserved": "Spectral parameter distributions, clean full-rank target, missingness, anchor, optimizer and 24k schedule",
        "validation": "Unchanged Haar spectral selector and broad holdout; MTEB report-only",
        "caution": "Geometry draws consume different RNG counts; per-step matrices are not identical to control",
        "rationale": "Repeated broad distributional regressions despite improving Haar spectral accuracy"}
    path = root / "geometry_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Geometry experiment changed; use a fresh queue name")
    write_json(path, manifest)
    runner.build_commands = build_commands
    sys.argv = [sys.argv[0], "--queue-name", args.queue_name, "--wait-for", args.wait_for, *rest]
    runner.main()


if __name__ == "__main__":
    main()
