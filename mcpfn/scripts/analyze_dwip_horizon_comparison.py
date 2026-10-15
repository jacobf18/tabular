"""CPU-only, equal-step comparison of the matched DWIP cosine-horizon trials."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_dwip_spectral_evidence import cached_comparison, compare_scores
from analyze_rowcol_controlled_evidence import read_jsonl
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, WORK, write_json

MATCHED_SOURCES = ("scripts/train_direct_pure_rabel_curriculum.py", "scripts/train_dwip_spectral_geometry.py",
    "scripts/general_spectral_geometry.py", "scripts/spectral_support_normalization.py",
    "scripts/rowcol_fixed_validation.py", "src/tabimpute/model/model_new_stable.py",
    "src/tabimpute/prior/spectrum_training_set_generation.py")


def verify_source_hashes(control, candidate):
    for path in MATCHED_SOURCES:
        if not control.get(path) or control[path] != candidate.get(path):
            raise ValueError(f"Recorded training/evaluation source hashes differ: {path}")
    return {path: control[path] for path in MATCHED_SOURCES}


def normalized_command(command):
    command = list(command)
    phases_index = command.index("--phases-json") + 1
    phases = json.loads(command[phases_index])
    phases[-1]["end_step"] = 0
    command[phases_index] = json.dumps(phases, sort_keys=True)
    for flag in ("--run-name", "--fixed-validation-cache", "--decay-steps"):
        command[command.index(flag) + 1] = "<comparison-specific>"
    if "--snapshot-steps" in command:
        start = command.index("--snapshot-steps")
        end = start + 1
        while end < len(command) and not command[end].startswith("--"):
            end += 1
        del command[start:end]
    return command


def verify_matched_commands(control, candidate):
    if normalized_command(control) != normalized_command(candidate):
        raise ValueError("Trials differ beyond the declared horizon and output locations")
    return {name: json.loads(command[command.index("--phases-json") + 1])[-1]["end_step"]
            for name, command in (("control", control), ("candidate", candidate))}


def collect(control_root, candidate_root):
    commands = {name: json.loads((root / "dwip1e5_command.json").read_text())
                for name, root in (("control", control_root), ("candidate", candidate_root))}
    horizons = verify_matched_commands(commands["control"], commands["candidate"])
    manifests = [json.loads((root / "manifest.json").read_text()) for root in (control_root, candidate_root)]
    matched_sources = verify_source_hashes(*(manifest["source_sha256"] for manifest in manifests))
    cache_paths = [Path(command[command.index("--fixed-validation-cache") + 1]) for command in commands.values()]
    if not cache_paths[0].samefile(cache_paths[1]) and sha256(cache_paths[0]) != sha256(cache_paths[1]):
        raise ValueError("Validation caches differ; equal-step score comparison is not matched")
    work = {name: WORK / command[command.index("--run-name") + 1] for name, command in commands.items()}
    validation = {name: read_jsonl(path / "validation_stream.jsonl") for name, path in work.items()}
    by_step = {name: {row["step"]: row for row in rows} for name, rows in validation.items()}
    shared = sorted(by_step["control"].keys() & by_step["candidate"].keys())
    report = {"updated_utc": datetime.now(timezone.utc).isoformat(), "horizons": horizons,
        "control": str(control_root), "candidate": str(candidate_root), "validation_cache_matched": True,
        "matched_source_sha256": matched_sources,
        "comparison": "Equal-step synthetic comparisons; later steps also use more training. No benchmark selection.",
        "same_budget_validation": {str(step): compare_scores(by_step["control"][step], by_step["candidate"][step])
                                   for step in shared}, "selected_independent_holdouts": {}}
    for domain, name in (("spectral", "dwip1e5_selected_predictions.pth"), ("general", "dwip1e5_general_predictions.pth")):
        before, after = control_root / name, candidate_root / name
        if before.exists() and after.exists():
            report["selected_independent_holdouts"][domain] = cached_comparison(before, after)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name, path in work.items():
        rows = validation[name]
        if rows:
            axes[0].plot([row["step"] for row in rows], [row["macro_rmse"] for row in rows],
                         "o-", label=f"{name}: {horizons[name]}-step schedule")
        training = read_jsonl(path / "metrics_stream.jsonl")
        if training:
            axes[1].plot([row["step"] for row in training], [row["lr"] for row in training], label=name)
    axes[0].set_ylabel("Fixed-selector macro RMSE")
    axes[1].set_ylabel("Learning rate")
    for axis in axes:
        axis.set_xlabel("Training step")
        axis.grid(alpha=.2)
        axis.legend()
    figure.suptitle("Same initialization, data and architecture; different cosine horizon")
    figure.savefig(candidate_root / "horizon_comparison.png", dpi=140)
    plt.close(figure)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", default="dwip_spectral_20260915")
    parser.add_argument("--candidate", default="dwip_spectral_horizon48k_20260916")
    args = parser.parse_args()
    queues = REPO / "runs/objective_queues"
    report = collect(queues / args.control, queues / args.candidate)
    write_json(queues / args.candidate / "horizon_comparison.json", report)
    print(json.dumps({step: {key: scores[key] for key in (
        "baseline_macro_rmse", "candidate_macro_rmse", "groups_improved", "group_count")}
        for step, scores in report["same_budget_validation"].items()}))


if __name__ == "__main__":
    main()
