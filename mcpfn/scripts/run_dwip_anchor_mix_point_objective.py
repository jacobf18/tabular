"""One fixed point-MSE factor under the unchanged25% rehearsal/64k recipe."""
import argparse
import json
from pathlib import Path

from run_dwip_anchor_mix_horizon import build_commands as control_commands
from run_dwip_budgeted_queue_v1 import artifact_budget, argument, immutable_json, run_queue
from run_dwip_continuous_spectrum import prepare_caches
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, WORK
from run_rowcol_output_remedies import replace_arg

CONTROL = "dwip_spectral_anchor_mix25_64k_20260916"
QUEUE = "dwip_spectral_anchor_mix25_point2_64k_20260916"


def verify_point_only_change(control, candidate):
    flag = "--point-loss-weight"
    if control.count(flag) != 1 or candidate.count(flag) != 1:
        raise ValueError("Require one explicit point-MSE coefficient")
    if float(argument(control, flag)) != .2 or float(argument(candidate, flag)) != 2.:
        raise ValueError("Only the preregistered point coefficient0.2 to2.0 is supported")
    restored = list(candidate)
    replace_arg(restored, flag, argument(control, flag))
    if restored != control:
        raise ValueError("Commands differ beyond the single point-MSE coefficient")


def build_commands(root):
    trials, commands, prior, anchor = control_commands(root)
    original = list(commands["dwip1e5"])
    replace_arg(commands["dwip1e5"], "--point-loss-weight", 2.)
    verify_point_only_change(original, commands["dwip1e5"])
    return trials, commands, prior, anchor


def require_complete_control(control):
    for folder in (control, *(control / f"{name}_evaluation" for name in ("transfer", "retention", "runtime"))):
        if json.loads((folder / "status.json").read_text())["state"] != "complete" or not (folder / "results.json").exists():
            raise RuntimeError("Finish all four matched-control stages before registering another factor")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-name", default=QUEUE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if Path(args.queue_name).name != args.queue_name or args.queue_name in {"", ".", ".."}:
        raise ValueError("Require a single fresh queue directory name")
    root = REPO / "runs/objective_queues" / args.queue_name
    control = root.parent / CONTROL
    if root == control:
        raise ValueError("The new point factor requires a distinct queue directory")
    require_complete_control(control)
    proposal_path = control / "next_objective_factor_proposal.json"
    proposal = json.loads(proposal_path.read_text())
    for path, expected in proposal["input_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Proposed factor evidence changed: {path}")
    control_manifest = json.loads((control / "manifest.json").read_text())
    common_sources = control_manifest["source_sha256"]
    for path, expected in common_sources.items():
        if sha256(REPO / path) != expected:
            raise ValueError(f"Frozen control source changed: {path}")
    _, expected_commands, _, _ = control_commands(control)
    if json.loads((control / "dwip1e5_command.json").read_text()) != expected_commands["dwip1e5"]:
        raise ValueError("Matched control command no longer reconstructs exactly")
    trials, commands, prior, anchor = build_commands(root)
    if sha256(anchor) != control_manifest["initial_checkpoint_sha256"]:
        raise ValueError("Original initialization changed")
    budget = artifact_budget(commands["dwip1e5"], WORK / f"tabimpute-{CONTROL}-dwip1e5")
    root.mkdir(parents=True, exist_ok=True)
    caches = prepare_caches(root, control, prior)
    identical = {name: sha256(root / name) == sha256(control / name) for name in ("validation.pth", "holdout.pth")}
    if not all(identical.values()):
        raise ValueError("Require byte-identical matched selector and independent caches")
    baseline, source = root / "baseline", (control / "baseline").resolve()
    if not baseline.exists() and not baseline.is_symlink():
        baseline.symlink_to(source, target_is_directory=True)
    elif not baseline.is_symlink() or baseline.resolve() != source:
        raise ValueError("Refusing to replace an existing baseline directory")
    sources = [*common_sources, "scripts/run_dwip_anchor_mix_point_objective.py"]
    manifest = {"control": CONTROL, "changed": "Only point-MSE coefficient0.2 to2.0",
        "preserved": proposal["preserved"], "synthetic_basis": proposal["synthetic_basis"],
        "risks": proposal["risks"], "selection": proposal["selection"],
        "design": "One within-mixture objective-factor test, not a new seed, prior, horizon or coefficient sweep. Original weights/fresh optimizer, not continuation of the64k control.",
        "rehearsal_scope": "Current explicit latent defaults and small shapes remain fixed. Not exact historical replay; original full configuration is unavailable. Rehearsal has not demonstrated protection for every latent80x40 group.",
        "followups": "Full transfer, exact frozen256-case retention and matched100/500/1000 square public runtime, all unconditional report-only. Add paired retention comparison with this matched64k control.",
        "budget": budget, "proposal_sha256": sha256(proposal_path),
        "control_manifest_sha256": sha256(control / "manifest.json"),
        "retention_control_sha256": sha256(control / "retention_evaluation/results.json"),
        "cache_equivalence": caches, "cache_files_byte_identical": identical,
        "source_sha256": {path: sha256(REPO / path) for path in sources}}
    immutable_json(root / "point_objective_manifest.json", manifest)
    run_queue(root, trials, commands, prior, anchor, budget, sources, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
