"""One fixed Huber1 point-objective test under the unchanged64k mixture recipe."""
import argparse
import json
from pathlib import Path

from run_dwip_anchor_mix_point_objective import build_commands as control_commands, require_complete_control
from run_dwip_budgeted_queue_v1 import artifact_budget, argument, immutable_json, run_queue
from run_dwip_continuous_spectrum import prepare_caches
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, WORK
from run_rowcol_output_remedies import replace_arg

CONTROL = "dwip_spectral_anchor_mix25_point2_64k_20260916"
QUEUE = "dwip_spectral_anchor_mix25_huber1_64k_20260916"


def verify_huber_only_change(control, candidate):
    for command in (control, candidate):
        for flag in ("--point-loss-kind", "--huber-delta", "--point-loss-weight"):
            if command.count(flag) != 1:
                raise ValueError(f"Require one explicit objective argument: {flag}")
        if float(argument(command, "--point-loss-weight")) != 2.:
            raise ValueError("Keep the point coefficient fixed at2.0")
    if argument(control, "--point-loss-kind") != "mse" or argument(candidate, "--point-loss-kind") != "huber":
        raise ValueError("Only the preregistered MSE-to-Huber objective is supported")
    if float(argument(control, "--huber-delta")) != .05 or float(argument(candidate, "--huber-delta")) != 1.:
        raise ValueError("Require inactive control delta0.05 and fixed candidate Huber delta1")
    restored = list(candidate)
    for flag in ("--point-loss-kind", "--huber-delta"):
        replace_arg(restored, flag, argument(control, flag))
    if restored != control:
        raise ValueError("Commands differ beyond the declared point-loss function")


def build_commands(root):
    trials, commands, prior, anchor = control_commands(root)
    original = list(commands["dwip1e5"])
    replace_arg(commands["dwip1e5"], "--point-loss-kind", "huber")
    replace_arg(commands["dwip1e5"], "--huber-delta", 1.)
    verify_huber_only_change(original, commands["dwip1e5"])
    return trials, commands, prior, anchor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-name", default=QUEUE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if Path(args.queue_name).name != args.queue_name or args.queue_name in {"", ".", "..", CONTROL}:
        raise ValueError("Require a fresh single queue directory name")
    root = REPO / "runs/objective_queues" / args.queue_name
    control = root.parent / CONTROL
    require_complete_control(control)
    proposal_path = control / "next_robust_objective_proposal.json"
    proposal = json.loads(proposal_path.read_text())
    for path, expected in proposal["input_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen proposal input changed: {path}")
    control_manifest = json.loads((control / "manifest.json").read_text())
    common_sources = control_manifest["source_sha256"]
    for path, expected in common_sources.items():
        if sha256(REPO / path) != expected:
            raise ValueError(f"Frozen control source changed: {path}")
    _, expected_commands, _, _ = control_commands(control)
    if json.loads((control / "dwip1e5_command.json").read_text()) != expected_commands["dwip1e5"]:
        raise ValueError("Matched control command does not reconstruct exactly")
    trials, commands, prior, anchor = build_commands(root)
    if sha256(anchor) != control_manifest["initial_checkpoint_sha256"]:
        raise ValueError("Original DWIP initialization changed")
    budget = artifact_budget(commands["dwip1e5"], WORK / f"tabimpute-{CONTROL}-dwip1e5")
    root.mkdir(parents=True, exist_ok=True)
    caches = prepare_caches(root, control, prior)
    identical = {name: sha256(root / name) == sha256(control / name) for name in ("validation.pth", "holdout.pth")}
    if not all(identical.values()):
        raise ValueError("Require byte-identical selector and independent caches")
    baseline, source = root / "baseline", (control / "baseline").resolve()
    if not baseline.exists() and not baseline.is_symlink():
        baseline.symlink_to(source, target_is_directory=True)
    elif not baseline.is_symlink() or baseline.resolve() != source:
        raise ValueError("Refusing to replace baseline artifacts")
    sources = [*common_sources, "scripts/run_dwip_anchor_mix_huber_objective.py"]
    manifest = {"control": CONTROL,
        "changed": "Point-loss function only: MSE to PyTorch Huber(delta1). Two explicit command values change: loss kind and previously inactive delta0.05 to1. Point coefficient remains2.0.",
        "synthetic_basis": proposal["synthetic_basis"], "risks": proposal["risks"],
        "preserved": proposal["preserved"], "selection": proposal["selection"],
        "design": "Original DWIP weights/fresh AdamW, not a warm restart. Same64k horizon, priors, seeds, mixture, data and LR trajectory; no coefficient or delta sweep.",
        "metric_semantics": "point_loss_missing now reports Huber, not raw MSE. Use rmse_missing_mean squared for descriptive raw training MSE comparisons, allowing float rounding.",
        "followups": "Unconditional full transfer, frozen256-case retention with extra paired matched-point2 control comparison, and matched100/500/1000 public runtime; all report-only.",
        "proposal_sha256": sha256(proposal_path), "control_manifest_sha256": sha256(control / "manifest.json"),
        "retention_control_sha256": sha256(control / "retention_evaluation/results.json"),
        "cache_equivalence": caches, "cache_files_byte_identical": identical, "budget": budget,
        "source_sha256": {path: sha256(REPO / path) for path in sources}}
    immutable_json(root / "huber_objective_manifest.json", manifest)
    run_queue(root, trials, commands, prior, anchor, budget, sources, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
