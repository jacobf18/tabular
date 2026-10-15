"""One algebraic local-curvature control for unit-Huber unchanged-DWIP training."""
import argparse
import json
from pathlib import Path

from diagnose_dwip_matched_objective_gradients import require_completed_stages
from run_dwip_anchor_mix_huber_objective import build_commands as control_commands
from run_dwip_budgeted_queue_v1 import artifact_budget, argument, immutable_json, run_queue
from run_dwip_continuous_spectrum import prepare_caches
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, WORK
from run_rowcol_output_remedies import replace_arg

CONTROL = "dwip_spectral_anchor_mix25_huber1_64k_20260916"
QUEUE = "dwip_spectral_anchor_mix25_huber1_point4_64k_20260917"
MSE_CONTROL = "dwip_spectral_anchor_mix25_point2_64k_20260916"


def verify_curvature_only_change(control, candidate):
    for command in (control, candidate):
        if argument(command, "--point-loss-kind") != "huber" or float(argument(command, "--huber-delta")) != 1.:
            raise ValueError("Require unchanged unit-Huber objectives")
    if float(argument(control, "--point-loss-weight")) != 2. or float(argument(candidate, "--point-loss-weight")) != 4.:
        raise ValueError("Only the algebraic2-to4 local-curvature control is registered")
    restored = list(candidate)
    replace_arg(restored, "--point-loss-weight", argument(control, "--point-loss-weight"))
    if restored != control:
        raise ValueError("Commands differ beyond the declared point coefficient")


def build_commands(root):
    trials, commands, prior, anchor = control_commands(root)
    original = list(commands["dwip1e5"])
    replace_arg(commands["dwip1e5"], "--point-loss-weight", 4.)
    verify_curvature_only_change(original, commands["dwip1e5"])
    return trials, commands, prior, anchor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-name", default=QUEUE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if Path(args.queue_name).name != args.queue_name or args.queue_name in {"", ".", "..", CONTROL, MSE_CONTROL}:
        raise ValueError("Require a fresh single queue directory name")
    root = REPO / "runs/objective_queues" / args.queue_name
    control = root.parent / CONTROL
    require_completed_stages(control)
    require_completed_stages(root.parent / MSE_CONTROL)
    proposal_path = control / "next_huber_curvature_proposal.json"
    proposal = json.loads(proposal_path.read_text())
    for path, expected in proposal["input_sha256"].items():
        if sha256(REPO / path) != expected:
            raise ValueError(f"Frozen curvature proposal input changed: {path}")
    control_manifest = json.loads((control / "manifest.json").read_text())
    common_sources = control_manifest["source_sha256"]
    for path, expected in common_sources.items():
        if sha256(REPO / path) != expected:
            raise ValueError(f"Frozen matched-control source changed: {path}")
    _, expected_commands, _, _ = control_commands(control)
    if json.loads((control / "dwip1e5_command.json").read_text()) != expected_commands["dwip1e5"]:
        raise ValueError("Matched Huber2 command no longer reconstructs exactly")
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
    sources = [*common_sources, "scripts/run_dwip_anchor_mix_huber_curvature.py",
        "scripts/diagnose_dwip_matched_objective_gradients.py"]
    manifest = {"control": CONTROL, "secondary_mse_control": MSE_CONTROL,
        "changed": proposal["factor"], "algebraic_choice": proposal["algebraic_choice"],
        "training_basis": proposal["training_basis"], "fixed_curve_context": proposal["fixed_curve_context"],
        "preserved": proposal["preserved"], "risks": proposal["risks"],
        "selection": proposal["selection"], "followups": proposal["followups"],
        "design": "Original DWIP weights and fresh AdamW, not a warm restart. Only point coefficient2 to4 changes; all data and LR trajectories match Huber2.",
        "metric_semantics": "point_loss_missing is unit-Huber, coefficient applied separately. Squared logged RMSE uses BF16-rounded means and is not exact FP32 mean-objective MSE.",
        "proposal_sha256": sha256(proposal_path), "control_manifest_sha256": sha256(control / "manifest.json"),
        "retention_control_sha256": {name: sha256(root.parent / name / "retention_evaluation/results.json")
            for name in (CONTROL, MSE_CONTROL)}, "cache_equivalence": caches,
        "cache_files_byte_identical": identical, "budget": budget,
        "source_sha256": {path: sha256(REPO / path) for path in sources}}
    immutable_json(root / "huber_curvature_manifest.json", manifest)
    run_queue(root, trials, commands, prior, anchor, budget, sources, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
