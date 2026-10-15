"""One continuous-exponent prior factor under the unchanged Huber2 rehearsal recipe."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from diagnose_dwip_matched_objective_gradients import require_completed_stages
from run_dwip_anchor_mix_huber_objective import build_commands as control_commands
from run_dwip_budgeted_queue_v1 import artifact_budget, immutable_json, run_queue
from run_dwip_continuous_spectrum import continuous_prior, prepare_caches
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, WORK
from run_rowcol_output_remedies import replace_arg

CONTROL = "dwip_spectral_anchor_mix25_huber1_64k_20260916"
QUEUE = "dwip_spectral_anchor_mix25_huber1_continuous_64k_20260917"
PROPOSAL = "dwip_gradient_reference1024_20260917_v2/next_continuous_prior_interaction_proposal.json"
HIGH_RANK_RANGES = {"slow_tail_mcar": (0., .5), "slow_tail_profile_mixed": (0., .5),
                    "steep_tail_mcar": (1., 2.5), "steep_tail_profile_mixed": (1., 2.5)}


def argument(command, flag):
    if command.count(flag) != 1:
        raise ValueError(f"Require one explicit argument: {flag}")
    return command[command.index(flag) + 1]


def continuous_rehearsal_prior(prior):
    components = prior["mixed_priors"]
    if len(components) != 8 or any(component["weight"] != 1. for component in components):
        raise ValueError("Require the original eight equal-mass components")
    if any(component["kind"] != "spectrum_decay" for component in components[:6]) or any(
            component["kind"] != "latent_factor" for component in components[6:]):
        raise ValueError("Require six spectral components followed by two unchanged latent components")
    found = {}
    for component in components[:6]:
        config = component["config"]
        if config["spectrum_rank_high"] > 16:
            found[component["name"]] = (config["spectrum_alpha_low"], config["spectrum_alpha_high"])
    if found != HIGH_RANK_RANGES:
        raise ValueError("Require exactly the original four disjoint high-rank exponent ranges")
    spectral = deepcopy(prior)
    spectral["mixed_priors"] = spectral["mixed_priors"][:6]
    updated = continuous_prior(spectral)
    updated["mixed_priors"].extend(deepcopy(components[6:]))
    return updated


def verify_continuous_only_change(control, candidate):
    prior = json.loads(argument(control, "--prior-overrides-json"))
    if json.loads(argument(candidate, "--prior-overrides-json")) != continuous_rehearsal_prior(prior):
        raise ValueError("Candidate prior differs beyond the four declared alpha values")
    restored = list(candidate)
    replace_arg(restored, "--prior-overrides-json", argument(control, "--prior-overrides-json"))
    if restored != control:
        raise ValueError("Commands differ beyond the declared continuous-prior factor")


def build_commands(root):
    trials, commands, prior, anchor = control_commands(root)
    original = list(commands["dwip1e5"])
    prior = continuous_rehearsal_prior(prior)
    replace_arg(commands["dwip1e5"], "--prior-overrides-json", json.dumps(prior))
    verify_continuous_only_change(original, commands["dwip1e5"])
    return trials, commands, prior, anchor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-name", default=QUEUE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if Path(args.queue_name).name != args.queue_name or args.queue_name in {"", ".", "..", CONTROL}:
        raise ValueError("Require a fresh single queue directory")
    root = REPO / "runs/objective_queues" / args.queue_name
    control = root.parent / CONTROL
    require_completed_stages(control)
    require_completed_stages(root.parent / "dwip_spectral_anchor_mix25_huber1_point4_64k_20260917")
    proposal_path = root.parent / PROPOSAL
    proposal = json.loads(proposal_path.read_text())
    if json.loads((proposal_path.parent / "status.json").read_text())["state"] != "complete":
        raise ValueError("Complete the frozen reference diagnostic first")
    for path, expected in proposal["input_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen proposal input changed: {path}")
    control_manifest = json.loads((control / "manifest.json").read_text())
    for path, expected in control_manifest["source_sha256"].items():
        if sha256(REPO / path) != expected:
            raise ValueError(f"Frozen matched-control source changed: {path}")
    _, expected_commands, _, _ = control_commands(control)
    if json.loads((control / "dwip1e5_command.json").read_text()) != expected_commands["dwip1e5"]:
        raise ValueError("Matched Huber2 command does not reconstruct exactly")
    trials, commands, prior, anchor = build_commands(root)
    if sha256(anchor) != control_manifest["initial_checkpoint_sha256"]:
        raise ValueError("Original DWIP checkpoint changed")
    budget = artifact_budget(commands["dwip1e5"], WORK / f"tabimpute-{CONTROL}-dwip1e5")
    root.mkdir(parents=True, exist_ok=True)
    caches = prepare_caches(root, control, prior)
    baseline, original = root / "baseline", (control / "baseline").resolve()
    if not baseline.exists() and not baseline.is_symlink():
        baseline.symlink_to(original, target_is_directory=True)
    elif not baseline.is_symlink() or baseline.resolve() != original:
        raise ValueError("Refusing to replace baseline artifacts")
    sources = [*control_manifest["source_sha256"], "scripts/run_dwip_anchor_mix_huber_continuous.py"]
    manifest = {"control": CONTROL, "factor": proposal["one_factor"], "basis": proposal["basis"],
        "changes": proposal["changes"], "mass_accounting": proposal["mass_accounting"],
        "preserved": proposal["training"], "selection": proposal["selection"], "risks": proposal["risks"],
        "data_matching": proposal["data_matching"], "followups": proposal["followups"],
        "cache_equivalence": caches, "cache_semantics": "Case tensors, masks, targets, IDs and stats must match. Training-prior metadata fingerprints may differ; do not require identical cache-file hashes.",
        "proposal_sha256": sha256(proposal_path), "control_manifest_sha256": sha256(control / "manifest.json"),
        "retention_control_sha256": {CONTROL: sha256(control / "retention_evaluation/results.json")},
        "budget": budget, "source_sha256": {path: sha256(REPO / path) for path in sources}}
    immutable_json(root / "continuous_huber_manifest.json", manifest)
    run_queue(root, trials, commands, prior, anchor, budget, sources, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
