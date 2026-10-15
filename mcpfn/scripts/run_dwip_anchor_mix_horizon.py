"""One preregistered64k cosine horizon, holding the25% anchor-family mixture fixed."""
import argparse
import json
from pathlib import Path

from run_dwip_anchor_mix import build_commands as control_commands
from run_dwip_budgeted_queue_v1 import artifact_budget, argument, immutable_json, run_queue
from run_dwip_continuous_spectrum import prepare_caches
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, WORK
from run_rowcol_output_remedies import replace_arg

CONTROL = "dwip_spectral_anchor_mix25_48k_20260916"
QUEUE = "dwip_spectral_anchor_mix25_64k_20260916"


def build_commands(root):
    trials, commands, prior, anchor = control_commands(root)
    command = commands["dwip1e5"]
    phases = json.loads(argument(command, "--phases-json"))
    phases[-1]["end_step"] = 64000
    replace_arg(command, "--phases-json", json.dumps(phases))
    replace_arg(command, "--decay-steps", 63400)
    position = command.index("--snapshot-steps") + 1
    while position < len(command) and not command[position].startswith("--"):
        position += 1
    command[position:position] = ["48000"]
    return trials, commands, prior, anchor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-name", default=QUEUE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if Path(args.queue_name).name != args.queue_name or args.queue_name in {"", ".", ".."}:
        raise ValueError("Require a single fresh queue directory name")
    root = REPO / "runs/objective_queues" / args.queue_name
    control = root.parent / CONTROL
    for path in (control, control / "transfer_evaluation", control / "retention_evaluation"):
        if json.loads((path / "status.json").read_text())["state"] != "complete":
            raise RuntimeError("Finish the48k control and both registered follow-ups first")
    control_manifest = json.loads((control / "anchor_mix_manifest.json").read_text())
    common_sources = {**json.loads((control / "manifest.json").read_text())["source_sha256"],
                      **control_manifest["source_sha256"]}
    for path, expected in common_sources.items():
        if sha256(REPO / path) != expected:
            raise ValueError(f"Frozen control source changed: {path}")
    trials, commands, prior, anchor = build_commands(root)
    _, expected_commands, _, _ = control_commands(control)
    if json.loads((control / "dwip1e5_command.json").read_text()) != expected_commands["dwip1e5"]:
        raise ValueError("Matched control command no longer reconstructs exactly")
    budget = artifact_budget(commands["dwip1e5"], WORK / f"tabimpute-{CONTROL}-dwip1e5")
    root.mkdir(parents=True, exist_ok=True)
    caches = prepare_caches(root, control, prior)
    baseline = root / "baseline"
    source = (control / "baseline").resolve()
    if not baseline.exists() and not baseline.is_symlink():
        baseline.symlink_to(source, target_is_directory=True)
    elif not baseline.is_symlink() or baseline.resolve() != source:
        raise ValueError("Refusing to replace an existing baseline directory")
    sources = [*common_sources, "scripts/run_dwip_anchor_mix_horizon.py"]
    manifest = {"control": CONTROL, "changed": "Only horizon48k to64k, decay47400 to63400, extra report-only48k snapshot",
        "reason": "Frozen25% rehearsal restores small latent/MCAR average approximately to original but sacrifices spectral accuracy.64k projects47852 spectral draws versus35951 at48k, approximately restoring the48000 no-rehearsal draws. Test duration without changing mixture, seeds, loss or prior.",
        "limits": "Draw counts do not establish causation or equal LR-weighted/gradient exposure. Current explicit latent defaults are not exact historical replay. No benchmark-driven mixture, seed, checkpoint or prior choices.",
        "preserved": "Original checkpoint, unchanged DWIP architecture, seed1150924/model seed1201, FP32 AdamW/BF16, LR1e-5,600 warmup and5% floor, NLL1/MSE0.2/CDF0.05,25% rehearsal, all geometry/support/df1 and50/50 missingness settings",
        "selection": "Unchanged fixed synthetic macro RMSE only. Independent spectral/broad, full real-world panel and frozen256-case retention follow-up report-only.",
        "design": "Fresh original initialization, not a warm restart. First48000 training examples match control; cosine LR trajectory differs.",
        "budget": budget, "cache_equivalence": caches, "source_sha256": {path: sha256(REPO / path) for path in sources},
        "control_manifest_sha256": sha256(control / "manifest.json"),
        "exposure_projection_sha256": sha256(control / "rehearsal_exposure_projection.json"),
        "retention_results_sha256": sha256(control / "retention_evaluation/results.json")}
    immutable_json(root / "horizon_manifest.json", manifest)
    run_queue(root, trials, commands, prior, anchor, budget, sources, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
