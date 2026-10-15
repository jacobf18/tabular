"""One fixed25% small latent-family rehearsal arm, with75% unchanged spectral priors."""
import argparse
from copy import deepcopy
import json
import sys

import run_spectral_query_experiments as queue
from diagnose_dwip_latent_retention import LATENT_CONFIG
from run_dwip_continuous_spectrum import prepare_caches
from run_dwip_spectral_geometry import SOURCES
from run_dwip_spectral_horizon import build_commands as control_commands
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, write_json
from run_rowcol_output_remedies import replace_arg
from tabimpute.model.model_new_stable import TabImputeModelStable

CONTROL = "dwip_spectral_horizon48k_20260916"
DEPENDENCY = "dwip_spectral_point2_seed2_48k_20260916/transfer_evaluation"
DIAGNOSTIC = "dwip_latent_retention_20260916"


def rehearsal_prior(prior):
    result = deepcopy(prior)
    components = result["mixed_priors"]
    if len(components) != 6 or any(component["kind"] != "spectrum_decay" for component in components):
        raise ValueError("Expected the six existing spectral components")
    weight = sum(component["weight"] for component in components) / 6
    for missingness in ("mcar", "profile_mixed"):
        components.append({"name": f"anchor_latent_{missingness}", "kind": "latent_factor",
            "weight": weight, "missingness_type": missingness,
            "config": {**LATENT_CONFIG, "p_missing": .4},
            "shape_override": {"num_rows_low": 10, "num_rows_high": 50, "num_cols_low": 5, "num_cols_high": 50}})
    return result


def build_commands(root):
    trials, commands, prior, anchor = control_commands(root)
    prior = rehearsal_prior(prior)
    command = commands["dwip1e5"]
    command[1] = "scripts/train_dwip_anchor_mix.py"
    replace_arg(command, "--prior-overrides-json", json.dumps(prior))
    return trials, commands, prior, anchor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-name", default="dwip_spectral_anchor_mix25_48k_20260916")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = REPO / "runs/objective_queues" / args.queue_name
    for name in (f"{CONTROL}/transfer_evaluation", DEPENDENCY):
        if json.loads((root.parent / name / "status.json").read_text())["state"] != "complete":
            raise RuntimeError("Finish the matched controls and existing transfer panel first")
    diagnostic = REPO / "runs/information_flow_diagnostics" / DIAGNOSTIC
    if json.loads((diagnostic / "status.json").read_text())["state"] != "complete":
        raise RuntimeError("Fresh generic retention diagnosis must be complete")
    average = root.parent / "dwip_point2_cross_seed_average_20260916_v2"
    if json.loads((average / "transfer_evaluation/status.json").read_text())["state"] != "skipped_rejected_average":
        raise RuntimeError("Review the average and finish its conditional panel before another trial")
    root.mkdir(parents=True, exist_ok=True)
    _, _, prior, _ = build_commands(root)
    caches = prepare_caches(root, root.parent / CONTROL, prior)
    baseline = root / "baseline"
    original_baseline = (root.parent / CONTROL / "baseline").resolve()
    if not baseline.exists() and not baseline.is_symlink():
        baseline.symlink_to(original_baseline, target_is_directory=True)
    elif not baseline.is_symlink() or baseline.resolve() != original_baseline:
        raise ValueError("Refusing to replace a baseline directory")
    sources = [*SOURCES, "scripts/run_dwip_spectral_horizon.py", "scripts/run_dwip_continuous_spectrum.py",
               "scripts/diagnose_dwip_latent_retention.py", "scripts/run_dwip_anchor_mix.py",
               "scripts/train_dwip_anchor_mix.py", "src/tabimpute/prior/training_set_generation.py"]
    manifest = {"control": CONTROL, "dependency": DEPENDENCY,
        "source_sha256": {path: sha256(REPO / path) for path in sources}, "cache_equivalence": caches,
        "diagnostic_manifest_sha256": sha256(diagnostic / "manifest.json"),
        "diagnostic_results_sha256": sha256(diagnostic / "results.json"),
        "anchor_provenance_sha256": sha256("/root/tabimpute_anchor3_named/anchor3_init_eval_manifest_20260420.json"),
        "changed": "One task-mixture intervention:25% latent-family small-shape examples,75% unchanged six-component spectral mixture. Fixed25%, not a weight sweep.",
        "reason": "All eight fine-tunes worsen fresh small latent/MCAR macro-case RMSE by3.0-5.3%, while improving the other seven family/shape/support groups. This is a localized generic family-shape interaction, not wholesale forgetting or a benchmark-derived failure profile.",
        "rehearsal": "Current explicit latent defaults, ranks1-11, rows10-50, cols5-50. Two equal MCAR/profile_mixed components preserve overall50/50 missingness-family mixture. Existing broad_support augmentation and df1 apply to ALL components, including rehearsal.",
        "limits": "Not exact historical replay: original full configuration is unavailable, profile_mixed and broad_support extend documented MCAR pretraining. Rehearsal dilutes spectral exposure by25% at fixed48k steps and may harm spectral generalization. No success is presumed.",
        "preserved": "Original DWIP checkpoint and architecture, seed1150924/model seed1201, LR1e-5, NLL1/MSE0.2/CDF0.05,600 warmup,48k cosine to5%, FP32 AdamW/BF16, batch2, original spectral component settings and shape ranges, identical fixed selector/cached independent cases.",
        "selection": "Unchanged fixed synthetic macro RMSE only; retention diagnostic and all independent/real-world scores report-only. No benchmark-specific priors, normalization or checkpoint/seed/mixture-weight choices.",
        "budget": "48k steps, expected under2h training, external24h timeout. Preserve existing files and the45GiB launch guard."}
    path = root / "anchor_mix_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Rehearsal experiment changed; use fresh artifacts")
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
