"""Complete the prior-coverage by horizon comparison without adding another factor."""
import argparse
import json
import sys

import torch

import run_spectral_query_experiments as queue
from run_dwip_continuous_spectrum import case_digest, continuous_prior
from run_dwip_spectral_geometry import SOURCES
from run_dwip_spectral_horizon import build_commands as control_commands, link_baseline
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, write_json
from run_rowcol_output_remedies import replace_arg
from tabimpute.model.model_new_stable import TabImputeModelStable

CONTROL = "dwip_spectral_horizon48k_20260916"
SHORT_CONTROL = "dwip_spectral_continuous_20260916"
DEPENDENCY = "dwip_spectral_point2_horizon48k_20260916/transfer_evaluation"


def build_commands(root):
    trials, commands, prior, anchor = control_commands(root)
    prior = continuous_prior(prior)
    replace_arg(commands["dwip1e5"], "--prior-overrides-json", json.dumps(prior))
    return trials, commands, prior, anchor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-name", default="dwip_spectral_continuous_horizon48k_20260916")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = REPO / "runs/objective_queues" / args.queue_name
    for name in (CONTROL, SHORT_CONTROL):
        if json.loads((root.parent / name / "transfer_evaluation/status.json").read_text())["state"] != "complete":
            raise RuntimeError(f"Matched control must be complete: {name}")
    diagnostic = REPO / "runs/information_flow_diagnostics/dwip_noise_sensitivity_20260916"
    if json.loads((diagnostic / "status.json").read_text())["state"] != "complete":
        raise RuntimeError("Review the independent noise diagnostic before registering training")
    root.mkdir(parents=True, exist_ok=True)
    link_baseline(root, root.parent / SHORT_CONTROL)
    cache_evidence = {}
    for name in ("validation.pth", "holdout.pth"):
        original = root.parent / CONTROL / name
        candidate = root / name
        original_cases = torch.load(original, map_location="cpu", weights_only=True)["cases"]
        candidate_cases = torch.load(candidate, map_location="cpu", weights_only=True)["cases"]
        if case_digest(original_cases) != case_digest(candidate_cases):
            raise ValueError("Fixed synthetic cases differ across prior-horizon controls")
        cache_evidence[name] = {"original_sha256": sha256(original), "candidate_sha256": sha256(candidate),
                                "identical_case_digest": case_digest(candidate_cases)}
    sources = [*SOURCES, "scripts/run_dwip_spectral_horizon.py", "scripts/run_dwip_continuous_spectrum.py",
               "scripts/run_dwip_continuous_horizon.py"]
    manifest = {"source_sha256": {path: sha256(REPO / path) for path in sources},
        "control": CONTROL, "short_control": SHORT_CONTROL, "cache_evidence": cache_evidence,
        "noise_diagnostic_sha256": sha256(diagnostic / "results.json"),
        "changed_vs_48k_control": "Only four high-rank exponent ranges become Uniform[0,2.5]; identical to the existing continuous24k prior",
        "changed_vs_continuous24k": "Only cosine horizon24k to48k, preserving600-step warmup and5% floor, with declared extra snapshots",
        "reason": "Continuous24k slightly improves fixed selector RMSE0.989744782 vs0.991709960 but not independent spectral RMSE. Its raw independent snapshots still improve1.107252->1.091058->1.080824 at4k/12k/24k. The original-prior48k control improves substantially over24k. Measure the prior-horizon interaction rather than infer it from unmatched schedules.",
        "noise_decision": "All five existing fine-tunes improve mean relative group RMSE over original in every generic noise regime/population; no evidence here supports a noise or corruption training change. No such factor is added.",
        "risk": "The short continuous-prior trial has worse independent spectral RMSE; a longer schedule may preserve or amplify that loss. Generic intermediate exponents are in support, not OOD interpolation. No improvement is presumed.",
        "preserved": "Original DWIP checkpoint and unchanged architecture; original data seed1150924 and model seed1201; NLL1/MSE0.2/CDF0.05; LR1e-5; fresh FP32 AdamW/BF16 autocast; shapes, missingness, df1 support, scales, noise and mixture masses unchanged",
        "selection": "Unchanged fixed synthetic macro RMSE only; all independent synthetic and real-world results report-only. No benchmark-specific generator, preprocessing, checkpoint or seed choice.",
        "budget": "48k steps with external24h timeout; about2h expected", "dependency": DEPENDENCY}
    path = root / "continuous_horizon_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Prior-horizon experiment changed; use a fresh queue")
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
