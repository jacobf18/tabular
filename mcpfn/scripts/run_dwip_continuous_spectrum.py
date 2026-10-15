"""One-factor continuous high-rank exponent coverage with identical synthetic cases."""
import argparse
from copy import deepcopy
import hashlib
import json
import sys

import torch

import run_spectral_query_experiments as queue
from rowcol_fixed_validation import load_or_build_validation
from run_dwip_spectral_geometry import build_commands as control_commands, SOURCES
from run_dwip_spectral_horizon import CONTROL, CACHED_BASELINE_FILES
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, write_json
from run_rowcol_output_remedies import replace_arg
from tabimpute.model.model_new_stable import TabImputeModelStable

DEPENDENCY = "dwip_spectral_point2_20260916/transfer_evaluation"


def continuous_prior(prior):
    updated = deepcopy(prior)
    changed = 0
    for component in updated["mixed_priors"]:
        config = component["config"]
        if config["spectrum_rank_high"] > 16:
            config.update(spectrum_alpha_low=0., spectrum_alpha_high=2.5)
            changed += 1
    if changed != 4:
        raise ValueError("Expected four high-rank components in the matched control")
    return updated


def build_commands(root):
    trials, commands, prior, anchor = control_commands(root)
    prior = continuous_prior(prior)
    replace_arg(commands["dwip1e5"], "--prior-overrides-json", json.dumps(prior))
    return trials, commands, prior, anchor


def case_digest(cases):
    digest = hashlib.sha256()
    for case in cases:
        for key, value in sorted(case.items()):
            digest.update(key.encode())
            if isinstance(value, torch.Tensor):
                digest.update(str((value.dtype, tuple(value.shape))).encode())
                digest.update(value.contiguous().numpy().tobytes())
            else:
                digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def prepare_caches(root, control, prior):
    evidence = {}
    for name in CACHED_BASELINE_FILES:
        source, destination = control / name, root / name
        if name in {"validation.pth", "holdout.pth"}:
            original = torch.load(source, map_location="cpu", weights_only=True)
            manifest = original["manifest"]
            if manifest["prior"]["validation_spectral_specs"] != prior["validation_spectral_specs"]:
                raise ValueError("Fixed validation distribution changed")
            cases = load_or_build_validation(destination, prior, seed=manifest["seed"],
                                             cases_per_group=manifest["cases_per_group"])
            original_digest, candidate_digest = case_digest(original["cases"]), case_digest(cases)
            if original_digest != candidate_digest:
                raise ValueError(f"Regenerated validation cases differ from the control: {name}")
            evidence[name] = {"source_sha256": sha256(source), "candidate_sha256": sha256(destination),
                              "identical_case_sha256": original_digest,
                              "note": "Metadata fingerprints differ with training prior; regenerated case tensors, IDs, masks and stats are byte-identical"}
        elif destination.is_symlink() and destination.resolve() == source.resolve():
            continue
        elif destination.exists() or destination.is_symlink():
            raise ValueError(f"Refusing to overwrite baseline artifact: {destination}")
        else:
            if not source.exists():
                raise FileNotFoundError(source)
            destination.symlink_to(source.resolve())
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-name", default="dwip_spectral_continuous_20260916")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = REPO / "runs/objective_queues" / args.queue_name
    control = root.parent / CONTROL
    if json.loads((control / "status.json").read_text())["state"] != "complete":
        raise RuntimeError("Matched 24k control must be complete")
    root.mkdir(parents=True, exist_ok=True)
    _, _, prior, _ = build_commands(root)
    caches = prepare_caches(root, control, prior)
    sources = [*SOURCES, "scripts/run_dwip_continuous_spectrum.py"]
    manifest = {"source_sha256": {path: sha256(REPO / path) for path in sources},
        "control": CONTROL, "changed": "High-rank exponents: equal mixture of Uniform[0,0.5] and Uniform[1,2.5] becomes Uniform[0,2.5]",
        "preserved": "Original checkpoint and unchanged DWIP; original24k cosine schedule, seed1150924, LR1e-5, NLL1/MSE0.2/CDF0.05, fresh FP32 AdamW/BF16 autocast; ranks, geometry, scales, noise, missingness, shape and mixture masses unchanged",
        "component_names": "Legacy slow_tail/steep_tail labels retained only for metadata; their identical continuous ranges preserve existing component counts and random sampling order. DWIP never sees labels.",
        "reason": "Previously documented generic coverage hypothesis: high-rank training omits the entire0.5-to1 exponent interval, while intermediate generic cases retain appreciable error. The point-weight trial did not improve spectral selector/independent RMSE, so do not bundle its loss change.",
        "evaluation_caches": caches,
        "interpretation": "Intermediate exponents are now in training support; this is not an out-of-support exponent-interpolation result. Matrices, missingness masks and targets remain identical and independently seeded.",
        "selection": "Unchanged synthetic macro-RMSE selector only; all independent synthetic and real-world results report-only; no benchmark-specific prior",
        "budget": "24k steps, expected about1h training; external24h timeout", "dependency": DEPENDENCY}
    path = root / "continuous_spectrum_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Continuous-spectrum experiment changed; use a new queue")
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
