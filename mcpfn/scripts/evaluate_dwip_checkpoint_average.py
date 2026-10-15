"""One prespecified late-checkpoint average, selected on synthetic validation only."""
import argparse
import fcntl
import importlib.resources
import json
from pathlib import Path

import numpy as np
import torch

from analyze_dwip_spectral_evidence import cached_comparison
from rowcol_fixed_validation import evaluate_fixed_validation
from run_dwip_transfer_evaluation import gpu_processes, sha256
from run_rowcol_controlled_experiments import REPO, WORK, write_json
from train_direct_pure_rabel_curriculum import atomic_torch_save
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModelStable

SOURCE_QUEUE = "dwip_spectral_horizon48k_20260916"
SOURCE_STEPS = (24000, 36000, 48000)


def average_states(states):
    if not states or not states[0]:
        raise ValueError("At least one nonempty model state is required")
    reference = states[0]
    if any(state.keys() != reference.keys() for state in states):
        raise ValueError("Checkpoint state keys differ")
    averaged = {}
    for name, tensor in reference.items():
        values = [state[name] for state in states]
        if any(value.shape != tensor.shape or value.dtype != tensor.dtype for value in values):
            raise ValueError(f"Checkpoint tensor shape/dtype differs: {name}")
        if tensor.is_floating_point():
            if tensor.dtype != torch.float32:
                raise ValueError(f"Expected FP32 model state: {name}")
            accumulator = torch.zeros_like(tensor, dtype=torch.float64)
            for value in values:
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite source checkpoint: {name}")
                accumulator.add_(value.double())
            averaged[name] = (accumulator / len(values)).float()
        else:
            if any(not torch.equal(value, tensor) for value in values):
                raise ValueError(f"Nonfloating checkpoint buffer differs: {name}")
            averaged[name] = tensor.clone()
    return averaged


def choose_average(raw_metrics, averaged_metrics):
    before, after = raw_metrics["macro_rmse"], averaged_metrics["macro_rmse"]
    if not np.isfinite([before, after]).all():
        raise ValueError("Nonfinite selector scores")
    return after < before


def prepare(root, source_root, work):
    source_config = json.loads((work / "config.json").read_text())
    if source_config["model_family"] != "new_stable":
        raise ValueError("Checkpoint averaging must preserve the DWIP architecture")
    paths = [work / f"checkpoint_{step}.pth" for step in SOURCE_STEPS]
    raw = work / "best_validation_checkpoint.pth"
    sources = [Path(__file__), REPO / "scripts/rowcol_fixed_validation.py",
        REPO / "src/tabimpute/model/model_new_stable.py", REPO / "src/tabimpute/model/bar_distribution.py"]
    manifest = {"source_queue": SOURCE_QUEUE, "source_steps": list(SOURCE_STEPS),
        "checkpoint_sha256": {str(path): sha256(path) for path in [*paths, raw]},
        "config_sha256": sha256(work / "config.json"),
        "borders_sha256": sha256(importlib.resources.files("tabimpute") / "data/borders.pt"),
        "validation_cache_sha256": sha256(source_root / "validation.pth"),
        "source_sha256": {str(path): sha256(path) for path in sources},
        "average": "Equal parameter weights, FP64 accumulation into one FP32 DWIP model",
        "selection": "Average must beat the original raw synthetic-selected checkpoint on the unchanged synthetic macro RMSE; no benchmark scores used",
        "reason": "Prespecified in DWIP_NEXT_EXPERIMENTS_20260916.md before the 36k/48k snapshots and their benchmark evaluation"}
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Checkpoint-average experiment changed; do not reuse these artifacts")
    write_json(manifest_path, manifest)
    candidate = root / "candidate"
    candidate.mkdir(exist_ok=True)
    checkpoint = candidate / "checkpoint.pth"
    integrity = root / "candidate_sha256.json"
    if not checkpoint.exists():
        states = [torch.load(path, map_location="cpu", weights_only=True, mmap=True)["model"] for path in paths]
        averaged = average_states(states)
        atomic_torch_save({"model": averaged, "source_steps": list(SOURCE_STEPS),
                           "artifact_kind": "equal_weight_parameter_average"}, checkpoint)
        write_json(integrity, {"sha256": sha256(checkpoint)})
    elif not integrity.exists() or json.loads(integrity.read_text())["sha256"] != sha256(checkpoint):
        raise ValueError("Averaged checkpoint integrity mismatch; do not reuse this artifact")
    write_json(candidate / "config.json", {"model_family": "new_stable", "model": source_config["model"],
                                           "artifact_kind": "equal_weight_parameter_average"})
    return checkpoint, raw, source_config["model"]


def evaluate(root, source_root, work, checkpoint, raw, config):
    if gpu_processes():
        raise RuntimeError("GPU is busy; no averaging evaluation started")
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    criterion = FullSupportBarDistribution(torch.load(
        importlib.resources.files("tabimpute") / "data/borders.pt", weights_only=True).cuda())
    cases = torch.load(source_root / "validation.pth", map_location="cpu", weights_only=True)["cases"]
    scores = {}
    for label, path in (("raw", raw), ("average", checkpoint)):
        model = TabImputeModelStable(**config).cuda().eval()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"], strict=True)
        scores[label] = evaluate_fixed_validation(model, criterion, cases, "cuda", "amp_bfloat16",
                                                  root / f"{label}_validation_predictions.pth")
        write_json(root / f"{label}_validation.json", scores[label])
        del model
        torch.cuda.empty_cache()
    recorded = json.loads((work / "best_validation_metrics.json").read_text())
    if not np.isclose(scores["raw"]["macro_rmse"], recorded["macro_rmse"], rtol=1e-5, atol=1e-6):
        raise RuntimeError("Fresh raw-model selector score does not reproduce its recorded validation")
    accepted = choose_average(scores["raw"], scores["average"])
    decision = {"selected_variant": "average" if accepted else "raw", "criterion": "synthetic macro RMSE only",
                "raw_macro_rmse": scores["raw"]["macro_rmse"], "average_macro_rmse": scores["average"]["macro_rmse"],
                "comparison": cached_comparison(root / "raw_validation_predictions.pth", root / "average_validation_predictions.pth")}
    write_json(root / "selection.json", decision)
    if accepted:
        model = TabImputeModelStable(**config).cuda().eval()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
        caches = {"spectral": source_root / "holdout.pth",
                  "general": REPO / "runs/objective_queues/rowcol_output_remedies_paired_20260907/independent_validation.pth"}
        for domain, path in caches.items():
            cases = torch.load(path, map_location="cpu", weights_only=True)["cases"]
            metrics = evaluate_fixed_validation(model, criterion, cases, "cuda", "amp_bfloat16",
                                                root / f"average_{domain}_predictions.pth")
            write_json(root / f"average_{domain}.json", metrics)
    write_json(root / "status.json", {"state": "synthetic_evaluation_complete", "selected_variant": decision["selected_variant"],
                                      "realworld_evaluation_allowed": accepted})
    print(json.dumps({key: value for key, value in decision.items() if key != "comparison"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="dwip_spectral_average_20260916")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = REPO / "runs/objective_queues" / args.experiment
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "experiment.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_root = REPO / "runs/objective_queues" / SOURCE_QUEUE
    work = WORK / f"tabimpute-{SOURCE_QUEUE}-dwip1e5"
    checkpoint, raw, config = prepare(root, source_root, work)
    if args.prepare_only:
        write_json(root / "status.json", {"state": "prepared", "gpu_evaluation_started": False})
        print(str(checkpoint), flush=True)
        return
    if json.loads((source_root / "transfer_evaluation/status.json").read_text())["state"] != "complete":
        raise RuntimeError("Wait for the existing real-world transfer queue to finish")
    evaluate(root, source_root, work, checkpoint, raw, config)


if __name__ == "__main__":
    main()
