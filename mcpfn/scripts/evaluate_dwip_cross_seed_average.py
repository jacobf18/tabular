"""The single preregistered cross-seed parameter average; synthetic acceptance only."""
import argparse
import fcntl
import importlib.resources
import json
from pathlib import Path

import numpy as np
import torch

from analyze_dwip_spectral_evidence import cached_comparison
from evaluate_dwip_checkpoint_average import average_states
from rowcol_fixed_validation import evaluate_fixed_validation
from run_dwip_transfer_evaluation import gpu_processes, sha256
from run_rowcol_controlled_experiments import REPO, WORK, write_json
from train_direct_pure_rabel_curriculum import atomic_torch_save
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModelStable

QUEUES = {"seed1": "dwip_spectral_horizon48k_20260916", "seed2": "dwip_spectral_seed2_48k_20260916"}


def select_variant(scores):
    if set(scores) != {"seed1", "seed2", "average"}:
        raise ValueError("Require both raw replicas and the one prespecified average")
    if not np.isfinite([metrics["macro_rmse"] for metrics in scores.values()]).all():
        raise ValueError("Nonfinite synthetic score")
    raw = min(("seed1", "seed2"), key=lambda name: scores[name]["macro_rmse"])
    return "average" if scores["average"]["macro_rmse"] < scores[raw]["macro_rmse"] else raw


def prepare(root):
    roots = {label: REPO / "runs/objective_queues" / name for label, name in QUEUES.items()}
    work = {label: WORK / f"tabimpute-{name}-dwip1e5" for label, name in QUEUES.items()}
    for label, path in roots.items():
        if json.loads((path / "status.json").read_text())["state"] not in {
                "evaluating", "evaluating_mteb", "complete"}:
            raise RuntimeError(f"Wait for {label} training to finish before preparing weights")
    configs = {label: json.loads((path / "config.json").read_text()) for label, path in work.items()}
    if any(config["model_family"] != "new_stable" for config in configs.values()) or (
            configs["seed1"]["model"] != configs["seed2"]["model"]):
        raise ValueError("The two replicas must share the unchanged DWIP architecture")
    if not (roots["seed1"] / "validation.pth").samefile(roots["seed2"] / "validation.pth"):
        raise ValueError("Replicas must use the same fixed synthetic selector")
    checkpoints = {label: path / "best_validation_checkpoint.pth" for label, path in work.items()}
    borders = importlib.resources.files("tabimpute") / "data/borders.pt"
    caches = {"validation": roots["seed1"] / "validation.pth", "spectral": roots["seed1"] / "holdout.pth",
              "general": REPO / "runs/objective_queues/rowcol_output_remedies_paired_20260907/independent_validation.pth"}
    sources = [Path(__file__), REPO / "scripts/evaluate_dwip_checkpoint_average.py",
               REPO / "scripts/rowcol_fixed_validation.py", REPO / "src/tabimpute/model/model_new_stable.py",
               REPO / "src/tabimpute/model/bar_distribution.py"]
    manifest = {"queues": QUEUES, "prespecification_sha256": sha256(root / "prespecification.json"),
        "checkpoint_sha256": {label: sha256(path) for label, path in checkpoints.items()},
        "config_sha256": {label: sha256(path / "config.json") for label, path in work.items()},
        "selection_metrics_sha256": {label: sha256(path / "best_validation_metrics.json") for label, path in work.items()},
        "cache_sha256": {label: sha256(path) for label, path in caches.items()}, "borders_sha256": sha256(borders),
        "source_sha256": {str(path): sha256(path) for path in sources},
        "selection": "Average must beat BOTH freshly reproduced raw synthetic-selected models on identical fixed macro RMSE; no benchmark-based choices"}
    path = root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Cross-seed experiment changed; do not reuse its artifacts")
    write_json(path, manifest)
    candidate = root / "candidate"
    candidate.mkdir(exist_ok=True)
    checkpoint, integrity = candidate / "checkpoint.pth", root / "candidate_sha256.json"
    if not checkpoint.exists():
        states = [torch.load(path, map_location="cpu", weights_only=True, mmap=True)["model"] for path in checkpoints.values()]
        atomic_torch_save({"model": average_states(states), "artifact_kind": "equal_weight_cross_seed_parameter_average"}, checkpoint)
        write_json(integrity, {"sha256": sha256(checkpoint)})
    elif not integrity.exists() or json.loads(integrity.read_text())["sha256"] != sha256(checkpoint):
        raise ValueError("Cross-seed average integrity mismatch")
    config = configs["seed1"]["model"]
    write_json(candidate / "config.json", {"model_family": "new_stable", "model": config,
                                           "artifact_kind": "equal_weight_cross_seed_parameter_average"})
    return roots, work, {**checkpoints, "average": checkpoint}, config, borders, caches


def evaluate(root, work, checkpoints, config, borders, caches):
    if gpu_processes():
        raise RuntimeError("GPU is busy; no cross-seed evaluation started")
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    criterion = FullSupportBarDistribution(torch.load(borders, weights_only=True).cuda())
    cases = torch.load(caches["validation"], map_location="cpu", weights_only=True)["cases"]
    scores = {}
    for label, path in checkpoints.items():
        write_json(root / "status.json", {"state": "evaluating_synthetic", "variant": label,
                                          "gpu_evaluation_started": True})
        model = TabImputeModelStable(**config).cuda().eval()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"], strict=True)
        scores[label] = evaluate_fixed_validation(model, criterion, cases, "cuda", "amp_bfloat16",
                                                  root / f"{label}_validation_predictions.pth")
        write_json(root / f"{label}_validation.json", scores[label])
        del model
        torch.cuda.empty_cache()
    for label, path in work.items():
        recorded = json.loads((path / "best_validation_metrics.json").read_text())
        if not np.isclose(scores[label]["macro_rmse"], recorded["macro_rmse"], rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"Fresh raw selector score fails reproduction: {label}")
    selected = select_variant(scores)
    decision = {"selected_variant": selected, "criterion": "Fixed synthetic macro RMSE only",
                "macro_rmse": {label: score["macro_rmse"] for label, score in scores.items()},
                "average_accepted": selected == "average"}
    write_json(root / "selection.json", decision)
    for label in work:
        write_json(root / f"average_vs_{label}.json", cached_comparison(
            root / f"{label}_validation_predictions.pth", root / "average_validation_predictions.pth"))
    if selected == "average":
        model = TabImputeModelStable(**config).cuda().eval()
        model.load_state_dict(torch.load(checkpoints["average"], map_location="cpu", weights_only=True)["model"], strict=True)
        for domain in ("spectral", "general"):
            cases = torch.load(caches[domain], map_location="cpu", weights_only=True)["cases"]
            metrics = evaluate_fixed_validation(model, criterion, cases, "cuda", "amp_bfloat16",
                                                root / f"average_{domain}_predictions.pth")
            write_json(root / f"average_{domain}.json", metrics)
    write_json(root / "status.json", {"state": "synthetic_evaluation_complete", "selected_variant": selected,
                                      "realworld_evaluation_allowed": selected == "average"})
    print(json.dumps(decision), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="dwip_spectral_cross_seed_average_20260916")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = REPO / "runs/objective_queues" / args.experiment
    lock = (root / "experiment.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    roots, work, checkpoints, config, borders, caches = prepare(root)
    if args.prepare_only:
        write_json(root / "status.json", {"state": "prepared", "gpu_evaluation_started": False})
        return
    if any(json.loads((path / "transfer_evaluation/status.json").read_text())["state"] != "complete" for path in roots.values()):
        raise RuntimeError("Wait for both existing transfer panels to complete")
    try:
        evaluate(root, work, checkpoints, config, borders, caches)
    except Exception as error:
        write_json(root / "status.json", {"state": "error", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
