"""One fixed-only complementarity check of the two best completed raw models."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import torch

from diagnose_dwip_matched_objective_gradients import require_completed_stages
from run_dwip_budgeted_queue_v1 import immutable_json
from run_dwip_continuous_spectrum import case_digest
from run_dwip_transfer_evaluation import sha256
from run_rowcol_controlled_experiments import REPO, WORK


INVENTORY = "dwip_spectral_anchor_mix25_huber1_continuous_64k_20260917/final_saved_model_inventory.json"
EXPERIMENT = "dwip_top2_fixed_complementarity_20260917"


def read_json(path):
    return json.loads(path.read_text())


def top_two(inventory):
    eligible = [item for item in inventory if item["kind"] == "fixed_selected_training_checkpoint"]
    if len(eligible) < 2 or len({item["queue"] for item in eligible}) != len(eligible):
        raise ValueError("Require at least two unique completed raw training selections")
    if any(item["state"] != "complete" or item["case_count"] != 72 or
           not math.isfinite(item["fixed_macro_rmse"]) or item["fixed_macro_rmse"] < 0 for item in eligible):
        raise ValueError("Require completed models with finite fixed72-case scores")
    if len({item["case_digest"] for item in eligible}) != 1:
        raise ValueError("Fixed case populations differ")
    return sorted(eligible, key=lambda item: (item["fixed_macro_rmse"], item["queue"]))[:2]


def compare_fixed_predictions(cases, first, second):
    labels = ("rank1", "rank2", "equal_prediction_mean")
    indexed = {case["id"]: case for case in cases}
    if not indexed or len(indexed) != len(cases) or any(
            payload["predictions"].keys() != indexed.keys() for payload in (first, second)):
        raise ValueError("Require identical nonempty unique fixed case IDs")
    totals = {}
    for name, case in indexed.items():
        before, after = first["predictions"][name], second["predictions"][name]
        target = case["target"][case["mask"]].float()
        indices = case["mask"].flatten().nonzero().flatten()
        for record in (before, after):
            if not torch.equal(record["target"], target) or not torch.equal(record["flat_indices"], indices):
                raise ValueError(f"Fixed targets or scored indices differ: {name}")
            if record["mean"].shape != target.shape or not target.numel() or record["mean"].dtype != torch.float32:
                raise ValueError(f"Invalid FP32 prediction shape: {name}")
            if not torch.isfinite(record["mean"]).all() or not torch.isfinite(target).all():
                raise ValueError(f"Nonfinite fixed prediction: {name}")
        averaged = ((before["mean"].double() + after["mean"].double()) / 2).float()
        errors = {label: (mean - target).double() for label, mean in zip(labels, (before["mean"], after["mean"], averaged))}
        if not all(torch.isfinite(error).all() for error in errors.values()):
            raise ValueError(f"Nonfinite fixed residual: {name}")
        group = totals.setdefault(case["group"], {"sse": dict.fromkeys(labels, 0.), "n": 0,
                                                 "monitor_only": case["monitor_only"], "residual_dot": 0.})
        if group["monitor_only"] != case["monitor_only"]:
            raise ValueError("Inconsistent monitor-only group membership")
        group["n"] += target.numel()
        for label in labels:
            group["sse"][label] += float(errors[label].square().sum())
        group["residual_dot"] += float((errors["rank1"] * errors["rank2"]).sum())
    for payload in (first, second):
        if payload["metrics"]["groups"].keys() != totals.keys():
            raise ValueError("Fixed metric groups differ")
    groups = {}
    for name, group in totals.items():
        scores = {label: math.sqrt(value / group["n"]) for label, value in group["sse"].items()}
        for label, payload in zip(labels, (first, second)):
            saved = payload["metrics"]["groups"][name]
            if saved["n"] != group["n"] or saved["monitor_only"] != group["monitor_only"] or not math.isclose(
                    saved["rmse"], scores[label], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Cached raw fixed group does not reproduce: {name}, {label}")
        denominator = math.sqrt(group["sse"]["rank1"] * group["sse"]["rank2"])
        groups[name] = {"rmse": scores, "n": group["n"], "monitor_only": group["monitor_only"],
                        "residual_cosine_uncentered": group["residual_dot"] / denominator if denominator else None,
                        "mean_beats_both_raw": scores["equal_prediction_mean"] < min(scores["rank1"], scores["rank2"])}
    selected = [group for group in groups.values() if not group["monitor_only"]]
    if not selected:
        raise ValueError("No fixed selector groups")
    macro = {label: float(np.mean([group["rmse"][label] for group in selected])) for label in labels}
    for label, payload in zip(labels, (first, second)):
        if not math.isclose(macro[label], payload["metrics"]["macro_rmse"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Cached raw fixed macro RMSE does not reproduce")
    return {"macro_rmse": macro, "groups": groups, "selector_group_count": len(selected),
            "selector_groups_mean_beats_both_raw": sum(group["mean_beats_both_raw"] for group in selected),
            "equal_prediction_mean_beats_both_raw": macro["equal_prediction_mean"] < min(macro["rank1"], macro["rank2"]),
            "scope": "Fixed-only diagnostic of one equal FP64 prediction mean rounded to FP32. Not a deployable single model, parameter-average result, independent score or reason to promote a checkpoint. No pair/weight grid."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    if Path(args.experiment).name != args.experiment or args.experiment in {"", ".", ".."}:
        raise ValueError("Require one experiment directory name")
    root = REPO / "runs/objective_queues" / args.experiment
    root.mkdir(exist_ok=True)
    if (root / "results.json").exists():
        raise ValueError("Diagnostic already complete; preserve existing evidence")
    inventory_path = root.parent / INVENTORY
    inventory = read_json(inventory_path)
    for path, expected in inventory["input_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Fixed inventory input changed: {path}")
    members = top_two(inventory["inventory"])
    inputs = {str(inventory_path): sha256(inventory_path), str(Path(__file__).resolve()): sha256(__file__)}
    paths = {}
    for label, item in zip(("rank1", "rank2"), members):
        queue = root.parent / item["queue"]
        require_completed_stages(queue)
        work = WORK / f"tabimpute-{item['queue']}-dwip1e5"
        if sha256(item["checkpoint"]) != item["sha256"]:
            raise ValueError("Selected model weights differ from fixed inventory")
        paths[label] = work / "validation_predictions_best.pth"
        for path in (Path(item["checkpoint"]), paths[label], work / "best_validation_metrics.json", work / "config.json", queue / "validation.pth"):
            inputs[str(path)] = sha256(path)
    registration = {"inventory_sha256": sha256(inventory_path), "members": members, "input_sha256": inputs,
                    "rule": "Take the two lowest identical fixed72-case macro RMSE raw completed selections; ties by queue name. Ignore benchmark fields. One equal-weight prediction diagnostic, no alternative pair/weight search.",
                    "known_outcomes": "Constituent benchmark reports already exist and are known; this is not a blinded benchmark claim. They do not enter the deterministic input ranking or arithmetic.",
                    "limits": "CPU-only cached synthetic diagnostic. No GPU inference, optimizer, training, parameter averaging, checkpoint promotion or new priors. All existing artifacts preserved."}
    immutable_json(root / "prespecification.json", registration)
    if args.register_only:
        immutable_json(root / "registration_status.json", {"state": "registered", "evaluation_started": False})
        print(json.dumps({"registered": str(root), "members": [item["queue"] for item in members]}, indent=2))
        return
    payloads = {label: torch.load(path, map_location="cpu", weights_only=True) for label, path in paths.items()}
    populations = [torch.load(root.parent / item["queue"] / "validation.pth", map_location="cpu", weights_only=True)["cases"] for item in members]
    if any(case_digest(cases) != members[0]["case_digest"] for cases in populations):
        raise ValueError("Fixed case contents differ")
    comparison = compare_fixed_predictions(populations[0], payloads["rank1"], payloads["rank2"])
    for path, expected in inputs.items():
        if sha256(path) != expected:
            raise ValueError(f"Input changed during read-only diagnostic: {path}")
    comparison.update(utc=datetime.now(timezone.utc).isoformat(), prespecification_sha256=sha256(root / "prespecification.json"))
    immutable_json(root / "results.json", comparison)
    immutable_json(root / "status.json", {"state": "complete", "gpu_evaluation_started": False,
                                          "checkpoint_created_or_promoted": False, "updated_utc": comparison["utc"]})
    print(json.dumps({key: value for key, value in comparison.items() if key != "groups"}, indent=2))


if __name__ == "__main__":
    main()
