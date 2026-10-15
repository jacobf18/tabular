"""Audit frozen architecture/inference, synthetic lineage and inner-only selection."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from dwip_bpr_evaluation import CAMPAIGN, INITIAL, INITIAL_SHA256, MF_BASELINE, ROOT, selection_key
from dwip_bpr_experiment import registered_baselines
from dwip_bpr_optimization import EXPOSURE_PARAMETER_NAMES, verify_frozen_parameter_state
from dwip_recsys_provenance import POLICY, TRAINING_DATA, require_synthetic_checkpoint
from evaluate_dwip_recsys import sha256, write_json


def function_digest(path, names):
    module = ast.parse(path.read_text())
    functions = {node.name: ast.dump(node, include_attributes=False) for node in module.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names}
    if set(functions) != set(names):
        raise ValueError("Missing frozen inference helper")
    return hashlib.sha256(json.dumps(functions, sort_keys=True).encode()).hexdigest()


def inference_signature():
    scripts = ROOT / "scripts"
    return {
        "model_sources": {str(path.relative_to(ROOT)): sha256(path) for path in sorted((ROOT / "src/tabimpute/model").glob("*.py"))},
        "interface_sha256": sha256(ROOT / "src/tabimpute/interface.py"),
        "normalization_functions": function_digest(scripts / "dwip_recsys_experiment.py",
            ("inference_scaling", "inference_components", "inference_row_scales")),
        "imputation_function": function_digest(scripts / "evaluate_dwip_recsys_candidate.py", ("impute_candidate_ratings",)),
        "ranking_audit_sha256": sha256(scripts / "dwip_recsys_ranking_audit.py"),
        "geometry": [2000, 1000], "statistic": "mean", "normalization": "two_way_user_scale",
    }


def lineage(checkpoint):
    result = []
    require_synthetic_checkpoint(checkpoint)
    while True:
        digest = sha256(checkpoint)
        result.append({"checkpoint": str(checkpoint.resolve()), "sha256": digest})
        sidecar = Path(str(checkpoint) + ".provenance.json")
        if not sidecar.is_file():
            break
        config = json.loads((checkpoint.parent / "config.json").read_text())
        checkpoint = Path(config["initial_checkpoint"])
    return result


def register(root):
    destination = root / "fixed_protocol.json"
    if destination.exists():
        raise ValueError("Do not overwrite a frozen protocol")
    require_synthetic_checkpoint(INITIAL)
    if sha256(INITIAL) != INITIAL_SHA256:
        raise ValueError("Initial checkpoint changed")
    _, _, baseline_registration = registered_baselines(root)
    original = INITIAL.parent / "context_2000x1000_validation/source_snapshot"
    for filename, current in (("interface.py", ROOT / "src/tabimpute/interface.py"),
                              ("model_new_stable.py", ROOT / "src/tabimpute/model/model_new_stable.py"),
                              ("bar_distribution.py", ROOT / "src/tabimpute/model/bar_distribution.py")):
        if sha256(original / filename) != sha256(current):
            raise ValueError(f"Previously registered inference source changed: {filename}")
    helper_names = ("inference_scaling", "inference_components", "inference_row_scales")
    if function_digest(original / "dwip_recsys_experiment.py", helper_names) != inference_signature()["normalization_functions"]:
        raise ValueError("Normalization differs from the MF winner")
    if function_digest(original / "evaluate_dwip_recsys_candidate.py", ("impute_candidate_ratings",)) != inference_signature()["imputation_function"]:
        raise ValueError("Public inference differs from the MF winner")
    report = {"initial_sha256": INITIAL_SHA256, "initial_lineage": lineage(INITIAL),
              "inference_signature": inference_signature(), "baseline_registration": baseline_registration,
              "source_sha256": sha256(Path(__file__)), "registered_before_first_new_checkpoint_selection": True}
    write_json(destination, report)
    baseline_source = ROOT / "scripts/dwip_bpr_evaluation.py"
    if sha256(baseline_source) != baseline_registration["script_sha256"]:
        raise ValueError("Cannot archive changed baseline evaluation source")
    shutil.copyfile(baseline_source, root / "bpr_reference/evaluation_source.py")
    print(json.dumps({"registered": True, "synthetic_lineage_checkpoints": len(report["initial_lineage"])}), flush=True)


def audit(root, checkpoint):
    frozen = json.loads((root / "fixed_protocol.json").read_text())
    if inference_signature() != frozen["inference_signature"]:
        raise ValueError("Frozen architecture/inference implementation changed")
    mf, bpr, registration = registered_baselines(root)
    if registration != frozen["baseline_registration"]:
        raise ValueError("Frozen comparison changed")
    ancestry = lineage(checkpoint)
    if not any(value["sha256"] == INITIAL_SHA256 for value in ancestry):
        raise ValueError("Require ancestry through the registered synthetic MF winner")
    config = json.loads((checkpoint.parent / "config.json").read_text())
    if config["training_policy"] != POLICY or config["training_data"] != TRAINING_DATA:
        raise ValueError("Non-synthetic training config")
    teacher = config.get("teacher")
    if teacher is not None:
        if teacher["frozen"] is not True or teacher["data"] != "Synthetic episodes only":
            raise ValueError("Invalid synthetic teacher policy")
        if require_synthetic_checkpoint(Path(teacher["checkpoint"])) != teacher["provenance"]:
            raise ValueError("Synthetic teacher lineage changed")
    for source, digest in config["source_sha256"].items():
        archived = checkpoint.parent / "source_snapshot" / Path(source).name
        if sha256(archived) != digest:
            raise ValueError("Archived training source mismatch")
    original = torch.load(INITIAL, map_location="cpu", weights_only=True, mmap=True)["model"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    scope = config.get("optimization_scope", {}).get("scope", "full")
    if scope != config["arguments"].get("trainable_scope", "full"):
        raise ValueError("Optimization scope differs from the registered arguments")
    frozen_state_tensors_verified = 0
    if scope != "full":
        if set(config["optimization_scope"]["parameter_names"]) != EXPOSURE_PARAMETER_NAMES:
            raise ValueError("Registered exposure-only parameter scope changed")
        initialization = torch.load(config["initial_checkpoint"], map_location="cpu", weights_only=True, mmap=True)["model"]
        frozen_state_tensors_verified = verify_frozen_parameter_state(payload["model"], initialization, scope)
    if original.keys() != payload["model"].keys():
        raise ValueError("Network state keys changed")
    for name, value in payload["model"].items():
        if value.shape != original[name].shape or value.dtype != original[name].dtype or not torch.isfinite(value).all():
            raise ValueError("Invalid or changed network state signature")
    changed_tensors = sum(not torch.equal(value, original[name]) for name, value in payload["model"].items())
    status = json.loads((checkpoint.parent / "status.json").read_text())
    validations = [json.loads(path.read_text()) for path in checkpoint.parent.glob("validation_[0-9]*.json")]
    selected = json.loads((checkpoint.parent / f"validation_{payload['step']}.json").read_text())
    for validation in validations:
        if validation["score"] != selection_key(validation["datasets"], mf["validation"], bpr["validation"]):
            raise ValueError("Inner selection score does not reproduce")
    if status["state"] != "complete" or not selected["improved"] or selected["score"] != min(value["score"] for value in validations):
        raise ValueError("Checkpoint is not the completed inner-selected winner")
    if selected["score"] != status["best_score"]:
        raise ValueError("Completed selection score differs")
    logs = [json.loads(line) for line in (checkpoint.parent / "training.jsonl").read_text().splitlines()]
    if not logs or any(value.get("gradient_data") != "synthetic" for value in logs):
        raise ValueError("Missing synthetic update provenance")
    report = {"passed": True, "checkpoint_sha256": sha256(checkpoint), "selected_step": payload["step"],
              "state_tensors": len(original), "state_elements": sum(value.numel() for value in original.values()),
              "changed_weight_tensors": changed_tensors,
              "optimization_scope": scope, "frozen_state_tensors_verified": frozen_state_tensors_verified,
              "lineage": ancestry, "inference_unchanged": True, "training_sources_verified": True,
              "selection_score": selected["score"], "logged_updates": len(logs),
              "all_logged_training_data_synthetic": True, "goal_completion_checked": False}
    write_json(checkpoint.parent / "campaign_audit.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "lineage"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "audit"))
    parser.add_argument("--root", type=Path, default=CAMPAIGN)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.action == "register":
        register(args.root)
    else:
        if args.checkpoint is None:
            raise ValueError("Audit requires a checkpoint")
        audit(args.root, args.checkpoint)


if __name__ == "__main__":
    main()
