"""One fresh, non-selecting synthetic replication of the frozen compression trio."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.resources
import json
from pathlib import Path
import shutil
import time

import torch

from diagnose_dwip_teacher_compression_gap import (
    EXPERIMENT as DIAGNOSTIC, compare_student, read_json, verify_inputs,
)
from rowcol_fixed_validation import evaluate_fixed_validation, load_or_build_validation, preserved_rng
from run_dwip_budgeted_queue_v1 import immutable_json, regular_bytes, RESERVE_BYTES
from run_dwip_continuous_spectrum import case_digest
from run_dwip_transfer_evaluation import gpu_processes, sha256
from run_rowcol_controlled_experiments import REPO
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model_new_stable import TabImputeModelStable


EXPERIMENT = "dwip_fresh_compression_replication_20260917"
PANELS = {"independent_spectral": {"seed": 2026091701, "case_count": 288, "groups": 18},
          "independent_general": {"seed": 2026091702, "case_count": 256, "groups": 16}}
CASES_PER_GROUP = 16
ALLOWANCE = 1024**3
SOURCES = (
    "scripts/evaluate_dwip_fresh_compression.py",
    "tests/test_dwip_fresh_compression.py",
    "scripts/diagnose_dwip_teacher_compression_gap.py",
    "scripts/rowcol_fixed_validation.py",
    "scripts/run_dwip_continuous_spectrum.py",
    "scripts/run_dwip_budgeted_queue_v1.py",
    "scripts/run_dwip_transfer_evaluation.py",
    "src/tabimpute/model/model_new_stable.py",
    "src/tabimpute/model/bar_distribution.py",
    "src/tabimpute/model/prediction_distribution.py",
    "src/tabimpute/prior/training_set_generation.py",
    "src/tabimpute/prior/spectrum_training_set_generation.py",
)


def storage_guard(free, allocated):
    if not 0 <= allocated <= ALLOWANCE:
        raise RuntimeError("Fresh diagnostic exceeds its1GiB artifact allowance")
    remaining = ALLOWANCE - allocated
    if free < RESERVE_BYTES + remaining:
        raise RuntimeError("Preserve20GiB reserve plus the remaining diagnostic allowance")
    return {"free_bytes": free, "allocated_bytes": allocated, "remaining_allowance_bytes": remaining,
            "reserve_bytes": RESERVE_BYTES}


def guard(root):
    return storage_guard(shutil.disk_usage(root).free, regular_bytes(root))


def validate_population(cases, registration, old_digests):
    digest = case_digest(cases)
    counts = {}
    if len(cases) != registration["case_count"] or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Fresh panel has wrong count or duplicate IDs")
    for case in cases:
        counts[case["group"]] = counts.get(case["group"], 0) + 1
    if len(counts) != registration["groups"] or set(counts.values()) != {CASES_PER_GROUP}:
        raise ValueError("Require all original groups with sixteen cases each")
    if digest in old_digests:
        raise ValueError("Fresh panel reuses a previous case population")
    return {"case_digest": digest, "case_count": len(cases), "group_counts": counts}


def register(root):
    previous_path = root.parent / DIAGNOSTIC / "prespecification.json"
    previous = read_json(previous_path)
    verify_inputs(previous["input_sha256"])
    inputs = {str(previous_path): sha256(previous_path), **previous["input_sha256"]}
    for path in (root.parent / DIAGNOSTIC / "results.json", *(REPO / source for source in SOURCES)):
        inputs[str(path.resolve())] = sha256(path)
    panels = {}
    for name, design in PANELS.items():
        old_path = Path(previous["panels"][name]["case_paths"][0])
        cached = torch.load(old_path, map_location="cpu", weights_only=True)
        old_manifest = cached["manifest"]
        if design["seed"] == old_manifest["seed"]:
            raise ValueError("Require a new preregistered generator seed")
        panels[name] = {**design, "prior": old_manifest["prior"], "old_manifest": old_manifest,
                        "cases_per_group": CASES_PER_GROUP}
    student = previous["student"]
    config_path = Path(student["checkpoint"]).parent / "config.json"
    inputs[str(config_path)] = sha256(config_path)
    borders = Path(str(importlib.resources.files("tabimpute") / "data/borders.pt"))
    inputs[str(borders)] = sha256(borders)
    registration = {
        "utc": datetime.now(timezone.utc).isoformat(), "input_sha256": inputs,
        "members": {**dict(zip(("rank1", "rank2"), previous["members"])), "student": student},
        "model_config": read_json(config_path)["model"], "borders": str(borders),
        "panels": panels, "old_case_digests": [panel["case_digest"] for panel in previous["panels"].values()],
        "storage": guard(root), "allowance_bytes": ALLOWANCE,
        "protocol": "One frozen trio, two fresh seeds, every existing spectral/general group, sixteen cases per group. The original generic prior configurations, shapes, missingness/support transforms and1024-target cap remain unchanged. Mixed and sparse-query cases share a generated underlying matrix;544 cases represent272 matrix draws, not544 independent draws. No added difficult-case weighting or geometry change.",
        "precision": "FP32 frozen model weights, BF16 forwards, existing evaluator. One equal FP64 teacher prediction mean rounded to FP32. Existing fixed selector is not changed.",
        "decision_rule": "Report all raw group/macro errors and teacher/student decompositions on both fresh panels. This is evidence about whether the cached compression gap replicates, not checkpoint selection, coefficient tuning or a promotion gate. No benchmark inference or training follows automatically.",
        "limits": "Known existing outcomes motivated replication; fresh outcomes are not inspected before registration. Small correlated group samples do not establish statistical significance. Preserve all existing files and jobs,1GiB owned artifacts and20GiB free reserve; GPU stage capped at1h externally.",
    }
    immutable_json(root / "prespecification.json", registration)
    immutable_json(root / "registration_status.json", {
        "state": "registered", "generation_started": False, "evaluation_started": False,
        "prespecification_sha256": sha256(root / "prespecification.json"),
    })
    return {"state": "registered", "case_count": 544, "verified_inputs": len(inputs)}


def verified_registration(root):
    path = root / "prespecification.json"
    if read_json(root / "registration_status.json")["prespecification_sha256"] != sha256(path):
        raise ValueError("Fresh registration changed")
    registration = read_json(path)
    verify_inputs(registration["input_sha256"])
    return registration


def prepare(root):
    registration = verified_registration(root)
    if (root / "prepared.json").exists():
        raise ValueError("Fresh data already prepared; preserve its original audit")
    panels = {}
    for name, design in registration["panels"].items():
        guard(root)
        path = root / f"{name}_cases.pth"
        cases = load_or_build_validation(path, design["prior"], seed=design["seed"], cases_per_group=CASES_PER_GROUP)
        panels[name] = {"path": str(path), "sha256": sha256(path),
                        **validate_population(cases, design, registration["old_case_digests"])}
    verify_inputs(registration["input_sha256"])
    result = {"panels": panels, "utc": datetime.now(timezone.utc).isoformat(),
              "prespecification_sha256": sha256(root / "prespecification.json"), "storage": guard(root)}
    immutable_json(root / "prepared.json", result)
    return {"state": "prepared", "panels": {name: values["case_count"] for name, values in panels.items()}}


def evaluate(root):
    registration = verified_registration(root)
    prepared = read_json(root / "prepared.json")
    if prepared["prespecification_sha256"] != sha256(root / "prespecification.json"):
        raise ValueError("Prepared data belongs to a different registration")
    if gpu_processes():
        raise RuntimeError("GPU is busy; preserve existing jobs")
    verify_inputs({panel["path"]: panel["sha256"] for panel in prepared["panels"].values()})
    if any(root.glob("*_predictions.pth")):
        raise ValueError("Preserve existing fresh predictions; audit a partial evaluation before resuming")
    started = time.monotonic()
    torch.backends.cuda.enable_cudnn_sdp(False)
    with preserved_rng(20260917):
        model = TabImputeModelStable(**registration["model_config"]).float().cuda().eval()
    model.requires_grad_(False)
    criterion = FullSupportBarDistribution(torch.load(registration["borders"], map_location="cpu", weights_only=True).cuda())
    artifacts, timing = {}, {}
    for label, member in registration["members"].items():
        guard(root)
        if sha256(member["checkpoint"]) != member["sha256"]:
            raise ValueError("Frozen checkpoint changed")
        checkpoint = torch.load(member["checkpoint"], map_location="cpu", weights_only=True)
        if any(value.is_floating_point() and (value.dtype != torch.float32 or not torch.isfinite(value).all())
               for value in checkpoint["model"].values()):
            raise ValueError("Require finite FP32 checkpoint weights")
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        for name, panel in prepared["panels"].items():
            guard(root)
            cases = torch.load(panel["path"], map_location="cpu", weights_only=True)["cases"]
            validate_population(cases, registration["panels"][name], registration["old_case_digests"])
            before = time.monotonic()
            prediction_path = root / f"{name}_{label}_predictions.pth"
            metrics = evaluate_fixed_validation(model, criterion, cases, "cuda", "amp_bfloat16", prediction_path)
            elapsed = time.monotonic() - before
            artifacts[str(prediction_path)] = sha256(prediction_path)
            timing[f"{name}/{label}"] = elapsed
            print(json.dumps({"panel": name, "model": label, "macro_rmse": metrics["macro_rmse"], "seconds": elapsed}), flush=True)
    del model, criterion
    torch.cuda.empty_cache()
    panels = {}
    for name, panel in prepared["panels"].items():
        cases = torch.load(panel["path"], map_location="cpu", weights_only=True)["cases"]
        payloads = [torch.load(root / f"{name}_{label}_predictions.pth", map_location="cpu", weights_only=True)
                    for label in ("rank1", "rank2", "student")]
        panels[name] = compare_student(name, cases, *payloads)
    verify_inputs(registration["input_sha256"])
    verify_inputs({panel["path"]: panel["sha256"] for panel in prepared["panels"].values()})
    verify_inputs(artifacts)
    result = {"utc": datetime.now(timezone.utc).isoformat(), "panels": panels,
              "prespecification_sha256": sha256(root / "prespecification.json"), "prepared_sha256": sha256(root / "prepared.json"),
              "artifact_sha256": artifacts, "evaluation_seconds": timing, "elapsed_seconds": time.monotonic() - started,
              "storage": guard(root), "checkpoint_created_or_promoted": False, "training_started": False}
    immutable_json(root / "results.json", result)
    immutable_json(root / "status.json", {"state": "complete", "utc": result["utc"],
                                          "checkpoint_created_or_promoted": False, "training_started": False})
    return {"state": "complete", "elapsed_seconds": result["elapsed_seconds"],
            "panels": {name: values["cached_macro_rmse_reproduced"] for name, values in panels.items()}}


def main():
    parser = argparse.ArgumentParser()
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--register-only", action="store_true")
    options.add_argument("--prepare", action="store_true")
    options.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = REPO / "runs/objective_queues" / EXPERIMENT
    root.mkdir(exist_ok=True)
    with (root / "experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "results.json").exists():
            raise ValueError("Fresh diagnostic already complete; preserve existing evidence")
        guard(root)
        result = register(root) if args.register_only else prepare(root) if args.prepare else evaluate(root)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
