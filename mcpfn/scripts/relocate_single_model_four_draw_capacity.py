"""Preserve one closed optimizer tail and four audited diagnostic vectors for V57."""

import argparse
import fcntl
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types

import torch

import recover_single_model_four_draw_mean_audit as recovery
import relocate_single_model_stronger_mean_capacity as template
from relocate_single_model_stronger_mean_capacity import readers, copying, sync_directory
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


probe = recovery.probe
ROOT = probe.ROOT
OUTPUT = ROOT / "four_draw_capacity_relocation_v1"
DESTINATION = copying.SCRATCH / "single_model_goal_20261005_owned/four_draw_capacity_v1"
TRAINING_ADMISSION_BYTES = 18 * 1024**3 + 4096 * 1024**2
SELECTED, SELECTED_SHA256 = probe.PARENT, probe.PARENT_SHA256
PROBES = ("standardized_teacher_huber_preflight_v1", "balanced_native_quarter_preflight_v1",
    "balanced_native_coverage_preflight_v1", "four_draw_preflight_v1")
TARGETS = {"balanced_native_long_v56/latest_checkpoint.pth": "tail",
    **{f"{name}/candidate_parameters.pt": "vector" for name in PROBES}}
EXPECTED_BYTES = 2184275535
TAIL_CONFIG_SHA256 = "ebd96a62be81a2f9737042fa49686390ed941ab0c8c9f2425d5899e9594002cf"
CLOSURE_HASHES = {
    str(recovery.OUTPUT / "result.json"): "a6923bb5cd25758545fbf3124a9f3ccd4792c9b13c133fc3fec305a715e3a6c0",
    str(recovery.mean.OUTPUT / "audit.json"): "bcc8fbc602a28391fec084ffe7e800501ecd32b98def457470b26ebb1947072a",
}
STATES = {**{name: "complete" for name in PROBES}, recovery.OUTPUT.name: "complete",
    **{f"balanced_native_long_v56{suffix}": "complete" if suffix else "training_complete"
        for suffix in ("", "_pipeline", "_audit_task_replay", "_endpoint_replay", "_confirmation")}}
POLICY = (
    "Storage-only capacity preparation after complete negative V56 closure and complete "
    "four-draw feasibility closure. Preserve exactly the936293291-byte unselected V56 "
    "optimizer tail and four311995561-byte independently audited candidate vectors, "
    "total2184275535 bytes. Keep the accepted and rejected diagnostic outcomes unchanged. "
    "The current four-draw vector is eligible only after the original optimizer, endpoint, "
    "reference, transfer and supplementary mean audits finish; never move an active artifact. "
    "Exclude the failed unaudited V28 tail. Do not move an initializer, teacher, selected "
    "checkpoint, mounted archive or ordinary probe checkpoint. Copy/fsync/hash/typed-load "
    "all five complete artifacts before atomically publishing canonical symlinks. Require "
    "stopped producers, free existing locks and no readers. Independently recheck in a "
    "separate CPU process. Restore18 GiB plus4 GiB checkpoint admission without lowering "
    "18/6/50 GiB reserves. No evidence deletion, training, inference, seed, qualification "
    "or benchmark outcome change. UUID-bound scratch remains machine-local, not external "
    "or provider-persistent backup."
)


def require_stopped():
    for name, expected in STATES.items():
        directory = ROOT / name
        state = read(directory / "status.json")
        owner = state.get("pid")
        if state["state"] != expected or type(owner) is not int or owner <= 0 or Path(f"/proc/{owner}").exists():
            raise ValueError("Require every original producer and completed audit coordinator to be stopped")
        lock = directory / "run.lock"
        if lock.exists():
            with lock.open("rb") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def validate_payload(payload, kind, *, full_parameters=77998496, model_tensors=254, optimizer_tensors=238):
    if kind == "vector":
        if (not isinstance(payload, torch.Tensor) or payload.shape != (full_parameters,)
                or payload.dtype != torch.float32 or not torch.isfinite(payload).all()):
            raise ValueError("Preserve every finite FP32 coordinate of the exact audited candidate")
        return
    expected = {"step": 256, "epoch": 256, "attempted_batches": 256, "gradient_updates_performed": 111,
        "parent_training_updates": 98, "parent_checkpoint_sha256": SELECTED_SHA256, "config_sha256": TAIL_CONFIG_SHA256,
        "best_step": 0, "preflight_only": False, "independent_confirmation_passed": False, "full_network_training": True}
    if kind != "tail" or not isinstance(payload, dict) or any(payload.get(name) != value for name, value in expected.items()):
        raise ValueError("Preserve the exact unqualified V56 terminal model, optimizer and metadata")
    model, optimizer, rng = payload.get("model"), payload.get("optimizer"), payload.get("rng")
    if (not isinstance(model, dict) or len(model) != model_tensors
            or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in model.values())
            or not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"}
            or len(optimizer["state"]) != optimizer_tensors or len(optimizer["param_groups"]) != 1
            or len(optimizer["param_groups"][0]["params"]) != optimizer_tensors
            or set(optimizer["param_groups"][0]["params"]) != set(optimizer["state"])):
        raise ValueError("Keep every original model tensor and optimizer coordinate")
    for state in optimizer["state"].values():
        if (set(state) != {"step", "exp_avg", "exp_avg_sq"}
                or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values())
                or state["step"].numel() != 1 or state["step"].item() != 111
                or state["exp_avg"].shape != state["exp_avg_sq"].shape):
            raise ValueError("Keep both original finite Adam moments and all111 accepted-update counters")
    if (not isinstance(rng, dict) or set(rng) != {"cpu", "cuda"} or not isinstance(rng["cuda"], list) or len(rng["cuda"]) != 1
            or any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or not value.numel()
                for value in [rng["cpu"], *rng["cuda"]])):
        raise ValueError("Keep complete original CPU and single-device CUDA RNG bytes")


def validate_file(path, kind):
    require_selected()
    validate_payload(torch.load(path, map_location="cpu", weights_only=True, mmap=True), kind)


def evidence():
    identities = probe.require_predecessor()
    recovery.verify()
    verify_hashes(CLOSURE_HASHES)
    identities.update(CLOSURE_HASHES)
    completed = read(recovery.OUTPUT / "result.json")
    if (completed["complete"] is not True or completed["feasibility_passed"] is not True
            or completed["all_original_failures_preserved"] is not True
            or any(completed[name] is not False for name in ("training_repeated", "neural_replay_repeated", "optimizer_audit_repeated", "full_trial_authorized"))):
        raise ValueError("Require complete original positive feasibility without relabeling any earlier failure")
    verify_hashes(completed["artifact_sha256"])
    identities.update(completed["artifact_sha256"])
    tail = read(ROOT / "balanced_native_long_v56_audit/audit.json")
    hashes = {str(ROOT / "balanced_native_long_v56/latest_checkpoint.pth"):
        tail["artifact_sha256"][str(ROOT / "balanced_native_long_v56/latest_checkpoint.pth")]}
    for name in PROBES:
        directory = ROOT / name
        audit = read(directory / "audit.json")
        gate = read(directory / "gate.json")
        expected_tasks = 56 if name == "four_draw_preflight_v1" else 28
        if (audit["complete"] is not True or audit["training_tasks"] != expected_tasks
                or audit["prediction_grids"] != 2 * expected_tasks
                or gate["audit_sha256"] != sha256(directory / "audit.json") or gate["real_benchmark_success"] is not False):
            raise ValueError("Preserve both complete scored endpoints and their original accepted or rejected decision")
        vector = str(directory / "candidate_parameters.pt")
        hashes[vector] = audit["artifact_sha256"][vector]
        for filename in ("config.json", "registration.json", "audit.json", "retention_audit.json", "gate.json", "result.json", "status.json"):
            identities[str(directory / filename)] = sha256(directory / filename)
    paths = [SELECTED, ROOT / "balanced_native_long_v56/config.json", ROOT / "balanced_native_long_v56_audit/audit.json"]
    paths.extend(ROOT / name / "status.json" for name in STATES)
    paths.extend(recovery.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "mean_audit.log"))
    identities.update({str(path): sha256(path) for path in paths})
    verify_hashes(identities)
    return hashes, identities


for name in ("read", "status", "require_selected", "inventory", "copy_and_verify", "recheck"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Never retain a prior capacity callback closure")
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
PRESERVE_SOURCE = replace_once(inspect.getsource(template.preserve), "test_single_model_stronger_mean_capacity.py",
    "test_single_model_four_draw_capacity.py")
PRESERVE_SOURCE = replace_once(PRESERVE_SOURCE, "    total = sum(row['bytes'] for row in records)",
    "    total = sum(row['bytes'] for row in records)\n"
    "    if total != EXPECTED_BYTES or len(records) != 5:\n"
    "        raise ValueError('Preserve the exact complete tail and all four audited vectors')")
exec(compile(PRESERVE_SOURCE, __file__, "exec"), globals())


def require_completed(receipt_sha256):
    receipt_path = OUTPUT / "external_recheck.json"
    if sha256(receipt_path) != receipt_sha256 or read(OUTPUT / "status.json")["state"] != "complete":
        raise ValueError("Require the completed independent five-file preservation receipt")
    saved, current = read(receipt_path), recheck()
    for key in ("complete", "canonical_artifacts_checked", "all_original_serialized_bytes_and_paths_preserved",
            "selected_checkpoints_unchanged", "bytes_relocated", "config_sha256", "training_admission_bytes"):
        if saved[key] != current[key]:
            raise ValueError("Retain every byte and canonical path of the original complete preservation")
    if saved["bytes_relocated"] != EXPECTED_BYTES or saved["canonical_artifacts_checked"] != 5:
        raise ValueError("Never substitute a smaller storage scope")
    config = read(OUTPUT / "config.json")
    paths = [OUTPUT / name for name in ("config.json", "registration.json", "status.json", "independent_recheck.json", "external_recheck.json")]
    return {**config["input_sha256"], **config["source_sha256"], **current["artifact_sha256"],
        **{str(path): sha256(path) for path in paths}}


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Preservation and its independent recheck must remain CPU-only")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preserve", "recheck"), required=True)
    arguments = parser.parse_args()
    with probe.diagnosis.original.arithmetic_recovery.previous.original_arithmetic():
        if arguments.stage == "preserve":
            preserve()
        else:
            receipt_path = OUTPUT / "external_recheck.json"
            if receipt_path.exists():
                raise FileExistsError("Never overwrite independent preservation evidence")
            write_json(receipt_path, recheck())


if __name__ == "__main__":
    main()
