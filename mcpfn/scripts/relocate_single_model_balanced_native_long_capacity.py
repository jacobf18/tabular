"""Preserve two closed optimizer-tail files before the longer fixed-budget trial."""

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

import diagnose_single_model_standardized_huber_initializer as previous_diagnosis
import diagnose_single_model_balanced_native_quarter_initializer as diagnosis
import preflight_single_model_balanced_native_quarter as probe
import relocate_single_model_stronger_mean_capacity as template
from relocate_single_model_stronger_mean_capacity import readers, copying, sync_directory
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


ROOT = diagnosis.original.ROOT
OUTPUT = ROOT / "balanced_native_long_capacity_relocation_v1"
DESTINATION = copying.SCRATCH / "single_model_goal_20261005_owned/balanced_native_long_capacity_v1"
TRAINING_ADMISSION_BYTES = 18 * 1024**3 + 4096 * 1024**2
SELECTED = diagnosis.training.PARENT
SELECTED_SHA256 = diagnosis.PARENT_SHA256
RUNS = {"v54_tail": "standardized_teacher_huber_v54", "v55_tail": "balanced_native_quarter_v55"}
TARGETS = {f"{name}/latest_checkpoint.pth": kind for kind, name in RUNS.items()}
EXPECTED_BYTES = 1872586582
EXPECTED = {
    "v54_tail": {"gradient_updates_performed": 57, "best_step": 50,
        "config_sha256": "97815ee6ea2e76008cfd99fa095416e98f614a9cc800689966e029fb86d3779e"},
    "v55_tail": {"gradient_updates_performed": 40, "best_step": 64,
        "config_sha256": "25609392201d6c3c8df698bddbd60257c5e53e147d4f3c6604b5a93cc80004ef"},
}
PROTECTED = {str(SELECTED): SELECTED_SHA256,
    str(ROOT / RUNS["v54_tail"] / "selected_checkpoint.pth"): previous_diagnosis.SELECTED_SHA256,
    str(ROOT / RUNS["v55_tail"] / "selected_checkpoint.pth"): diagnosis.SELECTED_SHA256}
STATES = {name + suffix: "training_complete" if not suffix else "complete"
    for name in RUNS.values() for suffix in ("", "_audit_task_replay", "_endpoint_replay", "_confirmation", "_pipeline", "_initializer_diagnosis_v1")}
POLICY = (
    "Storage only after complete negative V54/V55 closures and initializer attributions. "
    "Preserve exactly both936293291-byte latest optimizer-tail files, including model, optimizer, "
    "RNG and all metadata; total1872586582 bytes. Neither selected_checkpoint.pth path moves. "
    "V54's tail differs from its selected step50; V55's tail shares its selected step64 model "
    "but remains a separate full optimizer artifact. Preserve that distinction rather than "
    "pretending the V55 weights differ. Keep actual V29, all selected models, teachers, mounts "
    "and scientific outcomes untouched. Require stopped producers, unlocked existing locks "
    "and no open readers. Copy/fsync/hash/typed-load both complete files before publishing "
    "canonical symlinks; independently recheck in a separate CPU process. Restore18 GiB plus "
    "4096 MiB checkpoint admission without lowering18/6/50 GiB reserves. No evidence deletion, "
    "training, inference, seed, selection or qualification change. Scratch is UUID-bound "
    "machine-local storage, not external or provider-persistent backup."
)


def require_selected():
    verify_hashes(PROTECTED)


def require_stopped():
    for name, expected in STATES.items():
        directory = ROOT / name
        current = read(directory / "status.json")
        if (current["state"] != expected or type(current.get("pid")) is not int or current["pid"] <= 0
                or diagnosis.original.process_identity(current["pid"]) is not None):
            raise ValueError("Require every original producer and attribution to be terminal")
        lock = directory / "run.lock"
        if lock.exists():
            with lock.open("rb") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def validate_payload(payload, kind, *, model_tensors=254, optimizer_tensors=238, selected_model=None):
    if kind not in EXPECTED or not isinstance(payload, dict):
        raise ValueError("Preserve only the two declared complete optimizer tails")
    expected = {**EXPECTED[kind], "step": 64, "epoch": 64, "attempted_batches": 64,
        "parent_training_updates": 98, "parent_checkpoint_sha256": SELECTED_SHA256,
        "independent_confirmation_passed": False, "preflight_only": False, "full_network_training": True}
    if any(payload.get(name) != value for name, value in expected.items()):
        raise ValueError("Preserve exact original tail identity and unqualified training metadata")
    model, optimizer = payload.get("model"), payload.get("optimizer")
    if (not isinstance(model, dict) or len(model) != model_tensors or selected_model is None
            or set(model) != set(selected_model)
            or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in model.values())
            or not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"}
            or not optimizer["state"] or not optimizer["param_groups"] or "rng" not in payload):
        raise ValueError("Preserve all finite model tensors, complete optimizer and RNG state")
    equal = all(torch.equal(value, selected_model[name]) for name, value in model.items())
    if equal != (kind == "v55_tail"):
        raise ValueError("Preserve the original selected-versus-tail relationship for each run")
    if (len(optimizer["state"]) != optimizer_tensors or len(optimizer["param_groups"]) != 1
            or len(optimizer["param_groups"][0]["params"]) != optimizer_tensors
            or set(optimizer["param_groups"][0]["params"]) != set(optimizer["state"])):
        raise ValueError("Preserve every original optimizer parameter and both moment tensors")
    for state in optimizer["state"].values():
        if (set(state) != {"step", "exp_avg", "exp_avg_sq"}
                or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values())
                or state["step"].numel() != 1 or state["step"].item() != expected["gradient_updates_performed"]
                or state["exp_avg"].shape != state["exp_avg_sq"].shape):
            raise ValueError("Preserve only the original finite optimizer state")
    rng = payload["rng"]
    if (not isinstance(rng, dict) or set(rng) != {"cpu", "cuda"}
            or not isinstance(rng["cuda"], list) or len(rng["cuda"]) != 1
            or any(not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
                or value.ndim != 1 or value.numel() == 0 for value in [rng["cpu"], *rng["cuda"]])):
        raise ValueError("Preserve the complete original CPU and single-device CUDA RNG states")


def validate_file(path, kind):
    require_selected()
    selected_model = torch.load(ROOT / RUNS[kind] / "selected_checkpoint.pth",
        map_location="cpu", weights_only=True, mmap=True)["model"]
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    validate_payload(payload, kind, selected_model=selected_model)


def evidence():
    previous_diagnosis.require_closed_failure()
    diagnosis.require_closed_failure()
    identities, hashes = dict(PROTECTED), {}
    for kind, name in RUNS.items():
        run = ROOT / name
        audit_path = ROOT / f"{name}_audit/audit.json"
        audit = read(audit_path)
        attribution = ROOT / f"{name}_initializer_diagnosis_v1"
        report = read(attribution / "audit.json")
        if (not audit["complete"] or not audit["complete_training_audit"] or audit["through_step"] != 64
                or audit["accepted_updates"] != EXPECTED[kind]["gradient_updates_performed"]
                or not report["complete"] or report["prediction_grids"] != 284
                or len(report["comparisons"]) != 32 or report["training_performed"] is not False
                or report["confirmation_repeated"] is not False or report["cuda_initialized"] is not False):
            raise ValueError("Require complete original task audits and unchanged consumed-bank attributions")
        target = str(run / "latest_checkpoint.pth")
        hashes[target] = audit["artifact_sha256"][target]
        paths = [audit_path, run / "config.json", ROOT / f"{name}_pipeline/result.json"]
        paths.extend(attribution / filename for filename in ("config.json", "registration.json", "audit.json", "status.json"))
        paths.extend(ROOT / f"{name}{suffix}" / filename for suffix in ("_confirmation", "_endpoint_replay")
            for filename in ("gate.json", "audit.json", "status.json"))
        identities.update({str(path): sha256(path) for path in paths})
    identities.update({str(ROOT / name / "status.json"): sha256(ROOT / name / "status.json") for name in STATES})
    verify_hashes(identities)
    return hashes, identities


for name in ("read", "status", "inventory", "copy_and_verify"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Capacity helpers must bind only the declared two-tail scope")
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
RECHECK_SOURCE = replace_once(inspect.getsource(template.recheck), "'canonical_artifacts_checked': 5",
    "'canonical_artifacts_checked': 2")
exec(compile(RECHECK_SOURCE, __file__, "exec"), globals())
PRESERVE_SOURCE = replace_once(inspect.getsource(template.preserve), "test_single_model_stronger_mean_capacity.py",
    "test_single_model_balanced_native_long_capacity.py")
PRESERVE_SOURCE = replace_once(PRESERVE_SOURCE, "    total = sum(row['bytes'] for row in records)",
    "    total = sum(row['bytes'] for row in records)\n"
    "    if total != EXPECTED_BYTES or len(records) != 2:\n"
    "        raise ValueError('Require both exact complete optimizer tails before copying')")
exec(compile(PRESERVE_SOURCE, __file__, "exec"), globals())


def require_completed(receipt_sha256):
    receipt_path = OUTPUT / "external_recheck.json"
    if sha256(receipt_path) != receipt_sha256 or read(OUTPUT / "status.json")["state"] != "complete":
        raise ValueError("Require the completed independent capacity receipt")
    saved, current = read(receipt_path), recheck()
    for key in ("complete", "canonical_artifacts_checked", "all_original_serialized_bytes_and_paths_preserved",
            "selected_checkpoints_unchanged", "bytes_relocated", "config_sha256", "training_admission_bytes"):
        if saved[key] != current[key]:
            raise ValueError("Preserve the complete two-tail capacity proof")
    if saved["bytes_relocated"] != EXPECTED_BYTES or saved["canonical_artifacts_checked"] != 2:
        raise ValueError("Never substitute a smaller preservation scope")
    config = read(OUTPUT / "config.json")
    paths = [OUTPUT / name for name in ("config.json", "registration.json", "status.json", "independent_recheck.json", "external_recheck.json")]
    return {**config["input_sha256"], **config["source_sha256"], **current["artifact_sha256"],
        **{str(path): sha256(path) for path in paths}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preserve", "recheck"), required=True)
    arguments = parser.parse_args()
    with diagnosis.original.arithmetic_recovery.previous.original_arithmetic():
        if arguments.stage == "preserve":
            preserve()
        else:
            receipt_path = OUTPUT / "external_recheck.json"
            if receipt_path.exists():
                raise FileExistsError("Never overwrite independent preservation evidence")
            write_json(receipt_path, recheck())
