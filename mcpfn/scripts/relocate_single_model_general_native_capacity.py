"""Preserve three closed nonselected artifacts before the native-general trial."""

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

import preflight_single_model_general_native_retention as probe
import relocate_single_model_stronger_mean_capacity as blueprint
from relocate_single_model_stronger_mean_capacity import readers, copying, sync_directory
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


ROOT = probe.ROOT
OUTPUT = ROOT / "general_native_capacity_relocation_v1"
DESTINATION = copying.SCRATCH / "single_model_goal_20261005_owned/general_native_capacity_v1"
TRAINING_ADMISSION_BYTES = blueprint.TRAINING_ADMISSION_BYTES
SELECTED = ROOT / "stronger_mean_distillation_v46/selected_checkpoint.pth"
SELECTED_SHA256 = "122234a88e81ab3a3b9c88129ee4bd33e60030de097c669ea599e7a26821462b"
TARGETS = {
    "stronger_mean_distillation_v46/latest_checkpoint.pth": "tail",
    "general_native_retention_preflight_v1/candidate_parameters.pt": "vector",
    "decoder_rate_probe_v1/rate_1/endpoint.pt": "rate_endpoint",
}
POLICY = (
    "Storage only: preserve exactly the independently audited nonselected V46 step64 optimizer tail, "
    "completed native-general feasibility vector and closed decoder-rate1 diagnostic endpoint. Keep "
    "the original decoder probe failure and complete additive audit recovery unchanged. Never move "
    "selected/teacher checkpoints or unaudited V28 material. Require stopped owners, unlocked producer "
    "locks and no open readers. Copy/fsync/hash/typed-load every original byte before canonical symlink "
    "publication and independent recheck. Preserve all source/config/evidence hashes and restore "
    "18 GiB plus2432 MiB admission without lowering18/6/50 GiB reserves. Scratch is machine-local, "
    "not external backup. No new training, inference, seeds, selection or scientific success claim."
)
STATES = {
    "stronger_mean_distillation_v46": "training_complete",
    "stronger_mean_distillation_v46_cuda_entry_recovery_v1": "complete",
    "stronger_mean_distillation_v46_live_replay_continuation_v1": "complete",
    "stronger_mean_distillation_v46_confirmation": "complete",
    "stronger_mean_distillation_v46_initializer_diagnosis_v1": "complete",
    "general_native_retention_preflight_v1": "complete",
    "decoder_rate_probe_v1": "failed",
    "decoder_rate_probe_v1_audit_recovery1": "complete",
}


def require_stopped():
    for name, expected in STATES.items():
        state = read(ROOT / name / "status.json")
        if state["state"] != expected or probe.diagnosis.original.process_identity(state["pid"]) is not None:
            raise ValueError("Require every original stopped owner and preserved outcome")
        path = ROOT / name / "run.lock"
        if path.exists():
            with path.open("r") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def validate_payload(payload, kind, *, full_parameters=77998496, model_tensors=254,
                     rate_parameters=11227016, selected_model=None):
    if kind == "vector":
        return blueprint.validate_payload(payload, kind, full_parameters=full_parameters)
    if kind == "tail":
        if (not isinstance(payload, dict) or payload.get("step") != 64 or payload.get("best_step") != 10
                or payload.get("gradient_updates_performed") != 55 or payload.get("attempted_batches") != 64
                or payload.get("config_sha256") != "ac98d26801b73579928c510b6e76a7b8479cd4fa05c5690a5599e69127733c49"
                or payload.get("independent_confirmation_passed") is not False
                or payload.get("preflight_only") is not False or payload.get("full_network_training") is not True
                or not payload.get("optimizer") or not isinstance(payload.get("model"), dict)
                or len(payload["model"]) != model_tensors or selected_model is None
                or set(selected_model) != set(payload["model"])
                or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                       for value in payload["model"].values())
                or all(torch.equal(value, selected_model[name]) for name, value in payload["model"].items())):
            raise ValueError("Preserve only the distinct audited nonselected V46 terminal state")
        return
    if kind != "rate_endpoint":
        raise ValueError("Move only explicitly declared artifact kinds")
    if (payload["config_sha256"] != sha256(ROOT / "decoder_rate_probe_v1/config.json")
            or payload["record"]["rate"] != 2e-5 or payload["record"]["accepted"] is not True
            or payload["record"]["protected_tasks"] != 42 or not payload["optimizer"]
            or not payload["source_raw_sha256"]):
        raise ValueError("Preserve the original audited rate1 endpoint, never a model selection")
    for name in ("before", "tentative", "after", "combined_gradient", "applied_gradient"):
        value = payload["snapshot"][name]
        if value.shape != (rate_parameters,) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("Preserve every finite decoder-rate endpoint coordinate")


def validate_file(path, kind):
    selected = None
    if kind == "tail":
        require_selected()
        selected = torch.load(SELECTED, map_location="cpu", weights_only=True, mmap=True)["model"]
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    validate_payload(payload, kind, selected_model=selected)


def evidence():
    identities = probe.require_completed_v46()
    probe.verify(probe.OUTPUT)
    reports = {name: read(ROOT / name) for name in (
        "stronger_mean_distillation_v46_recovered_audit_v1/audit.json",
        "general_native_retention_preflight_v1/audit.json",
        "decoder_rate_probe_v1_audit_recovery1/audit.json")}
    tail, feasibility, rate = reports.values()
    retention = read(probe.OUTPUT / "retention_audit.json")
    gate = read(probe.OUTPUT / "gate.json")
    if (not tail["complete"] or tail["through_step"] != 64 or tail["best"]["step"] != 10
            or not feasibility["complete"] or feasibility["training_tasks"] != 28
            or feasibility["prediction_grids"] != 56 or not gate["passed"] or not gate["preflight_only"]
            or not retention["complete"] or retention["reference_role_counts"] != {"general": 2, "v8": 24}
            or not rate["complete"] or rate["checkpoint_selected"] is not False
            or rate["prediction_grids"] != 112 or rate["training_tasks"] != 14):
        raise ValueError("Require complete original audits for every nonselected artifact")
    hashes = {path: digest for report in reports.values() for path, digest in report["artifact_sha256"].items()}
    paths = [ROOT / name for name in reports]
    paths.extend(ROOT / name / "status.json" for name in STATES)
    paths.extend(probe.OUTPUT / name for name in ("config.json", "registration.json", "result.json",
                                                "retention_audit.json", "gate.json"))
    paths.extend([SELECTED, ROOT / "stronger_mean_distillation_v46/config.json",
                  ROOT / "decoder_rate_probe_v1/config.json"])
    identities.update({str(path): sha256(path) for path in paths})
    verify_hashes(identities)
    return hashes, identities


for name in ("read", "status", "require_selected", "inventory", "copy_and_verify"):
    function = getattr(blueprint, name)
    if function.__closure__ is not None:
        raise ValueError("Preservation helpers must bind only this registered scope")
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)

RECHECK_SOURCE = replace_once(inspect.getsource(blueprint.recheck), "'canonical_artifacts_checked': 5",
                              "'canonical_artifacts_checked': 3")
exec(compile(RECHECK_SOURCE, __file__, "exec"), globals())
PRESERVE_SOURCE = replace_once(inspect.getsource(blueprint.preserve),
    "tests/test_single_model_stronger_mean_capacity.py", "tests/test_single_model_general_native_capacity.py")
exec(compile(PRESERVE_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preserve", "recheck"), required=True)
    arguments = parser.parse_args()
    with probe.diagnosis.original.arithmetic_recovery.previous.original_arithmetic():
        if arguments.stage == "preserve":
            preserve()
        else:
            receipt = OUTPUT / "external_recheck.json"
            if receipt.exists():
                raise FileExistsError("Never overwrite independent preservation evidence")
            write_json(receipt, recheck())
