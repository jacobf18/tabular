"""Preserve three closed nonselected artifacts before the native-column full trial."""

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

import preflight_single_model_native_column_mean as probe
import relocate_single_model_general_native_capacity as blueprint
from relocate_single_model_general_native_capacity import readers, copying, sync_directory
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


ROOT = probe.ROOT
OUTPUT = ROOT / "native_column_capacity_relocation_v1"
DESTINATION = copying.SCRATCH / "single_model_goal_20261005_owned/native_column_capacity_v1"
TRAINING_ADMISSION_BYTES = blueprint.TRAINING_ADMISSION_BYTES
SELECTED = ROOT / "legacy_strata_v29/selected_checkpoint.pth"
SELECTED_SHA256 = "9747c69624220de33356145c5df9baa5256bc625719868d4f0e3740bce475c32"
TARGETS = {
    "general_native_retention_v47/latest_checkpoint.pth": "tail",
    "native_column_mean_preflight_v1/candidate_parameters.pt": "vector",
    "decoder_rate_probe_v1/rate_2/endpoint.pt": "rate_endpoint",
}
STATES = {
    "general_native_retention_v47": "training_complete",
    "general_native_retention_v47_audit_task_replay": "complete",
    "general_native_retention_v47_endpoint_replay": "complete",
    "general_native_retention_v47_confirmation": "complete",
    "general_native_retention_v47_pipeline": "complete",
    "native_column_mean_preflight_v1": "complete",
    "decoder_rate_probe_v1": "failed",
    "decoder_rate_probe_v1_audit_recovery1": "complete",
}
POLICY = (
    "Storage only: preserve exact original bytes of the fully audited nonselected V47 step64 "
    "tail with58 updates and no selection, the completed native-column feasibility vector, and "
    "the independently audited rejected decoder-rate2 endpoint with exact rollback. Preserve "
    "its original failed producer status and complete additive audit recovery. Keep V29, all "
    "selected/teacher checkpoints and unaudited V28 material untouched. Require stopped owners, "
    "unlocked producer locks and no open readers. Copy/fsync/hash/typed-load before canonical "
    "symlink publication and separate independent recheck. Restore18 GiB plus2432 MiB full-trial "
    "admission without lowering18/6/50 GiB reserves. Scratch is machine-local, not external backup. "
    "No new training, inference, seeds, checkpoint selection or scientific success claim."
)


def validate_payload(payload, kind, *, full_parameters=77998496, model_tensors=254,
                     rate_parameters=11227016, selected_model=None):
    if kind == "vector":
        return blueprint.validate_payload(payload, kind, full_parameters=full_parameters)
    if kind == "tail":
        if (not isinstance(payload, dict) or payload.get("step") != 64 or payload.get("best_step") != 0
                or payload.get("gradient_updates_performed") != 58 or payload.get("attempted_batches") != 64
                or payload.get("config_sha256") != "1a5410dda6c87086a35801e646688f408751eec0d99a0c8d01865d57f6d9c0df"
                or payload.get("parent_checkpoint_sha256") != SELECTED_SHA256
                or payload.get("independent_confirmation_passed") is not False
                or payload.get("preflight_only") is not False or payload.get("full_network_training") is not True
                or not payload.get("optimizer") or not isinstance(payload.get("model"), dict)
                or len(payload["model"]) != model_tensors or selected_model is None
                or set(selected_model) != set(payload["model"])
                or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in payload["model"].values())
                or all(torch.equal(value, selected_model[name]) for name, value in payload["model"].items())):
            raise ValueError("Preserve only the distinct audited V47 terminal state with no selected checkpoint")
        return
    if kind != "rate_endpoint":
        raise ValueError("Move only explicitly declared artifact kinds")
    if (payload["config_sha256"] != sha256(ROOT / "decoder_rate_probe_v1/config.json")
            or payload["record"]["rate"] != 1e-4 or payload["record"]["accepted"] is not False
            or payload["record"]["reason"] != "finite_metrics_rejected"
            or payload["record"]["protected_tasks"] != 42 or not payload["optimizer"]
            or not payload["source_raw_sha256"]):
        raise ValueError("Preserve the audited rejected rate2 endpoint, not an accepted or selected model")
    for name in ("before", "tentative", "after", "combined_gradient", "applied_gradient"):
        value = payload["snapshot"][name]
        if value.shape != (rate_parameters,) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("Preserve every original finite decoder coordinate")
    if not torch.equal(payload["snapshot"]["before"], payload["snapshot"]["after"]):
        raise ValueError("The original rejected endpoint must retain exact parameter rollback")


def evidence():
    identities = probe.require_completed_v47()
    probe.verify(probe.OUTPUT)
    reports = {name: read(ROOT / name) for name in (
        "general_native_retention_v47_audit/audit.json", "native_column_mean_preflight_v1/audit.json",
        "decoder_rate_probe_v1_audit_recovery1/audit.json")}
    tail, feasibility, rate = reports.values()
    retention, gate = read(probe.OUTPUT / "retention_audit.json"), read(probe.OUTPUT / "gate.json")
    if (tail["complete"] is not True or tail["through_step"] != 64 or tail["best"]["step"] != 0
            or tail["accepted_updates"] != 58 or feasibility["complete"] is not True
            or feasibility["training_tasks"] != 28 or feasibility["prediction_grids"] != 56
            or gate["passed"] is not True or gate["preflight_only"] is not True
            or retention["complete"] is not True or retention["reference_role_counts"] != {"general": 2, "v8": 24}
            or rate["complete"] is not True or rate["checkpoint_selected"] is not False
            or rate["prediction_grids"] != 112 or rate["training_tasks"] != 14):
        raise ValueError("Require complete original audits of all three nonselected artifacts")
    hashes = {path: digest for report in reports.values() for path, digest in report["artifact_sha256"].items()}
    paths = [ROOT / name for name in reports]
    paths.extend(ROOT / name / "status.json" for name in STATES)
    paths.extend(probe.OUTPUT / name for name in ("config.json", "registration.json", "result.json", "retention_audit.json", "gate.json"))
    paths.extend((SELECTED, ROOT / "general_native_retention_v47/config.json", ROOT / "decoder_rate_probe_v1/config.json"))
    identities.update({str(path): sha256(path) for path in paths})
    verify_hashes(identities)
    return hashes, identities


for name in ("read", "status", "require_selected", "require_stopped", "validate_file", "inventory", "copy_and_verify", "recheck"):
    function = getattr(blueprint, name)
    if function.__closure__ is not None:
        raise ValueError("Preservation callbacks must bind only this new registered scope")
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
PRESERVE_SOURCE = replace_once(blueprint.PRESERVE_SOURCE,
    "tests/test_single_model_general_native_capacity.py", "tests/test_single_model_native_column_capacity.py")
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
