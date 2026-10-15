"""Preserve four closed nonselected artifacts for the dual-parent full trial."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import types

import torch

import preflight_single_model_dual_parent_retention as probe
import relocate_single_model_parent_structured_capacity as blueprint
from relocate_single_model_parent_structured_capacity import readers, copying, sync_directory
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


ROOT = probe.ROOT
OUTPUT = ROOT / "dual_parent_capacity_relocation_v1"
DESTINATION = copying.SCRATCH / "single_model_goal_20261005_owned/dual_parent_capacity_v1"
TRAINING_ADMISSION_BYTES = blueprint.TRAINING_ADMISSION_BYTES
SELECTED = ROOT / "parent_structured_v49/selected_checkpoint.pth"
SELECTED_SHA256 = probe.diagnosis.SELECTED_SHA256
TARGETS = {
    "parent_structured_v49/latest_checkpoint.pth": "tail",
    "dual_parent_retention_preflight_v1/candidate_parameters.pt": "vector",
    "decoder_subspace_probe_v1/decoder.pt": "decoder",
    "decoder_subspace_probe_v1/output_layer.pt": "output_layer",
}
STATES = {
    "parent_structured_v49": "training_complete",
    "parent_structured_v49_audit_task_replay": "complete",
    "parent_structured_v49_endpoint_replay": "complete",
    "parent_structured_v49_confirmation": "complete",
    "parent_structured_v49_pipeline": "complete",
    "parent_structured_v49_initializer_diagnosis_v1": "complete",
    "dual_parent_retention_preflight_v1": "complete",
    "decoder_subspace_probe_v1": "complete",
}
POLICY = (
    "Storage only: preserve exact serialized bytes of V49's independently audited nonselected "
    "step64 tail with62 updates, the completed fixed dual-parent feasibility vector, and both "
    "already-audited decoder-subspace diagnostic projections. Preserve their original42-gradient "
    "scope and explicit absence of optimizer replay, finite-step neural accuracy and candidate "
    "selection. Keep V49's selected step25, actual V29, every selected/teacher checkpoint and "
    "unaudited V28 untouched. Require stopped owners, unlocked locks and no open readers. "
    "Copy/fsync/hash/typed-load before canonical symlink publication and separate independent "
    "recheck. Restore18 GiB plus2432 MiB full-trial admission without lowering18/6/50 GiB reserves. "
    "Scratch is machine-local, not external backup. No new training, inference, seeds, model "
    "selection, qualification or benchmark-success claim."
)


def validate_payload(payload, kind, *, full_parameters=77998496, model_tensors=254,
                     decoder_parameters=11227016, output_parameters=7685000, selected_model=None):
    if kind == "vector":
        return blueprint.validate_payload(payload, kind, full_parameters=full_parameters)
    if kind == "tail":
        if (not isinstance(payload, dict) or payload.get("step") != 64 or payload.get("best_step") != 25
                or payload.get("gradient_updates_performed") != 62 or payload.get("attempted_batches") != 64
                or payload.get("config_sha256") != "9eec9f2c3226153e68607fb42647fcc7c919940880c6530ef704cb2faae85b97"
                or payload.get("parent_checkpoint_sha256") != probe.PARENT_SHA256
                or payload.get("independent_confirmation_passed") is not False
                or payload.get("preflight_only") is not False or payload.get("full_network_training") is not True
                or not payload.get("optimizer") or not isinstance(payload.get("model"), dict)
                or len(payload["model"]) != model_tensors or selected_model is None
                or set(selected_model) != set(payload["model"])
                or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in payload["model"].values())
                or all(torch.equal(value, selected_model[name]) for name, value in payload["model"].items())):
            raise ValueError("Preserve only the distinct audited nonselected V49 terminal state")
        return
    if kind not in {"decoder", "output_layer"} or set(payload) != {"before", "tentative", "candidate", "ranges"}:
        raise ValueError("Preserve only both declared audited coordinate-subspace projections")
    parameters = decoder_parameters if kind == "decoder" else output_parameters
    for name in ("before", "tentative", "candidate"):
        value = payload[name]
        if not isinstance(value, torch.Tensor) or value.shape != (parameters,) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("Preserve every original finite FP32 subspace coordinate")
    ranges = payload["ranges"]
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("Require the complete original coordinate ranges")
    names, total, previous_stop = set(), 0, -1
    for entry in ranges:
        if (set(entry) != {"name", "shape", "start", "stop"}
                or not entry["name"].startswith("decoder.layers.4." if kind == "output_layer" else "decoder.")
                or entry["name"] in names or type(entry["start"]) is not int or type(entry["stop"]) is not int
                or entry["start"] < 0 or entry["start"] < previous_stop or entry["stop"] <= entry["start"]
                or not isinstance(entry["shape"], list) or not entry["shape"]
                or any(type(size) is not int or size <= 0 for size in entry["shape"])
                or math.prod(entry["shape"]) != entry["stop"] - entry["start"]):
            raise ValueError("Preserve ordered nonoverlapping named coordinate ranges")
        names.add(entry["name"])
        total += entry["stop"] - entry["start"]
        previous_stop = entry["stop"]
    if total != parameters:
        raise ValueError("The original coordinate ranges must cover every saved parameter")


def evidence():
    identities = probe.require_completed_v49()
    probe.verify(probe.OUTPUT)
    reports = {name: read(ROOT / name) for name in (
        "parent_structured_v49_audit/audit.json", "dual_parent_retention_preflight_v1/audit.json",
        "decoder_subspace_probe_v1/audit.json")}
    tail, feasibility, subspaces = reports.values()
    retention, gate = read(probe.OUTPUT / "retention_audit.json"), read(probe.OUTPUT / "gate.json")
    diagnostic = read(ROOT / "decoder_subspace_probe_v1/result.json")
    if (tail["complete"] is not True or tail["through_step"] != 64 or tail["best"]["step"] != 25
            or tail["accepted_updates"] != 62 or feasibility["complete"] is not True
            or feasibility["training_tasks"] != 28 or feasibility["prediction_grids"] != 56
            or gate["passed"] is not True or gate["preflight_only"] is not True
            or retention["complete"] is not True or retention["reference_grids"] != 26
            or retention["reference_role_counts"] != {"general": 2, "structured_mean": 24}
            or retention["component_role_counts"] != {"general": 2, "v8": 24, "v29": 24}
            or retention["component_grids"] != 50 or subspaces["complete"] is not True
            or subspaces["scopes"] != ["decoder", "output_layer"] or subspaces["source_gradient_vectors"] != 42
            or subspaces["cuda_initialized"] is not False or subspaces["optimizer_replayed"] is not False
            or subspaces["finite_step_accuracy_assessed"] is not False
            or subspaces["goal_success_not_assessed"] is not True
            or set(diagnostic["scopes"]) != {"decoder", "output_layer"}
            or any(diagnostic[key] is not False for key in ("neural_evaluation_performed", "optimizer_replayed", "candidate_selection_performed"))):
        raise ValueError("Require the complete original audits and limitations for all four nonselected artifacts")
    verify_hashes(subspaces["artifact_sha256"])
    hashes = {path: digest for report in reports.values() for path, digest in report["artifact_sha256"].items()}
    paths = [ROOT / name for name in reports]
    paths.extend(ROOT / name / "status.json" for name in STATES)
    paths.extend(probe.OUTPUT / name for name in ("config.json", "registration.json", "result.json", "retention_audit.json", "gate.json", "external_recheck.json"))
    paths.extend((SELECTED, probe.PARENT, ROOT / "parent_structured_v49/config.json",
                  ROOT / "decoder_subspace_probe_v1/config.json", ROOT / "decoder_subspace_probe_v1/result.json"))
    identities.update({str(path): sha256(path) for path in paths})
    verify_hashes(identities)
    return hashes, identities


for name in ("read", "status", "require_selected", "require_stopped", "validate_file", "inventory", "copy_and_verify"):
    function = getattr(blueprint, name)
    if function.__closure__ is not None:
        raise ValueError("Preservation callbacks must bind only this new registered scope")
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
RECHECK_SOURCE = replace_once(blueprint.RECHECK_SOURCE, "'canonical_artifacts_checked': 5", "'canonical_artifacts_checked': 4")
exec(compile(RECHECK_SOURCE, __file__, "exec"), globals())
PRESERVE_SOURCE = replace_once(blueprint.PRESERVE_SOURCE,
    "tests/test_single_model_parent_structured_capacity.py", "tests/test_single_model_dual_parent_capacity.py")
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
