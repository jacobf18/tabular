"""Preserve five audited nonselected artifacts for the parent-structured trial."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types

import torch

import preflight_single_model_parent_structured_retention as probe
import relocate_single_model_native_column_capacity as blueprint
from relocate_single_model_native_column_capacity import readers, copying, sync_directory
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


ROOT = probe.ROOT
OUTPUT = ROOT / "parent_structured_capacity_relocation_v1"
DESTINATION = copying.SCRATCH / "single_model_goal_20261005_owned/parent_structured_capacity_v1"
TRAINING_ADMISSION_BYTES = blueprint.TRAINING_ADMISSION_BYTES
SELECTED = ROOT / "native_column_mean_v48/selected_checkpoint.pth"
SELECTED_SHA256 = probe.diagnosis.SELECTED_SHA256
MATCHED_CASES = {
    "composite_1e-5": ("composite", 1e-5, True),
    "composite_2e-5": ("composite", 2e-5, False),
    "point_1e-5": ("point_inclusive", 1e-5, True),
}
TARGETS = {
    "native_column_mean_v48/latest_checkpoint.pth": "tail",
    "parent_structured_retention_preflight_v1/candidate_parameters.pt": "vector",
    **{f"point_proposal_matched_v1/{name}/endpoint.pt": name for name in MATCHED_CASES},
}
STATES = {
    "native_column_mean_v48": "training_complete",
    "native_column_mean_v48_audit_task_replay": "complete",
    "native_column_mean_v48_endpoint_replay": "complete",
    "native_column_mean_v48_confirmation": "complete",
    "native_column_mean_v48_pipeline": "complete",
    "native_column_mean_v48_initializer_diagnosis_v1": "complete",
    "parent_structured_retention_preflight_v1": "complete",
    "point_proposal_matched_v1": "complete",
}
POLICY = (
    "Storage only: preserve exact original serialized bytes of V48's fully audited nonselected "
    "step64 tail with56 accepted updates, the completed parent-structured feasibility vector, "
    "and all three independently audited matched-proposal diagnostic endpoints, including the "
    "rejected composite2e-5 control. Do not move V48's selected step10, V29, teachers, other "
    "selected checkpoints or unaudited V28 evidence. Require stopped owners, unlocked run locks "
    "and no open readers. Copy/fsync/hash/typed-load before canonical symlink publication and "
    "a separate independent recheck. Restore18 GiB plus2432 MiB full-trial admission without "
    "lowering18/6/50 GiB reserves. Scratch is machine-local, not external backup. No new training, "
    "inference, seeds, model selection or scientific success claim."
)


def validate_payload(payload, kind, *, full_parameters=77998496, model_tensors=254,
                     proposal_parameters=11227016, selected_model=None):
    if kind == "vector":
        return blueprint.validate_payload(payload, kind, full_parameters=full_parameters)
    if kind == "tail":
        if (not isinstance(payload, dict) or payload.get("step") != 64 or payload.get("best_step") != 10
                or payload.get("gradient_updates_performed") != 56 or payload.get("attempted_batches") != 64
                or payload.get("config_sha256") != "9f34ec4f80daae3609135eb41ffe18c58492f6db62ebecbb53759bb0d4e8e1fa"
                or payload.get("parent_checkpoint_sha256") != probe.PARENT_SHA256
                or payload.get("independent_confirmation_passed") is not False
                or payload.get("preflight_only") is not False or payload.get("full_network_training") is not True
                or not payload.get("optimizer") or not isinstance(payload.get("model"), dict)
                or len(payload["model"]) != model_tensors or selected_model is None
                or set(selected_model) != set(payload["model"])
                or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in payload["model"].values())
                or all(torch.equal(value, selected_model[name]) for name, value in payload["model"].items())):
            raise ValueError("Preserve only the distinct audited nonselected V48 terminal state")
        return
    if kind not in MATCHED_CASES:
        raise ValueError("Move only declared audited artifact kinds")
    method, rate, accepted = MATCHED_CASES[kind]
    expected_case = {"name": kind, "method": method, "rate": rate, "reused": False}
    expected_reference = {"path": str(ROOT / "point_inclusive_preflight_v1/raw_update.pt"),
        "sha256": "c17b357aa90ec33d66d9f11006c225e6f7354576537f1e3528ce13b63e501d0d"}
    record = payload["record"]
    if (payload["case"] != expected_case or payload["method"] != method
            or payload["reference"] != expected_reference
            or payload["final_optimizer_bitwise_verified"] is not True
            or payload["after_source"] != ("candidate" if accepted else "initialization")
            or record["rate"] != rate or record["accepted"] is not accepted or record["margin"] != .1
            or record["composite_tasks"] != 28 or record["protected_tasks"] != 84 or record["step"] != 1
            or record["accepted_updates"] != int(accepted)
            or record["reason"] != ("accepted" if accepted else "finite_metrics_rejected")
            or not payload["snapshot"]["optimizer_before"] or not payload["snapshot"]["optimizer_after"]
            or len(payload["frozen"]) != 3):
        raise ValueError("Preserve all original matched-proposal identities and acceptance outcomes")
    for name in ("tentative", "combined_gradient", "applied_gradient"):
        value = payload["snapshot"][name]
        if value.shape != (proposal_parameters,) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("Preserve every original finite matched-proposal coordinate")


def evidence():
    identities = probe.require_completed_v48()
    probe.verify(probe.OUTPUT)
    reports = {name: read(ROOT / name) for name in (
        "native_column_mean_v48_audit/audit.json", "parent_structured_retention_preflight_v1/audit.json",
        "point_proposal_matched_v1/audit.json")}
    tail, feasibility, matched = reports.values()
    retention, gate = read(probe.OUTPUT / "retention_audit.json"), read(probe.OUTPUT / "gate.json")
    if (tail["complete"] is not True or tail["through_step"] != 64 or tail["best"]["step"] != 10
            or tail["accepted_updates"] != 56 or feasibility["complete"] is not True
            or feasibility["training_tasks"] != 28 or feasibility["prediction_grids"] != 56
            or gate["passed"] is not True or gate["preflight_only"] is not True
            or retention["complete"] is not True or retention["reference_role_counts"] != {"general": 2, "v29": 24}
            or matched["complete"] is not True or matched["matched_cells"] != 4 or matched["new_controls"] != 3
            or matched["prediction_grids"] != 224 or matched["selection_performed"] is not False
            or matched["autograd_replayed"] is not False):
        raise ValueError("Require the complete original audits for all five nonselected artifacts")
    verify_hashes(matched["artifact_sha256"])
    hashes = {path: digest for report in reports.values() for path, digest in report["artifact_sha256"].items()}
    paths = [ROOT / name for name in reports]
    paths.extend(ROOT / name / "status.json" for name in STATES)
    paths.extend(probe.OUTPUT / name for name in ("config.json", "registration.json", "result.json", "retention_audit.json", "gate.json"))
    paths.extend((SELECTED, probe.PARENT, ROOT / "native_column_mean_v48/config.json",
                  ROOT / "point_proposal_matched_v1/config.json"))
    identities.update({str(path): sha256(path) for path in paths})
    verify_hashes(identities)
    return hashes, identities


for name in ("read", "status", "require_selected", "require_stopped", "validate_file", "inventory", "copy_and_verify"):
    function = getattr(blueprint, name)
    if function.__closure__ is not None:
        raise ValueError("Preservation callbacks must bind only this new registered scope")
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
RECHECK_SOURCE = replace_once(blueprint.blueprint.RECHECK_SOURCE, "'canonical_artifacts_checked': 3",
                              "'canonical_artifacts_checked': 5")
exec(compile(RECHECK_SOURCE, __file__, "exec"), globals())
PRESERVE_SOURCE = replace_once(blueprint.PRESERVE_SOURCE,
    "tests/test_single_model_native_column_capacity.py", "tests/test_single_model_parent_structured_capacity.py")
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
