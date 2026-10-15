"""Preserve two completed nonselected artifacts for a new column-mean trial."""

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

import preflight_single_model_all_imputation_column_mean as probe
import relocate_single_model_native_parent_capacity as blueprint
from relocate_single_model_native_parent_capacity import readers, copying, sync_directory
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


ROOT = probe.ROOT
OUTPUT = ROOT / "all_imputation_column_mean_capacity_relocation_v1"
DESTINATION = copying.SCRATCH / "single_model_goal_20261005_owned/all_imputation_column_mean_capacity_v1"
TRAINING_ADMISSION_BYTES = blueprint.TRAINING_ADMISSION_BYTES
SELECTED = probe.PARENT
SELECTED_SHA256 = probe.PARENT_SHA256
TAIL_CONFIG_SHA256 = "f8fcbee841f39f6c7bb33f02c5ce595b3734ded91e51a3ddb07c4c54fa6f5bd0"
TARGETS = {
    "native_parent_v51/latest_checkpoint.pth": "tail",
    "all_imputation_column_mean_preflight_v1/candidate_parameters.pt": "vector",
}
STATES = {
    "native_parent_v51": "training_complete",
    "native_parent_v51_audit_task_replay": "complete",
    "native_parent_v51_endpoint_replay": "complete",
    "native_parent_v51_confirmation": "complete",
    "native_parent_v51_pipeline": "complete",
    "all_imputation_column_mean_preflight_v1": "complete",
}
POLICY = (
    "Storage only: preserve exact serialized bytes of V51's audited unqualified step64 tail "
    "with58 accepted updates and the completed all-imputation column-mean probe vector. "
    "Keep actual V29, every selected/teacher checkpoint, all failed V28 evidence and all "
    "scientific outcomes untouched. Require stopped owners, unlocked producer locks and "
    "no open readers. Copy/fsync/hash/typed-load before canonical symlink publication and "
    "a separate independent recheck. Restore18 GiB plus2432 MiB admission without lowering "
    "18/6/50 GiB reserves. Scratch is machine-local, not external/provider-persistent backup. "
    "No new training, inference, seeds, selection, qualification or scientific success claim."
)


def validate_payload(payload, kind, *, full_parameters=77998496, model_tensors=254, selected_model=None):
    if kind == "vector":
        return blueprint.validate_payload(payload, kind, full_parameters=full_parameters)
    if kind != "tail":
        raise ValueError("Move only the declared two nonselected artifact kinds")
    if (not isinstance(payload, dict) or payload.get("step") != 64 or payload.get("best_step") != 0
            or payload.get("gradient_updates_performed") != 58 or payload.get("attempted_batches") != 64
            or payload.get("config_sha256") != TAIL_CONFIG_SHA256
            or payload.get("parent_checkpoint_sha256") != SELECTED_SHA256
            or payload.get("independent_confirmation_passed") is not False
            or payload.get("preflight_only") is not False or payload.get("full_network_training") is not True
            or not payload.get("optimizer") or not isinstance(payload.get("model"), dict)
            or len(payload["model"]) != model_tensors or selected_model is None
            or set(selected_model) != set(payload["model"])
            or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in payload["model"].values())
            or all(torch.equal(value, selected_model[name]) for name, value in payload["model"].items())):
        raise ValueError("Preserve only V51's distinct audited unqualified tail, never actual V29")


def evidence():
    identities = probe.require_completed_v51()
    probe.verify(probe.OUTPUT)
    paths = [ROOT / "native_parent_v51_audit/audit.json", probe.OUTPUT / "audit.json",
        probe.OUTPUT.with_name(probe.OUTPUT.name + "_mean_audit") / "audit.json"]
    tail, feasibility, mean_audit = [read(path) for path in paths]
    retention, gate = read(probe.OUTPUT / "retention_audit.json"), read(probe.OUTPUT / "gate.json")
    if (tail["complete"] is not True or tail["through_step"] != 64 or tail["best"]["step"] != 0
            or tail["accepted_updates"] != 58 or feasibility["complete"] is not True
            or feasibility["training_tasks"] != 28 or feasibility["prediction_grids"] != 56
            or gate["passed"] is not True or gate["preflight_only"] is not True
            or retention["complete"] is not True or retention["reference_grids"] != 26
            or retention["component_grids"] != 50 or retention["component_role_counts"] != {"v8": 24, "v29": 26}
            or mean_audit["complete"] is not True or mean_audit["tasks"] != 28
            or mean_audit["imputation_reweighted_tasks"] != 26 or mean_audit["structured_reweighted_tasks"] != 24
            or mean_audit["native_reweighted_tasks"] != 2 or mean_audit["unchanged_rating_tasks"] != 2):
        raise ValueError("Require stopped complete V51 and original/supplementary feasibility audits")
    hashes = {path: digest for report in (tail, feasibility, mean_audit) for path, digest in report["artifact_sha256"].items()}
    paths.extend(ROOT / name / "status.json" for name in STATES)
    paths.extend(probe.OUTPUT / name for name in ("config.json", "registration.json", "result.json", "retention_audit.json",
        "gate.json", "external_recheck.json", "OUTCOME.md"))
    paths.extend((SELECTED, ROOT / "native_parent_v51/config.json"))
    identities.update({str(path): sha256(path) for path in paths})
    verify_hashes(identities)
    return hashes, identities


for name in ("read", "status", "require_selected", "require_stopped", "validate_file", "inventory", "copy_and_verify"):
    function = getattr(blueprint, name)
    if function.__closure__ is not None:
        raise ValueError("Preservation callbacks must bind the exact two-artifact scope")
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
RECHECK_SOURCE = replace_once(blueprint.RECHECK_SOURCE, "'canonical_artifacts_checked': 8", "'canonical_artifacts_checked': 2")
exec(compile(RECHECK_SOURCE, __file__, "exec"), globals())
PRESERVE_SOURCE = replace_once(blueprint.PRESERVE_SOURCE,
    "tests/test_single_model_native_parent_capacity.py", "tests/test_single_model_all_imputation_column_mean_capacity.py")
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
