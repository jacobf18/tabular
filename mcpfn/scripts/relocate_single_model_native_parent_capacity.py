"""Preserve eight closed audited artifacts before the native-parent full trial."""

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

import preflight_single_model_native_parent_retention as probe
import relocate_single_model_dual_parent_capacity as blueprint
from relocate_single_model_dual_parent_capacity import readers, copying, sync_directory
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


ROOT = probe.ROOT
OUTPUT = ROOT / "native_parent_capacity_relocation_v1"
DESTINATION = copying.SCRATCH / "single_model_goal_20261005_owned/native_parent_capacity_v1"
TRAINING_ADMISSION_BYTES = blueprint.TRAINING_ADMISSION_BYTES
SELECTED = probe.PARENT
SELECTED_SHA256 = probe.PARENT_SHA256
REPLAYS = (
    "teacher_continuation_v22_cached_recovery_v1", "structured_teacher_v20", "replicated_decoder_v18",
    "point_inclusive_v21", "paired_decoder_v19", "compact_continuation_v23",
)
TARGETS = {
    "dual_parent_v50/latest_checkpoint.pth": "tail",
    "native_parent_retention_preflight_v1/candidate_parameters.pt": "vector",
    **{f"{name}_endpoint_replay/reconstructed_candidate.pt": "decoder_replay" for name in REPLAYS},
}
STATES = {
    "dual_parent_v50": "training_complete", "dual_parent_v50_audit_task_replay": "complete",
    "dual_parent_v50_endpoint_replay": "complete", "dual_parent_v50_confirmation": "complete",
    "dual_parent_v50_pipeline": "complete", "native_parent_retention_preflight_v1": "complete",
    **{f"{name}_endpoint_replay": "complete" for name in REPLAYS},
}
POLICY = (
    "Storage only: preserve exact serialized bytes of the audited nonselected V50 step64 "
    "optimizer tail with56 updates and no selection, the completed native-parent probe vector, "
    "and six completed independently audited decoder endpoint reconstruction records. Preserve "
    "their11227016 coordinates,84 saved products, norms and original selection/replay limitations; "
    "these diagnostic exports are not selected model checkpoints. Keep actual V29, all selected "
    "and teacher checkpoints, every failed V28 artifact and all scientific outcomes untouched. "
    "Require stopped owners, unlocked producer locks and no open readers. Copy/fsync/hash/typed "
    "load before canonical symlink publication and separate independent recheck. Preserve all "
    "source/config/evidence hashes and restore18 GiB plus2432 MiB admission with extra headroom, "
    "without lowering18/6/50 GiB reserves. Scratch is machine-local, not external backup. No new "
    "training, inference, seeds, selection, qualification or scientific success claim."
)


def validate_payload(payload, kind, *, full_parameters=77998496, model_tensors=254,
                     decoder_parameters=11227016, selected_model=None):
    if kind == "vector":
        return blueprint.validate_payload(payload, kind, full_parameters=full_parameters)
    if kind == "tail":
        if (not isinstance(payload, dict) or payload.get("step") != 64 or payload.get("best_step") != 0
                or payload.get("gradient_updates_performed") != 56 or payload.get("attempted_batches") != 64
                or payload.get("config_sha256") != "0d07291d0000fbf0fc6ff1b9b7cfd77ec298139dc29ec10900b91421f7561223"
                or payload.get("parent_checkpoint_sha256") != SELECTED_SHA256
                or payload.get("independent_confirmation_passed") is not False
                or payload.get("preflight_only") is not False or payload.get("full_network_training") is not True
                or not payload.get("optimizer") or not isinstance(payload.get("model"), dict)
                or len(payload["model"]) != model_tensors or selected_model is None
                or set(selected_model) != set(payload["model"])
                or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in payload["model"].values())
                or all(torch.equal(value, selected_model[name]) for name, value in payload["model"].items())):
            raise ValueError("Preserve only V50's distinct audited terminal state, not the actual V29 initializer")
        return
    if kind != "decoder_replay" or not isinstance(payload, dict) or set(payload) != {"parameters", "products", "norm"}:
        raise ValueError("Move only declared complete decoder endpoint reconstruction records")
    parameters = payload["parameters"]
    if (not isinstance(parameters, torch.Tensor) or parameters.shape != (decoder_parameters,)
            or parameters.dtype != torch.float32 or not torch.isfinite(parameters).all()
            or not isinstance(payload["products"], list) or len(payload["products"]) != 84
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in payload["products"])
            or type(payload["norm"]) not in (int, float) or not math.isfinite(payload["norm"]) or payload["norm"] < 0):
        raise ValueError("Preserve all finite decoder coordinates,84 products and the original norm")


def evidence():
    identities = probe.require_completed_v50()
    probe.verify(probe.OUTPUT)
    paths = [ROOT / "dual_parent_v50_audit/audit.json", probe.OUTPUT / "audit.json"]
    paths.extend(ROOT / f"{name}_endpoint_replay/audit.json" for name in REPLAYS)
    reports = [read(path) for path in paths]
    tail, feasibility = reports[:2]
    retention, gate = read(probe.OUTPUT / "retention_audit.json"), read(probe.OUTPUT / "gate.json")
    if (tail["complete"] is not True or tail["through_step"] != 64 or tail["best"]["step"] != 0
            or tail["accepted_updates"] != 56 or feasibility["complete"] is not True
            or feasibility["training_tasks"] != 28 or feasibility["prediction_grids"] != 56
            or gate["passed"] is not True or gate["preflight_only"] is not True
            or retention["complete"] is not True or retention["reference_grids"] != 26
            or retention["component_grids"] != 50 or retention["component_role_counts"] != {"v8": 24, "v29": 26}):
        raise ValueError("Require original complete V50 and native-parent probe audits")
    for report in reports[2:]:
        if (report["complete"] is not True or report["endpoint_grids"] != 56 or report["selected_grids"] not in (0, 152)
                or report["cuda_initialized"] is not False or report["autograd_replayed"] is not False
                or report["research_selection_tolerance"] != 0 or report["goal_success_not_assessed"] is not True):
            raise ValueError("Preserve each original decoder endpoint audit and its limited scope")
    hashes = {path: digest for report in reports for path, digest in report["artifact_sha256"].items()}
    paths.extend(ROOT / name / "status.json" for name in STATES)
    paths.extend(probe.OUTPUT / name for name in ("config.json", "registration.json", "result.json", "retention_audit.json",
        "gate.json", "external_recheck.json", "OUTCOME.md"))
    paths.extend(ROOT / f"{name}_endpoint_replay" / filename for name in REPLAYS for filename in ("config.json", "result.json", "gate.json"))
    paths.extend((SELECTED, ROOT / "dual_parent_v50/config.json"))
    identities.update({str(path): sha256(path) for path in paths})
    verify_hashes(identities)
    return hashes, identities


for name in ("read", "status", "require_selected", "require_stopped", "validate_file", "inventory", "copy_and_verify"):
    function = getattr(blueprint, name)
    if function.__closure__ is not None:
        raise ValueError("Preservation callbacks must bind only the declared eight-artifact scope")
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
RECHECK_SOURCE = replace_once(blueprint.RECHECK_SOURCE, "'canonical_artifacts_checked': 4", "'canonical_artifacts_checked': 8")
exec(compile(RECHECK_SOURCE, __file__, "exec"), globals())
PRESERVE_SOURCE = replace_once(blueprint.PRESERVE_SOURCE,
    "tests/test_single_model_dual_parent_capacity.py", "tests/test_single_model_native_parent_capacity.py")
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
