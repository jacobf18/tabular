"""Preserve three closed diagnostics before the separately registered legacy-blend trial."""

import inspect
import math
import os
from pathlib import Path
import shutil
import sys
import types

import torch

import relocate_single_model_all_imputation_capacity as previous
import relocate_single_model_column_anchor_capacity as original
from relocate_single_model_all_imputation_capacity import readers, sha256, verify_hashes, write_json, validate_vector, byte_equal
from relocate_single_model_gradient_archive import sync_directory
import preflight_single_model_legacy_blend_teacher as probe
import single_model_fixed_half_trial as fixed_trial
from single_model_registered_adaptation import replace_once


ROOT, REPO = previous.ROOT, previous.REPO
OUTPUT = ROOT / "legacy_blend_capacity_relocation_v1"
DESTINATION = previous.copying.SCRATCH / "single_model_goal_20261005_owned/legacy_blend_capacity_v1"
TRAINING_ADMISSION_BYTES = previous.TRAINING_ADMISSION_BYTES
PROBE_CONFIG_SHA256 = "325063059dd416d7eb3ffb7ef33226ad66335e0097c2cd92700d105d58ee7cad"
OWNERS = {
    "all_imputation_retention_v40": {"kind": "unselected_tail", "run": "all_imputation_retention_v40",
        "step": 64, "updates": 58, "selected": 0, "grids": 1368, "view_flag": "full_network_training"},
    "all_imputation_retention_preflight_v1": {"kind": "vector", "parameters": 77998496, "frozen": 16,
        "tasks": 28, "grids": 56},
    "decoder_training_preflight_v1": {"kind": "vector", "parameters": 11227016, "frozen": 248,
        "tasks": 14, "grids": 28},
}
POLICY = ("Preserve exactly V40's fully audited unselected terminal optimizer tail, the older all-imputation "
    "probe vector and the original14-task decoder feasibility vector. Preserve every original byte, outcome, "
    "canonical path and registered source. Never relocate a selected model, teacher or current legacy-blend "
    "probe. Require the exact new accepted probe, complete fixed-half CPU proof,28 independently replayed "
    "teacher targets, four raw legacy component grids with independently refitted observed weights, and26 "
    "retention references. Require stopped producers and no scientific workers/open readers. Copy/fsync/hash/"
    "mmap before atomic canonical symlinks and independently reopen all three. Preserve18/6/50 GiB reserves "
    "and restore18 GiB plus2432 MiB minimum training admission. A future trial must separately account for "
    "its full component replay. UUID-bound scratch is machine-local, not provider-persistent backup. This "
    "storage-only registration authorizes no training, new seed, selection, confirmation or benchmark run.")


for name in ("selected_copy", "checkpoint_identity", "copy_and_verify"):
    function = getattr(original, name)
    if function.__closure__ is not None:
        raise ValueError("Preservation helpers must not retain older experiment closures")
    globals()[name] = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)


def filename_for(expected):
    return {"vector": "candidate_parameters.pt", "unselected_tail": "latest_checkpoint.pth"}[expected["kind"]]


function = previous.require_idle
require_idle = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
PROBE_SOURCE = inspect.getsource(original.require_probe).replace("def require_probe():", "def original_probe_evidence():")
exec(compile(PROBE_SOURCE, __file__, "exec"), globals())


def validate_probe(result, audit, retention, gate, current):
    fixed_trial.check_preflight_evidence(result, audit)
    replay = audit["teacher_replay"]
    if (retention["reference_grids"] != 26 or current.get("state") != "complete"
            or current.get("retention_audited") is not True or audit["cuda_initialized"] is not False
            or replay["complete"] is not True or replay["full_grids"] != 28 or replay["bitwise_equal"] is not True
            or replay["blend_tasks"] != 2 or replay["component_grids"] != 4
            or replay["observed_weights_refitted"] is not True or gate["passed"] is not True
            or gate["preflight_only"] is not True or gate["real_benchmark_success"] is not False
            or gate["audit_sha256"] != sha256(probe.OUTPUT / "audit.json")):
        raise ValueError("Require the complete accepted CPU/component/reference audit, not local acceptance alone")


def require_probe():
    if sha256(probe.OUTPUT / "config.json") != PROBE_CONFIG_SHA256:
        raise ValueError("Preserve only after the exact registered legacy-blend feasibility experiment")
    evidence = original_probe_evidence()
    probe.require_exact_predecessor()
    records = [probe.read(probe.OUTPUT / name) for name in
        ("result.json", "audit.json", "retention_audit.json", "gate.json", "status.json")]
    validate_probe(*records)
    return {**evidence, str(probe.OUTPUT / "gate.json"): sha256(probe.OUTPUT / "gate.json")}


INVENTORY_SOURCE = replace_once(original.INVENTORY_SOURCE,
    'audit["training_tasks"] != 28 or audit["prediction_grids"] != 56',
    'audit["training_tasks"] != expected["tasks"] or audit["prediction_grids"] != expected["grids"]')
exec(compile(INVENTORY_SOURCE, __file__, "exec"), globals())
inventory_original = inventory


def inventory():
    evidence = require_probe()
    records, inputs = inventory_original()
    for name in OWNERS:
        config = previous.copying.read(ROOT / name / "config.json")
        for field in ("source_sha256", "input_sha256"):
            verify_hashes(config[field])
            inputs.update(config[field])
    return records, {**inputs, **evidence}


copying = types.SimpleNamespace(**{**vars(previous.copying), "copy_and_verify": copy_and_verify})
template = previous.template
MAIN_SOURCE = replace_once(previous.MAIN_SOURCE,
    'REPO / "tests/test_single_model_all_imputation_capacity.py",',
    'REPO / "tests/test_single_model_legacy_blend_capacity.py",')
MAIN_SOURCE = replace_once(MAIN_SOURCE, "copying_completed_all_imputation_capacity", "copying_completed_legacy_blend_capacity")
status = types.FunctionType(template.status.__code__, globals(), template.status.__name__, template.status.__defaults__)
exec(compile(MAIN_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise ValueError("Invoke this tested one-shot preservation once without stage arguments")
    main()
