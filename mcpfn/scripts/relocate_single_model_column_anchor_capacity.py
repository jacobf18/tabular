"""Preserve three completed unselected artifacts on scratch before the next full trial."""

import math
import os
from pathlib import Path
import shutil
import types

import torch

import relocate_single_model_anchor_retention_capacity as previous
from relocate_single_model_anchor_retention_capacity import readers, sha256, verify_hashes, write_json, validate_vector, byte_equal
from relocate_single_model_gradient_archive import sync_directory
import preflight_single_model_column_anchor_retention as probe
from single_model_registered_adaptation import replace_once


ROOT = previous.ROOT
OUTPUT = ROOT / "column_anchor_capacity_relocation_v1"
DESTINATION = previous.copying.SCRATCH / "single_model_goal_20261005_owned/column_anchor_capacity_v1"
TRAINING_ADMISSION_BYTES = previous.TRAINING_ADMISSION_BYTES
OWNERS = {
    "anchor_retention_v37": {"kind": "unselected_tail", "run": "anchor_retention_v37", "step": 64,
        "updates": 60, "selected": 0, "grids": 1368, "view_flag": "full_network_training"},
    "anchor_retention_preflight_v1": {"kind": "vector", "parameters": 77998496, "frozen": 16},
    "paired_decoder_preflight_v1": {"kind": "vector", "parameters": 11227016, "frozen": 248},
}
POLICY = ("Storage-only preservation of exactly V37's independently audited unselected terminal optimizer "
    "tail and two older completed raw-query-mean anchor-retention/paired-decoder probe vectors. Preserve the new equal-column "
    "probe, every selected model, teacher, published checkpoint, other endpoint and registered source. Require "
    "stopped producers, no scientific workers/open readers, original audit hashes and complete independently "
    "audited new-probe evidence. Copy/fsync/hash/mmap before atomic canonical symlinks, independently reopen "
    "all three artifacts and recheck every immutable input. Restore18 GiB plus2.25 GiB full-training capacity, "
    "preserving18/6/50 GiB reserves and every original serialized byte, parameter, metric, seed, selection and "
    "canonical path. UUID-bound machine-local scratch is not provider-persistent storage or backup.")


for name in ("selected_copy", "checkpoint_identity", "copy_and_verify"):
    function = getattr(previous, name)
    if function.__closure__ is not None:
        raise ValueError("Never retain a preceding relocation closure")
    globals()[name] = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)

IDLE_SOURCE = replace_once(previous.source, 'ROOT / "anchor_retention_preflight_v1/status.json"',
    'ROOT / "column_anchor_retention_preflight_v1/status.json"')
exec(compile(IDLE_SOURCE, __file__, "exec"), globals())
INVENTORY_SOURCE = previous.INVENTORY_SOURCE
exec(compile(INVENTORY_SOURCE, __file__, "exec"), globals())
inventory_original = inventory


def require_probe():
    probe.verify(probe.OUTPUT)
    readers.terminal(probe.OUTPUT / "status.json")
    result, original, retention, manifest = [probe.read(probe.OUTPUT / name) for name in
        ("result.json", "audit.json", "retention_audit.json", "retention_targets/manifest.json")]
    expected_grids = sum(record["eligible"] for record in manifest["records"])
    penalties = retention["endpoint_penalties"]
    if (result["data_admission_only"] is not False or result["record"]["accepted"] is not True
            or result["trainable_parameters"] != 77998496 or original["complete"] is not True
            or original["training_tasks"] != 28 or original["prediction_grids"] != 56
            or original["frozen"]["frozen_tensors"] != 16 or retention["complete"] is not True
            or retention["eligibility_decisions"] != 28 or len(manifest["records"]) != 28
            or manifest["complete"] is not True or manifest["specification"] != probe.retention.SPECIFICATION
            or expected_grids <= 0 or retention["reference_grids"] != expected_grids
            or len(penalties) != expected_grids or any(not math.isfinite(row[role]) or row[role] < 0
                for row in penalties for role in ("before", "candidate"))
            or retention["no_inference_ensemble"] is not True or retention["cuda_initialized"] is not False
            or retention["original_audit_sha256"] != sha256(probe.OUTPUT / "audit.json")):
        raise ValueError("Require the accepted column-retention probe and complete original and added-reference audits")
    verify_hashes(original["artifact_sha256"])
    verify_hashes(retention["artifact_sha256"])
    return {str(probe.OUTPUT / name): sha256(probe.OUTPUT / name) for name in
        ("config.json", "registration.json", "status.json", "result.json", "audit.json", "retention_audit.json",
         "retention_targets/manifest.json")}


def inventory():
    evidence = require_probe()
    records, inputs = inventory_original()
    return records, {**inputs, **evidence}


copying = types.SimpleNamespace(**{**vars(previous.copying), "copy_and_verify": copy_and_verify})
template, REPO = previous.template, previous.REPO
MAIN_SOURCE = previous.MAIN_SOURCE
for old, new in {
    'REPO / "tests/test_single_model_anchor_retention_capacity.py",':
        'REPO / "tests/test_single_model_column_anchor_capacity.py",',
    'copying_completed_anchor_retention_capacity': 'copying_completed_column_anchor_capacity',
}.items():
    MAIN_SOURCE = replace_once(MAIN_SOURCE, old, new)
status = types.FunctionType(template.status.__code__, globals(), template.status.__name__, template.status.__defaults__)
exec(compile(MAIN_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
