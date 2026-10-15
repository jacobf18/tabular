"""Preserve five completed unselected artifacts before count-shrunk full training."""

import math
import os
from pathlib import Path
import shutil
import sys
import types

import torch

import relocate_single_model_column_anchor_capacity as previous
from relocate_single_model_column_anchor_capacity import readers, sha256, verify_hashes, write_json, validate_vector, byte_equal
from relocate_single_model_gradient_archive import sync_directory
import preflight_single_model_shrunk_column_anchor_retention as probe
from single_model_registered_adaptation import replace_once


ROOT = previous.ROOT
OUTPUT = ROOT / "shrunk_column_capacity_relocation_v1"
DESTINATION = previous.copying.SCRATCH / "single_model_goal_20261005_owned/shrunk_column_capacity_v1"
TRAINING_ADMISSION_BYTES = previous.TRAINING_ADMISSION_BYTES
OWNERS = {
    "column_anchor_retention_v38": {"kind": "unselected_tail", "run": "column_anchor_retention_v38", "step": 64,
        "updates": 59, "selected": 0, "grids": 1368, "view_flag": "full_network_training"},
    "column_anchor_retention_preflight_v1": {"kind": "vector", "parameters": 77998496, "frozen": 16},
    "replicated_decoder_preflight_v1": {"kind": "vector", "parameters": 11227016, "frozen": 248},
    "structured_teacher_preflight_v1": {"kind": "vector", "parameters": 11227016, "frozen": 248},
    "point_inclusive_preflight_v1": {"kind": "vector", "parameters": 11227016, "frozen": 248},
}
POLICY = ("Storage-only preservation of exactly V38's independently audited unselected terminal optimizer "
    "tail, its older completed unshrunk column-retention probe vector, and three completed replicated-"
    "decoder/structured-teacher/point-inclusive decoder probe vectors. Preserve the new count-shrunk probe, "
    "every selected model, teacher, published checkpoint, other endpoint and registered source. Require "
    "stopped producers, no scientific workers/open readers, original complete audits and an accepted fully "
    "audited new count-shrunk probe. Copy/fsync/hash/mmap before atomic canonical symlinks, independently "
    "reopen all five artifacts and recheck every immutable input. Restore18 GiB plus2.25 GiB full-training "
    "admission capacity while preserving18/6/50 GiB reserves and all serialized bytes, parameter values, "
    "metrics, seeds, selections and canonical paths. UUID-bound machine-local scratch is not provider-"
    "persistent storage or backup. Invoke this one-shot operation once without stage arguments.")


for name in ("selected_copy", "checkpoint_identity", "copy_and_verify", "require_probe"):
    function = getattr(previous, name)
    if function.__closure__ is not None:
        raise ValueError("Never retain an older capacity operation closure")
    globals()[name] = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)

IDLE_SOURCE = replace_once(previous.IDLE_SOURCE, 'ROOT / "column_anchor_retention_preflight_v1/status.json"',
    'ROOT / "shrunk_column_anchor_retention_preflight_v1/status.json"')
exec(compile(IDLE_SOURCE, __file__, "exec"), globals())
INVENTORY_SOURCE = replace_once(previous.INVENTORY_SOURCE,
    "Move only the three declared regular single-link stopped artifacts",
    "Move only the five declared regular single-link stopped artifacts")
exec(compile(INVENTORY_SOURCE, __file__, "exec"), globals())
inventory_original = inventory


def inventory():
    evidence = require_probe()
    records, inputs = inventory_original()
    return records, {**inputs, **evidence}


copying = types.SimpleNamespace(**{**vars(previous.copying), "copy_and_verify": copy_and_verify})
template, REPO = previous.template, previous.REPO
MAIN_SOURCE = previous.MAIN_SOURCE
for old, new in {
    'REPO / "tests/test_single_model_column_anchor_capacity.py",':
        'REPO / "tests/test_single_model_shrunk_column_capacity.py",',
    'len(records) != 3': 'len(records) != 5',
    'Preserve exactly three completed artifacts': 'Preserve exactly five completed artifacts',
    '"canonical_artifacts_checked": 3': '"canonical_artifacts_checked": 5',
    'copying_completed_column_anchor_capacity': 'copying_completed_shrunk_column_capacity',
}.items():
    MAIN_SOURCE = replace_once(MAIN_SOURCE, old, new)
status = types.FunctionType(template.status.__code__, globals(), template.status.__name__, template.status.__defaults__)
exec(compile(MAIN_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise ValueError("This one-shot relocation accepts no stage arguments; invoke it once without arguments")
    main()
