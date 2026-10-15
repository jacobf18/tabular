"""Restore training capacity without moving selected models or the new retention probe."""

import inspect
import os
from pathlib import Path
import shutil
import types

import torch

import relocate_single_model_full_network_capacity as previous
from relocate_single_model_full_network_capacity import readers, sha256, verify_hashes, write_json, validate_vector, byte_equal
from relocate_single_model_gradient_archive import sync_directory
import preflight_single_model_anchor_retention as probe
from single_model_registered_adaptation import replace_once


ROOT = previous.ROOT
OUTPUT = ROOT / "anchor_retention_capacity_relocation_v1"
DESTINATION = previous.copying.SCRATCH / "single_model_goal_20261005_owned/anchor_retention_capacity_v1"
TRAINING_ADMISSION_BYTES = previous.TRAINING_ADMISSION_BYTES
OWNERS = {
    "full_network_cap90_v36": {"kind": "unselected_tail", "run": "full_network_cap90_v36", "step": 64,
        "updates": 61, "selected": 0, "grids": 1368, "view_flag": "full_network_training"},
    "full_network_preflight_v1": {"kind": "vector", "parameters": 77998496, "frozen": 16},
    "point_only_preflight_v1": {"kind": "vector", "parameters": 27766670, "frozen": 192},
}
POLICY = ("Storage-only preservation of exactly V36's fully audited unselected terminal optimizer tail and "
    "two older completed preflight candidate vectors. Preserve the new anchor-retention probe, every selected "
    "model, teacher, published checkpoint, failed V28 tail and registered source. Require stopped producers, "
    "no scientific workers or open readers, original audit hashes and complete independent new-probe evidence. "
    "Copy/fsync/hash/mmap before atomic canonical symlinks, then independently reopen and verify all three "
    "artifacts and immutable inputs. Restore18 GiB plus2.25 GiB training capacity while preserving18/6/50 GiB "
    "reserves and every original serialized byte. No model, optimizer, metric, selection, qualification, seed "
    "or canonical-path change. UUID-bound machine-local scratch is not provider-persistent storage or backup.")


for name in ("selected_copy", "checkpoint_identity", "copy_and_verify"):
    function = getattr(previous, name)
    if function.__closure__ is not None:
        raise ValueError("Never retain an older relocation closure")
    globals()[name] = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)

source = inspect.getsource(previous.require_idle)
source = replace_once(source, 'ROOT / "full_network_preflight_v1/status.json"',
                      'ROOT / "anchor_retention_preflight_v1/status.json"')
exec(compile(source, __file__, "exec"), globals())

INVENTORY_SOURCE = replace_once(inspect.getsource(previous.inventory),
    "Move only the four declared regular single-link stopped artifacts",
    "Move only the three declared regular single-link stopped artifacts")
exec(compile(INVENTORY_SOURCE, __file__, "exec"), globals())
inventory_original = inventory


def require_probe():
    probe.verify(probe.OUTPUT)
    readers.terminal(probe.OUTPUT / "status.json")
    result, original, retention = [probe.read(probe.OUTPUT / name) for name in
                                     ("result.json", "audit.json", "retention_audit.json")]
    if (result["data_admission_only"] is not False or result["record"]["accepted"] is not True
            or result["trainable_parameters"] != 77998496 or original["complete"] is not True
            or original["training_tasks"] != 28 or original["prediction_grids"] != 56
            or original["frozen"]["frozen_tensors"] != 16 or retention["complete"] is not True
            or retention["eligibility_decisions"] != 28 or retention["reference_grids"] != 16
            or retention["no_inference_ensemble"] is not True
            or retention["original_audit_sha256"] != sha256(probe.OUTPUT / "audit.json")):
        raise ValueError("Require the complete accepted new retention probe and both independent audits")
    verify_hashes(original["artifact_sha256"])
    verify_hashes(retention["artifact_sha256"])
    return {str(probe.OUTPUT / name): sha256(probe.OUTPUT / name) for name in
            ("config.json", "registration.json", "status.json", "result.json", "audit.json", "retention_audit.json")}


def inventory():
    evidence = require_probe()
    records, inputs = inventory_original()
    return records, {**inputs, **evidence}


copying = types.SimpleNamespace(**{**vars(previous.copying), "copy_and_verify": copy_and_verify})
template, REPO = previous.template, previous.REPO
MAIN_SOURCE = previous.MAIN_SOURCE
for old, new in {
    'REPO / "tests/test_single_model_full_network_capacity.py",':
        'REPO / "tests/test_single_model_anchor_retention_capacity.py", Path(probe.__file__).resolve(), '
        'Path(probe.retention.__file__).resolve(),',
    'len(records) != 4': 'len(records) != 3',
    'Preserve exactly four completed artifacts': 'Preserve exactly three completed artifacts',
    '"canonical_artifacts_checked": 4': '"canonical_artifacts_checked": 3',
    'copying_completed_full_network_capacity': 'copying_completed_anchor_retention_capacity',
}.items():
    MAIN_SOURCE = replace_once(MAIN_SOURCE, old, new)
status = types.FunctionType(template.status.__code__, globals(), template.status.__name__, template.status.__defaults__)
exec(compile(MAIN_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
