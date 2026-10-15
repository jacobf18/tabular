"""Preserve six audited artifacts before admitting fixed-half full training."""

import math
import os
from pathlib import Path
import shutil
import sys
import types

import torch

import relocate_single_model_shrunk_column_capacity as previous
from relocate_single_model_shrunk_column_capacity import readers, sha256, verify_hashes, write_json, validate_vector, byte_equal
from relocate_single_model_gradient_archive import sync_directory
import preflight_single_model_fixed_half_shrunk_retention as probe
import recover_single_model_shrunk_retention_cpu_audit as recovery
from single_model_registered_adaptation import replace_once


ROOT = previous.ROOT
OUTPUT = ROOT / "fixed_half_capacity_relocation_v1"
DESTINATION = previous.copying.SCRATCH / "single_model_goal_20261005_owned/fixed_half_capacity_v1"
TRAINING_ADMISSION_BYTES = previous.TRAINING_ADMISSION_BYTES
RECOVERY_SHA256 = "a37b726e5d46dad92de64d860b46f0661ac5b8f9ccbcfa13cb4a97902df6ba8e"
OWNERS = {name: dict(record) for name, record in previous.OWNERS.items()}
OWNERS[recovery.ORIGINAL.name] = {"kind": "vector", "parameters": 77998496, "frozen": 16, "cpu_audit_recovered": True}
POLICY = ("Storage-only preservation of exactly six artifacts: the unselected audited V38 terminal tail, "
    "four old completed diagnostic vectors and the original rejected count-shrunk probe's vector, whose "
    "first CPU audit completed in a separately registered recovery. Preserve that original failed status "
    "and negative result exactly; do not relabel them complete or accepted. Bind the exact recovery receipt, "
    "all original inputs and complete negative audit. Require a separately accepted fully audited fresh "
    "fixed-half probe, including independently checked uncontracted/contracted vectors and all reference "
    "grids. Leave the fresh probe, every selection, teacher and published checkpoint untouched. Require "
    "stopped producers and no scientific workers/readers. Copy/fsync/hash/mmap before atomic canonical "
    "symlinks; independently reopen all six and verify original bytes, paths, sources and failed/recovered "
    "statuses. Preserve18/6/50 GiB reserves and restore18 GiB plus2.25 GiB full-training admission. Scratch "
    "is UUID-bound machine-local storage, not provider-persistent backup. No science, seed, metric, "
    "qualification, model or optimizer change. Do not execute the older five-file helper or reinterpret "
    "its original failed prerequisite. Invoke this new one-shot operation once without stage arguments.")


def validate_recovered_failure(current):
    if (current.get("state") != "failed" or current.get("failed_stage") != "audit"
            or current.get("error") != "ValueError('Audit must be CPU only')"
            or type(current.get("pid")) is not int or current["pid"] <= 0):
        raise ValueError("Preserve only the exact original pre-audit environment failure")


def owner_terminal(directory, expected):
    if not expected.get("cpu_audit_recovered"):
        return readers.terminal(directory / "status.json")
    if directory != recovery.ORIGINAL or expected != OWNERS[recovery.ORIGINAL.name]:
        raise ValueError("No generic exception for failed diagnostics")
    current = previous.copying.read(directory / "status.json")
    validate_recovered_failure(current)
    if Path(f'/proc/{current["pid"]}').exists():
        raise ValueError("The original failed producer must be stopped")
    readers.terminal(recovery.OUTPUT / "status.json")
    return sha256(directory / "status.json")


def vector_evidence(directory, expected):
    if not expected.get("cpu_audit_recovered"):
        paths = [directory / name for name in ("audit.json", "result.json", "gate.json")]
        audit, result, gate = [previous.copying.read(path) for path in paths]
        return paths[0], audit, result, gate, paths
    owner_terminal(directory, expected)
    if sha256(recovery.OUTPUT / "recovery_result.json") != RECOVERY_SHA256:
        raise ValueError("Require the exact unchanged additive negative recovery")
    recovered, audit, gate, retention = [previous.copying.read(recovery.OUTPUT / name) for name in
        ("recovery_result.json", "audit.json", "gate.json", "retention_audit.json")]
    result = previous.copying.read(directory / "result.json")
    if (recovered["complete"] is not True or recovered["accepted"] is not False
            or recovered["original_failure_preserved"] is not True or recovered["original_rejection_preserved"] is not True
            or recovered["gate_changed"] is not False or recovered["cuda_initialized"] is not False
            or result["record"]["accepted"] is not False or result["record"]["accepted_updates"] != 0
            or result["record"]["reason"] != "finite_metrics_rejected" or gate["passed"] is not False
            or gate["reason"] != "finite_metrics_rejected" or retention["complete"] is not True
            or retention["cuda_initialized"] is not False or audit["cuda_initialized"] is not False
            or retention["original_audit_sha256"] != sha256(recovery.OUTPUT / "audit.json")
            or any((directory / name).exists() for name in recovery.AUDIT_FILES - {"status.json"})):
        raise ValueError("Never promote the original rejected proposal or spoof its failed CPU entry")
    configuration = previous.copying.read(recovery.OUTPUT / "recovery_config.json")
    verify_hashes(configuration["input_sha256"])
    verify_hashes(configuration["source_sha256"])
    verify_hashes(audit["artifact_sha256"])
    verify_hashes(retention["artifact_sha256"])
    paths = [directory / "result.json", *(Path(name) for name in configuration["input_sha256"]),
        *(Path(name) for name in configuration["source_sha256"])]
    paths.extend(path for path in recovery.OUTPUT.iterdir() if path.is_file())
    return recovery.OUTPUT / "audit.json", audit, result, gate, paths


for name in ("selected_copy", "copy_and_verify", "require_probe"):
    function = getattr(previous, name)
    if function.__closure__ is not None:
        raise ValueError("Do not retain an earlier capacity helper closure")
    globals()[name] = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
regular_checkpoint_identity = types.FunctionType(previous.checkpoint_identity.__code__, globals(),
    "regular_checkpoint_identity", previous.checkpoint_identity.__defaults__)


def checkpoint_identity(payload, current, expected, config_digest):
    if not expected.get("cpu_audit_recovered"):
        return regular_checkpoint_identity(payload, current, expected, config_digest)
    if expected != OWNERS[recovery.ORIGINAL.name] or config_digest != recovery.CONFIG_SHA256:
        raise ValueError("Require exactly the declared original recovered vector")
    validate_recovered_failure(current)
    validate_vector(payload, expected["parameters"])


require_accepted_retention_probe = require_probe


def require_probe():
    evidence = require_accepted_retention_probe()
    result, audit = [probe.read(probe.OUTPUT / name) for name in ("result.json", "audit.json")]
    contraction = audit["raw_endpoint"]["fixed_half_step"]
    if (result["record"]["displacement_fraction"] != .5 or contraction["evaluated"] is not True
            or contraction["fraction"] != .5 or contraction["parameters"] != 77998496
            or contraction["actual_saved_displacements_checked"] is not True):
        raise ValueError("Require the fresh accepted fixed-half proposal and its independent vector audit")
    return evidence


IDLE_SOURCE = replace_once(previous.IDLE_SOURCE, 'readers.terminal(ROOT / name / "status.json")',
    'owner_terminal(ROOT / name, expected)')
IDLE_SOURCE = replace_once(IDLE_SOURCE, 'ROOT / "shrunk_column_anchor_retention_preflight_v1/status.json"',
    'ROOT / "fixed_half_shrunk_retention_preflight_v1/status.json"')
exec(compile(IDLE_SOURCE, __file__, "exec"), globals())
INVENTORY_SOURCE = replace_once(previous.INVENTORY_SOURCE,
    "Move only the five declared regular single-link stopped artifacts",
    "Move only the six declared regular single-link stopped artifacts")
INVENTORY_SOURCE = replace_once(INVENTORY_SOURCE,
    '            audit_path = directory / "audit.json"\n'
    '            audit, result, gate = [previous.copying.read(directory / item) for item in ("audit.json", "result.json", "gate.json")]',
    '            audit_path, audit, result, gate, vector_paths = vector_evidence(directory, expected)')
INVENTORY_SOURCE = replace_once(INVENTORY_SOURCE,
    '            paths.extend(directory / item for item in ("audit.json", "result.json", "gate.json"))',
    '            paths.extend(vector_paths)')
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
    'REPO / "tests/test_single_model_shrunk_column_capacity.py",':
        'REPO / "tests/test_single_model_fixed_half_capacity.py", Path(recovery.__file__).resolve(),',
    'len(records) != 5': 'len(records) != 6',
    'Preserve exactly five completed artifacts': 'Preserve exactly six completely audited artifacts',
    '"canonical_artifacts_checked": 5': '"canonical_artifacts_checked": 6',
    'copying_completed_shrunk_column_capacity': 'copying_completed_fixed_half_capacity',
}.items():
    MAIN_SOURCE = replace_once(MAIN_SOURCE, old, new)
MAIN_SOURCE = replace_once(MAIN_SOURCE, 'def main():\n',
    'def main():\n'
    '    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():\n'
    '        raise ValueError("Preserve artifacts only in an explicitly CPU-only process")\n')
MAIN_SOURCE = replace_once(MAIN_SOURCE, '    config = {"policy": POLICY,',
    '    sources.extend(Path(module.__file__).resolve() for module in tuple(sys.modules.values())\n'
    '        if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")\n'
    '        and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file())\n'
    '    config = {"policy": POLICY,')
status = types.FunctionType(template.status.__code__, globals(), template.status.__name__, template.status.__defaults__)
exec(compile(MAIN_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise ValueError("Invoke this one-shot preservation once without stage arguments")
    main()
