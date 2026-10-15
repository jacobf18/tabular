"""Preserve three closed diagnostic artifacts before all-imputation training."""

import inspect
import math
import os
from pathlib import Path
import shutil
import sys
import types

import torch

import relocate_single_model_fixed_half_capacity as previous
import relocate_single_model_column_anchor_capacity as original
from relocate_single_model_fixed_half_capacity import readers, sha256, verify_hashes, write_json, validate_vector, byte_equal
from relocate_single_model_gradient_archive import sync_directory
import preflight_single_model_all_imputation_retention as probe
from single_model_registered_adaptation import replace_once


ROOT, REPO = previous.ROOT, previous.REPO
OUTPUT = ROOT / "all_imputation_capacity_relocation_v1"
DESTINATION = previous.copying.SCRATCH / "single_model_goal_20261005_owned/all_imputation_capacity_v1"
TRAINING_ADMISSION_BYTES = previous.TRAINING_ADMISSION_BYTES + 128 * 1024**2
OWNERS = {
    "fixed_half_shrunk_retention_v39": {"kind": "unselected_tail", "run": "fixed_half_shrunk_retention_v39",
        "step": 64, "updates": 59, "selected": 0, "grids": 1368, "view_flag": "full_network_training"},
    "fixed_half_shrunk_retention_preflight_v1": {"kind": "vector", "parameters": 77998496, "frozen": 16},
    "complete_missingness_training_preflight_v1": {"kind": "smoke", "step": 21, "tensors": 254, "elements": 77998688},
}
SMOKE_CONFIG = "6f63b13b27546ea96f4b4bc87bd040e5e21314cca369b4cc79ff8355b21677f8"
POLICY = ("Preserve exactly three closed, independently audited, nonselected artifacts: V39's unselected "
    "terminal optimizer tail, the older fixed-half feasibility vector and the original21-update noncandidate "
    "smoke checkpoint. Keep every selected model, teacher, current all-imputation probe and source untouched. "
    "Require the new accepted probe, all26 independently replayed imputation references and exact fixed-half "
    "vector proof. Require stopped owners and no scientific workers/open readers. Copy/fsync/hash/mmap before "
    "atomic canonical symlinks, then independently reopen all three and verify original hashes and outcomes. "
    "Keep18/6/50 GiB reserves; restore18 GiB plus2432 MiB training admission, including128 MiB additional "
    "checkpoint metadata allowance. No science or outcome changes. UUID-bound scratch is machine-local, "
    "not provider-persistent backup. Execute this tested one-shot preservation once without stage arguments.")


for name in ("selected_copy", "copy_and_verify"):
    function = getattr(original, name)
    globals()[name] = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
regular_checkpoint_identity = types.FunctionType(original.checkpoint_identity.__code__, globals(),
    "regular_checkpoint_identity", original.checkpoint_identity.__defaults__)


def checkpoint_identity(payload, current, expected, config_digest):
    if expected["kind"] != "smoke":
        return regular_checkpoint_identity(payload, current, expected, config_digest)
    if (config_digest != SMOKE_CONFIG or current.get("state") != "complete" or current.get("updates") != 21
            or set(payload) != {"model", "step", "not_a_candidate", "config_sha256"}
            or payload["step"] != 21 or payload["not_a_candidate"] is not True
            or payload["config_sha256"] != config_digest or not isinstance(payload["model"], dict)
            or len(payload["model"]) != expected["tensors"]):
        raise ValueError("Preserve the exact21-update noncandidate smoke checkpoint, never a selected model")
    values = list(payload["model"].values())
    if (any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or not torch.isfinite(value).all() for value in values)
            or sum(value.numel() for value in values) != expected["elements"]):
        raise ValueError("Preserve every finite original FP32 smoke tensor")


def smoke_evidence(directory):
    audit = previous.copying.read(directory / "training_audit.json")
    summary = previous.copying.read(directory / "summary.json")
    if (audit["complete"] is not True or audit["updates"] != 21 or audit["replayed_column_objectives"] != 20
            or audit["finite_original_fp32_inventory"] is not True or audit["tensors"] != 254
            or audit["elements"] != 77998688 or audit["changed_tensors"] != 190
            or audit["original_artifacts_unchanged"] is not True or audit["config_sha256"] != SMOKE_CONFIG
            or audit["gradient_computation_repeated"] is not False or audit["optimizer_trajectory_repeated"] is not False
            or audit["accuracy_assessed"] is not False or audit["not_a_candidate"] is not True
            or summary["complete"] is not True or summary["updates"] != 21
            or summary["not_a_candidate"] is not True or summary["checkpoint_selection_performed"] is not False
            or summary["accuracy_improvement_not_assessed"] is not True
            or summary["config_sha256"] != SMOKE_CONFIG
            or summary["checkpoint_sha256"] != sha256(directory / "smoke_test_checkpoint.pth")):
        raise ValueError("Require the complete original smoke audit without claiming accuracy or selection")
    verify_hashes(audit["original_artifact_sha256"])
    verify_hashes(summary["episode_sha256"])
    verify_hashes({str(directory / "audit_training.py"):
        "b771bb2f5e126f78cea5bc802cb2f6be1b5fda3362158e479711e6a8df988a7f"})
    return {**audit, "artifact_sha256": audit["original_artifact_sha256"]}, [
        directory / "training_audit.json", directory / "audit_training.py", directory / "summary.json",
        *(Path(name) for name in audit["original_artifact_sha256"]),
        *(Path(name) for name in summary["episode_sha256"])]


PROBE_SOURCE = inspect.getsource(original.require_probe).replace("def require_probe():", "def original_probe_evidence():")
exec(compile(PROBE_SOURCE, __file__, "exec"), globals())


def require_probe():
    evidence = original_probe_evidence()
    probe.require_exact_predecessor()
    result, audit, retention, gate, current = [probe.read(probe.OUTPUT / name) for name in
        ("result.json", "audit.json", "retention_audit.json", "gate.json", "status.json")]
    probe.completed.fixed_trial.check_preflight_evidence(result, audit)
    if (retention["reference_grids"] != 26 or current.get("retention_audited") is not True
            or gate["passed"] is not True or gate["preflight_only"] is not True
            or gate["real_benchmark_success"] is not False or gate["audit_sha256"] != sha256(probe.OUTPUT / "audit.json")):
        raise ValueError("Bind exactly26 audited references and the positive feasibility-only gate")
    return {**evidence, str(probe.OUTPUT / "gate.json"): sha256(probe.OUTPUT / "gate.json")}


def filename_for(expected):
    return {"smoke": "smoke_test_checkpoint.pth", "vector": "candidate_parameters.pt",
        "unselected_tail": "latest_checkpoint.pth"}[expected["kind"]]


def require_idle():
    protected = []
    for name, expected in OWNERS.items():
        readers.terminal(ROOT / name / "status.json")
        protected.append(str(ROOT / name / filename_for(expected)))
        if expected["kind"] == "unselected_tail":
            for suffix in ("_pipeline", "_audit_task_replay", "_endpoint_replay", "_confirmation"):
                readers.terminal(ROOT / f"{name}{suffix}/status.json")
    readers.terminal(probe.OUTPUT / "status.json")
    readers.require_idle(protected)


INVENTORY_SOURCE = original.INVENTORY_SOURCE
INVENTORY_SOURCE = replace_once(INVENTORY_SOURCE,
    'filename = "candidate_parameters.pt" if expected["kind"] == "vector" else "latest_checkpoint.pth"',
    'filename = filename_for(expected)')
INVENTORY_SOURCE = replace_once(INVENTORY_SOURCE,
    'paths = [directory / item for item in ("config.json", "registration.json", "status.json")]\n'
    '        if expected["kind"] == "vector":',
    'paths = [directory / item for item in (("config.json", "status.json") if expected["kind"] == "smoke"\n'
    '            else ("config.json", "registration.json", "status.json"))]\n'
    '        if expected["kind"] == "smoke":\n'
    '            audit, smoke_paths = smoke_evidence(directory)\n'
    '            paths.extend(smoke_paths)\n'
    '        elif expected["kind"] == "vector":')
exec(compile(INVENTORY_SOURCE, __file__, "exec"), globals())
original_inventory = inventory


def inventory():
    evidence = require_probe()
    records, inputs = original_inventory()
    for name in OWNERS:
        config = previous.copying.read(ROOT / name / "config.json")
        for field in ("source_sha256", "input_sha256"):
            verify_hashes(config[field])
            inputs.update(config[field])
    return records, {**inputs, **evidence}


copying = types.SimpleNamespace(**{**vars(previous.copying), "copy_and_verify": copy_and_verify})
template = previous.template
MAIN_SOURCE = previous.MAIN_SOURCE
for old, new in {
    "len(records) != 6": "len(records) != 3",
    "Preserve exactly six completely audited artifacts": "Preserve exactly three completely audited artifacts",
    '"canonical_artifacts_checked": 6': '"canonical_artifacts_checked": 3',
    "copying_completed_fixed_half_capacity": "copying_completed_all_imputation_capacity",
    'REPO / "tests/test_single_model_fixed_half_capacity.py", Path(recovery.__file__).resolve(),':
        'REPO / "tests/test_single_model_all_imputation_capacity.py",',
}.items():
    MAIN_SOURCE = replace_once(MAIN_SOURCE, old, new)
status = types.FunctionType(template.status.__code__, globals(), template.status.__name__, template.status.__defaults__)
exec(compile(MAIN_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise ValueError("Invoke this one-shot preservation once without stage arguments")
    main()
