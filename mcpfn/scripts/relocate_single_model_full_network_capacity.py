"""Preserve stopped optimizer tails and an old probe vector while retaining durable selected models."""

import os
from pathlib import Path
import shutil
import types

import torch

import relocate_single_model_top_three_capacity as previous
from relocate_single_model_top_three_capacity import readers, sha256, verify_hashes, write_json, validate_vector
from relocate_single_model_gradient_archive import sync_directory
from single_model_composite_gradient_storage import byte_equal
from single_model_registered_adaptation import replace_once


ROOT = previous.ROOT
OUTPUT = ROOT / "full_network_capacity_relocation_v1"
DESTINATION = previous.copying.SCRATCH / "single_model_goal_20261005_owned/full_network_capacity_v1"
TRAINING_ADMISSION_BYTES = 18 * 1024**3 + 2304 * 1024**2
OWNERS = {
    "top_three_blocks_v33": {"kind": "unselected_tail", "run": "top_three_blocks_v33", "step": 64,
        "updates": 64, "selected": 0, "grids": 1368, "view_flag": "top_three_block_training"},
    "top_two_blocks_v26": {"kind": "duplicate_selected_tail", "run": "top_two_blocks_v26", "step": 64,
        "updates": 64, "selected": 64, "grids": 1216, "view_flag": "top_two_block_training", "recovered": True},
    "top_block_v24": {"kind": "duplicate_selected_tail", "run": "top_block_v24", "step": 64,
        "updates": 61, "selected": 64, "grids": 1216, "view_flag": "top_block_training"},
    "top_three_blocks_preflight_v1": {"kind": "vector", "parameters": 36036497, "frozen": 164},
}
POLICY = ("Storage-only preservation of V33's unselected terminal optimizer state, V24/V26's separate "
    "optimizer tails and the old three-block diagnostic vector. V24/V26 terminal weights equal their selected "
    "step64: retain both distinct selected model-only checkpoint files physically on checkpoint storage and "
    "prove their exact model tensors equal each copied tail. Bind V26's original failed coordinator and its "
    "completed nonrepeating identity recovery, never pretend that original coordinator succeeded. Preserve "
    "every selected model, original teacher, recent full-network probe artifact and registered source. Require "
    "stopped owners, absent readers, original complete audit hashes, copy/fsync/hash/mmap verification before "
    "atomic canonical links and a full independent recheck. Preserve18/6/50 GiB reserves and restore18 GiB "
    "plus2.25 GiB full-training capacity without scientific changes or discarded bytes. Destination is UUID-"
    "bound machine-local scratch, not provider-persistent storage or external backup.")


def selected_copy(payload, expected, config_digest):
    path = ROOT / expected["run"] / "selected_checkpoint.pth"
    if (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
            or path.stat().st_dev != ROOT.stat().st_dev or path.resolve() != path):
        raise ValueError("Keep the distinct selected model-only checkpoint physically on checkpoint storage")
    selected = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if (selected["step"] != expected["selected"] or selected["config_sha256"] != config_digest
            or selected["preflight_only"] is not False or selected["independent_confirmation_passed"] is not False
            or "optimizer" in selected or set(selected["model"]) != set(payload["model"])):
        raise ValueError("Preserve the exact original model-only selection, not an optimizer file or diagnostic")
    for name, value in payload["model"].items():
        other = selected["model"][name]
        if (not isinstance(value, torch.Tensor) or not isinstance(other, torch.Tensor)
                or value.shape != other.shape or value.dtype != other.dtype
                or not torch.isfinite(value).all() or not byte_equal(value, other)):
            raise ValueError("Every optimizer-tail model tensor must equal the retained durable selected model bitwise")


def checkpoint_identity(payload, current, expected, config_digest):
    if expected["kind"] == "vector":
        if current["state"] != "complete":
            raise ValueError("Never move a live or incomplete diagnostic vector")
        validate_vector(payload, expected["parameters"])
        return
    step, updates, selected = (expected[key] for key in ("step", "updates", "selected"))
    if (expected["kind"] not in {"unselected_tail", "duplicate_selected_tail"}
            or current["state"] != "training_complete" or current["step"] != step
            or current["accepted_updates"] != updates or current["best"]["step"] != selected
            or payload["step"] != step or payload["attempted_batches"] != step
            or payload["gradient_updates_performed"] != updates or payload["best_step"] != selected
            or payload["best_score"] != current["best"]["score"] or payload["config_sha256"] != config_digest
            or payload["independent_confirmation_passed"] is not False or payload["preflight_only"] is not False
            or payload.get(expected["view_flag"]) is not True or not isinstance(payload["model"], dict)
            or not payload["model"] or not isinstance(payload["optimizer"], dict) or not payload["optimizer"]):
        raise ValueError("Require the exact audited terminal optimizer state and original scope")
    if expected["kind"] == "duplicate_selected_tail":
        if selected != step:
            raise ValueError("Only the explicitly audited equal-terminal selection may use duplicate-tail proof")
        selected_copy(payload, expected, config_digest)
    elif selected != 0 or (ROOT / expected["run"] / "selected_checkpoint.pth").exists():
        raise ValueError("V33 must retain its absent eligible selection")


def require_idle():
    protected = []
    for name, expected in OWNERS.items():
        readers.terminal(ROOT / name / "status.json")
        filename = "candidate_parameters.pt" if expected["kind"] == "vector" else "latest_checkpoint.pth"
        protected.append(str(ROOT / name / filename))
        if expected["kind"] != "vector":
            for suffix in ("_audit_task_replay", "_endpoint_replay", "_confirmation"):
                readers.terminal(ROOT / f"{name}{suffix}/status.json")
            if expected.get("recovered"):
                failed = previous.copying.read(ROOT / f"{name}_pipeline/status.json")
                if (failed["state"] != "failed" or type(failed["pid"]) is not int or failed["pid"] <= 0
                        or Path(f'/proc/{failed["pid"]}').exists()):
                    raise ValueError("Preserve the stopped original V26 coordinator failure")
                readers.terminal(ROOT / f"{name}_identity_recovery/status.json")
            else:
                readers.terminal(ROOT / f"{name}_pipeline/status.json")
            if expected["selected"]:
                protected.append(str(ROOT / name / "selected_checkpoint.pth"))
    readers.terminal(ROOT / "full_network_preflight_v1/status.json")
    readers.require_idle(protected)


def inventory():
    previous.copying.require_mount()
    require_idle()
    records, inputs = [], {str(previous.copying.UNIT): sha256(previous.copying.UNIT)}
    for name, expected in OWNERS.items():
        directory = ROOT / name
        filename = "candidate_parameters.pt" if expected["kind"] == "vector" else "latest_checkpoint.pth"
        path = directory / filename
        if (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
                or path.stat().st_dev != ROOT.stat().st_dev or path.resolve() != path):
            raise ValueError("Move only the four declared regular single-link stopped artifacts")
        paths = [directory / item for item in ("config.json", "registration.json", "status.json")]
        if expected["kind"] == "vector":
            audit_path = directory / "audit.json"
            audit, result, gate = [previous.copying.read(directory / item) for item in ("audit.json", "result.json", "gate.json")]
            if (audit["complete"] is not True or audit["training_tasks"] != 28 or audit["prediction_grids"] != 56
                    or result["preflight_only"] is not True or result["trainable_parameters"] != expected["parameters"]
                    or audit["frozen"]["frozen_tensors"] != expected["frozen"] or gate["preflight_only"] is not True
                    or gate["audit_sha256"] != sha256(audit_path)):
                raise ValueError("Require the original complete nonselectable three-block diagnostic audit")
            paths.extend(directory / item for item in ("audit.json", "result.json", "gate.json"))
        else:
            audit_path = ROOT / f"{name}_audit/audit.json"
            audit = previous.copying.read(audit_path)
            confirmation = ROOT / f"{name}_confirmation"
            gate = previous.copying.read(confirmation / "gate.json")
            completion = ROOT / (f"{name}_identity_recovery" if expected.get("recovered") else f"{name}_pipeline")
            result = previous.copying.read(completion / "result.json")
            if (audit["complete"] is not True or audit["complete_training_audit"] is not True
                    or audit["through_step"] != expected["step"] or audit["accepted_updates"] != expected["updates"]
                    or audit["best"]["step"] != expected["selected"] or audit["prediction_grids"] != expected["grids"]
                    or gate["passed"] is not False or gate["selected_step"] != expected["selected"]
                    or result["complete"] is not True or result["gate"] != gate
                    or result["training_audit_sha256"] != sha256(audit_path)
                    or result["campaign"] is not None or result["review"] is not None
                    or (directory / "best_validation_checkpoint.pth").exists()):
                raise ValueError("Require complete original failed qualification with its exact audited terminal state")
            if expected["selected"]:
                selected = directory / "selected_checkpoint.pth"
                confirmed = previous.copying.read(confirmation / "audit.json")
                if (path.samefile(selected) or gate["confirmation_evaluated"] is not True
                        or gate["checkpoint_sha256"] != sha256(selected) or confirmed["complete"] is not True
                        or confirmed["cases"] != 284 or confirmed["prediction_grids"] != 1136
                        or gate["audit_sha256"] != sha256(confirmation / "audit.json")
                        or audit["artifact_sha256"].get(str(selected)) != sha256(selected)):
                    raise ValueError("Preserve each distinct audited selected model and its failed confirmation")
                paths.extend([selected, confirmation / "audit.json"])
            elif gate["confirmation_evaluated"] is not False or (directory / "selected_checkpoint.pth").exists():
                raise ValueError("Never fabricate V33 selection or confirmation")
            if expected.get("recovered"):
                failure = ROOT / f"{name}_pipeline/status.json"
                if (result["original_failed_pipeline"] != str(failure.parent)
                        or result["original_failure_sha256"] != sha256(failure)
                        or result["scientific_stages_repeated"] is not False):
                    raise ValueError("Bind V26's original failure and its completed nonrepeating identity recovery")
            paths.extend([audit_path, confirmation / "gate.json"])
            paths.extend(completion / item for item in ("status.json", "config.json", "registration.json", "result.json"))
            paths.extend(ROOT / f"{name}{suffix}/status.json"
                         for suffix in ("_pipeline", "_audit_task_replay", "_endpoint_replay", "_confirmation"))
        digest = sha256(path)
        if audit["artifact_sha256"].get(str(path)) != digest:
            raise ValueError("Never move bytes that differ from the original complete independent audit")
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        checkpoint_identity(payload, previous.copying.read(directory / "status.json"), expected, sha256(directory / "config.json"))
        del payload
        records.append({"source": str(path), "destination": str(DESTINATION / name / filename),
                        "sha256": digest, "bytes": path.stat().st_size})
        inputs.update({str(source): sha256(source) for source in paths})
    return records, inputs


copy_and_verify = types.FunctionType(previous.copy_and_verify.__code__, globals(), "copy_and_verify")
copying = types.SimpleNamespace(**{**vars(previous.copying), "copy_and_verify": copy_and_verify})
template, REPO = previous.template, previous.REPO
MAIN_SOURCE = previous.MAIN_SOURCE
for old, new in {
    'REPO / "tests/test_single_model_top_three_capacity.py",':
        'REPO / "tests/test_single_model_full_network_capacity.py", REPO / "scripts/single_model_composite_gradient_storage.py",',
    'copying_completed_three_block_capacity': 'copying_completed_full_network_capacity',
    'records, inputs = inventory()\n    total = sum(row["bytes"] for row in records)':
        'records, inputs = inventory()\n    total = sum(row["bytes"] for row in records)\n'
        '    if shutil.disk_usage(ROOT).free + total < TRAINING_ADMISSION_BYTES:\n'
        '        raise OSError("The declared preservation must restore the full unchanged reserve plus expanded allowance")',
    'verify_hashes(config["source_sha256"])\n        write_json':
        'verify_hashes(config["source_sha256"])\n'
        '        if shutil.disk_usage(ROOT).free < TRAINING_ADMISSION_BYTES:\n'
        '            raise OSError("Preservation completed but full-training capacity is not yet restored")\n        write_json',
    '"scientific_training_or_evaluation_performed": False':
        '"scientific_training_or_evaluation_performed": False, "full_training_capacity_ready": True,\n'
        '                   "selected_model_only_checkpoints_retained_on_original_volume": True',
}.items():
    MAIN_SOURCE = replace_once(MAIN_SOURCE, old, new)
status = types.FunctionType(template.status.__code__, globals(), template.status.__name__, template.status.__defaults__)
exec(compile(MAIN_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
