"""Preserve two unselected tails and two old diagnostic vectors for three-block training."""

import os
from pathlib import Path
import shutil
import types

import torch

import relocate_single_model_point_only_capacity as previous
from relocate_single_model_point_only_capacity import readers, sha256, verify_hashes, write_json, validate_vector
from relocate_single_model_gradient_archive import sync_directory
from single_model_registered_adaptation import replace_once


ROOT = previous.ROOT
OUTPUT = ROOT / "top_three_capacity_relocation_v1"
DESTINATION = previous.copying.SCRATCH / "single_model_goal_20261005_owned/top_three_capacity_v1"
OWNERS = {
    "point_only_v32": {"kind": "tail", "step": 64, "updates": 61, "selected": 10, "view_flag": "top_two_block_training"},
    "top_block_continuation_v25": {"kind": "tail", "step": 96, "updates": 89, "selected": 32, "view_flag": "top_block_training"},
    "top_two_blocks_preflight_v1": {"kind": "vector", "parameters": 27766670, "frozen": 192},
    "top_block_preflight_v1": {"kind": "vector", "parameters": 19496843, "frozen": 220},
}
POLICY = ("Storage-only preservation of exactly V32/V25's audited unselected terminal optimizer tails and "
    "two old accepted diagnostic vectors from the one/two-block feasibility probes. Retain every selected "
    "checkpoint, teacher, published model, recent three-block preflight artifact and registered source. "
    "Require stopped terminal owners, absent scientific workers/readers, original complete audits and "
    "exact serialized hashes. Copy/fsync/hash/mmap each file before atomic canonical-link publication, "
    "then independently reopen/recheck all four files and bound provenance. Preserve18/6/50 GiB reserves "
    "and leave headroom for the larger ordinary optimizer checkpoint. No model, optimizer, metric, "
    "training, selection, qualification, source or canonical-path change. Destination is UUID-bound "
    "machine-local scratch, not provider-persistent storage or external backup.")


def checkpoint_identity(payload, current, expected, config_digest):
    if expected["kind"] == "vector":
        if current["state"] != "complete":
            raise ValueError("Never move a live or incomplete diagnostic vector")
        validate_vector(payload, expected["parameters"])
        return
    step, updates, selected = (expected[key] for key in ("step", "updates", "selected"))
    if (expected["kind"] != "tail" or current["state"] != "training_complete" or current["step"] != step
            or current["accepted_updates"] != updates or current["best"]["step"] != selected
            or payload["step"] != step or payload["attempted_batches"] != step
            or payload["gradient_updates_performed"] != updates or payload["best_step"] != selected or selected >= step
            or payload["best_score"] != current["best"]["score"] or payload["config_sha256"] != config_digest
            or payload["independent_confirmation_passed"] is not False or payload["preflight_only"] is not False
            or payload.get(expected["view_flag"]) is not True or not isinstance(payload["model"], dict)
            or not payload["model"] or not isinstance(payload["optimizer"], dict) or not payload["optimizer"]):
        raise ValueError("Require the exact audited nonselected terminal optimizer state with its original view")


def require_idle():
    protected = []
    for name, expected in OWNERS.items():
        readers.terminal(ROOT / name / "status.json")
        filename = "candidate_parameters.pt" if expected["kind"] == "vector" else "latest_checkpoint.pth"
        protected.append(str(ROOT / name / filename))
        if expected["kind"] == "tail":
            for suffix in ("_pipeline", "_audit_task_replay", "_endpoint_replay", "_confirmation"):
                readers.terminal(ROOT / f"{name}{suffix}/status.json")
    readers.terminal(ROOT / "top_three_blocks_preflight_v1/status.json")
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
            raise ValueError("Move only the exact four regular single-link stopped artifacts")
        paths = [directory / item for item in ("config.json", "registration.json", "status.json")]
        if expected["kind"] == "vector":
            audit_path = directory / "audit.json"
            audit, result, gate = [previous.copying.read(directory / item) for item in ("audit.json", "result.json", "gate.json")]
            if (audit["complete"] is not True or audit["training_tasks"] != 28 or audit["prediction_grids"] != 56
                    or result["preflight_only"] is not True or result["trainable_parameters"] != expected["parameters"]
                    or audit["frozen"]["frozen_tensors"] != expected["frozen"] or gate["preflight_only"] is not True
                    or gate["audit_sha256"] != sha256(audit_path)):
                raise ValueError("Require each original complete nonselectable diagnostic audit")
            paths.extend(directory / item for item in ("audit.json", "result.json", "gate.json"))
        else:
            audit_path = ROOT / f"{name}_audit/audit.json"
            audit = previous.copying.read(audit_path)
            confirmation = ROOT / f"{name}_confirmation"
            gate, confirmed = [previous.copying.read(confirmation / item) for item in ("gate.json", "audit.json")]
            result_path = ROOT / f"{name}_pipeline/result.json"
            result = previous.copying.read(result_path)
            selected = directory / "selected_checkpoint.pth"
            if (path.samefile(selected) or audit["complete"] is not True or audit["complete_training_audit"] is not True
                    or audit["through_step"] != expected["step"] or audit["accepted_updates"] != expected["updates"]
                    or audit["best"]["step"] != expected["selected"] or audit["prediction_grids"] != 1368
                    or gate["passed"] is not False or gate["confirmation_evaluated"] is not True
                    or gate["selected_step"] != expected["selected"] or gate["checkpoint_sha256"] != sha256(selected)
                    or confirmed["complete"] is not True or confirmed["cases"] != 284 or confirmed["prediction_grids"] != 1136
                    or gate["audit_sha256"] != sha256(confirmation / "audit.json") or result["complete"] is not True
                    or result["gate"] != gate or result["campaign"] is not None or result["review"] is not None
                    or (directory / "best_validation_checkpoint.pth").exists()):
                raise ValueError("Require complete failed qualification and preserve every selected checkpoint")
            paths.extend([audit_path, selected, result_path, confirmation / "gate.json", confirmation / "audit.json"])
            paths.extend(ROOT / f"{name}{suffix}/status.json"
                         for suffix in ("_pipeline", "_audit_task_replay", "_endpoint_replay", "_confirmation"))
        digest = sha256(path)
        if audit["artifact_sha256"].get(str(path)) != digest:
            raise ValueError("Never move bytes that differ from the original independent audit")
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        checkpoint_identity(payload, previous.copying.read(directory / "status.json"), expected, sha256(directory / "config.json"))
        del payload
        records.append({"source": str(path), "destination": str(DESTINATION / name / filename),
                        "sha256": digest, "bytes": path.stat().st_size})
        inputs.update({str(source): sha256(source) for source in paths})
    return records, inputs


def copy_and_verify(row):
    source, destination = Path(row["source"]), Path(row["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024**2)
        writer.flush()
        os.fsync(writer.fileno())
    sync_directory(destination.parent)
    if destination.stat().st_size != row["bytes"] or sha256(destination) != row["sha256"]:
        raise ValueError("Never publish a copy without preserving every original serialized byte")
    payload = torch.load(destination, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_identity(payload, previous.copying.read(source.parent / "status.json"), OWNERS[source.parent.name],
                        sha256(source.parent / "config.json"))
    del payload


copying = types.SimpleNamespace(**{**vars(previous.copying), "copy_and_verify": copy_and_verify})
template, REPO = previous.template, previous.REPO
MAIN_SOURCE = previous.MAIN_SOURCE
for old, new in {
    'REPO / "tests/test_single_model_point_only_capacity.py",':
        'Path(previous.previous.__file__).resolve(), REPO / "tests/test_single_model_top_three_capacity.py",',
    'len(records) != 7': 'len(records) != 4',
    'Preserve exactly seven completed artifacts': 'Preserve exactly four completed artifacts',
    '"canonical_artifacts_checked": 7': '"canonical_artifacts_checked": 4',
    'copying_completed_point_only_capacity': 'copying_completed_three_block_capacity',
}.items():
    MAIN_SOURCE = replace_once(MAIN_SOURCE, old, new)
status = types.FunctionType(template.status.__code__, globals(), template.status.__name__, template.status.__defaults__)
exec(compile(MAIN_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    main()
