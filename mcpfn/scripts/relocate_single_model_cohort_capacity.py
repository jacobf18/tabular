"""Preserve seven closed nonselected artifacts before a separately registered cohort trial."""

import inspect
import math
import os
from pathlib import Path
import shutil
import sys
import types

import torch

import preflight_single_model_cohort_anchor_retention as probe
import relocate_single_model_legacy_blend_capacity as previous
from relocate_single_model_legacy_blend_capacity import readers, sha256, verify_hashes, write_json, validate_vector, byte_equal
from relocate_single_model_gradient_archive import sync_directory
from single_model_registered_adaptation import replace_once


ROOT, REPO = previous.ROOT, previous.REPO
OUTPUT = ROOT / "cohort_anchor_capacity_relocation_v1"
DESTINATION = previous.copying.SCRATCH / "single_model_goal_20261005_owned/cohort_anchor_capacity_v1"
TRAINING_ADMISSION_BYTES = previous.TRAINING_ADMISSION_BYTES
PROBE_CONFIG_SHA256 = "cdeab141bf4f234134ee05c35156dfb3e8f6da529594b2c52502ae20efa3c1f9"
SELECTED_SHA256 = "43d5a07c16812967a7ff0453581e5a0705e2d203ae68ba9403f3482f1f492c81"
REPLAY_AUDIT_SHA256 = "2a3af6af1c7ec970cd7a418427a62e793bd38e79d1f530dd13eee0dd5afca6d6"
OWNERS = {
    "legacy_blend_teacher_v41": {"kind": "unselected_distinct_tail", "step": 64, "updates": 56, "selected": 50,
        "filename": "latest_checkpoint.pth"},
    "legacy_blend_teacher_preflight_v1": {"kind": "vector", "parameters": 77998496,
        "filename": "candidate_parameters.pt"},
    "finite_guarded_v13_endpoint_replay": {"kind": "replayed_vector", "parameters": 77998496,
        "norm": 0.011502062124006824, "filename": "reconstructed_candidate.pt", "selected": 0,
        "audit_sha256": REPLAY_AUDIT_SHA256},
    "decoder_repair_v14_endpoint_replay": {"kind": "replayed_vector", "parameters": 11227016,
        "norm": 0.0069013797168205274, "filename": "reconstructed_candidate.pt", "selected": 0,
        "audit_sha256": "4f2acfc43226e3d1b1c08a378bc89ed8e11cfed4de7172b476fbc28d7c802fa6"},
    "targeted_decoder_v15_endpoint_replay": {"kind": "replayed_vector", "parameters": 11227016,
        "norm": 0.0005675151242596617, "filename": "reconstructed_candidate.pt", "selected": 0,
        "audit_sha256": "16ca20972443b3cdb33c5088bb65d915eb6bc4642872a840672b72df6f30edbe"},
    "stronger_decoder_v16_endpoint_replay": {"kind": "replayed_vector", "parameters": 11227016,
        "norm": 0.02634428894945959, "filename": "reconstructed_candidate.pt", "selected": 0,
        "audit_sha256": "8729abcf71363b640b42538f5cc19cb6520101b4ea6eb2864a035980db8bba2d"},
    "v8_decoder_v17_endpoint_replay": {"kind": "replayed_vector", "parameters": 11227016,
        "norm": 0.022548632575288188, "filename": "reconstructed_candidate.pt", "selected": 50,
        "audit_sha256": "213744177ee9f271ea3183b7589e83f7d3325cba7c30706a3139c73fe83025e7"},
}
POLICY = ("Storage-only preservation of exactly V41's fully audited unselected step64 optimizer tail, the "
    "older legacy-blend feasibility vector and five independently reconstructed V13–V17 endpoint vectors. "
    "V41's distinct step50 selection stays physically on checkpoint storage, as do all other selected models "
    "and original teachers. Preserve V41's failed confirmation, V13–V16's absent selections, V17's original "
    "step50 development success and every original "
    "byte, metric, source and canonical path. Require the exact accepted cohort-reference probe, complete "
    "fixed-half proof,28 independent teacher grids/four legacy component grids and26 independently replayed "
    "retention grids with both cohort identities. Stop all owners/readers before copy/fsync/hash/mmap and "
    "atomic canonical symlinks; independently reopen all seven. Preserve18/6/50 GiB reserves and restore18 "
    "GiB plus2432 MiB training admission. UUID-bound scratch is machine-local, not provider-persistent backup. "
    "This registration authorizes no training, new seed, checkpoint selection, confirmation or real benchmark.")


def checkpoint_identity(payload, current, expected, config_digest):
    if expected["kind"] == "replayed_vector":
        if (current["state"] != "complete" or current["selected_validation_passed"] is not bool(expected["selected"])
                or not isinstance(payload, dict) or set(payload) != {"parameters", "products", "norm"}
                or payload["norm"] != expected["norm"] or not isinstance(payload["products"], list)
                or len(payload["products"]) != 42
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in payload["products"])):
            raise ValueError("Preserve the complete original nonselected reconstructed endpoint and norm")
        validate_vector(payload["parameters"], expected["parameters"])
        return
    if expected["kind"] == "vector":
        if current["state"] != "complete":
            raise ValueError("Preserve only stopped complete diagnostic vectors")
        validate_vector(payload, expected["parameters"])
        return
    if (expected["kind"] != "unselected_distinct_tail" or current["state"] != "training_complete"
            or current["step"] != 64 or current["accepted_updates"] != 56 or current["best"]["step"] != 50
            or payload["step"] != 64 or payload["attempted_batches"] != 64 or payload["gradient_updates_performed"] != 56
            or payload["best_step"] != 50 or payload["best_score"] != current["best"]["score"]
            or payload["config_sha256"] != config_digest or payload["independent_confirmation_passed"] is not False
            or payload["preflight_only"] is not False or payload.get("full_network_training") is not True
            or not isinstance(payload["model"], dict) or len(payload["model"]) != 254
            or not isinstance(payload["optimizer"], dict) or not payload["optimizer"]):
        raise ValueError("Preserve the exact audited step64 tail, never relabel it as the selected step50")


def require_selected_preserved():
    path = ROOT / "legacy_blend_teacher_v41/selected_checkpoint.pth"
    tail = ROOT / "legacy_blend_teacher_v41/latest_checkpoint.pth"
    if (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1 or path.resolve() != path
            or path.stat().st_dev != ROOT.stat().st_dev or path.samefile(tail) or sha256(path) != SELECTED_SHA256):
        raise ValueError("Keep the distinct original selected model physically on checkpoint storage")


def require_idle():
    protected = []
    for name, expected in OWNERS.items():
        readers.terminal(ROOT / name / "status.json")
        protected.append(str(ROOT / name / expected["filename"]))
    for directory in (probe.OUTPUT, probe.diagnosis.recovery.OUTPUT, probe.diagnosis.recovery.REPLAY,
            probe.diagnosis.original.CONFIRMATION, probe.diagnosis.OUTPUT):
        readers.terminal(directory / "status.json")
    require_selected_preserved()
    protected.append(str(ROOT / "legacy_blend_teacher_v41/selected_checkpoint.pth"))
    readers.require_idle(protected)


function = previous.original_probe_evidence
original_probe_evidence = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
fixed_trial = previous.fixed_trial
function = previous.validate_probe
validate_original_probe = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)


def validate_probe(result, audit, retention, gate, current):
    validate_original_probe(result, audit, retention, gate, current)
    expected = {role: {**identity, "state_tensors": 254, "bitwise_equal": True}
        for role, identity in probe.reference_identities().items()}
    if (retention["reference"] != expected or retention["reference_role_counts"] != {"v8": 4, "v29": 22}
            or len(retention["endpoint_penalties"]) != 26):
        raise ValueError("Require both exact replayed references and every cohort assignment")


def require_probe():
    if sha256(probe.OUTPUT / "config.json") != PROBE_CONFIG_SHA256:
        raise ValueError("Bind the sole registered cohort-reference probe")
    evidence = original_probe_evidence()
    records = [probe.read(probe.OUTPUT / name) for name in
        ("result.json", "audit.json", "retention_audit.json", "gate.json", "status.json")]
    validate_probe(*records)
    return {**evidence, str(probe.OUTPUT / "gate.json"): sha256(probe.OUTPUT / "gate.json")}


def inventory():
    previous.copying.require_mount()
    require_idle()
    inputs = {str(previous.copying.UNIT): sha256(previous.copying.UNIT), **require_probe(), **probe.require_completed_v41()}
    records = []
    for name, expected in OWNERS.items():
        directory = ROOT / name
        path = directory / expected["filename"]
        if (path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1
                or path.stat().st_dev != ROOT.stat().st_dev or path.resolve() != path):
            raise ValueError("Preserve exactly seven original regular single-link stopped artifacts")
        paths = [directory / filename for filename in ("config.json", "status.json")]
        if name == "legacy_blend_teacher_v41":
            audit_path = probe.diagnosis.original.AUDIT / "audit.json"
            paths.extend([audit_path, directory / "registration.json", directory / "selected_checkpoint.pth"])
            audit = probe.read(audit_path)
        elif name == "legacy_blend_teacher_preflight_v1":
            audit_path = directory / "audit.json"
            audit, result, gate = [probe.read(directory / filename) for filename in ("audit.json", "result.json", "gate.json")]
            if (audit["complete"] is not True or audit["training_tasks"] != 28 or audit["prediction_grids"] != 56
                    or audit["frozen"]["frozen_tensors"] != 16 or result["preflight_only"] is not True
                    or result["trainable_parameters"] != 77998496 or gate["preflight_only"] is not True
                    or gate["audit_sha256"] != sha256(audit_path)):
                raise ValueError("Require the complete original nonselected legacy-blend feasibility audit")
            paths.extend(directory / filename for filename in ("registration.json", "audit.json", "result.json", "gate.json"))
        else:
            audit_path = directory / "audit.json"
            audit, gate = [probe.read(directory / filename) for filename in ("audit.json", "gate.json")]
            if (sha256(audit_path) != expected["audit_sha256"] or audit["complete"] is not True
                    or audit["endpoint_grids"] != 28 or audit["selected_grids"] != (152 if expected["selected"] else 0)
                    or audit["cuda_initialized"] is not False or gate["endpoint_reproduced"] is not True
                    or gate["selected_step"] != expected["selected"] or gate["selected_validation_passed"] is not bool(expected["selected"])
                    or gate["audit_sha256"] != expected["audit_sha256"]):
                raise ValueError("Keep the exact independently replayed endpoint and its original selection outcome")
            paths.extend([audit_path, directory / "gate.json"])
        verify_hashes(audit["artifact_sha256"])
        config = probe.read(directory / "config.json")
        for field in ("source_sha256", "input_sha256"):
            verify_hashes(config.get(field, {}))
            inputs.update(config.get(field, {}))
        digest = sha256(path)
        if audit["artifact_sha256"].get(str(path)) != digest:
            raise ValueError("Copy only bytes bound by the original independent audit")
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        checkpoint_identity(payload, probe.read(directory / "status.json"), expected, sha256(directory / "config.json"))
        del payload
        records.append({"source": str(path), "destination": str(DESTINATION / name / expected["filename"]),
            "sha256": digest, "bytes": path.stat().st_size})
        inputs.update({str(current): sha256(current) for current in paths})
    return records, inputs


function = previous.copy_and_verify
copy_and_verify = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
copying = types.SimpleNamespace(**{**vars(previous.copying), "copy_and_verify": copy_and_verify})
template = previous.template
MAIN_SOURCE = replace_once(previous.MAIN_SOURCE, 'REPO / "tests/test_single_model_legacy_blend_capacity.py",',
    'REPO / "tests/test_single_model_cohort_capacity.py",')
MAIN_SOURCE = replace_once(MAIN_SOURCE, "copying_completed_legacy_blend_capacity", "copying_completed_cohort_capacity")
MAIN_SOURCE = replace_once(MAIN_SOURCE, "len(records) != 3", "len(records) != 7")
MAIN_SOURCE = replace_once(MAIN_SOURCE, "Preserve exactly three completely audited artifacts", "Preserve exactly seven completely audited artifacts")
MAIN_SOURCE = replace_once(MAIN_SOURCE, '"canonical_artifacts_checked": 3', '"canonical_artifacts_checked": 7')
MAIN_SOURCE = replace_once(MAIN_SOURCE, '        verify_hashes(hashes)', '        require_selected_preserved()\n        verify_hashes(hashes)')
status = types.FunctionType(template.status.__code__, globals(), template.status.__name__, template.status.__defaults__)
exec(compile(MAIN_SOURCE, __file__, "exec"), globals())


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise ValueError("Invoke the tested storage-only preservation once without stage arguments")
    main()
