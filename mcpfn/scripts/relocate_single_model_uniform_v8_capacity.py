"""Preserve three closed nonselected artifacts before a uniform-reference full trial."""

import os
from pathlib import Path
import shutil
import time

import torch

import preflight_single_model_uniform_v8_retention as probe
import relocate_single_model_idle_diagnostics_to_root as readers
import relocate_single_model_idle_diagnostics_to_scratch as copying
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from relocate_single_model_gradient_archive import sync_directory


ROOT = probe.ROOT
OUTPUT = ROOT / "uniform_v8_capacity_relocation_v1"
DESTINATION = copying.SCRATCH / "single_model_goal_20261005_owned/uniform_v8_capacity_v1"
TRAINING_ADMISSION_BYTES = 18 * 1024**3 + 2432 * 1024**2
SELECTED = ROOT / "family_balanced_retention_v43/selected_checkpoint.pth"
SELECTED_SHA256 = "461664c2bafc8fa89131684b2a19346564cef6e31263c7b5437fb5e63cce1db2"
PROBE_CONFIG_SHA256 = "f6ce842851d8320e1e0543920838ceaa6b057c2d29b7eab20ad64dd77ca319b6"
TARGETS = {
    "family_balanced_retention_v43/latest_checkpoint.pth": "tail",
    "uniform_v8_retention_preflight_v1/candidate_parameters.pt": "vector",
    "decoder_rate_probe_v1/rate_0/endpoint.pt": "rate_endpoint",
}
POLICY = ("Storage only: preserve exactly the fully audited unselected V43 step64 optimizer tail, completed "
    "uniform-V8 feasibility vector and original decoder-rate probe's independently audited nonselected rate0 "
    "endpoint. Preserve the older probe's original failed status and successful nonrepeating audit recovery; "
    "do not imply scientific qualification. Keep V43's distinct step25 selection physically on checkpoint "
    "storage and all original teachers, selected models, sources, serialized bytes and canonical paths "
    "unchanged. Require stopped owners and no open readers. Copy/fsync/hash/mmap before atomic canonical "
    "symlinks, then independently recheck all three and every input. Restore18 GiB plus2432 MiB full-training "
    "admission without weakening18/6/50 GiB reserves. UUID-bound scratch is machine-local, not an external "
    "backup. No training, model selection, new seed, qualification retry or benchmark evaluation.")


def read(path):
    return probe.read(path)


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def require_stopped():
    for name, expected in {
        "family_balanced_retention_v43": "training_complete",
        "family_balanced_retention_v43_pipeline": "failed",
        "family_balanced_retention_v43_orphan_recovery_v1": "complete",
        "family_balanced_retention_v43_audit_task_replay": "complete",
        "family_balanced_retention_v43_endpoint_replay": "complete",
        "family_balanced_retention_v43_confirmation": "complete",
        "family_balanced_retention_v43_initializer_diagnosis_v1": "complete",
        "uniform_v8_retention_preflight_v1": "complete",
        "decoder_rate_probe_v1": "failed",
        "decoder_rate_probe_v1_audit_recovery1": "complete",
    }.items():
        current = read(ROOT / name / "status.json")
        if current["state"] != expected or probe.diagnosis.original.process_identity(current["pid"]) is not None:
            raise ValueError("Preserve every exact stopped producer outcome, including both original failures")


def validate_payload(payload, kind, *, rate_parameters=11227016, full_parameters=77998496):
    if kind == "vector":
        if (not isinstance(payload, torch.Tensor) or payload.shape != (full_parameters,)
                or payload.dtype != torch.float32 or not torch.isfinite(payload).all()):
            raise ValueError("Preserve the full finite ordinary nonpromotable probe vector")
    elif kind == "tail":
        current = read(ROOT / "family_balanced_retention_v43/status.json")
        if (payload["step"] != 64 or payload["attempted_batches"] != 64
                or payload["gradient_updates_performed"] != 56 or payload["best_step"] != 25
                or payload["best_score"] != current["best"]["score"]
                or payload["config_sha256"] != sha256(ROOT / "family_balanced_retention_v43/config.json")
                or payload["independent_confirmation_passed"] is not False or payload["preflight_only"] is not False
                or payload["full_network_training"] is not True or not payload["model"] or not payload["optimizer"]):
            raise ValueError("Preserve only the exact unselected V43 terminal optimizer state")
        if (SELECTED.is_symlink() or SELECTED.stat().st_dev != ROOT.stat().st_dev
                or sha256(SELECTED) != SELECTED_SHA256):
            raise ValueError("Never move or change the distinct original selected checkpoint")
        selected = torch.load(SELECTED, map_location="cpu", weights_only=True, mmap=True)
        if (selected["step"] != 25 or "optimizer" in selected or set(selected["model"]) != set(payload["model"])
                or all(torch.equal(value, selected["model"][name]) for name, value in payload["model"].items())):
            raise ValueError("Keep the distinct selected model, not a duplicate or promoted tail")
        del selected
    elif kind == "rate_endpoint":
        if (payload["config_sha256"] != sha256(ROOT / "decoder_rate_probe_v1/config.json")
                or payload["record"]["rate"] != 2e-6 or payload["record"]["accepted"] is not True
                or payload["record"]["protected_tasks"] != 42 or not payload["optimizer"]
                or not payload["source_raw_sha256"]):
            raise ValueError("Preserve the original audited rate0 diagnostic, never a model candidate")
        for name in ("before", "tentative", "after", "combined_gradient", "applied_gradient"):
            value = payload["snapshot"][name]
            if value.shape != (rate_parameters,) or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError("Preserve every full finite decoder-rate endpoint vector")
    else:
        raise ValueError("Only the three prospectively declared artifact kinds may move")


def inventory():
    copying.require_mount()
    require_stopped()
    with probe.diagnosis.original.arithmetic_recovery.previous.original_arithmetic():
        probe.verify(probe.OUTPUT)
    if sha256(probe.OUTPUT / "config.json") != PROBE_CONFIG_SHA256:
        raise ValueError("Require the exact accepted uniform-reference probe")
    probe_result, probe_audit, retention, gate = [read(probe.OUTPUT / name) for name in
        ("result.json", "audit.json", "retention_audit.json", "gate.json")]
    if (not probe_result["record"]["accepted"] or not probe_audit["complete"] or probe_audit["training_tasks"] != 28
            or probe_audit["prediction_grids"] != 56 or not retention["complete"]
            or retention["reference_role_counts"] != {"v8": 26} or not gate["passed"] or not gate["preflight_only"]):
        raise ValueError("Require completed original and26-reference audits without promoting the probe")
    tail_audit = read(ROOT / "family_balanced_retention_v43_audit/audit.json")
    rate_audit = read(ROOT / "decoder_rate_probe_v1_audit_recovery1/audit.json")
    if (not tail_audit["complete"] or tail_audit["through_step"] != 64 or tail_audit["best"]["step"] != 25
            or not rate_audit["complete"] or rate_audit["checkpoint_selected"] is not False
            or rate_audit["prediction_grids"] != 112 or rate_audit["training_tasks"] != 14):
        raise ValueError("Require the exact independently audited closed artifacts")
    hashes = {**tail_audit["artifact_sha256"], **probe_audit["artifact_sha256"], **rate_audit["artifact_sha256"]}
    paths = [ROOT / name for name in (
        "family_balanced_retention_v43_audit/audit.json", "family_balanced_retention_v43_orphan_recovery_v1/result.json",
        "decoder_rate_probe_v1/config.json", "decoder_rate_probe_v1/status.json",
        "decoder_rate_probe_v1_audit_recovery1/audit.json", "decoder_rate_probe_v1_audit_recovery1/status.json")]
    paths.extend(probe.OUTPUT / name for name in ("config.json", "registration.json", "result.json", "audit.json",
        "retention_audit.json", "gate.json", "status.json"))
    paths.extend([SELECTED, ROOT / "family_balanced_retention_v43/config.json", ROOT / "family_balanced_retention_v43/status.json"])
    records = []
    for name, kind in TARGETS.items():
        source = ROOT / name
        if (source.is_symlink() or not source.is_file() or source.stat().st_nlink != 1
                or source.stat().st_dev != ROOT.stat().st_dev or source.samefile(SELECTED)
                or str(source) not in hashes or sha256(source) != hashes[str(source)]):
            raise ValueError("Move only the exact three audited single-link nonselected artifacts")
        payload = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
        validate_payload(payload, kind)
        del payload
        records.append({"source": str(source), "destination": str(DESTINATION / name), "kind": kind,
                        "bytes": source.stat().st_size, "sha256": hashes[str(source)]})
    readers.require_no_artifact_readers([row["source"] for row in records])
    return records, {str(path): sha256(path) for path in paths}


def copy_and_verify(row):
    source, destination = Path(row["source"]), Path(row["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024**2)
        writer.flush()
        os.fsync(writer.fileno())
    sync_directory(destination.parent)
    if destination.stat().st_size != row["bytes"] or sha256(destination) != row["sha256"]:
        raise ValueError("Never publish an inexact diagnostic copy")
    payload = torch.load(destination, map_location="cpu", weights_only=False, mmap=True)
    validate_payload(payload, row["kind"])
    del payload


def recheck():
    config = read(OUTPUT / "config.json")
    if (read(OUTPUT / "registration.json")["config_sha256"] != sha256(OUTPUT / "config.json")
            or config["policy"] != POLICY or config["scratch_uuid"] != copying.UUID):
        raise ValueError("Never alter the registered preservation operation")
    copying.require_mount()
    require_stopped()
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    hashes = {}
    for row in config["records"]:
        source, destination = Path(row["source"]), Path(row["destination"])
        if not source.is_symlink() or source.resolve() != destination or destination.is_symlink():
            raise ValueError("Preserve every canonical path to its original exact bytes")
        payload = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
        validate_payload(payload, row["kind"])
        del payload
        hashes.update({str(source): row["sha256"], str(destination): row["sha256"]})
    verify_hashes(hashes)
    if (shutil.disk_usage(ROOT).free < TRAINING_ADMISSION_BYTES or shutil.disk_usage("/").free < 6 * 1024**3
            or shutil.disk_usage(copying.SCRATCH).free < 50 * 1024**3):
        raise OSError("Restore full original admission without lowering any reserve")
    return {"complete": True, "canonical_artifacts_checked": 3, "artifact_sha256": hashes,
        "all_original_serialized_bytes_and_paths_preserved": True, "selected_checkpoints_unchanged": True,
        "bytes_relocated": sum(row["bytes"] for row in config["records"]),
        "checkpoint_available_bytes": shutil.disk_usage(ROOT).free, "provider_persistent_backup": False,
        "config_sha256": sha256(OUTPUT / "config.json"), "training_admission_bytes": TRAINING_ADMISSION_BYTES}


def main():
    if OUTPUT.exists() or DESTINATION.exists():
        raise FileExistsError("Never restart or overwrite a begun preservation")
    records, inputs = inventory()
    total = sum(row["bytes"] for row in records)
    if (shutil.disk_usage(ROOT).free < 18 * 1024**3 + 16 * 1024**2
            or shutil.disk_usage(ROOT).free + total < TRAINING_ADMISSION_BYTES + 16 * 1024**2
            or shutil.disk_usage("/").free < 6 * 1024**3 + 16 * 1024**2
            or shutil.disk_usage(copying.SCRATCH).free - total < 50 * 1024**3):
        raise OSError("Preserve all reserves and enough exact-byte recovery for training")
    sources = [Path(__file__).resolve(), Path(probe.__file__).resolve(), Path(readers.__file__).resolve(),
               Path(copying.__file__).resolve(), REPO / "tests/test_single_model_uniform_v8_capacity.py"]
    OUTPUT.mkdir()
    DESTINATION.mkdir(parents=True)
    write_json(OUTPUT / "config.json", {"policy": POLICY, "records": records, "input_sha256": inputs,
        "source_sha256": {str(path): sha256(path) for path in sources}, "scratch_uuid": copying.UUID})
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json"), "registered_unix": time.time()})
    write_json(DESTINATION / "OWNERSHIP.json", {"registered_at": str(OUTPUT), "config_sha256": sha256(OUTPUT / "config.json")})
    status("copying_closed_nonselected_artifacts")
    try:
        for row in records:
            copy_and_verify(row)
        require_stopped()
        readers.require_no_artifact_readers([row["source"] for row in records])
        for row in records:
            copying.publish(row)
        write_json(OUTPUT / "independent_recheck.json", recheck())
        status("complete", bytes_relocated=total)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
