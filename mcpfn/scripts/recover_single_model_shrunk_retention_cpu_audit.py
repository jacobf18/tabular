"""Audit one completed rejected probe without changing its failed entry or rerunning science."""

import inspect
import os
from pathlib import Path
import shutil
import sys
import types

import torch

from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
import preflight_single_model_shrunk_column_anchor_retention as probe


ORIGINAL = probe.OUTPUT
OUTPUT = ORIGINAL.with_name("shrunk_column_anchor_retention_preflight_v1_cpu_audit_recovery_v1")
CONFIG_SHA256 = "86f399a6304d1a8f18a860a722341df5323b3ed38d0b1615f37f7effc2014d79"
AUDIT_FILES = {"audit.json", "gate.json", "retention_audit.json", "RESULTS.md", "status.json"}
POLICY = ("Finish only the previously unexecuted CPU audit of the original one-shot count-shrunk probe. "
    "The original run and independent GPU teacher/reference replay completed; its optimizer proposal was "
    "rejected by the unchanged finite metrics. The audit CLI then failed at its first CPU-environment guard, "
    "before any audit artifact was written. Preserve the failed original status, log, configuration and every "
    "original artifact byte/path. Use unmodified audit function code in an isolated globals dictionary with "
    "the original input paths and a strict view routing only the five audit report/status filenames to this "
    "separate directory. Do not spoof or edit the failed original status: this recovery's own status records "
    "a fresh pending CPU audit. Require CUDA hidden/uninitialized, stopped original producers, complete GPU "
    "replay, unchanged sources and exact original rejection. No model inference, training, optimizer "
    "proposal, reference inference, selection or benchmark is repeated. The unchanged CPU auditor may "
    "independently regenerate already-consumed synthetic admission inputs; no new seeds or data are used. Preserve all original "
    "arithmetic tolerances and guards. A completed audit of rejection does not qualify a training trial.")


class AuditView(os.PathLike):
    def __init__(self, original, destination):
        self.original = Path(original)
        self.destination = Path(destination)

    def __truediv__(self, name):
        name = Path(name)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Keep all audit-view paths relative to their declared owner")
        return (self.destination if str(name) in AUDIT_FILES else self.original) / name

    def __fspath__(self):
        return str(self.original)

    def __str__(self):
        return str(self.original)


def require_cpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Recovery must start with CUDA hidden and uninitialized")


def validate_failure(failed, result, replay, manifest, log, existing_reports):
    if (failed.get("state") != "failed" or failed.get("failed_stage") != "audit"
            or failed.get("error") != "ValueError('Audit must be CPU only')"
            or "ValueError: Audit must be CPU only" not in log or existing_reports):
        raise ValueError("Recover only the exact audit-entry failure before any CPU audit outputs")
    if (result.get("data_admission_only") is not False or result.get("preflight_only") is not True
            or result.get("trainable_parameters") != 77998496
            or result["record"].get("accepted") is not False
            or result["record"].get("accepted_updates") != 0
            or result["record"].get("reason") != "finite_metrics_rejected"
            or replay.get("complete") is not True or manifest.get("complete") is not True
            or len(manifest.get("records", [])) != 28
            or manifest.get("specification") != probe.retention.SPECIFICATION
            or len(replay.get("records", [])) != sum(row["eligible"] for row in manifest["records"])
            or replay.get("reference") != manifest.get("reference")):
        raise ValueError("Preserve the completed original rejected proposal and every independently replayed reference")


def original_inventory():
    result, resolved_directories = {}, set()
    for directory, folders, files in os.walk(ORIGINAL, followlinks=True):
        directory = Path(directory)
        resolved = directory.resolve()
        if resolved in resolved_directories or not any(resolved.is_relative_to(root) for root in (ORIGINAL, probe.SCRATCH)):
            raise ValueError("Require the acyclic declared original probe and scratch tree")
        resolved_directories.add(resolved)
        for name in files:
            path = directory / name
            result[str(path)] = sha256(path)
    return result


def failure_proof():
    require_cpu()
    if sha256(ORIGINAL / "config.json") != CONFIG_SHA256:
        raise ValueError("Require the exact registered one-shot probe, not a new experiment")
    probe.verify(ORIGINAL)
    failed, result, replay, manifest = [probe.read(ORIGINAL / name) for name in
        ("status.json", "result.json", "retention_replay/manifest.json", "retention_targets/manifest.json")]
    reports = {name for name in AUDIT_FILES - {"status.json"} if (ORIGINAL / name).exists()}
    validate_failure(failed, result, replay, manifest, (ORIGINAL / "run_and_audit.log").read_text(), reports)
    for pid in (failed["pid"], result["training_pid"], replay["pid"]):
        if type(pid) is not int or pid <= 0 or Path(f"/proc/{pid}").exists():
            raise ValueError("Original run, replay and failed entry must be stopped")
    if replay["input_manifest_sha256"] != sha256(ORIGINAL / "retention_targets/manifest.json"):
        raise ValueError("Bind the complete actual reference replay to the original target manifest")
    for record in replay["records"]:
        if sha256(record["artifact"]) != record["sha256"]:
            raise ValueError("Do not replace any completed independently replayed reference")
    return original_inventory()


def namespace(destination):
    current = dict(vars(probe))
    for name, value in vars(probe).items():
        if inspect.isfunction(value) and value.__globals__ is vars(probe) and value.__closure__ is None:
            cloned = types.FunctionType(value.__code__, current, value.__name__, value.__defaults__)
            cloned.__kwdefaults__ = value.__kwdefaults__
            current[name] = cloned

    def audit_write(path, payload):
        path = Path(path)
        if path.parent != destination or path.name not in AUDIT_FILES - {"RESULTS.md"}:
            raise ValueError("The isolated auditor may write only its new declared report/status files")
        if path.name != "status.json" and path.exists():
            raise FileExistsError("Never repeat or overwrite an audit report")
        return write_json(path, payload)

    current["write_json"] = audit_write
    return current


def recover():
    require_cpu()
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("Use one new recovery directory; never repeat a completed or failed audit")
    if shutil.disk_usage(OUTPUT.parent).free < 18 * 1024**3 + 16 * 1024**2:
        raise OSError("Preserve18 GiB reserve and16 MiB bounded audit output allowance")
    inputs = failure_proof()
    sources = {str(Path(module.__file__).resolve()): sha256(module.__file__) for module in tuple(sys.modules.values())
        if getattr(module, "__file__", None) and str(module.__file__).endswith(".py")
        and Path(module.__file__).resolve().is_relative_to(REPO) and Path(module.__file__).is_file()}
    test_path = REPO / "tests/test_single_model_shrunk_retention_cpu_audit_recovery.py"
    sources[str(test_path)] = sha256(test_path)
    OUTPUT.mkdir()
    write_json(OUTPUT / "recovery_config.json", {"policy": POLICY, "original": str(ORIGINAL),
        "original_config_sha256": CONFIG_SHA256, "input_sha256": inputs, "source_sha256": sources,
        "report_names": sorted(AUDIT_FILES), "scientific_configuration_changed": False})
    write_json(OUTPUT / "recovery_registration.json", {
        "recovery_config_sha256": sha256(OUTPUT / "recovery_config.json"),
        "original_failure_preceded_cpu_audit": True, "model_or_reference_inference_repeated": False})
    write_json(OUTPUT / "status.json", {"state": "awaiting_independent_audit", "pid": os.getpid(),
        "original_failed_entry_preserved": True, "recovery_role": "first CPU audit of the original completed proposal"})
    try:
        current = namespace(OUTPUT)
        current["audit"](AuditView(ORIGINAL, OUTPUT))
        original, retention, gate = [probe.read(OUTPUT / name) for name in ("audit.json", "retention_audit.json", "gate.json")]
        if (original["complete"] is not True or original["training_tasks"] != 28 or original["prediction_grids"] != 56
                or original["frozen"]["frozen_tensors"] != 16 or original["cuda_initialized"] is not False
                or retention["complete"] is not True or retention["eligibility_decisions"] != 28
                or retention["cuda_initialized"] is not False or retention["no_inference_ensemble"] is not True
                or retention["original_audit_sha256"] != sha256(OUTPUT / "audit.json")
                or gate["passed"] is not False or gate["reason"] != "finite_metrics_rejected"
                or gate["preflight_only"] is not True or gate["real_benchmark_success"] is not False
                or gate["audit_sha256"] != sha256(OUTPUT / "audit.json")):
            raise ValueError("Keep complete unchanged audit coverage and the original negative feasibility gate")
        verify_hashes(original["artifact_sha256"])
        verify_hashes(retention["artifact_sha256"])
        verify_hashes(inputs)
        verify_hashes(sources)
        if original_inventory() != inputs:
            raise ValueError("Every original path, file and failed status must remain unchanged")
        require_cpu()
        write_json(OUTPUT / "recovery_result.json", {"complete": True, "accepted": False,
            "original_failure_preserved": True, "original_rejection_preserved": True,
            "training_repeated": False, "model_or_reference_inference_repeated": False,
            "new_seeds_consumed": False, "selection_or_benchmarks_evaluated": False,
            "cuda_initialized": False, "gate_changed": False,
            "original_audit_sha256": sha256(OUTPUT / "audit.json"),
            "retention_audit_sha256": sha256(OUTPUT / "retention_audit.json"),
            "recovery_config_sha256": sha256(OUTPUT / "recovery_config.json")})
        write_json(OUTPUT / "status.json", {"state": "complete", "pid": os.getpid(), "accepted": False,
            "original_failed_entry_preserved": True, "retention_audited": True})
    except BaseException as error:
        write_json(OUTPUT / "status.json", {"state": "failed", "pid": os.getpid(), "error": repr(error),
            "original_failed_entry_preserved": True})
        raise


if __name__ == "__main__":
    recover()
