"""Restore the original four-Torch/24-BLAS-thread context for the CPU audit."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import types

import torch

import recover_single_model_four_draw_audit as previous
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


probe = previous.probe
arithmetic = probe.diagnosis.original.arithmetic_recovery.previous
OUTPUT = probe.OUTPUT.with_name(probe.OUTPUT.name + "_audit_recovery_v2")
DIAGNOSIS = probe.ROOT / "four_draw_preflight_v1_audit_thread_diagnosis.json"
POLICY = previous.POLICY + (
    " Preserve the failed first count recovery and its tests: they reached the corrected helper "
    "but omitted the original main's arithmetic context. Import-time Torch configuration was24 "
    "threads despite OMP_NUM_THREADS=4. CPU admission then differed only in normalized metadata "
    "and episode digests for three tasks; all raw-array digests remained identical. Restore the "
    "already registered original_arithmetic context (four Torch threads, both OpenBLAS pools24) "
    "around every entry. Independently require exact complete regenerated admission and all48 "
    "captured-episode digests before registration; no numerical tolerance or identity relaxation. "
    "Keep all earlier errors and sources. This is another administrative audit-entry repair, "
    "not another training attempt, neural replay or full-trial authorization."
)


def digest_record(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def diagnose():
    if DIAGNOSIS.exists() or DIAGNOSIS.is_symlink():
        raise FileExistsError("Never replace an arithmetic diagnosis")
    previous.verify()
    previous.require_unentered()
    config = probe.verify(probe.OUTPUT)
    saved = probe.read(probe.OUTPUT / "data_admission.json")
    original_threads = torch.get_num_threads()
    default_profile = arithmetic.blas_profile()
    default = probe.admission.TrainingBatchAdmission(probe.StructuredSampler(config["base"], device="cpu"), config["batch"])
    record = default.record
    changed = [{"index": original["task"]["index"], "fields": [key for key in original if original[key] != regenerated[key]]}
        for original, regenerated in zip(saved["tasks"], record["tasks"]) if original != regenerated]
    del default
    with arithmetic.original_arithmetic() as restored_profile:
        restored_threads = torch.get_num_threads()
        admitted = previous.audit_namespace()["audit_admission"](probe.OUTPUT, config)
    if (original_threads != 24 or restored_threads != 4
            or changed != [{"index": index, "fields": ["metadata", "episode_content_sha256"]} for index in (18, 24, 40)]
            or [key for key in saved if saved[key] != record[key]] != ["tasks"]
            or not admitted["complete"] or not admitted["eligible"] or admitted["structured_draws"] != 48):
        raise ValueError("Require the exact diagnosed thread-only mismatch and complete unchanged admission checks")
    paths = [previous.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "audit.log", "failed_original_status.json")]
    paths.extend((Path(__file__), REPO / "tests/test_single_model_four_draw_audit_arithmetic.py", probe.OUTPUT / "status.json"))
    report = {"complete": True, "default_torch_threads": original_threads, "default_blas_profile": default_profile,
        "restored_torch_threads": restored_threads, "restored_blas_profile": restored_profile,
        "saved_record_sha256": digest_record(saved), "default_record_sha256": digest_record(record),
        "changed_tasks": changed, "raw_arrays_identical": True, "restored_exact_record_equal": True,
        "captured_structured_episodes_checked": 48, "restored_admission": admitted,
        "artifact_sha256": {str(path): sha256(path) for path in paths}, "cuda_initialized": torch.cuda.is_initialized(),
        "training_repeated": False, "neural_replay_repeated": False, "goal_success_not_assessed": True}
    write_json(DIAGNOSIS, report)
    print({key: report[key] for key in ("changed_tasks", "restored_exact_record_equal", "captured_structured_episodes_checked")}, flush=True)


def require_closed():
    config = previous.verify()
    failed = probe.read(previous.OUTPUT / "status.json")
    state = probe.read(probe.OUTPUT / "status.json")
    if (failed["state"] != "failed" or len(failed["completed_stages"]) != 1
            or failed["completed_stages"][0]["stage"] != "audit" or failed["completed_stages"][0]["returncode"] != 1
            or failed["completed_stages"][0]["log_sha256"] != sha256(previous.OUTPUT / "audit.log")
            or state["state"] != "failed" or state["failed_stage"] != "audit_recovery"
            or state["error"] != "ValueError('Independently reproduce the complete registered admission attempt')"
            or state["recovery"] != str(previous.OUTPUT)):
        raise ValueError("Require the preserved first count recovery's exact stopped admission-identity failure")
    for owner in (failed["pid"], state["pid"]):
        if type(owner) is not int or owner <= 0 or Path(f"/proc/{owner}").exists():
            raise ValueError("Never overlap an earlier coordinator or audit")
    diagnosis = probe.read(DIAGNOSIS)
    if (diagnosis["complete"] is not True or diagnosis["restored_exact_record_equal"] is not True
            or diagnosis["captured_structured_episodes_checked"] != 48 or diagnosis["raw_arrays_identical"] is not True
            or diagnosis["restored_torch_threads"] != 4 or diagnosis["cuda_initialized"] is not False
            or any(diagnosis[name] is not False for name in ("training_repeated", "neural_replay_repeated"))):
        raise ValueError("Require the complete exact admission and captured-episode diagnosis")
    previous.verify_hashes(diagnosis["artifact_sha256"])
    previous.verify_hashes(diagnosis["restored_admission"]["artifact_sha256"])
    previous.require_unentered()
    return config


def namespace():
    result = {**vars(previous), "__file__": __file__, "OUTPUT": OUTPUT, "POLICY": POLICY,
        "require_closed": require_closed, "arithmetic_predecessor": previous, "DIAGNOSIS": DIAGNOSIS}
    for name, function in vars(previous).items():
        if inspect.isfunction(function) and function.__globals__ is vars(previous) and name != "require_closed":
            if function.__closure__ is not None:
                raise ValueError("Never preserve a first-recovery callback closure")
            cloned = types.FunctionType(function.__code__, result, function.__name__, function.__defaults__)
            cloned.__kwdefaults__ = function.__kwdefaults__
            result[name] = cloned
    source = inspect.getsource(previous.register)
    source = replace_once(source, 'sources = [Path(__file__), REPO / "tests/test_single_model_four_draw_audit_recovery.py"]',
        'paths.extend(arithmetic_predecessor.OUTPUT / name for name in '
        '("config.json", "registration.json", "status.json", "audit.log", "failed_original_status.json", "status_transition.json"))\n'
        '    paths.append(DIAGNOSIS)\n'
        '    sources = [Path(__file__), REPO / "tests/test_single_model_four_draw_audit_arithmetic.py"]')
    exec(compile(source, __file__, "exec"), result)
    return result


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("The arithmetic recovery and diagnosis must remain CPU-only")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("diagnose", "register", "pipeline", "audit"), required=True)
    if parser.parse_args().stage == "diagnose":
        diagnose()
    else:
        with arithmetic.original_arithmetic():
            namespace()["main"]()


if __name__ == "__main__":
    main()
