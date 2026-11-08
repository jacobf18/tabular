"""Repair only the unentered replay primitive after the original wiring failure."""

import argparse
import fcntl
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import torch

import preflight_single_model_four_draw as probe
import run_single_model_four_draw_probe_pipeline as original_pipeline
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


OUTPUT = probe.OUTPUT.with_name(probe.OUTPUT.name + "_replay_recovery_v1")
DIAGNOSIS = probe.ROOT / "four_draw_preflight_v1_replay_wiring_diagnosis.json"
POLICY = (
    "Administrative entry-point recovery only. Preserve the original completed56-task/167-gradient "
    "accepted optimizer attempt, every original task/component/grid/checkpoint and all registered "
    "sources. The original alias overwrote the registered replay_admitted primitive with the "
    "outer replay_targets wrapper, recursively verifying the configuration before any replay "
    "directory, model setup or teacher inference. Preserve the diagnosed SIGTERM and original "
    "coordinator failure. Recompile the already registered teacher replay source into a separate "
    "namespace, retain its declared replay_admitted name, and rebind only the two outer replay "
    "wrappers to that namespace. Never edit the registered predecessor, repeat training/admission, "
    "change a seed, parameter, task, arithmetic rule, finite guard or original audit. Perform the "
    "previously unentered complete teacher/reference replay once, then the original CPU audits "
    "and mean audit. This is not a scientific retry or authorization for a full training trial. "
    "Fail closed on any prior replay artifacts, live original owner or changed original evidence. "
    "Retain all original memory/disk reserves and the full unmet held-out/real-benchmark objective."
)


def replay_namespace():
    namespace = dict(vars(probe))
    exec(compile(probe.TEACHER_REPLAY_SOURCE, __file__, "exec"), namespace)
    for name in ("replay_original_teachers", "replay_cohort_targets"):
        function = getattr(probe, name)
        if function.__closure__ is not None:
            raise ValueError("Never preserve a historical wrapper closure")
        cloned = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
        cloned.__kwdefaults__ = function.__kwdefaults__
        namespace[name] = cloned
    primitive = inspect.unwrap(namespace["replay_admitted"])
    if (primitive.__code__.co_name != "replay_admitted" or primitive.__globals__ is not namespace
            or "replay_cohort_targets" in primitive.__code__.co_names
            or "original_setup" not in primitive.__code__.co_names):
        raise ValueError("Bind the original neural replay primitive, not another outer wrapper")
    return namespace


def validate_original(diagnosis, result, failed):
    stages = failed["completed_stages"]
    if (failed["state"] != "failed" or len(stages) != 2
            or [(row["stage"], row["returncode"]) for row in stages] != [("run", 0), ("replay", -15)]
            or diagnosis["pid"] != 257967 or diagnosis["parent_pid"] != failed["pid"]
            or diagnosis["teacher_replay_directory_exists"] is not False
            or diagnosis["retention_replay_directory_exists"] is not False
            or diagnosis["training_repeated"] is not False or diagnosis["registered_source_modified"] is not False
            or diagnosis["new_scientific_seed"] is not False
            or diagnosis["installed_replay_admitted_code_name"] != "replay_targets"
            or diagnosis["installed_replay_admitted_global_calls"] != ["replay_cohort_targets"]
            or diagnosis["registered_primitive_header"] != "def replay_admitted(output):"
            or result.get("data_admission_only") is not False or result["record"]["accepted"] is not True
            or result["record"]["accepted_updates"] != 1 or result["record"]["composite_tasks"] != 56
            or result["record"]["protected_tasks"] != 167 or result["record"]["finite"]["defined_slots"] != 34):
        raise ValueError("Recover only this diagnosed unentered replay, preserving its exact completed training result")


def require_original_closed():
    config = probe.verify(probe.OUTPUT)
    diagnosis = probe.read(DIAGNOSIS)
    result = probe.read(probe.OUTPUT / "result.json")
    failed = probe.read(original_pipeline.OUTPUT / "status.json")
    validate_original(diagnosis, result, failed)
    for owner in (failed["pid"], diagnosis["pid"], result["training_pid"]):
        if type(owner) is not int or owner <= 0 or Path(f"/proc/{owner}").exists():
            raise ValueError("The original training, broken replay and coordinator must all be stopped")
    for name in ("teacher_replay", "retention_replay", "audit.json", "retention_audit.json", "gate.json"):
        path = probe.OUTPUT / name
        if path.exists() or path.is_symlink():
            raise FileExistsError("Do not repeat or replace a previously entered scientific audit stage")
    if (probe.read(probe.OUTPUT / "status.json")["state"] != "awaiting_independent_audit"
            or diagnosis["configuration_sha256"] != sha256(probe.OUTPUT / "config.json")
            or diagnosis["training_result_sha256"] != sha256(probe.OUTPUT / "result.json")
            or diagnosis["registered_source_sha256"] != config["source_sha256"][str(Path(probe.__file__))]):
        raise ValueError("Preserve original scientific identities and the unentered-replay boundary")
    replay_namespace()
    return config


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def register():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("Never overwrite a recovery registration")
    original = require_original_closed()
    probe.require_space()
    paths = [DIAGNOSIS, original_pipeline.OUTPUT / "status.json", original_pipeline.OUTPUT / "config.json",
        original_pipeline.OUTPUT / "run.log", original_pipeline.OUTPUT / "replay.log"]
    paths.extend(probe.OUTPUT / name for name in ("config.json", "registration.json", "result.json",
        "gradient_transfer.json", "data_admission.json", "training_batch/manifest.json", "before_predictions.json",
        "candidate_predictions.json", "retention_targets/manifest.json"))
    sources = [Path(__file__), REPO / "tests/test_single_model_four_draw_replay_recovery.py"]
    config = {"policy": POLICY, "registered_unix": time.time(), "original_config_sha256": sha256(probe.OUTPUT / "config.json"),
        "teacher_replay_source_sha256": probe.TEACHER_REPLAY_SOURCE_SHA256,
        "original_training_state": probe.read(probe.OUTPUT / "status.json"),
        "input_sha256": {str(path): sha256(path) for path in paths},
        "source_sha256": {**original["source_sha256"], **{str(path): sha256(path) for path in sources}},
        "raw_update_sha256": probe.read(probe.OUTPUT / "result.json")["raw_sha256"],
        "scientific_seeds_added": False, "training_repeated": False, "full_trial_authorized": False}
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json")})
    status("registered")


def verify():
    config = probe.read(OUTPUT / "config.json")
    if (sha256(OUTPUT / "config.json") != probe.read(OUTPUT / "registration.json")["config_sha256"]
            or config["policy"] != POLICY or config["teacher_replay_source_sha256"] != probe.TEACHER_REPLAY_SOURCE_SHA256
            or config["original_config_sha256"] != sha256(probe.OUTPUT / "config.json")
            or config["training_repeated"] is not False or config["scientific_seeds_added"] is not False):
        raise ValueError("Never alter the registered entry-point-only recovery")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    probe.verify(probe.OUTPUT)
    return config


def replay():
    verify()
    state = probe.read(OUTPUT / "status.json")
    if state["state"] != "running" or state["stage"] != "replay":
        raise ValueError("Only the original recovery coordinator may enter replay once")
    require_original_closed()
    probe.require_space()
    with (probe.OUTPUT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with probe.diagnosis.original.arithmetic_recovery.previous.original_arithmetic():
            replay_namespace()["replay_cohort_targets"](probe.OUTPUT)


def pipeline():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Keep the recovery coordinator CUDA-hidden")
    config = verify()
    if probe.read(OUTPUT / "status.json")["state"] != "registered":
        raise ValueError("Never restart a recovery pipeline")
    require_original_closed()
    commands = [("replay", [sys.executable, __file__, "--stage", "replay"], "0"),
        ("audit", [sys.executable, str(Path(probe.__file__)), "--stage", "audit"], ""),
        ("mean_audit", [sys.executable, str(REPO / "scripts/audit_single_model_four_draw_probe.py"),
            "--config-sha256", config["original_config_sha256"]], "")]
    outcomes = []
    try:
        for stage, command, visible in commands:
            with (OUTPUT / f"{stage}.log").open("x") as log:
                process = subprocess.Popen(command, cwd=REPO, env={**os.environ, "CUDA_VISIBLE_DEVICES": visible},
                    stdout=log, stderr=subprocess.STDOUT)
                status("running", stage=stage, child_pid=process.pid)
                print(f"{stage}: pid={process.pid}", flush=True)
                code = process.wait()
            outcomes.append({"stage": stage, "returncode": code, "log_sha256": sha256(OUTPUT / f"{stage}.log")})
            if code:
                raise RuntimeError(f"The original recovery {stage} exited {code}; do not retry")
        result, audit, retention, transfer = [probe.read(probe.OUTPUT / name) for name in (
            "result.json", "audit.json", "retention_audit.json", "gradient_transfer_audit.json")]
        mean = probe.read(original_pipeline.MEAN / "audit.json")
        report = original_pipeline.summarize(result, audit, retention, transfer, mean)
        gate = probe.read(probe.OUTPUT / "gate.json")
        if gate["passed"] is not result["record"]["accepted"] or gate["audit_sha256"] != sha256(probe.OUTPUT / "audit.json"):
            raise ValueError("Bind the unchanged finite decision to the original independent audit")
        verify()
        paths = [probe.OUTPUT / name for name in ("result.json", "audit.json", "retention_audit.json", "gate.json",
            "gradient_transfer_audit.json")]
        paths.append(original_pipeline.MEAN / "audit.json")
        write_json(OUTPUT / "result.json", {"complete": True, **report, "stages": outcomes,
            "artifact_sha256": {str(path): sha256(path) for path in paths}, "original_pipeline_failure_preserved": True,
            "training_repeated": False, "independent_replay_stage_repeated": False,
            "full_trial_authorized": False, "goal_success_not_assessed": True})
        status("complete", feasibility_passed=report["feasibility_passed"])
        print(report, flush=True)
    except BaseException as error:
        status("failed", error=repr(error), completed_stages=outcomes)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "pipeline", "replay"), required=True)
    stage = parser.parse_args().stage
    if stage == "pipeline":
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pipeline()
    else:
        {"register": register, "replay": replay}[stage]()


if __name__ == "__main__":
    main()
