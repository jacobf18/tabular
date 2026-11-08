"""Recover only the CPU audit's inherited two-draw admission cardinalities."""

import argparse
import fcntl
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import torch

import preflight_single_model_four_draw as probe
import recover_single_model_four_draw_replay as previous
import run_single_model_four_draw_probe_pipeline as original_pipeline
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


OUTPUT = probe.OUTPUT.with_name(probe.OUTPUT.name + "_audit_recovery_v1")
REPLACEMENTS = (
    ('saved["task_count"] != 28', 'saved["task_count"] != 56'),
    ('saved["structured_draws"] != 24', 'saved["structured_draws"] != 48'),
    ('saved["deferred_task_indices"] != [0, 13, 14, 27]',
        'saved["deferred_task_indices"] != [0, 13, 14, 27, 28, 41, 42, 55]'),
)
POLICY = (
    "CPU-only administrative audit recovery. Change exactly the three inherited two-draw "
    "admission cardinalities to the already registered56-task/48-structured/eight-deferred "
    "four-draw schedule. Preserve exact independently regenerated admission equality, hash, "
    "distinct process identity and every captured-episode digest check. Rebind only the four "
    "audit-chain wrappers in a separate namespace; never edit registered predecessors. Preserve "
    "both failed coordinators, original logs and a byte-identical snapshot of the failed audit "
    "status before an explicit recovery transition. Reuse the completed bitwise neural teacher "
    "and reference replays; do not rerun inference, training, sampling for training or any "
    "scientific attempt. Execute the original raw/endpoint/retention/transfer and mean audits "
    "without weakening arithmetic or finite guards. No new seed, selectable checkpoint, full "
    "trial authorization or goal success. Keep every original source and scientific artifact "
    "immutable and retain the18/6/50 GiB reserves. Fail closed rather than retry this pipeline."
)


def corrected_source():
    source = inspect.getsource(probe.audit_admission)
    for before, after in REPLACEMENTS:
        source = replace_once(source, before, after)
    return source


def audit_namespace():
    namespace = dict(vars(probe))
    exec(compile(corrected_source(), __file__, "exec"), namespace)
    for name in ("audit_admitted", "audit_original", "retention_audit", "audit"):
        function = getattr(probe, name)
        if function.__closure__ is not None:
            raise ValueError("Never retain a predecessor audit closure")
        cloned = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
        cloned.__kwdefaults__ = function.__kwdefaults__
        namespace[name] = cloned
    return namespace


def validate_evidence(failed, state, teacher, retention, result):
    if (failed["state"] != "failed"
            or [(row["stage"], row["returncode"]) for row in failed["completed_stages"]] != [("replay", 0), ("audit", 1)]
            or state["state"] != "failed" or state["failed_stage"] != "audit"
            or state["error"] != "ValueError('Independently reproduce the complete registered admission attempt')"
            or teacher["complete"] is not True or len(teacher["rows"]) != 56
            or teacher["full_grid_bitwise_replay"] is not True
            or teacher["blend_tasks"] != 52 or teacher["component_grids"] != 104
            or teacher["student_or_validation_evaluated"] is not False
            or retention["complete"] is not True or len(retention["records"]) != 52
            or retention["component_grids"] != 100 or retention["component_role_counts"] != {"v8": 48, "v29": 52}
            or result.get("data_admission_only") is not False or result["record"]["accepted"] is not True
            or result["record"]["accepted_updates"] != 1 or result["record"]["composite_tasks"] != 56
            or result["record"]["protected_tasks"] != 167):
        raise ValueError("Recover only the diagnosed CPU count failure after complete independent neural replay")


def require_unentered():
    for name in ("audit.json", "retention_audit.json", "gate.json", "gradient_transfer_audit.json"):
        path = probe.OUTPUT / name
        if path.exists() or path.is_symlink():
            raise FileExistsError("Never replace a completed or partly completed CPU audit")
    if original_pipeline.MEAN.exists() or original_pipeline.MEAN.is_symlink():
        raise FileExistsError("Never repeat the independent mean audit")


def require_closed():
    original = previous.verify()
    failed = probe.read(previous.OUTPUT / "status.json")
    state = probe.read(probe.OUTPUT / "status.json")
    teacher = probe.read(probe.OUTPUT / "teacher_replay/result.json")
    retention = probe.read(probe.OUTPUT / "retention_replay/manifest.json")
    result = probe.read(probe.OUTPUT / "result.json")
    validate_evidence(failed, state, teacher, retention, result)
    for owner in (failed["pid"], state["pid"], teacher["pid"], retention["pid"], result["training_pid"]):
        if type(owner) is not int or owner <= 0 or Path(f"/proc/{owner}").exists():
            raise ValueError("Original scientific workers and both failed recovery workers must be stopped")
    for row in failed["completed_stages"]:
        if row["log_sha256"] != sha256(previous.OUTPUT / f'{row["stage"]}.log'):
            raise ValueError("Preserve the completed replay and failed audit logs")
    require_unentered()
    audit_namespace()
    return original


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def register():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("Never overwrite an audit recovery registration")
    original = require_closed()
    probe.require_space()
    failed_bytes = (probe.OUTPUT / "status.json").read_bytes()
    paths = [previous.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "audit.log", "replay.log")]
    paths.extend(probe.OUTPUT / name for name in ("config.json", "registration.json", "result.json", "data_admission.json",
        "gradient_transfer.json", "training_batch/manifest.json", "teacher_replay/result.json", "retention_replay/manifest.json",
        "retention_targets/manifest.json", "before_predictions.json", "candidate_predictions.json"))
    sources = [Path(__file__), REPO / "tests/test_single_model_four_draw_audit_recovery.py"]
    config = {"policy": POLICY, "registered_unix": time.time(), "original_config_sha256": sha256(probe.OUTPUT / "config.json"),
        "corrected_admission_source_sha256": hashlib.sha256(corrected_source().encode()).hexdigest(),
        "cardinality_replacements": [list(pair) for pair in REPLACEMENTS],
        "failed_status_sha256": hashlib.sha256(failed_bytes).hexdigest(),
        "input_sha256": {str(path): sha256(path) for path in paths},
        "source_sha256": {**original["source_sha256"], **{str(path): sha256(path) for path in sources}},
        "training_repeated": False, "neural_replay_repeated": False, "scientific_seeds_added": False,
        "full_trial_authorized": False}
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    OUTPUT.mkdir()
    with (OUTPUT / "failed_original_status.json").open("xb") as snapshot:
        snapshot.write(failed_bytes)
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json")})
    status("registered")


def verify():
    config = probe.read(OUTPUT / "config.json")
    if (sha256(OUTPUT / "config.json") != probe.read(OUTPUT / "registration.json")["config_sha256"]
            or config["policy"] != POLICY or config["cardinality_replacements"] != [list(pair) for pair in REPLACEMENTS]
            or config["corrected_admission_source_sha256"] != hashlib.sha256(corrected_source().encode()).hexdigest()
            or config["original_config_sha256"] != sha256(probe.OUTPUT / "config.json")
            or config["failed_status_sha256"] != sha256(OUTPUT / "failed_original_status.json")
            or any(config[name] is not False for name in ("training_repeated", "neural_replay_repeated",
                "scientific_seeds_added", "full_trial_authorized"))):
        raise ValueError("Never alter the registered CPU-only audit recovery")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    previous.verify()
    return config


def audit():
    config = verify()
    state = probe.read(OUTPUT / "status.json")
    if state["state"] != "running" or state["stage"] != "audit":
        raise ValueError("Enter this audit only through its one-shot coordinator")
    with (probe.OUTPUT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_closed()
        if sha256(probe.OUTPUT / "status.json") != config["failed_status_sha256"]:
            raise ValueError("Require the exact preserved failed audit status before transitioning")
        write_json(OUTPUT / "status_transition.json", {"from_sha256": config["failed_status_sha256"],
            "to_state": "awaiting_independent_audit", "recovery_config_sha256": sha256(OUTPUT / "config.json"),
            "pid": os.getpid(), "original_failure_preserved": True})
        probe.status(probe.OUTPUT, "awaiting_independent_audit", recovery=str(OUTPUT), failed_status_sha256=config["failed_status_sha256"])
        try:
            audit_namespace()["audit"](probe.OUTPUT)
        except BaseException as error:
            probe.status(probe.OUTPUT, "failed", failed_stage="audit_recovery", error=repr(error), recovery=str(OUTPUT))
            raise


def pipeline():
    config = verify()
    if probe.read(OUTPUT / "status.json")["state"] != "registered":
        raise ValueError("Never restart this audit recovery pipeline")
    require_closed()
    probe.require_space()
    commands = [("audit", [sys.executable, __file__, "--stage", "audit"]),
        ("mean_audit", [sys.executable, str(REPO / "scripts/audit_single_model_four_draw_probe.py"),
            "--config-sha256", config["original_config_sha256"]])]
    outcomes = []
    try:
        for stage, command in commands:
            status("running", stage=stage)
            with (OUTPUT / f"{stage}.log").open("x") as log:
                process = subprocess.Popen(command, cwd=REPO, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                    stdout=log, stderr=subprocess.STDOUT)
                status("running", stage=stage, child_pid=process.pid)
                print(f"{stage}: pid={process.pid}", flush=True)
                code = process.wait()
            outcomes.append({"stage": stage, "returncode": code, "log_sha256": sha256(OUTPUT / f"{stage}.log")})
            if code:
                raise RuntimeError(f"CPU-only recovery {stage} exited {code}; do not retry")
        result, admitted, retention, transfer = [probe.read(probe.OUTPUT / name) for name in (
            "result.json", "audit.json", "retention_audit.json", "gradient_transfer_audit.json")]
        mean = probe.read(original_pipeline.MEAN / "audit.json")
        report = original_pipeline.summarize(result, admitted, retention, transfer, mean)
        gate = probe.read(probe.OUTPUT / "gate.json")
        if (gate["passed"] is not result["record"]["accepted"] or gate["real_benchmark_success"] is not False
                or gate["audit_sha256"] != sha256(probe.OUTPUT / "audit.json")):
            raise ValueError("Require the unchanged finite decision and independent audit binding")
        verify()
        paths = [probe.OUTPUT / name for name in ("result.json", "audit.json", "retention_audit.json", "gate.json",
            "gradient_transfer_audit.json", "teacher_replay/result.json", "retention_replay/manifest.json")]
        paths.extend((original_pipeline.MEAN / "audit.json", OUTPUT / "failed_original_status.json", OUTPUT / "status_transition.json"))
        write_json(OUTPUT / "result.json", {"complete": True, **report, "stages": outcomes,
            "artifact_sha256": {str(path): sha256(path) for path in paths}, "both_original_failures_preserved": True,
            "training_repeated": False, "neural_replay_repeated": False, "full_trial_authorized": False,
            "goal_success_not_assessed": True})
        status("complete", feasibility_passed=report["feasibility_passed"])
        print(report, flush=True)
    except BaseException as error:
        status("failed", error=repr(error), completed_stages=outcomes)
        raise


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Every recovery entry point must remain CPU-only")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("register", "pipeline", "audit"), required=True)
    stage = parser.parse_args().stage
    if stage == "pipeline":
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pipeline()
    else:
        {"register": register, "audit": audit}[stage]()


if __name__ == "__main__":
    main()
