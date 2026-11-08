"""Finish the unentered supplementary audit with the registered100 reference components."""

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

import audit_single_model_four_draw_probe as mean
import recover_single_model_four_draw_audit_arithmetic as previous
import run_single_model_four_draw_probe_pipeline as pipeline_blueprint
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from single_model_registered_adaptation import replace_once


probe = mean.probe
OUTPUT = probe.OUTPUT.with_name(probe.OUTPUT.name + "_mean_recovery_v1")
BEFORE = 'probe.read(probe.OUTPUT / "retention_audit.json")["component_grids"] != 50'
AFTER = 'probe.read(probe.OUTPUT / "retention_audit.json")["component_grids"] != 100'
POLICY = (
    "Supplementary CPU mean-audit entry recovery only. The independent full admission, "
    "teacher, raw optimizer,112 endpoint grids,52 reference/100 component grids and exact "
    "FP32 transfer audits have completed. Preserve all their original bytes and every "
    "earlier pipeline failure. The original supplementary entry still required50 retention "
    "components although the registered56-task schedule and completed audit contain100. "
    "Change exactly that one prerequisite constant in an isolated namespace. Preserve "
    "every source identity, task, endpoint, column, original independent arithmetic and "
    "adverse mean contribution. Enter the previously unentered descriptive audit once; "
    "do not rerun training, sampling, neural replay or raw/endpoint audits. No new seed, "
    "checkpoint promotion, full-trial authorization or goal success. Keep the original "
    "supplementary capacity check and CPU-only execution, with exclusive coordinator lock."
)


def corrected_source():
    return replace_once(mean.AUDIT_SOURCE, BEFORE, AFTER)


def namespace(config_sha256):
    result = {**vars(mean), "__file__": __file__, "CONFIG_SHA256": config_sha256}
    exec(compile(corrected_source(), __file__, "exec"), result)
    return result


def validate_completed(result, audit, retention, transfer):
    if (result.get("data_admission_only") is not False or result["record"]["accepted"] is not True
            or result["record"]["accepted_updates"] != 1 or result["record"]["protected_tasks"] != 167
            or audit["complete"] is not True or audit["training_tasks"] != 56 or audit["prediction_grids"] != 112
            or audit["raw_endpoint"]["gradient_vectors"] != 167 or audit["raw_endpoint"]["accepted"] is not True
            or audit["raw_endpoint"]["accepted_updates"] != 1 or retention["complete"] is not True
            or retention["eligibility_decisions"] != 56 or retention["reference_grids"] != 52
            or retention["component_grids"] != 100 or retention["component_role_counts"] != {"v8": 48, "v29": 52}
            or retention["reference_role_counts"] != {"v29": 4, "structured_mean": 48}
            or retention["endpoint_roles"] != ["before", "candidate"] or transfer["complete"] is not True
            or transfer["transfer_sha256"] != result["gradient_transfer_sha256"]
            or transfer["raw_sha256"] != result["raw_sha256"]):
        raise ValueError("Preserve all completed raw, endpoint, reference and transfer audits before the mean recheck")


def require_closed():
    config = previous.namespace()["verify"]()
    failed = probe.read(previous.OUTPUT / "status.json")
    state = probe.read(probe.OUTPUT / "status.json")
    if (failed["state"] != "failed"
            or [(row["stage"], row["returncode"]) for row in failed["completed_stages"]] != [("audit", 0), ("mean_audit", 1)]
            or state["state"] != "complete" or state.get("retention_audited") is not True):
        raise ValueError("Require completed independent audits followed by the preserved unentered mean-audit failure")
    for owner in (failed["pid"], state["pid"]):
        if type(owner) is not int or owner <= 0 or Path(f"/proc/{owner}").exists():
            raise ValueError("Never overlap the completed audit or its failed coordinator")
    for row in failed["completed_stages"]:
        if row["log_sha256"] != sha256(previous.OUTPUT / f'{row["stage"]}.log'):
            raise ValueError("Preserve both original audit logs")
    if "Require the original complete independent probe audits" not in (previous.OUTPUT / "mean_audit.log").read_text():
        raise ValueError("Bind the original supplementary prerequisite failure")
    if mean.OUTPUT.exists() or mean.OUTPUT.is_symlink():
        raise FileExistsError("Never repeat a partly entered supplementary audit")
    result, audit, retention, transfer = [probe.read(probe.OUTPUT / name) for name in (
        "result.json", "audit.json", "retention_audit.json", "gradient_transfer_audit.json")]
    validate_completed(result, audit, retention, transfer)
    gate = probe.read(probe.OUTPUT / "gate.json")
    if (gate["passed"] is not True or gate["real_benchmark_success"] is not False
            or gate["audit_sha256"] != sha256(probe.OUTPUT / "audit.json")):
        raise ValueError("Require the accepted original optimizer decision bound to its completed audit")
    return config


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def register():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("Never overwrite the mean-entry recovery")
    original = require_closed()
    paths = [previous.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "audit.log", "mean_audit.log")]
    paths.extend(probe.OUTPUT / name for name in ("config.json", "registration.json", "status.json", "result.json", "audit.json",
        "retention_audit.json", "gradient_transfer_audit.json", "gate.json", "teacher_replay/result.json", "retention_replay/manifest.json"))
    sources = (Path(__file__), REPO / "tests/test_single_model_four_draw_mean_recovery.py", Path(mean.__file__))
    config = {"policy": POLICY, "registered_unix": time.time(), "original_config_sha256": sha256(probe.OUTPUT / "config.json"),
        "corrected_source_sha256": hashlib.sha256(corrected_source().encode()).hexdigest(), "replacement": [BEFORE, AFTER],
        "input_sha256": {str(path): sha256(path) for path in paths},
        "source_sha256": {**original["source_sha256"], **{str(path): sha256(path) for path in sources}},
        "training_repeated": False, "neural_replay_repeated": False, "optimizer_audit_repeated": False,
        "scientific_seeds_added": False, "full_trial_authorized": False}
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json")})
    status("registered")


def verify():
    config = probe.read(OUTPUT / "config.json")
    if (sha256(OUTPUT / "config.json") != probe.read(OUTPUT / "registration.json")["config_sha256"]
            or config["policy"] != POLICY or config["replacement"] != [BEFORE, AFTER]
            or config["corrected_source_sha256"] != hashlib.sha256(corrected_source().encode()).hexdigest()
            or config["original_config_sha256"] != sha256(probe.OUTPUT / "config.json")
            or any(config[name] is not False for name in ("training_repeated", "neural_replay_repeated", "optimizer_audit_repeated",
                "scientific_seeds_added", "full_trial_authorized"))):
        raise ValueError("Preserve the exact one-constant supplementary recovery")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    previous.namespace()["verify"]()
    return config


def audit():
    config = verify()
    state = probe.read(OUTPUT / "status.json")
    if state["state"] != "running" or state["stage"] != "mean_audit":
        raise ValueError("Enter the supplementary audit only from its original one-shot coordinator")
    require_closed()
    with (probe.OUTPUT / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        namespace(config["original_config_sha256"])["main"]()


def pipeline():
    verify()
    if probe.read(OUTPUT / "status.json")["state"] != "registered":
        raise ValueError("Never restart the supplementary recovery")
    require_closed()
    outcomes = []
    try:
        status("running", stage="mean_audit")
        with (OUTPUT / "mean_audit.log").open("x") as log:
            process = subprocess.Popen([sys.executable, __file__, "--stage", "audit"], cwd=REPO,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, stdout=log, stderr=subprocess.STDOUT)
            status("running", stage="mean_audit", child_pid=process.pid)
            print(f"mean_audit: pid={process.pid}", flush=True)
            code = process.wait()
        outcomes.append({"stage": "mean_audit", "returncode": code, "log_sha256": sha256(OUTPUT / "mean_audit.log")})
        if code:
            raise RuntimeError(f"Supplementary recovery exited {code}; do not retry")
        result, admitted, retention, transfer = [probe.read(probe.OUTPUT / name) for name in (
            "result.json", "audit.json", "retention_audit.json", "gradient_transfer_audit.json")]
        report = pipeline_blueprint.summarize(result, admitted, retention, transfer, probe.read(mean.OUTPUT / "audit.json"))
        verify()
        paths = [probe.OUTPUT / name for name in ("result.json", "audit.json", "retention_audit.json", "gradient_transfer_audit.json", "gate.json")]
        paths.extend(mean.OUTPUT / name for name in ("config.json", "audit.json", "tasks.csv", "columns.csv"))
        write_json(OUTPUT / "result.json", {"complete": True, **report, "stages": outcomes,
            "artifact_sha256": {str(path): sha256(path) for path in paths}, "all_original_failures_preserved": True,
            "training_repeated": False, "neural_replay_repeated": False, "optimizer_audit_repeated": False,
            "full_trial_authorized": False, "goal_success_not_assessed": True})
        status("complete", feasibility_passed=report["feasibility_passed"])
        print(report, flush=True)
    except BaseException as error:
        status("failed", error=repr(error), completed_stages=outcomes)
        raise


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Keep every supplementary recovery stage CPU-only")
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
