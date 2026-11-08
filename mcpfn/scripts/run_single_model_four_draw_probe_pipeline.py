"""Execute the one registered probe and its audits, never retrying scientific stages."""

import os
from pathlib import Path
import subprocess
import sys
import time

import torch

import preflight_single_model_four_draw as probe
from recsys_evaluation_common import sha256, write_json


OUTPUT = probe.OUTPUT.with_name(probe.OUTPUT.name + "_pipeline")
MEAN = probe.OUTPUT.with_name(probe.OUTPUT.name + "_mean_audit")


def summarize(result, audit, retention, transfer, mean):
    if result.get("data_admission_only"):
        if (result["accepted_updates"] != 0 or result["optimizer_proposal_performed"] is not False
                or not audit["complete"] or not retention["complete"] or transfer is not None or mean is not None
                or audit["data_admission_only"] is not True or retention["data_admission_only"] is not True
                or audit["data_admission"]["eligible"] is not False
                or audit["data_admission"]["structured_draws"] != 48
                or audit["data_admission"]["deferred_task_indices"] != [0, 13, 14, 27, 28, 41, 42, 55]):
            raise ValueError("A data-only rejection must remain completely audited and untrained")
        return {"feasibility_passed": False, "reason": "data_admission_rejected", "mean_audit_evaluated": False}
    record = result["record"]
    roles = probe.endpoint_roles(record, mean is not None)
    if (not audit["complete"] or audit["training_tasks"] != 56 or audit["prediction_grids"] != 56 * len(roles)
            or not retention["complete"] or retention["eligibility_decisions"] != 56
            or retention["reference_grids"] != 52 or retention["component_grids"] != 100
            or retention["component_role_counts"] != {"v8": 48, "v29": 52}
            or retention["reference_role_counts"] != {"v29": 4, "structured_mean": 48}
            or retention["endpoint_roles"] != list(roles)
            or not transfer["complete"] or transfer["transfer_sha256"] != result["gradient_transfer_sha256"]
            or transfer["raw_sha256"] != result["raw_sha256"]):
        raise ValueError("Retain the full captured task, endpoint, component and transfer audits")
    if mean is not None and (not mean["complete"] or mean["tasks"] != 56 or mean["imputation_reweighted_tasks"] != 52
            or mean["native_reweighted_tasks"] != 4 or mean["structured_reweighted_tasks"] != 48
            or mean["unchanged_rating_tasks"] != 4 or mean["cuda_initialized"] is not False):
        raise ValueError("Require every independent teacher-mean contribution without selecting favorable columns")
    allocated = result["peak_gib"] * 1024**3
    reserved = result["peak_reserved_gib"] * 1024**3
    memory_passed = (result["allocator_fraction"] == .9 and 0 < allocated <= reserved <= .9 * result["cuda_total_bytes"]
        and 0 < result["peak_host_rss_bytes"] <= probe.HOST_WORKING_BYTES)
    return {"feasibility_passed": bool(record["accepted"] and memory_passed and mean is not None),
        "reason": record["reason"], "accepted_updates": record["accepted_updates"],
        "memory_within_registered_allowances": memory_passed, "peak_gib": result["peak_gib"],
        "peak_reserved_gib": result["peak_reserved_gib"], "peak_host_rss_bytes": result["peak_host_rss_bytes"],
        "mean_audit_evaluated": mean is not None, "prediction_grids": audit["prediction_grids"]}


def status(state, **fields):
    write_json(OUTPUT / "status.json", {"state": state, "pid": os.getpid(), "updated_unix": time.time(), **fields})


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("The coordinator must remain CUDA-hidden")
    config = probe.verify(probe.OUTPUT)
    if OUTPUT.exists() or OUTPUT.is_symlink() or probe.read(probe.OUTPUT / "status.json")["state"] != "registered":
        raise FileExistsError("Use only the original unstarted registration; never retry a scientific attempt")
    probe.require_space()
    probe.require_predecessor()
    config_hash = sha256(probe.OUTPUT / "config.json")
    commands = [(stage, [sys.executable, str(Path(probe.__file__)), "--stage", stage], visible)
        for stage, visible in (("run", "0"), ("replay", "0"), ("audit", ""))]
    commands.append(("mean_audit", [sys.executable, str(probe.REPO / "scripts/audit_single_model_four_draw_probe.py"),
        "--config-sha256", config_hash], ""))
    OUTPUT.mkdir()
    write_json(OUTPUT / "config.json", {"probe_config_sha256": config_hash, "commands": commands,
        "source_sha256": config["source_sha256"], "scientific_retries_allowed": False,
        "full_trial_authorized": False, "goal_success_not_assessed": True})
    outcomes = []
    try:
        for stage, command, visible in commands:
            if stage == "mean_audit" and not (probe.OUTPUT / "candidate_predictions.json").exists():
                outcomes.append({"stage": stage, "evaluated": False, "reason": "no_measured_candidate_endpoint"})
                continue
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": visible}
            with (OUTPUT / f"{stage}.log").open("x") as log:
                process = subprocess.Popen(command, cwd=probe.REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
                status("running", stage=stage, child_pid=process.pid)
                print(f"{stage}: pid={process.pid}", flush=True)
                code = process.wait()
            outcomes.append({"stage": stage, "evaluated": True, "returncode": code,
                "log_sha256": sha256(OUTPUT / f"{stage}.log")})
            if code:
                raise RuntimeError(f"Original {stage} exited {code}; do not retry")
        result, audit, retention = [probe.read(probe.OUTPUT / name) for name in ("result.json", "audit.json", "retention_audit.json")]
        gate = probe.read(probe.OUTPUT / "gate.json")
        accepted = False if result.get("data_admission_only") else result["record"]["accepted"]
        if (gate["passed"] is not accepted or gate["audit_sha256"] != sha256(probe.OUTPUT / "audit.json")
                or gate["real_benchmark_success"] is not False):
            raise ValueError("Bind the original finite decision to its independently audited negative or positive feasibility gate")
        transfer = None if result.get("data_admission_only") else probe.read(probe.OUTPUT / "gradient_transfer_audit.json")
        mean = probe.read(MEAN / "audit.json") if MEAN.exists() else None
        report = summarize(result, audit, retention, transfer, mean)
        paths = [probe.OUTPUT / name for name in ("config.json", "result.json", "audit.json", "retention_audit.json", "gate.json")]
        if transfer is not None:
            paths.append(probe.OUTPUT / "gradient_transfer_audit.json")
        if mean is not None:
            paths.append(MEAN / "audit.json")
        write_json(OUTPUT / "result.json", {"complete": True, **report, "stages": outcomes,
            "artifact_sha256": {str(path): sha256(path) for path in paths}, "full_trial_authorized": False,
            "scientific_stages_repeated": False, "goal_success_not_assessed": True})
        status("complete", feasibility_passed=report["feasibility_passed"])
        print(report, flush=True)
    except BaseException as error:
        status("failed", error=repr(error), completed_stages=outcomes)
        raise


if __name__ == "__main__":
    main()
