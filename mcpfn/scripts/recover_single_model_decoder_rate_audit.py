"""Audit the preserved rate probe without confusing reduction rounding with exact log identity."""

import inspect
import os
from pathlib import Path

import torch

from dwip_teacher_mean_training import verify_hashes
import probe_single_model_decoder_rates as probe
from recsys_evaluation_common import REPO, sha256, write_json


SOURCE = probe.OUTPUT
OUTPUT = SOURCE.parent / "decoder_rate_probe_v1_audit_recovery1"
POLICY = ("Preserve the failed original CPU audit and all registered neural evidence. Re-run only its complete "
          "CPU checks, comparing exact logged records to logged records, and independently remeasured scores "
          "with the existing 1e-12 scalar tolerances. Require exact per-metric improvement/retention signs and "
          "all strict finite acceptance decisions in both repetitions. Keep the original one-ULP projection "
          "allowance and optimizer checks. No new prediction, optimizer update, validation, selection, "
          "checkpoint nomination, guard relaxation or benchmark.")


def compare_repeated_decisions(actual, expected):
    probe.compare_nested(actual, expected)
    if actual is None:
        return
    for repeat, decision in expected["repetitions"].items():
        for name, change in decision["changes"].items():
            observed = actual["repetitions"][repeat]["changes"][name]
            if change is not None and ((observed < 0) != (change < 0) or (observed <= 0) != (change <= 0)):
                raise ValueError("Every metric must preserve its exact improvement and retention signs")


def adapt_audit_source(source):
    replacements = {
        'def audit(output):': 'def recovered_audit(output, recovery):',
        'read(output / "status.json")["state"] != "awaiting_independent_audit"':
            'read(output / "status.json")["state"] != "failed"',
        'checks.check_record(record, [task["weight"] for task in config["batch"]["tasks"]], before["0"], 42)':
            'checks.check_record(record, [task["weight"] for task in config["batch"]["tasks"]], result["before"]["0"], 42)',
        'if repeated != row["repeated"] or row["finite_feasible"] != bool(record["accepted"] and repeated and repeated["finite_feasible"]):':
            'compare_repeated_decisions(repeated, row["repeated"])\n        if row["finite_feasible"] != bool(record["accepted"] and repeated and repeated["finite_feasible"]):',
        'write_json(output / "audit.json",': 'write_json(recovery / "audit.json",',
        '(output / "RESULTS.md").write_text': '(recovery / "RESULTS.md").write_text',
        'status(output, "complete")': 'status(recovery, "complete")',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError("Adapt only the exact original audit expressions")
        source = source.replace(before, after)
    return source


def recovered_auditor():
    source = adapt_audit_source(inspect.getsource(probe.audit))
    namespace = {**probe.__dict__, "compare_repeated_decisions": compare_repeated_decisions}
    exec(compile(source, str(Path(__file__).resolve()), "exec"), namespace)
    return namespace["recovered_audit"]


def recover():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Recovery is a separate CPU-only audit")
    if OUTPUT.exists():
        raise FileExistsError("Preserve every previous recovery attempt")
    probe.verify(SOURCE)
    failed = probe.read(SOURCE / "status.json")
    failure = (SOURCE / "audit.log").read_text()
    message = "Complete gradient/metric coverage and the registered margin are mandatory"
    if failed["state"] != "failed" or message not in failure or message not in failed.get("error", ""):
        raise ValueError("Recover only the observed exact-record versus CPU-reduction audit failure")
    result = probe.read(SOURCE / "result.json")
    if not result["complete"] or (SOURCE / "audit.json").exists():
        raise ValueError("Require the original completed neural panel and unfinished audit")
    paths = [SOURCE / name for name in ("config.json", "registration.json", "status.json", "audit.log", "run.log", "result.json")]
    inputs = {str(path): sha256(path) for path in paths}
    for row in result["rates"]:
        inputs[row["endpoint"]] = row["endpoint_sha256"]
        directory = SOURCE / f"rate_{row['index']}"
        inputs[str(directory / "result.json")] = sha256(directory / "result.json")
    sources = [Path(__file__).resolve(), REPO / "tests/test_single_model_decoder_rate_audit_recovery.py"]
    config = {"policy": POLICY, "input_sha256": inputs,
              "source_sha256": {str(path): sha256(path) for path in sources},
              "original_registration_sha256": sha256(SOURCE / "registration.json"),
              "neural_evaluation_performed": False, "research_selection_tolerance": 0}
    verify_hashes(inputs)
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "recovery_config.json", config)
    probe.status(OUTPUT, "auditing")
    try:
        recovered_auditor()(SOURCE, OUTPUT)
        verify_hashes(inputs)
        verify_hashes(config["source_sha256"])
        probe.verify(SOURCE)
        audit = probe.read(OUTPUT / "audit.json")
        write_json(OUTPUT / "recovery.json", {"complete": True, "original_artifacts_unchanged": True,
                   "neural_evaluation_repeated": False, "prediction_grids": audit["prediction_grids"],
                   "research_selection_tolerance": 0, "per_metric_decision_signs_exact": True,
                   "recovery_config_sha256": sha256(OUTPUT / "recovery_config.json"),
                   "audit_sha256": sha256(OUTPUT / "audit.json"), "goal_success_not_assessed": True})
        probe.status(OUTPUT, "complete", original_failed_status_preserved=True)
    except BaseException as error:
        probe.status(OUTPUT, "failed", error=repr(error))
        raise


if __name__ == "__main__":
    recover()
