"""Preserve the failed decoder audit and handle nested report lists in a CPU-only recovery."""

from contextlib import contextmanager
import os
from pathlib import Path
import shutil

import torch

import check_single_model_decoder_finite_steps as finite
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json


SOURCE = finite.OUTPUT
OUTPUT = SOURCE.parent / "decoder_finite_steps_v1_audit_recovery1"
SCALAR_COMPARE = finite.probe.alignment.compare_metric_records
POLICY = ("Recover only the nested-list report comparison after the original CPU audit failed with a dict-subtraction "
          "TypeError. Preserve every original source, candidate, prediction, registration and failed log/status. "
          "Repeat the full84-grid CPU audit with identical scalar tolerances and zero-tolerance finite decisions. "
          "No neural replay, new data, optimizer update, checkpoint selection, guard relaxation or benchmark.")


def compare_nested(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("Nested report keys changed")
        for key in expected:
            compare_nested(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError("Nested report coverage changed")
        for observed, reference in zip(actual, expected):
            compare_nested(observed, reference)
    else:
        SCALAR_COMPARE(actual, expected)


@contextmanager
def nested_report_comparator():
    previous = finite.probe.alignment.compare_metric_records
    finite.probe.alignment.compare_metric_records = compare_nested
    try:
        yield
    finally:
        finite.probe.alignment.compare_metric_records = previous


def recover():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Recovery must be CPU only")
    if OUTPUT.exists():
        raise FileExistsError("Preserve the first recovery attempt")
    finite.verify(SOURCE)
    failed = finite.probe.read(SOURCE / "status.json")
    failure = (SOURCE / "audit.log").read_text()
    if failed["state"] != "failed" or "TypeError: unsupported operand type(s) for -: 'dict' and 'dict'" not in failure:
        raise ValueError("Recover only the observed nested-list audit failure")
    result = finite.probe.read(SOURCE / "result.json")
    if not result["complete"] or len(result["prediction_sha256"]) != 84 or (SOURCE / "audit.json").exists():
        raise ValueError("Require the complete original neural panel and its unfinished audit")
    inputs = {str(SOURCE / name): sha256(SOURCE / name) for name in
              ("config.json", "registration.json", "status.json", "audit.log", "result.json", "v8/cases.jsonl", "v8/summary.json")}
    inputs.update(result["prediction_sha256"])
    verify_hashes(inputs)
    sources = [Path(__file__).resolve(), REPO / "tests/test_single_model_decoder_audit_recovery.py"]
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT / "recovery_config.json", {"policy": POLICY, "input_sha256": inputs,
               "source_sha256": {str(path): sha256(path) for path in sources},
               "original_registration_sha256": sha256(SOURCE / "registration.json"), "neural_forward_performed": False})
    for name in ("config.json", "registration.json"):
        shutil.copy2(SOURCE / name, OUTPUT / name)
    for name in ("result.json", "v8"):
        (OUTPUT / name).symlink_to(SOURCE / name)
    finite.probe.status(OUTPUT, "awaiting_independent_audit", recovery_policy=POLICY)
    try:
        with nested_report_comparator():
            finite.audit(OUTPUT)
        verify_hashes(inputs)
        recovery = finite.probe.read(OUTPUT / "recovery_config.json")
        verify_hashes(recovery["source_sha256"])
        write_json(OUTPUT / "recovery.json", {"complete": True, "original_artifacts_unchanged": True,
                   "neural_evaluation_repeated": False, "prediction_grids": 84, "research_selection_tolerance": 0,
                   "recovery_config_sha256": sha256(OUTPUT / "recovery_config.json"), "audit_sha256": sha256(OUTPUT / "audit.json"),
                   "goal_success_not_assessed": True})
    except BaseException as error:
        finite.probe.status(OUTPUT, "failed", error=repr(error))
        raise


if __name__ == "__main__":
    recover()
