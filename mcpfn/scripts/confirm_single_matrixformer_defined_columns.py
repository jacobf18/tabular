"""Recover V8's fixed confirmation bank with an explicit truth-defined auxiliary guard."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

import confirm_single_matrixformer_query_budget as original
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from retry_single_model_retained_confirmation import FAILURE, require_terminal, verify_coverage_records
import single_model_truth_defined_columns as corrected
from train_single_matrixformer import storage_guard
from train_single_matrixformer_continuation import read, verify_run
from train_single_matrixformer_query_budget import validate_recipe
from train_streamed_large_matrices import wait_for_gpu


def register(run, failed, coverage, after, output):
    base = read(run / "config.json")
    verify_run(base, run)
    validate_recipe(base)
    status = require_terminal(run, "training_complete")
    failure = require_terminal(failed, "failed")
    if status["step"] != 400 or FAILURE not in failure.get("error", ""):
        raise ValueError("Require completed original V8 and its specific pre-student metric failure")
    if any((failed / name).exists() for name in ("gate.json", "student.json", "synthetic_bank.pt")):
        raise ValueError("Never rescreen a scored candidate or replace a bank")
    manifest = read(failed / "manifest.json")
    checkpoint = run / "best_validation_checkpoint.pth"
    if (manifest["checkpoint_sha256"] != sha256(checkpoint) or manifest["selected_step"] != status["best_step"]
            or manifest["cases"] != base["confirmation_cases"]):
        raise ValueError("The actual selected checkpoint and original 92 cases must remain unchanged")
    probe, summary = read(coverage / "config.json"), read(coverage / "summary.json")
    if (not summary["complete"] or summary["config_sha256"] != sha256(coverage / "config.json")
            or probe["banks"]["selected"] != base["confirmation_cases"] or Path(probe["producer"]).resolve() != run
            or summary["model_inference_performed"] or summary["case_replacement_performed"] or summary["gate_changed"]):
        raise ValueError("Correction requires unchanged model-free truth-coverage evidence")
    verify_coverage_records(probe, read(coverage / "cases.json"))
    verify_hashes(probe["input_sha256"])
    verify_hashes(probe["source_sha256"])
    after_config = read(after / "cache_launcher_config.json")
    if Path(after_config["run"]).resolve() != run.parent / "query_budget_v8_tail_average":
        raise ValueError("Require sequential scheduling after this producer's independent average campaign")
    artifacts = [run / name for name in ("config.json", "status.json", "progress_audit_400.json", "best_validation_checkpoint.pth")]
    artifacts += [failed / name for name in ("manifest.json", "status.json")]
    artifacts += [coverage / name for name in ("config.json", "summary.json", "cases.json", "empty_common_cases.json", "selected_case_6.pt")]
    artifacts.append(after / "cache_launcher_config.json")
    sources = {Path(__file__).resolve()}
    sources.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                   if getattr(module, "__file__", None) and str(Path(module.__file__).resolve()).startswith(str(REPO))
                   and str(module.__file__).endswith(".py") and Path(module.__file__).is_file())
    config = {"run": str(run), "failed_confirmation": str(failed), "after_campaign": str(after),
              "cases": base["confirmation_cases"], "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
              "selected_step": status["best_step"], "policy": corrected.POLICY,
              "fixed_truth_undefined_case_indices": [6], "canonical_imputation_cases": 60, "defined_common_guard_cases": 59,
              "training_performed": False, "checkpoint_reselection_performed": False,
              "canonical_metrics_or_real_protocol_changed": False,
              "auxiliary_guard_definition_clarified": True,
              "input_sha256": {str(path): sha256(path) for path in artifacts},
              "source_sha256": {**base["source_sha256"], **{str(path): sha256(path) for path in sources}}}
    write_json(output / "config.json", config)
    write_json(output / "registration.json", {"config_sha256": sha256(output / "config.json"),
                                              "registered_unix": time.time(), "student_evaluated": False})
    for path in sources:
        destination = output / "sources" / path.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return config


def verify(config, output):
    verify_hashes(config["source_sha256"])
    verify_hashes(config["input_sha256"])
    if sha256(output / "config.json") != read(output / "registration.json")["config_sha256"]:
        raise ValueError("Correction registration changed")


def campaign_terminal(after):
    status = read(after / "status.json")
    if status["state"] in {"complete", "not_promoted", "failed", "interrupted"}:
        try:
            require_terminal(after, status["state"])
        except RuntimeError:
            return False
        return True
    try:
        os.kill(status["pid"], 0)
        if Path(f"/proc/{status['pid']}/stat").read_text().split(") ", 1)[1].split()[0] == "Z":
            raise ProcessLookupError()
    except (ProcessLookupError, FileNotFoundError):
        raise RuntimeError("Average campaign handle disappeared without a terminal result; never restart it here") from None
    return False


def role_coverage(reference, initialization, student):
    reports = {"initialization": (initialization["cases"], "student", initialization["metrics"]),
               "student": (student["cases"], "student", student["metrics"]),
               "general": (reference["cases"], "general", reference["general"]),
               "teacher": (reference["cases"], "teacher", reference["teacher"])}
    for records, field, metrics in reports.values():
        if len(records) != 92 or [case["index"] for case in records] != list(range(92)):
            raise ValueError("Every role must retain the entire ordered bank")
        imputation = [case for case in records if case["domain"] == "imputation"]
        empty = [case["index"] for case in imputation if case[field]["column_coverage"]["common_nonconstant_columns"] == 0]
        if (len(imputation) != 60 or empty != [6] or metrics["imputation"]["canonical_metric_cases"] != 60
                or metrics["imputation"]["common_nrmse_cases"] != 59):
            raise ValueError("No role may drop cases or use prediction-dependent auxiliary coverage")
        for case in imputation:
            if (case[field]["common_nrmse"] is None) != (case["index"] == 6):
                raise ValueError("Only the fixed truth-undefined case may have an undefined guard")
    return {"roles": list(reports), "cases_per_role": 92, "canonical_imputation_cases_per_role": 60,
            "common_guard_cases_per_role": 59, "truth_undefined_case_indices": [6], "coverage_equal_across_all_roles": True}


def confirm(config, output):
    verify(config, output)
    with corrected.confirmation_metrics():
        original.confirm(Path(config["run"]), output)
    reference, initial, student = [read(output / name) for name in ("reference.json", "initialization.json", "student.json")]
    coverage = role_coverage(reference, initial, student)
    bank = torch.load(output / "synthetic_bank.pt", map_location="cpu", weights_only=True)
    empty = []
    for index, cached in enumerate(bank):
        if cached["metadata"]["seed"] != config["cases"][index]["seed"]:
            raise ValueError("Original bank seed changed")
        if cached["metadata"]["domain"] != "imputation":
            continue
        truth, mask = cached["target"][0].numpy(), cached["mask"][0].numpy()
        count = sum(float(np.std(truth[mask[:, column], column])) >= 1e-3
                    for column in range(truth.shape[1]) if mask[:, column].any())
        if count == 0:
            empty.append(index)
    if empty != [6]:
        raise ValueError("Actual evaluated bank differs from the model-free coverage diagnosis")
    gate = read(output / "gate.json")
    write_json(output / "gate_without_coverage_metadata.json", gate)
    gate.update(correction_config_sha256=sha256(output / "config.json"), auxiliary_guard_definition_clarified=True,
                canonical_metrics_or_real_protocol_changed=False, **coverage)
    verify(config, output)
    write_json(output / "coverage_audit.json", {"complete": True, **coverage,
        "raw_bank_sha256": sha256(output / "synthetic_bank.pt"), "raw_predictions_rescored": False,
        "policy": corrected.POLICY, "real_benchmark_success": False})
    write_json(output / "gate.json", gate)
    print(json.dumps({"corrected_gate": gate}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "failed-confirmation", "coverage", "after-campaign", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            torch.set_num_threads(4)
            storage_guard(output)
            write_json(output / "status.json", {"state": "registering", "pid": os.getpid(), "updated_unix": time.time()})
            config = register(arguments.run.resolve(), arguments.failed_confirmation.resolve(), arguments.coverage.resolve(),
                              arguments.after_campaign.resolve(), output)
            while not campaign_terminal(Path(config["after_campaign"])):
                write_json(output / "status.json", {"state": "waiting_for_average_campaign", "pid": os.getpid(), "updated_unix": time.time()})
                time.sleep(30)
            wait_for_gpu(output, True, False)
            confirm(config, output)
            write_json(output / "status.json", {"state": "complete", "pid": os.getpid(), "updated_unix": time.time()})
        except BaseException as error:
            write_json(output / "status.json", {"state": "failed", "error": repr(error), "pid": os.getpid(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    main()
