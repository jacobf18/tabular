"""Attribute V46's failed confirmation without repeating its selected-model evaluation."""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import types

import torch

import diagnose_single_model_legacy_blend_initializer as blueprint
import run_single_model_stronger_mean_pipeline as original
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from train_streamed_large_matrices import wait_for_gpu


training = original.training
mixture = training.original.alignment.mixture
fixed = training.original.alignment.fixed
OUTPUT = original.ROOT / "stronger_mean_distillation_v46_initializer_diagnosis_v1"
SCRATCH = training.SCRATCH / "initializer_diagnosis_v1"
PARENT_SHA256 = "9747c69624220de33356145c5df9baa5256bc625719868d4f0e3740bce475c32"
CLOSURE = original.ROOT / "stronger_mean_distillation_v46_live_replay_continuation_v1/result.json"
CLOSURE_SHA256 = "a770350950579119058fb2c463c05801e209bc8eb25eba2732417803e95de9d2"
CONFIRMATION_GATE_SHA256 = "bd14d54664ae071731633ff2f09b035821cf382ac6822dc382919bf870be3777"
CONFIRMATION_AUDIT_SHA256 = "217b726a4ea82320a83597282590b4813d49b8cf409ba33552ceca42ff107811"
TRAINING_AUDIT = original.ROOT / "stronger_mean_distillation_v46_recovered_audit_v1/audit.json"
TRAINING_AUDIT_SHA256 = "343707444737561b5edf451b2b71c00e76dcfe5407464f299b92b3f75c711310"
RECOVERIES = tuple(original.ROOT / f"stronger_mean_distillation_v46_{suffix}" for suffix in (
    "cuda_entry_recovery_v1", "qualification_recovery_v1", "idle_owner_recovery_v1", "live_replay_continuation_v1"))
WRITE_BUDGETS = {"checkpoint": 1024**2, "root": 16 * 1024**2, "scratch": 1024**3}
POLICY = (
    "Post-hoc synthetic diagnosis after the fully audited negative V46 step10 confirmation. "
    "Evaluate only the unchanged actual V29 initializer on all284 already-consumed contexts, "
    "with identical full matrices, masks, queries, normalization and inference. Never regenerate "
    "episodes, train, reselect, reevaluate V46 or its teachers, repeat confirmation or publish a "
    "candidate. Independently CPU-rescore all284 new full grids and attribute all32 original "
    "retention comparisons, including22 against V8 and10 against the general parent. Preserve "
    "all four original failures, successful cohorts and both regression directions. Keep the "
    "complete audit and every operational failure/registration immutable. Verification of closed "
    "receipts is pure CPU and must not invoke launcher-only CUDA visibility checks. Use24 BLAS/4 "
    "Torch threads,18/6/50 GiB reserves and1 GiB scratch allowance. These consumed synthetic data "
    "may inform a hypothesis but cannot count as fresh qualification or establish benchmark parity. "
    "No real benchmark evaluation, inference ensemble or goal-success claim."
)


def require_closed_failure():
    original.verify(original.OUTPUT)
    for directory in RECOVERIES:
        config = original.read(directory / "config.json")
        if original.read(directory / "registration.json")["config_sha256"] != sha256(directory / "config.json"):
            raise ValueError("Preserve every original operational recovery registration")
        verify_hashes(config["source_sha256"])
        verify_hashes(config["input_sha256"])
    verify_hashes({str(training.PARENT): PARENT_SHA256, str(CLOSURE): CLOSURE_SHA256,
                   str(original.CONFIRMATION / "gate.json"): CONFIRMATION_GATE_SHA256,
                   str(original.CONFIRMATION / "audit.json"): CONFIRMATION_AUDIT_SHA256,
                   str(TRAINING_AUDIT): TRAINING_AUDIT_SHA256})
    gate = original.read(original.CONFIRMATION / "gate.json")
    audit = original.read(original.CONFIRMATION / "audit.json")
    closure = original.read(CLOSURE)
    if (gate["passed"] is not False or gate["cases"] != 284 or gate["selected_step"] != 10
            or audit["complete"] is not True or audit["prediction_grids"] != 1136 or audit["cases"] != 284
            or audit["cuda_initialized"] is not False or closure["complete"] is not True or closure["gate"] != gate
            or closure["training_repeated"] is not False or closure["scientific_stages_repeated"] is not False
            or closure["cuda_initialized"] is not False):
        raise ValueError("Require the completely audited negative V46 and no repeated scientific stage")
    for directory in (CLOSURE.parent, original.CONFIRMATION, training.OUTPUT):
        current = original.read(directory / "status.json")
        expected = "training_complete" if directory == training.OUTPUT else "complete"
        if current["state"] != expected or original.process_identity(current["pid"]) is not None:
            raise ValueError("Require all original scientific producers to be terminal")
    if original.CAMPAIGN.exists() or original.REVIEW.exists() or (training.OUTPUT / "best_validation_checkpoint.pth").exists():
        raise ValueError("Keep the original negative outcome without publication or a real campaign")


for name in ("require_storage", "verify", "require_full_case_identity", "require_frozen_state", "audit"):
    function = getattr(blueprint, name)
    if function.__closure__ is not None:
        raise ValueError("Diagnostic functions must not retain historical closures")
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)

function = blueprint.run.__wrapped__
if function.__closure__ is not None:
    raise ValueError("Bind the unchanged inference procedure to this registered diagnosis")
run = torch.inference_mode()(types.FunctionType(function.__code__, globals(), "run", function.__defaults__))


def register():
    if any(path.exists() or path.is_symlink() for path in (OUTPUT, SCRATCH)):
        raise FileExistsError("Never repeat a begun consumed-bank initializer diagnosis")
    require_closed_failure()
    require_storage(False)
    inputs = [training.PARENT, training.OUTPUT / "config.json", CLOSURE, TRAINING_AUDIT]
    inputs.extend(original.CONFIRMATION / name for name in ("manifest.json", "synthetic_bank.pt", "reference.json",
        "initialization.json", "student.json", "audit.json", "gate.json", "status.json"))
    inputs.extend(directory / name for directory in RECOVERIES for name in ("config.json", "registration.json", "status.json"))
    sources = [Path(__file__).resolve(), Path(blueprint.__file__).resolve(),
               REPO / "tests/test_single_model_stronger_mean_initializer_diagnosis.py"]
    config = {"policy": POLICY, "cases": 284, "checkpoint": str(training.PARENT), "checkpoint_sha256": PARENT_SHA256,
        "bank": str(original.CONFIRMATION / "synthetic_bank.pt"), "scratch": str(SCRATCH),
        "write_budget_bytes": WRITE_BUDGETS, "reserves": original.diagnostics.RESERVES,
        "input_sha256": {str(path): sha256(path) for path in inputs},
        "source_sha256": {str(path): sha256(path) for path in sources}}
    SCRATCH.mkdir()
    OUTPUT.symlink_to(SCRATCH, target_is_directory=True)
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json"),
        "model_inference_started": False, "confirmation_repeated": False})
    original.status(OUTPUT, "registered")


def comparisons(initializer, candidate, baseline, base):
    general = original.read(original.CONFIRMATION / "reference.json")["general"]
    roles = {role: {(group, metric): value for group, metric, value in fixed.retention_values(metrics, base)}
             for role, metrics in (("V29", initializer), ("V46", candidate), ("V8", baseline), ("general", general))}
    if len(roles["V29"]) != 22 or any(set(values) != set(roles["V29"]) for values in roles.values()):
        raise ValueError("Preserve every original22 retention metric in every model role")
    rows = []
    for reference in ("V8", "general"):
        for group, metric in roles["V29"]:
            if reference == "general" and group not in {"imputation", "recsys", "structured_overall"}:
                continue
            key = group, metric
            parent_value, value, target = roles["V29"][key], roles["V46"][key], roles[reference][key]
            rows.append({"reference": reference, "group": group, "metric": metric,
                "V29_initializer": parent_value, "V46_step10": value, "reference_value": target,
                "V46_minus_V29": value - parent_value, "V29_minus_reference": parent_value - target,
                "V46_minus_reference": value - target, "initializer_failed": parent_value > target,
                "candidate_failed": value > target})
    if len(rows) != 32:
        raise ValueError("Attribute all32 original comparisons, not only the four failed checks")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("register", "run", "audit"))
    stage = parser.parse_args().stage
    with original.arithmetic_recovery.previous.original_arithmetic():
        if stage == "register":
            register()
            return
        verify()
        with (OUTPUT / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                {"run": run, "audit": audit}[stage]()
            except BaseException as error:
                original.status(OUTPUT, "failed", error=repr(error))
                raise


if __name__ == "__main__":
    main()
