"""Attribute V48's failed confirmation using only its unchanged V29 initializer."""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import types

import torch

import diagnose_single_model_stronger_mean_initializer as predecessor
import diagnose_single_model_legacy_blend_initializer as blueprint
import run_single_model_native_column_mean_pipeline as original
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from train_streamed_large_matrices import wait_for_gpu


training = original.training
mixture = training.original.alignment.mixture
fixed = training.original.alignment.fixed
OUTPUT = original.ROOT / "native_column_mean_v48_initializer_diagnosis_v1"
SCRATCH = training.SCRATCH / "initializer_diagnosis_v1"
PARENT_SHA256 = predecessor.PARENT_SHA256
SELECTED_SHA256 = "7e8b6d6c6b29bb339744ecd62f9d080a2c70de7e0c82c0ca7fa181bcc8f88b54"
CLOSURE = original.OUTPUT / "result.json"
CLOSURE_SHA256 = "8c7742bfadb54a27d5fe1681a125a1faeb2dec482a2f49e9e60714d0b2a47b7c"
TRAINING_AUDIT = original.ROOT / "native_column_mean_v48_audit/audit.json"
TRAINING_AUDIT_SHA256 = "9b1c91649d171d1a79c38c49d7b4e416db8a43d5a88fb9128e0938a578fafef5"
REPLAY = original.ROOT / "native_column_mean_v48_endpoint_replay"
REPLAY_AUDIT_SHA256 = "a0054e6290eb152933a90b05f918df0ba1ee6b64338b1fd9cc461413ca7933e8"
REPLAY_GATE_SHA256 = "820dc1b436bba15ad7755e42c2747e4d984faf2716094db4948d29a6e08e1be2"
CONFIRMATION_GATE_SHA256 = "d62175eaacb476ef3f9f31a9e04741c68743b5f33c4c3f862ebd71408adf9dc1"
CONFIRMATION_AUDIT_SHA256 = "4aa6289010d3dd0e8c367554b1292dfe709bf6c054ac909e14b8ca6a1fd9d699"
WRITE_BUDGETS = dict(predecessor.WRITE_BUDGETS)
POLICY = (
    "Post-hoc synthetic attribution after V48's fully audited failed step10 confirmation. "
    "Evaluate only the unchanged actual V29 initializer on all284 already-consumed contexts, "
    "with identical full matrices, masks, queries, normalization, inference and original metrics. "
    "Preserve both orientations, all cohorts, all32 retention comparisons and every adverse "
    "direction. Never regenerate episodes, train, reselect, reevaluate V48 or any teacher, repeat "
    "confirmation, relax a guard, publish a candidate or run a real benchmark. Independently "
    "CPU-rescore all284 new full grids and compare V29/V48 against both V8 and general. Preserve "
    "the original four failed comparisons and complete negative closure, including the fact "
    "that historical full-autograd/optimizer-trajectory replay was not performed. Verification "
    "is CUDA-hidden CPU work; only the separate initializer inference process owns CUDA. Use "
    "24 BLAS/4 Torch threads, unchanged18/6/50 GiB reserves and1 GiB scratch allowance. Consumed "
    "bank attribution is descriptive, not fresh qualification, causal evidence for a proposed "
    "training change or real-benchmark success. No new training recipe or seed is authorized."
)


def validate_closed_evidence(gate, audit, closure, training_audit, replay_gate, replay_audit):
    if (gate["passed"] is not False or gate["cases"] != 284 or gate["selected_step"] != 10
            or gate["checkpoint_sha256"] != SELECTED_SHA256 or gate["real_benchmark_success"] is not False
            or gate["confirmation_evaluated"] is not True
            or audit["complete"] is not True or audit["prediction_grids"] != 1136 or audit["cases"] != 284
            or audit["cuda_initialized"] is not False or closure["complete"] is not True or closure["gate"] != gate
            or closure["campaign"] is not None or closure["review"] is not None
            or closure["no_inference_ensemble"] is not True or closure["goal_success_not_assessed"] is not True
            or closure["training_audit_sha256"] != TRAINING_AUDIT_SHA256
            or training_audit["complete"] is not True or training_audit["complete_training_audit"] is not True
            or training_audit["through_step"] != 64 or training_audit["attempted_batches"] != 64
            or training_audit["accepted_updates"] != 56 or training_audit["training_episodes"] != 1792
            or training_audit["full_network_training"] is not True
            or training_audit["full_optimizer_trajectory_replayed"] is not False
            or replay_gate["endpoint_reproduced"] is not True
            or replay_gate["selected_validation_passed"] is not True or replay_gate["selected_step"] != 10
            or replay_gate["audit_sha256"] != REPLAY_AUDIT_SHA256
            or replay_audit["complete"] is not True or replay_audit["endpoint_grids"] != 56
            or replay_audit["selected_grids"] != 152 or replay_audit["cuda_initialized"] is not False
            or gate["training_audit_sha256"] != TRAINING_AUDIT_SHA256
            or gate["replay_audit_sha256"] != REPLAY_AUDIT_SHA256
            or gate["audit_sha256"] != CONFIRMATION_AUDIT_SHA256):
        raise ValueError("Require V48's complete unchanged negative closure and original audit scope")


def require_closed_failure():
    original.verify(original.OUTPUT)
    verify_hashes({str(training.PARENT): PARENT_SHA256, str(CLOSURE): CLOSURE_SHA256,
        str(TRAINING_AUDIT): TRAINING_AUDIT_SHA256, str(REPLAY / "audit.json"): REPLAY_AUDIT_SHA256,
        str(REPLAY / "gate.json"): REPLAY_GATE_SHA256,
        str(original.CONFIRMATION / "gate.json"): CONFIRMATION_GATE_SHA256,
        str(original.CONFIRMATION / "audit.json"): CONFIRMATION_AUDIT_SHA256})
    validate_closed_evidence(*(original.read(path) for path in (
        original.CONFIRMATION / "gate.json", original.CONFIRMATION / "audit.json", CLOSURE,
        TRAINING_AUDIT, REPLAY / "gate.json", REPLAY / "audit.json")))
    for directory in (original.OUTPUT, original.CONFIRMATION, training.OUTPUT, REPLAY):
        current = original.read(directory / "status.json")
        expected = "training_complete" if directory == training.OUTPUT else "complete"
        if current["state"] != expected or original.process_identity(current["pid"]) is not None:
            raise ValueError("Require all original scientific producers to be terminal")
    if (original.read(original.OUTPUT / "status.json")["confirmed"] is not False
            or original.read(original.CONFIRMATION / "status.json")["passed"] is not False
            or original.CAMPAIGN.exists() or original.REVIEW.exists()
            or (training.OUTPUT / "best_validation_checkpoint.pth").exists()):
        raise ValueError("Keep the negative outcome without publication or real evaluation")


for name in ("require_storage", "verify", "require_full_case_identity", "require_frozen_state", "audit"):
    function = getattr(blueprint, name)
    if function.__closure__ is not None:
        raise ValueError("Diagnostic functions must not retain historical closures")
    globals()[name] = types.FunctionType(function.__code__, globals(), name, function.__defaults__)

function = blueprint.run.__wrapped__
if function.__closure__ is not None:
    raise ValueError("Bind the unchanged inference procedure to this diagnosis")
run = torch.inference_mode()(types.FunctionType(function.__code__, globals(), "run", function.__defaults__))


def register():
    if any(path.exists() or path.is_symlink() for path in (OUTPUT, SCRATCH)):
        raise FileExistsError("Never repeat a begun consumed-bank initializer diagnosis")
    require_closed_failure()
    require_storage(False)
    inputs = [training.PARENT, training.OUTPUT / "config.json", training.OUTPUT / "status.json", CLOSURE, TRAINING_AUDIT]
    inputs.extend(original.CONFIRMATION / name for name in ("manifest.json", "synthetic_bank.pt", "reference.json",
        "initialization.json", "student.json", "audit.json", "gate.json", "status.json"))
    inputs.extend(original.OUTPUT / name for name in ("config.json", "registration.json", "status.json"))
    inputs.extend(REPLAY / name for name in ("config.json", "registration.json", "audit.json", "gate.json", "status.json"))
    sources = [Path(__file__).resolve(), Path(blueprint.__file__).resolve(), Path(predecessor.__file__).resolve(),
        REPO / "tests/test_single_model_native_column_initializer_diagnosis.py"]
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
        for role, metrics in (("V29", initializer), ("V48", candidate), ("V8", baseline), ("general", general))}
    if len(roles["V29"]) != 22 or any(set(values) != set(roles["V29"]) for values in roles.values()):
        raise ValueError("Preserve all22 original retention metrics in every model role")
    rows = []
    for reference in ("V8", "general"):
        for group, metric in roles["V29"]:
            if reference == "general" and group not in {"imputation", "recsys", "structured_overall"}:
                continue
            key = group, metric
            parent_value, value, target = roles["V29"][key], roles["V48"][key], roles[reference][key]
            rows.append({"reference": reference, "group": group, "metric": metric,
                "V29_initializer": parent_value, "V48_step10": value, "reference_value": target,
                "V48_minus_V29": value - parent_value, "V29_minus_reference": parent_value - target,
                "V48_minus_reference": value - target, "initializer_failed": parent_value > target,
                "candidate_failed": value > target})
    if len(rows) != 32:
        raise ValueError("Attribute all32 comparisons, not just the four failed checks")
    return rows


main = types.FunctionType(predecessor.main.__code__, globals(), "main", predecessor.main.__defaults__)


if __name__ == "__main__":
    main()
