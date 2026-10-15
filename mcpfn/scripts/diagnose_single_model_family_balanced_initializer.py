"""Attribute V43's failed synthetic retention to inherited versus newly introduced drift."""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import types

import torch

import diagnose_single_model_legacy_blend_initializer as blueprint
import recover_single_model_family_balanced_orphaned_chain as recovery
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from train_streamed_large_matrices import wait_for_gpu


original = recovery.original
training = original.training
mixture = training.original.alignment.mixture
fixed = training.original.alignment.fixed
OUTPUT = original.ROOT / "family_balanced_retention_v43_initializer_diagnosis_v1"
SCRATCH = training.SCRATCH / "initializer_diagnosis_v1"
PARENT_SHA256 = "9747c69624220de33356145c5df9baa5256bc625719868d4f0e3740bce475c32"
CONFIRMATION_GATE_SHA256 = "2cd03d3a11b59c10fd38ad562b468ed0597df82ac1678cf55c14048fb8b78ad5"
CLOSURE = recovery.OUTPUT / "result.json"
CLOSURE_SHA256 = "24db7b0d712033174d05443eb29e860f99c7b949e08534f980528a2d8e29ffd3"
CONFIRMATION_AUDIT_SHA256 = "30d75cfbbf2bc904625d80cbaed717db01e4d92e074d30354b548da112ce3c44"
WRITE_BUDGETS = {"checkpoint": 1024**2, "root": 16 * 1024**2, "scratch": 1024**3}
POLICY = ("Post-hoc synthetic diagnosis only, after V43's fully audited failed confirmation. Evaluate only "
    "the unchanged V29 initializer on every one of the284 already-consumed synthetic contexts. Keep original "
    "full matrices, observations, normalizations, masks, queries, inference and metrics; do not regenerate "
    "data, change seeds, train, reselect, reevaluate V43 or its teachers, or repeat confirmation. Independently "
    "CPU-rescore all284 new full prediction grids and compare all22 original retention metrics to saved V43 "
    "step25 and V8 results, including every adverse cohort. Preserve the complete original six-failure gate "
    "and operational recovery, all registered bytes,24 BLAS/4 Torch threads,18/6/50 GiB reserves and1 GiB "
    "scratch allowance. Consumed data may inform a later hypothesis but cannot count as fresh qualification. "
    "No selected or qualified checkpoint is published, no real benchmark runs, and no goal success is implied.")


def require_closed_failure():
    recovery.verify()
    verify_hashes({str(training.PARENT): PARENT_SHA256,
        str(original.CONFIRMATION / "gate.json"): CONFIRMATION_GATE_SHA256,
        str(original.CONFIRMATION / "audit.json"): CONFIRMATION_AUDIT_SHA256,
        str(CLOSURE): CLOSURE_SHA256})
    gate, closure = original.read(original.CONFIRMATION / "gate.json"), original.read(CLOSURE)
    audit = original.read(original.CONFIRMATION / "audit.json")
    if (gate["passed"] is not False or gate["cases"] != 284 or gate["selected_step"] != 25
            or closure["complete"] is not True or closure["gate"] != gate
            or audit["complete"] is not True or audit["prediction_grids"] != 1136 or audit["cases"] != 284
            or audit["cuda_initialized"] is not False or closure["training_repeated"] is not False
            or closure["scientific_stages_repeated"] is not False):
        raise ValueError("Require the fully audited original negative V43 closure")
    for directory in (recovery.OUTPUT, original.CONFIRMATION):
        current = original.read(directory / "status.json")
        if current["state"] != "complete" or original.process_identity(current["pid"]) is not None:
            raise ValueError("Require every original qualification producer to be terminal")
    if original.CAMPAIGN.exists() or original.REVIEW.exists() or (training.OUTPUT / "best_validation_checkpoint.pth").exists():
        raise ValueError("Keep the original negative result without qualification or a real campaign")


for name in ("require_storage", "verify", "require_full_case_identity", "require_frozen_state", "audit"):
    function = getattr(blueprint, name)
    if function.__closure__ is not None:
        raise ValueError("Reused diagnostic functions must not retain predecessor closures")
    globals()[name] = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)

function = blueprint.run.__wrapped__
if function.__closure__ is not None:
    raise ValueError("The initializer inference must bind only this registered diagnostic")
run = torch.inference_mode()(types.FunctionType(function.__code__, globals(), "run", function.__defaults__))


def register():
    if any(path.exists() or path.is_symlink() for path in (OUTPUT, SCRATCH)):
        raise FileExistsError("Never repeat a begun consumed-bank initializer diagnosis")
    require_closed_failure()
    require_storage(False)
    inputs = [training.PARENT, training.OUTPUT / "config.json", CLOSURE, recovery.OUTPUT / "config.json"]
    inputs.extend(original.CONFIRMATION / name for name in ("manifest.json", "synthetic_bank.pt", "reference.json",
        "initialization.json", "student.json", "audit.json", "gate.json", "status.json"))
    sources = [Path(__file__).resolve(), Path(blueprint.__file__).resolve(), Path(recovery.__file__).resolve(),
               REPO / "tests/test_single_model_family_balanced_initializer_diagnosis.py"]
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
    roles = {role: {(group, metric): value for group, metric, value in fixed.retention_values(metrics, base)}
        for role, metrics in (("initializer", initializer), ("candidate", candidate), ("baseline", baseline))}
    if any(set(values) != set(roles["initializer"]) for values in roles.values()) or len(roles["initializer"]) != 22:
        raise ValueError("Compare all original22 metrics without selecting favorable cohorts")
    return [{"group": group, "metric": metric, "V29_initializer": roles["initializer"][group, metric],
        "V43_step25": roles["candidate"][group, metric], "V8_retention_baseline": roles["baseline"][group, metric],
        "V43_minus_V29": roles["candidate"][group, metric] - roles["initializer"][group, metric],
        "V29_minus_V8": roles["initializer"][group, metric] - roles["baseline"][group, metric],
        "V43_minus_V8": roles["candidate"][group, metric] - roles["baseline"][group, metric]}
        for group, metric in roles["initializer"]]


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
