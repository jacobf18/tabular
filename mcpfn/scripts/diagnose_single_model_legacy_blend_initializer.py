"""Measure inherited versus V41-induced drift on the already-consumed confirmation bank."""

import argparse
import fcntl
import os
from pathlib import Path
import shutil

import torch

import recover_single_model_legacy_blend_blas_stable as recovery
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import REPO, sha256, write_json
from train_streamed_large_matrices import wait_for_gpu


original = recovery.previous.original
training = original.training
mixture = training.original.alignment.mixture
fixed = training.original.alignment.fixed
OUTPUT = original.ROOT / "legacy_blend_teacher_v41_initializer_diagnosis_v1"
SCRATCH = training.SCRATCH / "initializer_diagnosis_v1"
PARENT_SHA256 = "9747c69624220de33356145c5df9baa5256bc625719868d4f0e3740bce475c32"
CONFIRMATION_GATE_SHA256 = "2d1d4bb7b5aafb3af03371abe832dffb2844ebb7479246c7edc4eb0c2936b443"
CLOSURE = original.ROOT / "legacy_blend_teacher_v41_closure_audit_v1/audit.json"
CLOSURE_SHA256 = "d300c76d007eabded3318c8a9b80c488d71cbe0b141091d724b60a1dd85cf916"
WRITE_BUDGETS = {"checkpoint": 1024**2, "root": 16 * 1024**2, "scratch": 1024**3}
POLICY = ("Post-hoc synthetic diagnosis only, after V41's completed failed confirmation. Evaluate only the "
    "unchanged V29 initializer on all284 already-consumed confirmation contexts, using the original saved "
    "normalization, masks, full matrices, inference and metric procedures. Do not regenerate episodes, "
    "change seeds, train, reselect, reevaluate V41 or either original teacher, repeat confirmation, publish "
    "a candidate, run real benchmarks or relax the five failed comparisons. Independently CPU-rescore all284 "
    "new full prediction grids and compare all original22 retention metrics with the saved V41 and V8 roles. "
    "Report inherited and newly introduced regressions, including adverse results outside the failing cohorts. "
    "These consumed data may inform a future hypothesis but cannot count as fresh qualification. Preserve all "
    "original and recovery bytes,18/6/50 GiB reserves and the declared1 GiB scratch allowance. Scratch is not "
    "provider-persistent storage. This diagnosis does not establish goal success.")


def require_closed_failure():
    recovery.namespace()["verify"](recovery.OUTPUT)
    verify_hashes({str(training.PARENT): PARENT_SHA256,
        str(original.CONFIRMATION / "gate.json"): CONFIRMATION_GATE_SHA256, str(CLOSURE): CLOSURE_SHA256})
    gate = original.read(original.CONFIRMATION / "gate.json")
    closure = original.read(CLOSURE)
    if (gate["passed"] is not False or gate["cases"] != 284 or gate["selected_step"] != 50
            or not closure["complete"] or closure["prediction_grids"] != 1136
            or len(closure["comparisons"]) != 32 or len(closure["failures"]) != 5
            or closure["passed_original_confirmation"] is not False):
        raise ValueError("Require the complete original five-failure confirmation, never a partial result")
    for directory in (recovery.OUTPUT, original.CONFIRMATION):
        state = original.read(directory / "status.json")
        if state["state"] != "complete" or original.process_identity(state["pid"]) is not None:
            raise ValueError("Original scientific producers must be terminal before post-hoc inference")
    if original.CAMPAIGN.exists() or original.REVIEW.exists() or (training.OUTPUT / "best_validation_checkpoint.pth").exists():
        raise ValueError("Do not confuse an unqualified diagnostic with a publication or benchmark campaign")


def require_storage(prepared):
    original.diagnostics.storage.require_mount()
    if (SCRATCH.parent.resolve() != SCRATCH.parent or not SCRATCH.parent.is_dir()
            or SCRATCH.parent.stat().st_dev != original.diagnostics.storage.SCRATCH.stat().st_dev):
        raise ValueError("Keep diagnostic predictions on the original scratch filesystem")
    if prepared and (not OUTPUT.is_symlink() or OUTPUT.readlink() != SCRATCH or not SCRATCH.is_dir()
            or SCRATCH.is_symlink() or SCRATCH.resolve() != SCRATCH):
        raise ValueError("Preserve the declared canonical diagnostic location")
    volumes = {"checkpoint": original.ROOT, "root": Path("/"), "scratch": SCRATCH.parent}
    for name, path in volumes.items():
        if shutil.disk_usage(path).free < original.diagnostics.RESERVES[name] + WRITE_BUDGETS[name]:
            raise RuntimeError(f"Insufficient diagnostic space on {name}")


def register():
    if any(path.exists() or path.is_symlink() for path in (OUTPUT, SCRATCH)):
        raise FileExistsError("Never repeat a registered post-hoc diagnostic")
    require_closed_failure()
    require_storage(False)
    inputs = [training.PARENT, training.OUTPUT / "config.json", CLOSURE, recovery.OUTPUT / "result.json"]
    inputs.extend(original.CONFIRMATION / name for name in ("manifest.json", "synthetic_bank.pt", "reference.json",
        "initialization.json", "student.json", "audit.json", "gate.json", "status.json"))
    config = {"policy": POLICY, "cases": 284, "checkpoint": str(training.PARENT), "checkpoint_sha256": PARENT_SHA256,
        "bank": str(original.CONFIRMATION / "synthetic_bank.pt"), "scratch": str(SCRATCH),
        "write_budget_bytes": WRITE_BUDGETS, "reserves": original.diagnostics.RESERVES,
        "input_sha256": {str(path): sha256(path) for path in inputs},
        "source_sha256": {str(path): sha256(path) for path in (Path(__file__).resolve(),
            REPO / "tests/test_single_model_legacy_blend_initializer_diagnosis.py")}}
    SCRATCH.mkdir()
    OUTPUT.symlink_to(SCRATCH, target_is_directory=True)
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "registration.json", {"config_sha256": sha256(OUTPUT / "config.json"),
        "model_inference_started": False, "confirmation_repeated": False})
    original.status(OUTPUT, "registered")


def verify():
    config = original.read(OUTPUT / "config.json")
    if (original.read(OUTPUT / "registration.json")["config_sha256"] != sha256(OUTPUT / "config.json")
            or config["policy"] != POLICY or config["cases"] != 284 or config["checkpoint"] != str(training.PARENT)
            or config["checkpoint_sha256"] != PARENT_SHA256
            or config["bank"] != str(original.CONFIRMATION / "synthetic_bank.pt") or config["scratch"] != str(SCRATCH)
            or config["write_budget_bytes"] != WRITE_BUDGETS or config["reserves"] != original.diagnostics.RESERVES):
        raise ValueError("Preserve the single frozen initializer and complete consumed-bank scope")
    verify_hashes(config["input_sha256"])
    verify_hashes(config["source_sha256"])
    require_closed_failure()
    require_storage(True)
    return config


def require_full_case_identity(cases, baseline):
    if len(cases) != len(baseline) or len(cases) != 284:
        raise ValueError("Require all284 original cases without a favorable subset")
    fields = ("index", "seed", "rows", "columns", "targets", "domain", "prior", "component", "family", "structured")
    for index, (current, previous) in enumerate(zip(cases, baseline)):
        if current["index"] != index or any(current.get(name) != previous.get(name) for name in fields):
            raise ValueError("Preserve every original case identity, full shape and query population")


def require_frozen_state(model, state):
    current = model.state_dict()
    if set(current) != set(state) or any(not torch.equal(current[name].detach().cpu(), value) for name, value in state.items()):
        raise ValueError("The initializer must remain bitwise unchanged, including every buffer")


@torch.inference_mode()
def run():
    config = verify()
    if original.read(OUTPUT / "status.json")["state"] != "registered":
        raise ValueError("Never repeat a begun diagnostic inference")
    wait_for_gpu(OUTPUT, True, False)
    original.status(OUTPUT, "evaluating_frozen_initializer")
    bank = torch.load(config["bank"], map_location="cpu", weights_only=True)
    if len(bank) != 284:
        raise ValueError("Require the entire original confirmation bank")
    base = original.read(training.OUTPUT / "config.json")["base"]
    model, optimizer, criterion, teachers, sampler = mixture.original_setup(base)
    del optimizer, teachers, sampler
    parent = torch.load(training.PARENT, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(parent["model"], strict=True)
    model.eval().requires_grad_(False)
    require_frozen_state(model, parent["model"])
    with mixture.runtime(base, OUTPUT):
        metrics, cases = mixture.backend.evaluate_bank(model, criterion, bank)
    require_full_case_identity(cases, original.read(original.CONFIRMATION / "student.json")["cases"])
    require_frozen_state(model, parent["model"])
    verify_hashes(config["input_sha256"])
    write_json(OUTPUT / "initializer.json", {"student": metrics, "cases": cases, "checkpoint_sha256": PARENT_SHA256,
        "model_state_unchanged": True, "training_performed": False, "confirmation_repeated": False})
    original.status(OUTPUT, "awaiting_independent_audit")


def comparisons(initializer, candidate, baseline, base):
    roles = {role: {(group, metric): value for group, metric, value in fixed.retention_values(metrics, base)}
        for role, metrics in (("initializer", initializer), ("candidate", candidate), ("baseline", baseline))}
    if any(set(values) != set(roles["initializer"]) for values in roles.values()) or len(roles["initializer"]) != 22:
        raise ValueError("Compare all original22 metrics, not only failed families")
    return [{"group": group, "metric": metric, "V29_initializer": roles["initializer"][group, metric],
        "V41_step50": roles["candidate"][group, metric], "V8_retention_baseline": roles["baseline"][group, metric],
        "V41_minus_V29": roles["candidate"][group, metric] - roles["initializer"][group, metric],
        "V29_minus_V8": roles["initializer"][group, metric] - roles["baseline"][group, metric],
        "V41_minus_V8": roles["candidate"][group, metric] - roles["baseline"][group, metric]}
        for group, metric in roles["initializer"]]


def audit():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("Independently audit the diagnostic with CUDA hidden and uninitialized")
    config = verify()
    if original.read(OUTPUT / "status.json")["state"] != "awaiting_independent_audit" or (OUTPUT / "audit.json").exists():
        raise ValueError("Audit only the completed new inference once")
    bank = torch.load(config["bank"], map_location="cpu", weights_only=True)
    report = original.read(OUTPUT / "initializer.json")
    candidate = original.read(original.CONFIRMATION / "student.json")
    baseline = original.read(original.CONFIRMATION / "initialization.json")
    require_full_case_identity(report["cases"], candidate["cases"])
    if report["checkpoint_sha256"] != PARENT_SHA256 or report["model_state_unchanged"] is not True:
        raise ValueError("Bind the unchanged original V29 initializer")
    count, hashes = fixed.raw_audit(bank, [(report, ("student",))])
    if count != 284 or mixture.aggregate(report["cases"], "student") != report["student"]:
        raise ValueError("Every full initializer grid and aggregate must independently reproduce")
    rows = comparisons(report["student"], candidate["student"], baseline["student"],
        original.read(training.OUTPUT / "config.json")["base"])
    hashes.update(config["input_sha256"])
    hashes.update({str(OUTPUT / name): sha256(OUTPUT / name) for name in ("config.json", "initializer.json")})
    verify_hashes(hashes)
    write_json(OUTPUT / "audit.json", {"complete": True, "policy": POLICY, "prediction_grids": count,
        "comparisons": rows, "artifact_sha256": hashes, "cuda_initialized": False,
        "post_hoc_consumed_bank": True, "checkpoint_selection_performed": False, "confirmation_repeated": False,
        "training_performed": False, "real_benchmarks_evaluated": False, "goal_success_not_assessed": True})
    original.status(OUTPUT, "complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("register", "run", "audit"))
    stage = parser.parse_args().stage
    with recovery.previous.original_arithmetic():
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
