"""Recover CPU-auditor assumptions without modifying registered V12 training."""

from contextlib import contextmanager
import inspect
import os
from pathlib import Path
import signal
import sys
import time

import torch

import audit_single_model_family_projection_training as audit
from dwip_teacher_mean_training import verify_hashes
from recsys_evaluation_common import sha256, write_json
import run_single_model_family_projection_pipeline as pipeline


ORIGINAL_PIPELINE = pipeline.OUTPUT
OUTPUT = pipeline.ROOT / "family_projection_v12_pipeline_recovery1"
SOURCES = {
    str(Path(audit.__file__).resolve()): "68b24b26798d43f0002877425961c5d96e6e8b984b7ebf66b0e18e9c85b7c610",
    str(Path(pipeline.__file__).resolve()): "c85b9907e4b2c373546fd623109e4d81330b5b16a1b583ced1e029770043d9ab",
    str(Path(torch.__file__).parent / "include/ATen/native/cuda/fused_adam_utils.cuh"): "5ead25c9fbd21201cd78a89ff5a996dbd133e64e4a2da0f94c62af9421de0538",
}
INPUTS = {str(ORIGINAL_PIPELINE / "config.json"): "3895c6cc16174712a416f744334d6eeccca38e91cff8729d9e684fb6adac9d6a"}
OLD_PREDICATE = "not torch.isfinite(episode.source[~episode.mask]).all()"
NEW_PREDICATE = "(torch.isinf(episode.source).any() or not torch.isfinite(episode.source).any())"
REASON = "The scoring mask is a query subset, not the complement of all observed context. Legacy support and sparse rating episodes legitimately retain NaNs outside scored cells. Check source values are finite-or-NaN with some observed context; retain all finite targets, teachers, query coverage, provenance and all numerical/research tolerances."
OPTIMIZER_REASON = "The installed fused CUDA AdamW kernel casts betas, LR, decay and epsilon to FP32, uses nested FMA for its moments, and computes bias corrections in FP32. The first CPU audit incorrectly used decimal-double .001 instead of 1-FP32(.999), producing a 1.28746e-5 relative second-moment discrepancy. Reconstruct the actual kernel scalar and moment arithmetic without changing the four-epsilon rounding envelope or any other audit/selection tolerance."
PAIR_REASON = "The registered trainer restores identical cached inputs/teachers and seeds, but does not enable deterministic CUDA algorithms. The first paired forward losses and starting weights match exactly, while several backward vectors do not. Preserve exact starting-weight equality and all independent per-arm update audits; measure and disclose the actual gradient/proposal discrepancies instead of falsely requiring bitwise CUDA backward reproducibility. This changes an auditor assumption, not training, seeds, the 22 retention guards, minimax selection, confirmation or benchmarks. Small between-arm differences cannot be attributed solely to projection without replication."


def corrected_checker():
    verify_hashes(SOURCES)
    source = inspect.getsource(audit.check_episode)
    if source.count(OLD_PREDICATE) != 1:
        raise ValueError("Recover exactly one source-observation predicate, not another audit rule")
    namespace = dict(audit.__dict__)
    exec(compile(source.replace(OLD_PREDICATE, NEW_PREDICATE), str(Path(__file__).resolve()), "exec"), namespace)
    return namespace["check_episode"]


def fused_moment(moment, applied, beta, *, squared=False):
    gradient = applied.float().square().double() if squared else applied
    inner = ((1 - beta) * gradient).float().double()
    return (beta * moment + inner).float().double()


def fused_adaptive(state, group, step):
    beta1, beta2 = [torch.tensor(value, dtype=torch.float32) for value in group["betas"]]
    correction1 = 1 - beta1.pow(float(step))
    correction2_sqrt = (1 - beta2.pow(float(step))).sqrt()
    rate = torch.tensor(group["lr"], dtype=torch.float32) / correction1
    denominator = state["exp_avg_sq"].flatten().sqrt() / correction2_sqrt + torch.tensor(group["eps"], dtype=torch.float32)
    return float(rate) * state["exp_avg"].flatten().double() / denominator.double()


def corrected_optimizer():
    verify_hashes(SOURCES)
    original = audit.check_optimizer
    source = inspect.getsource(original)
    replacements = {
        "offset, maximum_error = 0, 0.": "beta1, beta2 = [float(torch.tensor(value, dtype=torch.float32)) for value in group['betas']]\n    offset, maximum_error = 0, 0.",
        "first_expected = .9 * first + .1 * applied": "first_expected = fused_moment(first, applied, beta1)",
        "second_expected = .999 * second + .001 * applied.square()": "second_expected = fused_moment(second, applied, beta2, squared=True)",
        "first_bound = 4 * epsilon * (.9 * first.abs() + .1 * applied.abs())": "first_bound = 4 * epsilon * (beta1 * first.abs() + (1 - beta1) * applied.abs())",
        "second_bound = 4 * epsilon * (.999 * second.abs() + .001 * applied.square())": "second_bound = 4 * epsilon * (beta2 * second.abs() + (1 - beta2) * applied.float().square().double())",
        "adaptive = group[\"lr\"] * (state[\"exp_avg\"].flatten().double() / (1 - .9 ** step)) / (\n            (state[\"exp_avg_sq\"].flatten().double() / (1 - .999 ** step)).sqrt() + 1e-8)": "adaptive = fused_adaptive(state, group, step)",
        "decay = group[\"lr\"] * weight_decay * original": "decay = float(torch.tensor(group['lr'], dtype=torch.float32) * torch.tensor(weight_decay, dtype=torch.float32)) * original",
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError("Correct only the exact registered optimizer arithmetic expressions")
        source = source.replace(before, after)
    namespace = {**audit.__dict__, "fused_moment": fused_moment, "fused_adaptive": fused_adaptive}
    exec(compile(source, str(Path(__file__).resolve()), "exec"), namespace)
    corrected = namespace["check_optimizer"]

    def check(snapshot, sizes, step, weight_decay):
        target = corrected if snapshot["optimizer_before"]["param_groups"][0].get("fused") else original
        return target(snapshot, sizes, step, weight_decay)

    return check


def pair_diagnostics(initial):
    if not torch.equal(initial["control"]["before"], initial["projected"]["before"]):
        raise ValueError("Both arms must start from exactly the same selected V8 parameters")
    result = {"initial_parameters_exactly_equal": True, "reason": PAIR_REASON, "vectors": {}, "tasks": []}
    for key in ("combined_gradient", "applied_gradient", "tentative"):
        control, projected = initial["control"][key], initial["projected"][key]
        difference = control.double() - projected.double()
        result["vectors"][key] = {"exactly_equal": torch.equal(control, projected), "unequal_coordinates": int((control != projected).sum()),
            "maximum_absolute_difference": float(difference.abs().max()), "difference_norm": float(difference.norm()),
            "control_norm": float(control.double().norm())}
    for index in range(initial["control"]["task_gradients"].shape[0]):
        control, projected = initial["control"]["task_gradients"][index], initial["projected"]["task_gradients"][index]
        difference = control.double() - projected.double()
        norm = float(control.double().norm())
        result["tasks"].append({"index": index, "exactly_equal": torch.equal(control, projected),
            "unequal_coordinates": int((control != projected).sum()), "maximum_absolute_difference": float(difference.abs().max()),
            "difference_norm": float(difference.norm()), "relative_l2_difference": float(difference.norm()) / norm if norm else None})
    result["all_task_gradients_exactly_equal"] = all(task["exactly_equal"] for task in result["tasks"])
    return result


def corrected_audit(checker):
    verify_hashes(SOURCES)
    source = inspect.getsource(audit.audit)
    before = '''for key in ("before", "task_gradients", "combined_gradient", "applied_gradient", "tentative"):
        if not torch.equal(initial["control"][key], initial["projected"][key]):
            raise ValueError("The first paired Adam proposal must be exactly matched before projection")'''
    if source.count(before) != 1:
        raise ValueError("Replace only the extra bitwise-backward assumption, not per-arm arithmetic or selection")
    source = source.replace(before, '''initial_pair = pair_diagnostics(initial)
    first_tasks = training.read(pair / "training_batches/step_001/manifest.json")["tasks"]
    initial_pair["first_forward_losses_exactly_equal"] = all(task["losses"]["control"] == task["losses"]["projected"] for task in first_tasks)
    initial_pair["first_objective_diagnostics_exactly_equal"] = all(task["diagnostics"]["control"] == task["diagnostics"]["projected"] for task in first_tasks)''')
    before = '"goal_success_not_assessed": True, "source_sha256":'
    if source.count(before) != 1:
        raise ValueError("Attach the actual paired-gradient diagnostics to the independent audit")
    source = source.replace(before, '"initial_pair_diagnostics": initial_pair, "audit_assumption_recovery": True, "goal_success_not_assessed": True, "source_sha256":')
    namespace = {**audit.__dict__, "check_episode": checker, "pair_diagnostics": pair_diagnostics, "__file__": str(Path(__file__).resolve())}
    exec(compile(source, str(Path(__file__).resolve()), "exec"), namespace)
    return namespace["audit"]


@contextmanager
def runtime():
    verify_hashes(SOURCES)
    verify_hashes(INPUTS)
    replacement, optimizer_replacement = corrected_checker(), corrected_optimizer()
    audit_replacement = corrected_audit(replacement)
    original_checker, original_audit_file = audit.check_episode, audit.__file__
    original_optimizer = audit.check_optimizer
    original_audit = audit.audit
    names = ("__file__", "OUTPUT", "AUDIT", "CONFIRMATIONS", "CAMPAIGNS", "REVIEWS")
    previous = {name: getattr(pipeline, name) for name in names}
    audit.check_episode, audit.__file__ = replacement, str(Path(__file__).resolve())
    audit.check_optimizer = optimizer_replacement
    audit.audit = audit_replacement
    pipeline.__file__ = str(Path(__file__).resolve())
    pipeline.OUTPUT = OUTPUT
    pipeline.AUDIT = pipeline.ROOT / "family_projection_v12_audit_recovery1"
    pipeline.CONFIRMATIONS = {name: pipeline.ROOT / f"family_projection_v12_{name}_confirmation_recovery1" for name in pipeline.ORDER}
    pipeline.CAMPAIGNS = {name: pipeline.ROOT / f"family_projection_v12_{name}_benchmarks_recovery1" for name in pipeline.ORDER}
    pipeline.REVIEWS = {name: pipeline.ROOT / f"family_projection_v12_{name}_review_recovery1" for name in pipeline.ORDER}
    try:
        yield
    finally:
        audit.check_episode, audit.__file__ = original_checker, original_audit_file
        audit.check_optimizer = original_optimizer
        audit.audit = original_audit
        for name, value in previous.items():
            setattr(pipeline, name, value)


def stop_original_queue():
    verify_hashes(SOURCES)
    verify_hashes(INPUTS)
    original = pipeline.training.read(ORIGINAL_PIPELINE / "status.json")
    if original["state"] != "waiting_for_paired_training":
        raise ValueError("Stop only the identified waiting queue, never training or a started evaluation")
    process = Path(f'/proc/{original["pid"]}/cmdline').read_bytes().split(b"\0")
    if str(Path(pipeline.__file__).resolve()).encode() not in process or b"run" not in process:
        raise ValueError("Queue PID no longer identifies the original orchestrator")
    manifest_path = pipeline.training.PAIR / "training_batches/step_001/manifest.json"
    manifest = pipeline.training.read(manifest_path)
    task = pipeline.training.read(pipeline.training.PAIR / "config.json")["batches"][0]["tasks"][0]
    logged = manifest["tasks"][0]
    verify_hashes({logged["artifact"]: logged["sha256"]})
    payload = torch.load(logged["artifact"], map_location="cpu", weights_only=True)
    try:
        audit.check_episode(payload, task, logged)
    except ValueError as error:
        if str(error) != "Incomplete or invalid synthetic training episode":
            raise
        original_error = repr(error)
    else:
        raise ValueError("Original audit defect does not reproduce; do not stop its queue")
    corrected_checker()(payload, task, logged)
    source, mask = payload["episode"]["source"], payload["episode"]["mask"]
    record = {"reason": REASON, "original_error": original_error, "original_status": original,
        "source_nan_outside_scoring_mask": int(torch.isnan(source[~mask]).sum()), "finite_context_cells": int(torch.isfinite(source).sum()),
        "artifact_sha256": {logged["artifact"]: logged["sha256"], str(manifest_path): sha256(manifest_path)},
        "original_source_sha256": SOURCES, "original_input_sha256": INPUTS,
        "training_restarted": False, "training_signaled": False, "selection_tolerance_changed": False,
        "confirmation_consumed": False, "real_benchmarks_evaluated": False, "updated_unix": time.time()}
    write_json(ORIGINAL_PIPELINE / "early_audit_defect.json", record)
    os.kill(original["pid"], signal.SIGTERM)
    for attempt in range(100):
        state = Path(f'/proc/{original["pid"]}/stat')
        if not state.exists() or state.read_text().split()[2] == "Z":
            break
        time.sleep(.05)
    else:
        raise RuntimeError("Original waiting queue did not stop")
    pipeline.status(ORIGINAL_PIPELINE, "interrupted", reason=REASON, stopped_queue_pid=original["pid"],
                    training_pid=original["training_pid"], training_restarted=False,
                    early_audit_defect_sha256=sha256(ORIGINAL_PIPELINE / "early_audit_defect.json"))


def main():
    if "--stop-original-queue" in sys.argv:
        stop_original_queue()
        return
    with runtime():
        if "--pair" in sys.argv:
            audit.main()
        else:
            pipeline.main()
            if "register" in sys.argv:
                write_json(OUTPUT / "recovery.json", {"reason": REASON, "original_source_sha256": SOURCES,
                    "optimizer_arithmetic_correction": OPTIMIZER_REASON,
                    "paired_backward_assumption_correction": PAIR_REASON,
                    "original_input_sha256": INPUTS, "old_predicate": OLD_PREDICATE, "new_predicate": NEW_PREDICATE,
                    "source_predicates_changed": 1, "training_restarted": False, "per_arm_roundoff_tolerances_changed": False,
                    "bitwise_cross_arm_backward_requirement_removed_and_reported": True,
                    "research_selection_tolerance_changed": False, "original_artifacts_preserved": True,
                    "same_reserved_confirmation_seeds": True, "independent_recovery_config_sha256": sha256(OUTPUT / "config.json")})


if __name__ == "__main__":
    main()
