"""Independent checks for finite-guarded log decisions and a full raw endpoint."""

from contextlib import contextmanager
from copy import deepcopy
import numpy as np
import torch

import audit_single_model_family_projection_training as legacy
from recover_single_model_family_projection_pipeline import corrected_optimizer
import single_model_finite_guarded_update as guarded
from single_model_point_proposal_grid import direction_products, displacement_decision


def check_record(record, weights, point_before, protected_tasks):
    if (record["composite_tasks"] != len(weights) or record["protected_tasks"] != protected_tasks
            or record["margin"] != guarded.MARGIN or record["point_before"] != point_before
            or type(record["accepted"]) is not bool):
        raise ValueError("Complete gradient/metric coverage and the registered margin are mandatory")
    adam = record["composite_adam_record"]
    legacy.check_record(adam, weights, projected=False)
    if record["rate"] != adam["lr"] or record["adam_norm"] != adam["proposed_step_norm"]:
        raise ValueError("Protect exactly the logged original Adam proposal")
    gram, products = np.asarray(record["joint_gram"]), np.asarray(record["joint_products_before"])
    if (gram.shape != (protected_tasks, protected_tasks) or products.shape != (protected_tasks,)
            or not np.isfinite(gram).all() or not np.isfinite(products).all() or np.any(np.diag(gram) < 0)):
        raise ValueError("Require the full finite protected geometry")
    np.testing.assert_allclose(gram, gram.T, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(gram[:len(weights), :len(weights)], adam["raw_gram"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(products[:len(weights)], adam["raw_products_before"], rtol=1e-9, atol=1e-12)
    if record["reason"] == "projection_infeasible":
        if record["accepted"] or record["point_after"] is not None or not record.get("solver_error") or "projection" in record:
            raise ValueError("Infeasible projections must reject without a finite-metric claim")
        return
    legacy.training.preflight.verify_certificate(gram, products, record["adam_norm"], record["projection"], guarded.MARGIN)
    candidate = np.asarray(record["candidate_products"])
    if candidate.shape != products.shape:
        raise ValueError("Never drop a protected direction after projection")
    displacement = displacement_decision(candidate, record["candidate_norm"], record["adam_norm"])
    if displacement != record["displacement"]:
        raise ValueError("Actual rounded-displacement decision changed")
    if not displacement["admissible"]:
        if record["accepted"] or record["point_after"] is not None or record["reason"] != "rounded_direction_rejected":
            raise ValueError("Inadmissible rounded directions cannot be accepted")
        return
    finite = guarded.metric_decision(point_before, record["point_after"])
    expected_reason = "accepted" if finite["accepted"] else "finite_metrics_rejected"
    if finite != record["finite"] or record["accepted"] != finite["accepted"] or record["reason"] != expected_reason:
        raise ValueError("Strict finite-metric acceptance or rollback decision changed")


def optimizer_equal(actual, expected):
    if actual["param_groups"] != expected["param_groups"] or set(actual["state"]) != set(expected["state"]):
        raise ValueError("Optimizer groups or state inventory changed")
    for key, state in expected["state"].items():
        if set(actual["state"][key]) != set(state):
            raise ValueError("Optimizer state keys changed")
        for field, value in state.items():
            if not torch.equal(actual["state"][key][field], value):
                raise ValueError("Optimizer moments/counts must match the accepted or rolled-back state exactly")


@contextmanager
def fused_cpu_checker():
    previous = legacy.check_optimizer
    legacy.check_optimizer = corrected_optimizer()
    try:
        yield
    finally:
        legacy.check_optimizer = previous


def check_endpoint(payload, record, sizes, weight_decay, rollback_rate, final_optimizer=None):
    snapshot, gradients = payload["snapshot"], payload["gradients"]
    if any(record[key] != value for key, value in payload["record"].items()):
        raise ValueError("Raw endpoint and logged attempt differ")
    if snapshot["record"] != record["composite_adam_record"]:
        raise ValueError("Raw composite Adam proposal changed")
    if any(record[key] != value for key, value in snapshot["protected_record"].items()):
        raise ValueError("Raw protected decision changed")
    if (gradients.shape != (record["protected_tasks"], sum(sizes)) or gradients.dtype != torch.float32
            or not torch.isfinite(gradients).all()
            or not torch.equal(gradients[:record["composite_tasks"]], snapshot["task_gradients"])):
        raise ValueError("Require every original full FP32 composite and point gradient")
    if snapshot["rollback_rates"] != [rollback_rate]:
        raise ValueError("Rollback must restore the preceding accepted learning rate")
    proposed_count = record["accepted_updates"] - int(record["accepted"]) + 1
    proposal_view = {**snapshot, "after": snapshot["tentative"]}
    with fused_cpu_checker():
        adam_audit = legacy.check_snapshot(proposal_view, snapshot["record"], sizes, proposed_count, weight_decay)
    gram, products, norm = legacy.training.projection.gram_and_products(
        gradients, snapshot["before"].double() - snapshot["tentative"].double())
    np.testing.assert_allclose(gram, record["joint_gram"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(products, record["joint_products_before"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(norm, record["adam_norm"], rtol=1e-12, atol=1e-12)
    if record["accepted"]:
        expected = legacy.training.projection.projected_parameters(snapshot["tentative"], gradients,
                                                                  record["projection"]["raw_gradient_coefficients"])
        lower, upper = torch.nextafter(expected, torch.full_like(expected, -torch.inf)), torch.nextafter(expected, torch.full_like(expected, torch.inf))
        if not ((snapshot["after"] >= lower) & (snapshot["after"] <= upper)).all():
            raise ValueError("Protected endpoint arithmetic exceeds one FP32 output ULP")
        actual, actual_norm = direction_products(gradients, snapshot["before"].double() - snapshot["after"].double())
        np.testing.assert_allclose(actual, record["candidate_products"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(actual_norm, record["candidate_norm"], rtol=1e-12, atol=1e-12)
        if not displacement_decision(actual, actual_norm, norm)["admissible"]:
            raise ValueError("Accepted actual endpoint violates protection")
        expected_optimizer = snapshot["optimizer_after"]
    else:
        if not torch.equal(snapshot["after"], snapshot["before"]):
            raise ValueError("A rejected attempt must restore exact original parameters")
        expected_optimizer = deepcopy(snapshot["optimizer_before"])
        for group, rate in zip(expected_optimizer["param_groups"], snapshot["rollback_rates"]):
            group["lr"] = rate
    if final_optimizer is not None:
        optimizer_equal(final_optimizer, expected_optimizer)
    return {"adam": adam_audit, "proposed_optimizer_count": proposed_count,
            "accepted_updates": record["accepted_updates"], "gradient_vectors": gradients.shape[0],
            "accepted": record["accepted"], "actual_final_parameters_checked": True,
            "rejected_candidate_requires_gpu_reconstruction": not record["accepted"] and "projection" in record,
            "finite_point_scores_require_neural_replay": True, "autograd_replayed": False}
