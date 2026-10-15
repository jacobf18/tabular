"""Contract one point-protected proposal without retries or changing Adam's moments."""

import hashlib

import numpy as np
import torch

import single_model_point_only_protection as previous
from single_model_point_proposal_grid import direction_products
from single_model_registered_adaptation import replace_once


FRACTION = .5
POLICY = ("Keep the original half-composite/half-point AdamW proposal, native legacy strata and paired "
    "structured/rating point protection. Solve the unchanged0.1 inward-margin projection, then contract "
    "its rounded FP32 displacement ONCE by the fixed fraction0.5, using FP64 before+0.5*(projected-before) "
    "and a final FP32 cast. This is not a smaller Adam learning rate: accepted moments are the original "
    "proposal moments. Save both uncontracted and contracted full vectors and their actual84 gradient "
    "products. Independently check the30 actual contracted directions and original2x Adam norm cap, then "
    "the unchanged finite point guard. Reject with exact parameter/moment/rate rollback. No fraction "
    "search, retry, fallback, new inference operation, ensemble or relaxed held-out requirement. A fresh "
    "synthetic test is needed; local success on a previously consumed batch is not training qualification.")
SCOPE = previous.SCOPE
proposal_weights = previous.proposal_weights
gradient_layout = previous.gradient_layout
validate_layout = previous.validate_layout
metric_decision = previous.metric_decision
paired_solve = previous.paired_solve
paired_verify = previous.paired_verify
paired_displacement = previous.paired_displacement
StepEvaluationError = previous.StepEvaluationError


def half_step(before, projected):
    if (before.ndim != 1 or before.shape != projected.shape or not before.numel()
            or before.dtype != torch.float32 or projected.dtype != torch.float32
            or before.device != projected.device or not torch.isfinite(before).all()
            or not torch.isfinite(projected).all()):
        raise ValueError("Require complete aligned finite FP32 proposal vectors")
    return (before.double() + FRACTION * (projected.double() - before.double())).float()


def independent_half_step(before, projected):
    return ((1. - FRACTION) * before.double() + FRACTION * projected.double()).float()


def within_ulp(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not torch.isfinite(actual).all():
        raise ValueError("Preserve every finite full-vector coordinate")
    lower = torch.nextafter(expected, torch.full_like(expected, -torch.inf))
    upper = torch.nextafter(expected, torch.full_like(expected, torch.inf))
    if not torch.isfinite(expected).all() or not ((actual >= lower) & (actual <= upper)).all():
        raise ValueError("Independent endpoint arithmetic exceeded one output ULP")
    return int((actual != expected).sum())


def check_contraction(snapshot, gradients, record):
    if record.get("displacement_fraction") != FRACTION:
        raise ValueError("Never change the fixed contraction or call it an Adam rate")
    if "projection" not in record:
        if any(name in snapshot for name in ("uncontracted", "contracted")):
            raise ValueError("No rounded vector exists for an infeasible projection")
        return {"evaluated": False, "fraction": FRACTION}
    before, uncontracted, contracted = (snapshot[name] for name in ("before", "uncontracted", "contracted"))
    expected = previous._audit_namespace["legacy"].training.projection.projected_parameters(
        snapshot["tentative"], gradients, record["projection"]["raw_gradient_coefficients"])
    uncontracted_differences = within_ulp(uncontracted, expected)
    contracted_differences = within_ulp(contracted, independent_half_step(before, uncontracted))
    for parameters, prefix in ((uncontracted, "uncontracted"), (contracted, "candidate")):
        products, norm = direction_products(gradients, before.double() - parameters.double())
        np.testing.assert_allclose(products, record[f"{prefix}_products"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(norm, record[f"{prefix}_norm"], rtol=1e-12, atol=1e-12)
    uncontracted_decision = paired_displacement(record["uncontracted_products"], record["uncontracted_norm"],
        record["adam_norm"], record["point_before"], record["composite_tasks"])
    if uncontracted_decision != record["uncontracted_displacement"]:
        raise ValueError("Keep the full uncontracted geometry diagnostic")
    if record["accepted"] and not torch.equal(snapshot["after"], contracted):
        raise ValueError("Accepted parameters must equal the actual contracted endpoint exactly")
    return {"evaluated": True, "fraction": FRACTION, "parameters": before.numel(),
        "uncontracted_non_bitwise_elements": uncontracted_differences,
        "contracted_non_bitwise_elements": contracted_differences, "allowed_output_ulps": 1,
        "actual_saved_displacements_checked": True}


PROTECTED_SOURCE = previous.PROTECTED_SOURCE
PROTECTED_SOURCE = replace_once(PROTECTED_SOURCE,
    '"proposal_policy": POLICY, "proposal_weights": proposal,',
    '"proposal_policy": POLICY, "proposal_weights": proposal, "displacement_fraction": FRACTION,')
PROTECTED_SOURCE = replace_once(PROTECTED_SOURCE,
    '            actual, actual_norm = direction_products(gradients, before.double() - parameters.double())',
    '            snapshot["uncontracted"] = parameters.detach().cpu().clone()\n'
    '            uncontracted_products, uncontracted_norm = direction_products(gradients, before.double() - parameters.double())\n'
    '            record.update(uncontracted_products=uncontracted_products.tolist(), uncontracted_norm=uncontracted_norm,\n'
    '                uncontracted_displacement=paired_displacement(uncontracted_products, uncontracted_norm, norm, before_metrics, len(weights)))\n'
    '            parameters = half_step(before, parameters)\n'
    '            snapshot["contracted"] = parameters.detach().cpu().clone()\n'
    '            actual, actual_norm = direction_products(gradients, before.double() - parameters.double())')
PROTECTED_SOURCE_SHA256 = hashlib.sha256(PROTECTED_SOURCE.encode()).hexdigest()
_update_namespace = {**previous._update_namespace, "POLICY": POLICY, "FRACTION": FRACTION, "half_step": half_step}
exec(compile(PROTECTED_SOURCE, __file__, "exec"), _update_namespace)
protected_update = _update_namespace["protected_update"]

AUDIT_SOURCE = previous.AUDIT_SOURCE
AUDIT_SOURCE = replace_once(AUDIT_SOURCE,
    '    proposal = proposal_weights(point_before, weights)',
    '    if record.get("displacement_fraction") != FRACTION:\n'
    '        raise ValueError("Require the prospectively fixed half displacement")\n'
    '    proposal = proposal_weights(point_before, weights)')
AUDIT_SOURCE = replace_once(AUDIT_SOURCE,
    '    snapshot, gradients = payload["snapshot"], payload["gradients"]',
    '    snapshot, gradients = payload["snapshot"], payload["gradients"]\n'
    '    contraction = check_contraction(snapshot, gradients, record)')
AUDIT_SOURCE = replace_once(AUDIT_SOURCE,
    '        expected = legacy.training.projection.projected_parameters(snapshot["tentative"], gradients,\n'
    '                                                                  record["projection"]["raw_gradient_coefficients"])',
    '        expected = snapshot["contracted"]')
AUDIT_SOURCE = replace_once(AUDIT_SOURCE,
    '    return {"adam": adam_audit,', '    return {"fixed_half_step": contraction, "adam": adam_audit,')
AUDIT_SOURCE_SHA256 = hashlib.sha256(AUDIT_SOURCE.encode()).hexdigest()
_audit_namespace = {**previous._audit_namespace, "POLICY": POLICY, "FRACTION": FRACTION, "check_contraction": check_contraction}
exec(compile(AUDIT_SOURCE, __file__, "exec"), _audit_namespace)
check_record = _audit_namespace["check_record"]
check_endpoint = _audit_namespace["check_endpoint"]
