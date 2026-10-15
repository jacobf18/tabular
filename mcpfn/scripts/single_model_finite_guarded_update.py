"""One Adam proposal with strong point protection and actual training-metric rollback."""

from copy import deepcopy
import math

import torch

import preflight_single_model_family_projection as preflight
import single_model_family_projection as projection
from single_model_point_proposal_grid import direction_products, displacement_decision
import train_single_matrixformer_family_projection as original


MARGIN = .1
POLICY = ("Use the unchanged domain-weighted clipped composite AdamW proposal; protect all sampled composite and "
          "defined point gradients with normalized inward margin0.1. Require the actual FP32 step to be non-adverse "
          "to every protected direction and at most twice the original norm. Accept only if all actual finite-step "
          "training-task point metrics retain and at least one strictly improves. Otherwise restore both parameters "
          "and optimizer moments exactly. No fallback margin or step-size search.")


def metric_decision(before, after):
    if not before or set(before) != set(after):
        raise ValueError("Retain every original training task/metric slot")
    changes = {}
    retained, improved, defined = True, False, 0
    for key in before:
        initial, current = before[key], after[key]
        if initial is None or current is None:
            if initial is not None or current is not None:
                raise ValueError("Undefined truth-only coverage cannot change")
            changes[key] = None
            continue
        if not all(math.isfinite(value) and value >= 0 for value in (initial, current)):
            raise ValueError("Every defined metric must remain finite and nonnegative")
        difference = current - initial
        changes[key] = difference
        retained &= difference <= 0
        improved |= difference < 0
        defined += 1
    return {"changes": changes, "defined_slots": defined, "all_retained": bool(retained),
            "some_improved": bool(improved), "accepted": bool(defined and retained and improved)}


class StepEvaluationError(RuntimeError):
    def __init__(self, error, snapshot, record):
        super().__init__(f"Protected-step evaluation failed: {error!r}")
        self.snapshot = snapshot
        self.record = record


def protected_update(model, optimizer, gradients, weights, rate, before_metrics, evaluate):
    if gradients.ndim != 2 or len(weights) < 1 or gradients.shape[0] <= len(weights):
        raise ValueError("Require all composite gradients followed by defined point gradients")
    rollback_rates = [group["lr"] for group in optimizer.param_groups]
    _, snapshot = original.update_model(model, optimizer, gradients[:len(weights)], weights, False, rate, raw=True)
    snapshot["rollback_rates"] = rollback_rates
    before = snapshot["before"].to(gradients.device)
    tentative = snapshot["tentative"].to(gradients.device)
    record = {"rate": rate, "margin": MARGIN, "composite_tasks": len(weights), "protected_tasks": gradients.shape[0],
              "point_before": deepcopy(before_metrics), "point_after": None, "accepted": False,
              "composite_adam_record": deepcopy(snapshot["record"])}

    def rollback():
        preflight.assign(model, before)
        optimizer.load_state_dict(snapshot["optimizer_before"])
        for group, previous_rate in zip(optimizer.param_groups, rollback_rates):
            group["lr"] = previous_rate
        optimizer.zero_grad(set_to_none=True)
        if not torch.equal(preflight.flatten(model), before):
            raise ValueError("Rollback must restore the exact original FP32 parameters")

    try:
        gram, products, norm = projection.gram_and_products(gradients, before.double() - tentative.double())
        record.update(joint_gram=gram.tolist(), joint_products_before=products.tolist(), adam_norm=norm)
        try:
            solution = projection.solve_projection(gram, products, norm, margin_ratio=MARGIN)
            preflight.verify_certificate(gram, products, norm, solution, MARGIN)
        except (ValueError, AssertionError) as error:
            record.update(reason="projection_infeasible", solver_error=repr(error))
            rollback()
        else:
            parameters = projection.projected_parameters(tentative, gradients, solution["raw_gradient_coefficients"])
            actual, actual_norm = direction_products(gradients, before.double() - parameters.double())
            displacement = displacement_decision(actual, actual_norm, norm)
            record.update(projection=solution, candidate_products=actual.tolist(), candidate_norm=actual_norm,
                          displacement=displacement)
            if not displacement["admissible"]:
                record["reason"] = "rounded_direction_rejected"
                rollback()
            else:
                preflight.assign(model, parameters)
                after_metrics = evaluate(model)
                finite = metric_decision(before_metrics, after_metrics)
                record.update(point_after=after_metrics, finite=finite, accepted=finite["accepted"],
                              reason="accepted" if finite["accepted"] else "finite_metrics_rejected")
                if not record["accepted"]:
                    rollback()
        snapshot["after"] = preflight.flatten(model).cpu().clone()
        snapshot["protected_record"] = deepcopy(record)
        return record, snapshot
    except BaseException as error:
        rollback()
        snapshot["after"] = before.cpu().clone()
        record.update(reason="evaluation_failed", error=repr(error), accepted=False)
        raise StepEvaluationError(error, snapshot, record) from error
