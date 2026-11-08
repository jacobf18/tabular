"""Four-draw point protection with unchanged quarter-step optimizer arithmetic."""

from collections import defaultdict
import math
import types

import numpy as np

import single_model_fixed_quarter_protection as quarter
import single_model_finite_guarded_update as original
from single_model_four_draw_batching import COHORTS


REPLICAS = 4
SCOPE = "point_only_native_strata_other_four_draw_means"
POLICY = (
    "Prospective four-draw integration, not authorization for a scientific run. Preserve all56 "
    "composite gradients and every defined individual point gradient, up to168 vectors. Protect "
    "each of the four native draws separately and equal-weight four-draw structured/rating "
    "means, retaining all34 finite slots including truth-defined nulls. Keep half-composite/"
    "half-point proposal mass, per-gradient clipping, AdamW,0.1 inward margin,2x norm cap, "
    "fixed0.25 rounded displacement and exact parameter/moment/rate rollback. Undefined "
    "NRMSE is never a measured pass or a fabricated zero; its point mass goes to defined RMSE. "
    "Record all individual regressions even when a cohort mean retains. Keep every original "
    "held-out requirement and single-network inference. Require fresh registration, admission, "
    "capacity checks and complete capture/replay integration before a feasibility update."
)
FRACTION = quarter.FRACTION
StepEvaluationError = quarter.StepEvaluationError
quarter_step = quarter.quarter_step
independent_quarter_step = quarter.independent_quarter_step
within_ulp = quarter.within_ulp


def identity(name):
    parts = name.rsplit("_draw_", 1)
    if len(parts) != 2 or parts[0] not in COHORTS or parts[1] not in {"0", "1", "2", "3"}:
        raise ValueError("Require one of the original fourteen cohorts and four explicit draws")
    return parts[0], int(parts[1])


def metric_groups(values):
    original.metric_decision(values, values)
    tasks, objectives, groups = {}, defaultdict(set), defaultdict(list)
    for name, value in values.items():
        parts = name.rsplit("/", 1)
        if len(parts) != 2:
            raise ValueError("Require an explicit task/objective identity")
        task, kind = parts
        cohort, draw = identity(task)
        allowed = {"rmse", "mae" if cohort == "recsys" else "common_nrmse"}
        if kind not in allowed or (value is None and kind != "common_nrmse"):
            raise ValueError("Preserve original point objectives and truth-defined null semantics")
        tasks[task] = cohort, draw
        objectives[task].add(kind)
        groups[f"{cohort}/{kind}"].append((draw, name, value))
    expected_tasks = {f"{cohort}_draw_{draw}" for draw in range(REPLICAS) for cohort in COHORTS}
    if set(tasks) != expected_tasks or any(len(kinds) != 2 for kinds in objectives.values()):
        raise ValueError("Retain all56 tasks and112 point-metric slots")
    if any(sorted(draw for draw, name, value in members) != list(range(REPLICAS))
            for members in groups.values()):
        raise ValueError("Never omit or duplicate one of four metric draws")
    return tasks, groups


def mean_metrics(values):
    tasks, groups = metric_groups(values)
    result = {}
    for name, members in groups.items():
        if name.startswith("legacy/"):
            result.update((key, value) for draw, key, value in members)
        else:
            defined = [value for draw, key, value in members if value is not None]
            result[name] = math.fsum(defined) / len(defined) if defined else None
    return result


def metric_decision(before, after):
    individual = original.metric_decision(before, after)
    result = original.metric_decision(mean_metrics(before), mean_metrics(after))
    return {**result, "individual_draw_diagnostics": individual}


def gradient_layout(before, composite_count):
    tasks, groups = metric_groups(before)
    if type(composite_count) is not int or composite_count != len(tasks):
        raise ValueError("Preserve all56 raw composite gradients")
    labels = [(task, "composite") for task in tasks]
    labels.extend(tuple(name.rsplit("/", 1)) for name, value in before.items() if value is not None)
    slots = defaultdict(list)
    for index, (task, kind) in enumerate(labels[composite_count:], start=composite_count):
        cohort, draw = identity(task)
        group = task if cohort == "legacy" else cohort
        slots[f"{group}/{kind}"].append(index)
    matrix = np.zeros((len(slots), len(labels)))
    for index, members in enumerate(slots.values()):
        matrix[index, members] = 1. / len(members)
    return labels, list(slots), matrix


def proposal_weights(before, weights):
    labels, names, matrix = gradient_layout(before, len(weights))
    if (any(isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in weights)
            or not math.isclose(math.fsum(weights), 1., rel_tol=0, abs_tol=1e-15)):
        raise ValueError("Require the complete positive unit-mass task mixture")
    tasks = [task for task, kind in labels[:len(weights)]]
    masses = dict(zip(tasks, weights))
    for cohort in COHORTS:
        if len({masses[f"{cohort}_draw_{draw}"] for draw in range(REPLICAS)}) != 1:
            raise ValueError("Average only equally weighted four-draw cohorts")
    counts = {task: sum(name == task for name, kind in labels[len(weights):]) for task in tasks}
    result = [mass * .5 for mass in weights]
    result.extend(masses[task] * .5 / counts[task] for task, kind in labels[len(weights):])
    if not math.isclose(math.fsum(result), 1., rel_tol=0, abs_tol=1e-15):
        raise ValueError("Never change the total proposal mass")
    return result


def validate_layout(entries, tasks, before):
    labels, names, matrix = gradient_layout(before, len(tasks))
    expected = [{"task": task, "kind": kind, "index": index} for index, (task, kind) in enumerate(labels)]
    if (entries != expected or [task["name"] for task in tasks] != [task for task, kind in labels[:len(tasks)]]
            or any(type(task["index"]) is not int or task["index"] != index for index, task in enumerate(tasks))):
        raise ValueError("Keep complete ordered raw gradient/task identities")
    proposal_weights(before, [task["weight"] for task in tasks])
    return matrix


def paired_solve(gram, products, norm, before, count, margin):
    labels, names, matrix = gradient_layout(before, count)
    gram, products = np.asarray(gram), np.asarray(products)
    if gram.shape != (len(labels), len(labels)) or products.shape != (len(labels),):
        raise ValueError("Preserve every individual raw gradient in four-draw projection")
    reduced = original.projection.solve_projection(matrix @ gram @ matrix.T, matrix @ products, norm,
        margin_ratio=margin)
    return {"scope": SCOPE, "groups": names, "paired_projection": reduced,
        "raw_gradient_coefficients": (matrix.T @ np.asarray(reduced["raw_gradient_coefficients"])).tolist()}


def paired_verify(gram, products, norm, solution, margin, before, count):
    labels, names, matrix = gradient_layout(before, count)
    if solution["scope"] != SCOPE or solution["groups"] != names:
        raise ValueError("Never substitute protection scope or omit a four-draw objective")
    original.preflight.verify_certificate(matrix @ np.asarray(gram) @ matrix.T, matrix @ np.asarray(products),
        norm, solution["paired_projection"], margin)
    expected = matrix.T @ np.asarray(solution["paired_projection"]["raw_gradient_coefficients"])
    np.testing.assert_allclose(solution["raw_gradient_coefficients"], expected, rtol=1e-12, atol=1e-15)


def paired_displacement(products, norm, proposed_norm, before, count):
    labels, names, matrix = gradient_layout(before, count)
    products = np.asarray(products)
    if products.shape != (len(labels),) or not np.isfinite(products).all():
        raise ValueError("Every realized individual gradient product must remain explicit")
    result = original.displacement_decision(matrix @ products, norm, proposed_norm)
    return {**result, "individual_adverse_directions": int(np.count_nonzero(products < 0)),
        "individual_directions": len(labels), "paired_directions": len(names)}


_bindings = {name: globals()[name] for name in ("POLICY", "proposal_weights", "paired_solve", "paired_verify",
    "paired_displacement", "metric_decision")}
PROTECTED_SOURCE = quarter.PROTECTED_SOURCE
PROTECTED_SOURCE_SHA256 = quarter.PROTECTED_SOURCE_SHA256
_update_namespace = {**quarter._update_namespace, **_bindings}
exec(compile(PROTECTED_SOURCE, __file__, "exec"), _update_namespace)
protected_update = _update_namespace["protected_update"]
_contraction_namespace = {**quarter.__dict__, **_bindings}
check_contraction = types.FunctionType(quarter.check_contraction.__code__, _contraction_namespace,
    "check_contraction", quarter.check_contraction.__defaults__)
AUDIT_SOURCE = quarter.AUDIT_SOURCE
AUDIT_SOURCE_SHA256 = quarter.AUDIT_SOURCE_SHA256
_audit_namespace = {**quarter._audit_namespace, **_bindings, "check_contraction": check_contraction}
exec(compile(AUDIT_SOURCE, __file__, "exec"), _audit_namespace)
check_record = _audit_namespace["check_record"]
check_endpoint = _audit_namespace["check_endpoint"]
