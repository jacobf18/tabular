"""An additive paired-example training rule; held-out qualification is unchanged."""

from collections import defaultdict
import hashlib
import inspect
import math

import numpy as np

import single_model_finite_guarded_audit as original_audit
import single_model_finite_guarded_update as original
from single_model_registered_adaptation import replace_once


POLICY = ("Keep every raw composite/point gradient and the original per-draw clipped Adam proposal. "
          "Protect equal-weight within-cohort/objective means with the same0.1 inward margin and2x step bound. "
          "Accept only if each actual paired point-metric mean retains and one improves; otherwise restore "
          "parameters, moments and prior rate exactly. Preserve individual-draw losses and regressions in "
          "the record. Truth-defined nulls remain null and contribute no fabricated gradient or score. "
          "This changes training acceptance, not any development, independent-confirmation or benchmark gate.")


def identity(name):
    parts = name.rsplit("_draw_", 1)
    if len(parts) != 2 or not parts[0] or parts[1] not in {"0", "1"}:
        raise ValueError("Require explicit paired-draw identities")
    return parts[0], int(parts[1])


def metric_groups(values):
    original.metric_decision(values, values)
    groups, tasks, objectives = defaultdict(list), {}, defaultdict(set)
    for name, value in values.items():
        parts = name.rsplit("/", 1)
        if len(parts) != 2 or parts[1] not in {"rmse", "mae", "common_nrmse"}:
            raise ValueError("Preserve explicit task/objective metric slots")
        task, kind = parts
        cohort, draw = identity(task)
        tasks[task] = cohort, draw
        objectives[task].add(kind)
        if value is None and kind != "common_nrmse":
            raise ValueError("Only truth-defined common NRMSE may be undefined")
        groups[f"{cohort}/{kind}"].append((draw, name, value))
    for kinds in objectives.values():
        if kinds not in ({"rmse", "mae"}, {"rmse", "common_nrmse"}):
            raise ValueError("Every draw must retain both registered point objectives")
    for members in groups.values():
        if sorted(draw for draw, name, value in members) != [0, 1]:
            raise ValueError("Never omit or duplicate a paired metric draw")
    return tasks, groups


def mean_metrics(values):
    tasks, groups = metric_groups(values)
    result = {}
    for name, members in groups.items():
        defined = [value for draw, key, value in members if value is not None]
        result[name] = math.fsum(defined) / len(defined) if defined else None
    return result


def metric_decision(before, after):
    individual = original.metric_decision(before, after)
    result = original.metric_decision(mean_metrics(before), mean_metrics(after))
    return {**result, "individual_draw_diagnostics": individual}


def gradient_layout(before, composite_count):
    tasks, groups = metric_groups(before)
    if len(tasks) != composite_count or composite_count % 2:
        raise ValueError("Every composite gradient must match a paired task")
    labels = [(name, "composite") for name in tasks]
    labels.extend(tuple(name.rsplit("/", 1)) for name, value in before.items() if value is not None)
    slots = defaultdict(list)
    for index, (task, kind) in enumerate(labels):
        cohort, draw = identity(task)
        slots[f"{cohort}/{kind}"].append(index)
    matrix = np.zeros((len(slots), len(labels)))
    for index, members in enumerate(slots.values()):
        matrix[index, members] = 1. / len(members)
    return labels, list(slots), matrix


def validate_layout(entries, tasks, before):
    labels, names, matrix = gradient_layout(before, len(tasks))
    expected = [{"task": task, "kind": kind, "index": index} for index, (task, kind) in enumerate(labels)]
    if (entries != expected or [task["name"] for task in tasks] != [task for task, kind in labels[:len(tasks)]]
            or [task["index"] for task in tasks] != list(range(len(tasks)))):
        raise ValueError("Raw gradient and task order must bind the inferred paired layout")
    paired_tasks = defaultdict(list)
    for task in tasks:
        paired_tasks[identity(task["name"])[0]].append(task)
    for members in paired_tasks.values():
        if len(members) != 2 or members[0]["weight"] != members[1]["weight"]:
            raise ValueError("Average only equally weighted paired training draws")
    return matrix


def paired_solve(gram, products, norm, before, count, margin):
    labels, names, matrix = gradient_layout(before, count)
    gram, products = np.asarray(gram), np.asarray(products)
    if gram.shape != (len(labels), len(labels)) or products.shape != (len(labels),):
        raise ValueError("Preserve every raw gradient in the paired projection")
    reduced = original.projection.solve_projection(matrix @ gram @ matrix.T, matrix @ products, norm,
                                                   margin_ratio=margin)
    return {"scope": "paired_means", "groups": names, "paired_projection": reduced,
            "raw_gradient_coefficients": (matrix.T @ np.asarray(reduced["raw_gradient_coefficients"])).tolist()}


def paired_verify(gram, products, norm, solution, margin, before, count):
    labels, names, matrix = gradient_layout(before, count)
    if solution["scope"] != "paired_means" or solution["groups"] != names:
        raise ValueError("Never substitute a protection scope or omit a paired objective")
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


PROTECTED_SOURCE = inspect.getsource(original.protected_update)
for before, after in {
    "solution = projection.solve_projection(gram, products, norm, margin_ratio=MARGIN)":
        "solution = paired_solve(gram, products, norm, before_metrics, len(weights), MARGIN)",
    "preflight.verify_certificate(gram, products, norm, solution, MARGIN)":
        "paired_verify(gram, products, norm, solution, MARGIN, before_metrics, len(weights))",
    "displacement_decision(actual, actual_norm, norm)":
        "paired_displacement(actual, actual_norm, norm, before_metrics, len(weights))",
}.items():
    PROTECTED_SOURCE = replace_once(PROTECTED_SOURCE, before, after)
PROTECTED_SOURCE_SHA256 = hashlib.sha256(PROTECTED_SOURCE.encode()).hexdigest()
_update_namespace = {**original.__dict__, **{name: globals()[name] for name in
                     ("paired_solve", "paired_verify", "paired_displacement", "metric_decision")}}
exec(compile(PROTECTED_SOURCE, __file__, "exec"), _update_namespace)
protected_update = _update_namespace["protected_update"]
StepEvaluationError = original.StepEvaluationError

AUDIT_SOURCE = inspect.getsource(original_audit.check_record) + "\n" + inspect.getsource(original_audit.check_endpoint)
for before, after in {
    "legacy.training.preflight.verify_certificate(gram, products, record[\"adam_norm\"], record[\"projection\"], guarded.MARGIN)":
        "paired_verify(gram, products, record[\"adam_norm\"], record[\"projection\"], guarded.MARGIN, point_before, len(weights))",
    "displacement_decision(candidate, record[\"candidate_norm\"], record[\"adam_norm\"])":
        "paired_displacement(candidate, record[\"candidate_norm\"], record[\"adam_norm\"], point_before, len(weights))",
    "guarded.metric_decision(point_before, record[\"point_after\"])":
        "metric_decision(point_before, record[\"point_after\"])",
    "displacement_decision(actual, actual_norm, norm)":
        "paired_displacement(actual, actual_norm, norm, record[\"point_before\"], record[\"composite_tasks\"])",
}.items():
    AUDIT_SOURCE = replace_once(AUDIT_SOURCE, before, after)
AUDIT_SOURCE_SHA256 = hashlib.sha256(AUDIT_SOURCE.encode()).hexdigest()
_audit_namespace = {**original_audit.__dict__, **{name: globals()[name] for name in
                    ("paired_verify", "paired_displacement", "metric_decision")}}
exec(compile(AUDIT_SOURCE, __file__, "exec"), _audit_namespace)
check_record, check_endpoint = _audit_namespace["check_record"], _audit_namespace["check_endpoint"]
