"""Protect legacy original-coverage and anchor draws separately during training only."""

from collections import defaultdict
import hashlib
import inspect

import numpy as np

import single_model_paired_protection as template


POLICY = ("Retain the original paired protection for structured missingness and ratings, but protect each "
          "legacy draw independently. With anchor rehearsal, legacy draw0 follows original component coverage "
          "and draw1 is native anchor-latent: their point metrics and composite/point gradient constraints "
          "must not average away a within-pair regression. Preserve all raw gradients, task weights, "
          "individual diagnostics, the0.1 inward margin,2x displacement cap and exact rollback. This is "
          "a training-only hypothesis about spectral/latent interference, not a new held-out gate or proof "
          "of improved accuracy. Other paired cohorts still allow individual regressions and report them.")

original = template.original
StepEvaluationError = template.StepEvaluationError


def mean_metrics(values):
    template.metric_groups(values)
    result = template.mean_metrics(values)
    for name, value in values.items():
        task, kind = name.rsplit("/", 1)
        cohort, draw = template.identity(task)
        if cohort == "legacy":
            result.pop(f"{cohort}/{kind}", None)
            result[name] = value
    return result


def metric_decision(before, after):
    individual = original.metric_decision(before, after)
    result = original.metric_decision(mean_metrics(before), mean_metrics(after))
    return {**result, "individual_draw_diagnostics": individual}


def gradient_layout(before, composite_count):
    labels, previous_names, previous_matrix = template.gradient_layout(before, composite_count)
    slots = defaultdict(list)
    for index, (task, kind) in enumerate(labels):
        cohort, draw = template.identity(task)
        group = task if cohort == "legacy" else cohort
        slots[f"{group}/{kind}"].append(index)
    matrix = np.zeros((len(slots), len(labels)))
    for index, members in enumerate(slots.values()):
        matrix[index, members] = 1. / len(members)
    return labels, list(slots), matrix


def validate_layout(entries, tasks, before):
    template.validate_layout(entries, tasks, before)
    return gradient_layout(before, len(tasks))[2]


for name in ("paired_solve", "paired_verify", "paired_displacement"):
    source = inspect.getsource(getattr(template, name)).replace('"paired_means"', '"legacy_strata_other_paired_means"')
    exec(compile(source, __file__, "exec"), globals())

PROTECTED_SOURCE = template.PROTECTED_SOURCE
PROTECTED_SOURCE_SHA256 = hashlib.sha256(PROTECTED_SOURCE.encode()).hexdigest()
_update_namespace = {**original.__dict__, **{name: globals()[name] for name in
                     ("paired_solve", "paired_verify", "paired_displacement", "metric_decision")}}
exec(compile(PROTECTED_SOURCE, __file__, "exec"), _update_namespace)
protected_update = _update_namespace["protected_update"]

AUDIT_SOURCE = template.AUDIT_SOURCE
AUDIT_SOURCE_SHA256 = hashlib.sha256(AUDIT_SOURCE.encode()).hexdigest()
_audit_namespace = {**template.original_audit.__dict__, **{name: globals()[name] for name in
                    ("paired_verify", "paired_displacement", "metric_decision")}}
exec(compile(AUDIT_SOURCE, __file__, "exec"), _audit_namespace)
check_record, check_endpoint = _audit_namespace["check_record"], _audit_namespace["check_endpoint"]
