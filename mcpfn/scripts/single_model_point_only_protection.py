"""Keep point-inclusive proposals but constrain only the original point-error directions."""

import hashlib
import inspect

import single_model_legacy_point_proposal as previous
import single_model_legacy_stratum_protection as strata
from single_model_registered_adaptation import replace_once


POLICY = ("Retain all raw composite and defined point gradients and the original half-composite/half-point "
          "Adam proposal weights. Change only training projection constraints: protect each legacy draw's "
          "defined original point metrics separately, and paired structured/rating point metrics, without "
          "constraining composite-loss directions. Composite gradients remain optimized and fully recorded; "
          "their adverse directions are allowed and reported, not silently discarded. Preserve0.1 inward "
          "margin,2x displacement bound, original finite point-metric acceptance, individual diagnostics and "
          "exact parameter/moment/rate rollback. Keep every held-out qualification gate unchanged. This "
          "hypothesis concerns first-order training interference, not proof of generalization or causality.")
SCOPE = "point_only_legacy_strata_other_paired_means"
proposal_weights = previous.proposal_weights
metric_decision = strata.metric_decision
StepEvaluationError = previous.StepEvaluationError


def gradient_layout(before, composite_count):
    labels, names, matrix = strata.gradient_layout(before, composite_count)
    indices = [index for index, name in enumerate(names) if not name.endswith("/composite")]
    if not indices or any(kind != "composite" for task, kind in labels[:composite_count]):
        raise ValueError("Preserve the complete original composite prefix and every defined point metric")
    return labels, [names[index] for index in indices], matrix[indices]


def validate_layout(entries, tasks, before):
    strata.validate_layout(entries, tasks, before)
    return gradient_layout(before, len(tasks))[2]


original = strata.original
np = strata.np
for name in ("paired_solve", "paired_verify", "paired_displacement"):
    source = inspect.getsource(getattr(strata.template, name))
    if '"paired_means"' in source:
        source = replace_once(source, '"paired_means"', repr(SCOPE))
    exec(compile(source, __file__, "exec"), globals())

PROTECTED_SOURCE = previous.PROTECTED_SOURCE
PROTECTED_SOURCE_SHA256 = hashlib.sha256(PROTECTED_SOURCE.encode()).hexdigest()
_update_namespace = {**previous._update_namespace, "POLICY": POLICY,
    **{name: globals()[name] for name in ("paired_solve", "paired_verify", "paired_displacement", "metric_decision")}}
exec(compile(PROTECTED_SOURCE, __file__, "exec"), _update_namespace)
protected_update = _update_namespace["protected_update"]

AUDIT_SOURCE = previous.AUDIT_SOURCE
AUDIT_SOURCE_SHA256 = hashlib.sha256(AUDIT_SOURCE.encode()).hexdigest()
_audit_namespace = {**previous._audit_namespace, "POLICY": POLICY,
    **{name: globals()[name] for name in ("paired_verify", "paired_displacement", "metric_decision")}}
exec(compile(AUDIT_SOURCE, __file__, "exec"), _audit_namespace)
check_record = _audit_namespace["check_record"]
check_endpoint = _audit_namespace["check_endpoint"]
