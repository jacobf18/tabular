"""Add point-loss proposal mass while retaining separate legacy-draw protection."""

import hashlib

import single_model_legacy_stratum_protection as strata
import single_model_point_inclusive_proposal as inclusive


POLICY = ("Assign half of each original task's Adam proposal mass to its composite loss and divide the "
          "other half among its defined original point metrics. Preserve every raw gradient, task mass, "
          "teacher and normalization. Retain separate legacy-draw protection and paired protection for "
          "structured/rating draws, including individual regression diagnostics,0.1 inward margin,2x "
          "displacement bound and exact parameter/moment/rate rollback. Undefined truth-only NRMSE has "
          "no gradient or weight; its task's point share goes to defined RMSE. No held-out gate change.")

proposal_weights = inclusive.proposal_weights
gradient_layout = strata.gradient_layout
validate_layout = strata.validate_layout
metric_decision = strata.metric_decision
StepEvaluationError = strata.StepEvaluationError

PROTECTED_SOURCE = inclusive.PROTECTED_SOURCE
PROTECTED_SOURCE_SHA256 = hashlib.sha256(PROTECTED_SOURCE.encode()).hexdigest()
_update_namespace = {**strata._update_namespace, "proposal_weights": proposal_weights, "POLICY": POLICY}
exec(compile(PROTECTED_SOURCE, __file__, "exec"), _update_namespace)
protected_update = _update_namespace["protected_update"]

AUDIT_SOURCE = inclusive.AUDIT_SOURCE
AUDIT_SOURCE_SHA256 = hashlib.sha256(AUDIT_SOURCE.encode()).hexdigest()
_audit_namespace = {**strata._audit_namespace, "proposal_weights": proposal_weights, "POLICY": POLICY}
exec(compile(AUDIT_SOURCE, __file__, "exec"), _audit_namespace)
check_record = _audit_namespace["check_record"]
check_endpoint = _audit_namespace["check_endpoint"]
