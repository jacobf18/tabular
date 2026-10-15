"""Optimize ground-truth point metrics alongside distillation with unchanged paired protection."""

import hashlib
import math

import single_model_paired_protection as paired
from single_model_registered_adaptation import replace_once


POLICY = ("Keep every original composite and point gradient, all prior/cohort masses, teachers and normalization. "
          "Assign half of each task's Adam proposal weight to its original Huber/NLL/CDF/query composite and "
          "divide the remaining half equally among its defined original point metrics: pooled RMSE/common "
          "NRMSE for imputation, equal-user clamped RMSE/MAE for ratings. Clip each gradient with the original "
          "per-gradient rule before combination, then use the unchanged global clip and AdamW. An undefined "
          "truth-only common NRMSE has no gradient or weight; its point share goes to that task's defined RMSE. "
          "Preserve equal-draw paired composite/point projection,0.1 margin,2x norm cap, actual finite paired "
          "acceptance and exact parameter/moment/rate rollback. No held-out gate or inference change.")


def proposal_weights(before, weights):
    labels, names, matrix = paired.gradient_layout(before, len(weights))
    if (any(not math.isfinite(value) or value <= 0 for value in weights)
            or not math.isclose(math.fsum(weights), 1., rel_tol=0, abs_tol=1e-15)):
        raise ValueError("Require the complete positive unit-mass task mixture")
    tasks = [task for task, kind in labels[:len(weights)]]
    masses = dict(zip(tasks, weights))
    for task in tasks:
        cohort, draw = paired.identity(task)
        if masses[task] != masses[f"{cohort}_draw_{1 - draw}"]:
            raise ValueError("Preserve equal original paired-draw weights")
    counts = {task: sum(name == task for name, kind in labels[len(weights):]) for task in tasks}
    if any(count not in (1, 2) for count in counts.values()):
        raise ValueError("Require every task's defined original point metrics")
    result = [mass * .5 for mass in weights]
    result.extend(masses[task] * .5 / counts[task] for task, kind in labels[len(weights):])
    if not math.isclose(math.fsum(result), 1., rel_tol=0, abs_tol=1e-15):
        raise ValueError("Never change the total proposal mass")
    return result


PROTECTED_SOURCE = paired.PROTECTED_SOURCE
for old, new in {
    '_, snapshot = original.update_model(model, optimizer, gradients[:len(weights)], weights, False, rate, raw=True)':
        'proposal = proposal_weights(before_metrics, weights)\n'
        '    _, snapshot = original.update_model(model, optimizer, gradients, proposal, False, rate, raw=True)',
    '"point_before": deepcopy(before_metrics), "point_after": None, "accepted": False,':
        '"point_before": deepcopy(before_metrics), "point_after": None, "accepted": False,\n'
        '              "proposal_policy": POLICY, "proposal_weights": proposal,',
}.items():
    PROTECTED_SOURCE = replace_once(PROTECTED_SOURCE, old, new)
PROTECTED_SOURCE_SHA256 = hashlib.sha256(PROTECTED_SOURCE.encode()).hexdigest()
_update_namespace = {**paired._update_namespace, "proposal_weights": proposal_weights, "POLICY": POLICY}
exec(compile(PROTECTED_SOURCE, __file__, "exec"), _update_namespace)
protected_update = _update_namespace["protected_update"]
StepEvaluationError = paired.StepEvaluationError


AUDIT_SOURCE = paired.AUDIT_SOURCE
for old, new in {
    'legacy.check_record(adam, weights, projected=False)':
        'proposal = proposal_weights(point_before, weights)\n'
        '    if record["proposal_policy"] != POLICY or record["proposal_weights"] != proposal:\n'
        '        raise ValueError("Original point-inclusive proposal policy or weights changed")\n'
        '    legacy.check_record(adam, proposal, projected=False)',
    'gram[:len(weights), :len(weights)]': 'gram',
    'products[:len(weights)]': 'products',
    'gradients[:record["composite_tasks"]]': 'gradients',
}.items():
    AUDIT_SOURCE = replace_once(AUDIT_SOURCE, old, new)
AUDIT_SOURCE_SHA256 = hashlib.sha256(AUDIT_SOURCE.encode()).hexdigest()
_audit_namespace = {**paired._audit_namespace, "proposal_weights": proposal_weights, "POLICY": POLICY}
exec(compile(AUDIT_SOURCE, __file__, "exec"), _audit_namespace)
check_record, check_endpoint = _audit_namespace["check_record"], _audit_namespace["check_endpoint"]
