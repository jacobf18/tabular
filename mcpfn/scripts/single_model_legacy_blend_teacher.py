"""Distill the frozen observed-error blend on legacy priors during training only."""

import hashlib

import single_model_general_imputation_teacher as predecessor
from single_model_distillation import teacher_target as observed_blend_target
from single_model_registered_adaptation import replace_assignment, replace_once


POLICY = {"legacy": "observed_blend", "structured": "general", "recsys": "specialist"}
POLICY_TEXT = ("Prospective training-only teacher-policy experiment after V40's audited negative result. "
    "For every original legacy-imputation component, distill the original two frozen teachers' observed-error "
    "blend on the same normalized synthetic input, using the existing observed_scales and fit_routing "
    "implementation. Fit its single convex weight only to finite observed source cells, never query truth, "
    "validation errors or component-specific outcomes. Preserve the original general teacher for structured "
    "imputation and original specialist for ratings, their checkpoint verification and exact original "
    "callbacks. Bind both actual teacher identities for blended targets. Keep evaluation references and "
    "inference unchanged: the student still makes one ordinary network prediction with no inference ensemble. "
    "This revisits the pilot's legacy target under a different, separately registered training recipe; it "
    "does not establish that blend distillation succeeds. This standalone implementation authorizes no "
    "new seeds, GPU probe, training, selection, teacher replacement in V40 or benchmark evaluation.")


ADAPTED_SOURCE = replace_assignment(predecessor.ADAPTED_SOURCE, "POLICY", POLICY)
ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE,
    '"""Training-only general-imputation distillation; original validation references stay unchanged."""',
    '"""Training-only legacy observed-blend distillation; original validation references stay unchanged."""')
ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE,
    '        chosen = teachers[0] if role == "general" else initialization\n'
    '        prediction, routing = parent_teacher_target(chosen, teachers, criterion, episode, options)\n'
    '        return prediction, {**routing, "target": role, "policy": POLICY,\n'
    '                            "teacher_identity": teacher_identities[role]}',
    '        if role == "observed_blend":\n'
    '            prediction, routing = observed_blend_target(teachers, criterion, episode, options)\n'
    '            identity = {name: teacher_identities[name] for name in ("general", "specialist")}\n'
    '        else:\n'
    '            chosen = teachers[0] if role == "general" else initialization\n'
    '            prediction, routing = parent_teacher_target(chosen, teachers, criterion, episode, options)\n'
    '            identity = teacher_identities[role]\n'
    '        return prediction, {**routing, "target": role, "policy": POLICY, "teacher_identity": identity}')
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())
