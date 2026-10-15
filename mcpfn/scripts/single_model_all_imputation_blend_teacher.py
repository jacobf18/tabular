"""Prospective training-only blend targets for every synthetic imputation task."""

import hashlib

import single_model_legacy_blend_teacher as legacy_blueprint
from single_model_distillation import teacher_target as observed_blend_target
from single_model_registered_adaptation import replace_assignment, replace_once


POLICY = {"legacy": "observed_blend", "structured": "observed_blend", "recsys": "specialist"}
POLICY_TEXT = (
    "Prospective teacher-fidelity hypothesis: use the original observed-only blend for all synthetic "
    "imputation families and components, with the original specialist callback for ratings. "
    "The blend is not an established better synthetic target: on V44's consumed confirmation bank, "
    "structured common NRMSE is 0.6764002341944885 versus general's 0.6487101563731997, "
    "with censoring about 28.60 percent worse. Keep all original qualification comparisons. "
    "Preserve checkpoint identity checks, normalization, supervised weights, evaluation references "
    "and single-network inference. Never select targets by hidden truth, family performance or seed. "
    "This standalone helper authorizes no scientific run, fresh seeds, checkpoint selection or "
    "changes to any previously registered experiment.")

ADAPTED_SOURCE = replace_assignment(legacy_blueprint.ADAPTED_SOURCE, "POLICY", POLICY)
ADAPTED_SOURCE = replace_once(ADAPTED_SOURCE,
    '"""Training-only legacy observed-blend distillation; original validation references stay unchanged."""',
    '"""Training-only all-imputation observed-blend distillation; original validation references stay unchanged."""')
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
exec(compile(ADAPTED_SOURCE, __file__, "exec"), globals())
