"""Prospective full-task teacher replay with complete all-imputation component evidence."""

import inspect

import single_model_legacy_blend_full_replay as full_replay_blueprint
from single_model_registered_adaptation import replace_assignment, replace_once


POLICY_TEXT = (
    "During the independent full-training audit, preserve every admitted task's original role and "
    "complete typed payload digest. For all26 imputation tasks per batch, independently replay both "
    "original frozen teachers, refit the observed-only blend weight and reproduce the exact full "
    "cached target. Archive the observed source, target and both raw component grids, never query "
    "truth; independently reconstruct every saved target on CPU and repeat at qualification. "
    "Preserve rating-specialist checks, all families/components, frozen checkpoint identities and "
    "the original endpoint/reference audits. CPU grid reconstruction occurs inside the GPU-capable "
    "auditor, not a separate CPU-only process. This helper authorizes no run, seeds, selection, "
    "storage migration, reduced coverage, inference ensemble or historical full-autograd claim.")

SOURCE = inspect.getsource(full_replay_blueprint)
SOURCE = replace_once(SOURCE,
    '"""Preserve and independently reconstruct every full-trial legacy blend target."""',
    '"""Preserve and independently reconstruct every full-trial imputation blend target."""')
SOURCE = replace_once(SOURCE, "import single_model_legacy_blend_replay as replay",
    "import single_model_all_imputation_blend_replay as replay")
SOURCE = replace_once(SOURCE, "import single_model_legacy_blend_teacher as teacher_policy",
    "import single_model_all_imputation_blend_teacher as teacher_policy")
SOURCE = replace_assignment(SOURCE, "POLICY", POLICY_TEXT)
SOURCE = SOURCE.replace("legacy_blend_teacher_components", "all_imputation_blend_teacher_components")
SOURCE = SOURCE.replace("legacy_blend_replay", "all_imputation_blend_replay")
SOURCE = SOURCE.replace("legacy_tasks", "blend_tasks")
SOURCE = replace_once(SOURCE, "if len(selected) != 2:", "if len(selected) != 26:")
SOURCE = replace_once(SOURCE, '"Budget both legacy draws in every attempt"',
    '"Budget all26 imputation draws in every original attempt"')
SOURCE = replace_once(SOURCE, 'shape = task["specification"]["shape"]',
    'specification = task["specification"]\n'
    '            shape = specification.get("structured", specification)["shape"]')
SOURCE = replace_once(SOURCE, 'if blend_tasks != 2 * len(counts["admitted_steps"]):',
    'if blend_tasks != 26 * len(counts["admitted_steps"]):')
for old, new in (
    ("separately registered legacy-blend policy", "separately registered all-imputation blend policy"),
    ("never only favorable legacy targets", "never only favorable imputation targets"),
    ("Keep unchanged structured and rating teachers outside blend routing", "Keep rating teachers outside blend routing"),
    ("Every legacy task must retain", "Every imputation task must retain"),
    ("both original legacy components", "both original imputation teacher components"),
    ("Never omit a legacy draw", "Never omit an imputation draw"),
):
    SOURCE = replace_once(SOURCE, old, new)
exec(compile(SOURCE, __file__, "exec"), globals())
