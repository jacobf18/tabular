"""Preserve per-task loss geometry while halving four-draw mixture weights."""

from pathlib import Path

import single_model_standardized_teacher_huber as template
from single_model_registered_adaptation import replace_once, replace_assignment


FOUR_DRAW_POLICY = (
    "Prospective four-draw loss integration, not scientific execution authorization. Preserve "
    "every native/structured/rating per-task raw gradient and standardized teacher Huber/NLL/"
    "CDF/query/point computation. Halve each original supervised task weight and each effective "
    "retention task weight; their ratio is unchanged. All four native tasks retain actual V29; "
    "all48 structured tasks retain fixed FP32 half V8/V29 targets. Four rating tasks use the "
    "unchanged specialist callback without retention. Keep.5/.25/.25 domain masses and total "
    "structured retention mass.25. No teacher, normalization, architecture, held-out guard or "
    "inference change. A fresh complete capture/replay and capacity-admitted feasibility test "
    "are still required before authorizing training."
)
SOURCE = Path(template.__file__).read_text()
SOURCE = replace_once(SOURCE, "REPLICAS = raw_mean.REPLICAS", "REPLICAS = 4")
SOURCE = replace_once(SOURCE, "STRUCTURED_TASK_MASS = raw_mean.STRUCTURED_TASK_MASS",
    "STRUCTURED_TASK_MASS = STRUCTURED_MASS / (len(FAMILIES) * REPLICAS)")
SOURCE = replace_assignment(SOURCE, "POLICY", FOUR_DRAW_POLICY)
exec(compile(SOURCE, __file__, "exec"), globals())
SPECIFICATION.update(
    structured_retention_family_weighting="equal over all twelve original families and all four replicas",
    structured_retention_task_mass=STRUCTURED_TASK_MASS,
    supervised_task_weighting="one half of each original two-draw supervised task weight",
)


def task_weight(cohort):
    if cohort == "legacy":
        return .125
    if cohort == "recsys":
        return .0625
    if cohort not in FAMILIES:
        raise ValueError("Never add or reinterpret a synthetic cohort")
    return .025 if cohort in ("mar", "mnar_skip_logic") else .00125


def retention_weighting(metadata):
    role = reference_role(metadata)
    cohort = "recsys" if role is None else metadata["family"] if metadata.get("structured", False) else "legacy"
    weight = task_weight(cohort)
    effective = 0. if role is None else STRUCTURED_TASK_MASS if cohort in FAMILIES else weight
    return {"task_weight": weight, "multiplier": effective / weight, "effective_task_weight": effective}


def validate_task_mixture(tasks):
    expected = {"legacy": REPLICAS, "recsys": REPLICAS, **{family: REPLICAS for family in FAMILIES}}
    if Counter(task["cohort"] for task in tasks) != expected:
        raise ValueError("Require four draws from every original cohort")
    for task in tasks:
        if isinstance(task["weight"], bool) or task["weight"] != task_weight(task["cohort"]):
            raise ValueError("Require exactly half the original per-task supervised weight")
