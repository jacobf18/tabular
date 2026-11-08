"""Prospective fixed-duration extension of the unchanged V55 training recipe."""

from copy import deepcopy

from single_model_registered_adaptation import replace_once


STEPS = 256
BOUNDARIES = (0, 10, 25, 50, 64, 128, 192, 256)
TEMPLATE_STEPS = 64
DATA_SEED, GRADIENT_SEED = 1481100901, 1482100901
CONFIRMATION_SEEDS = (1491100901, 1492100901)
SPECIFICATION = {
    "attempts": STEPS,
    "boundaries": list(BOUNDARIES),
    "template_attempts": TEMPLATE_STEPS,
    "template_cycles": 4,
    "tasks_per_attempt": 28,
    "total_training_tasks": 7168,
    "rejected_attempts_consume_budget": True,
    "initializer": "unchanged actual V29",
    "optimizer_restart": "fresh AdamW, not a resumed closed trajectory",
    "adaptive_stopping": False,
}
POLICY = (
    "After V55's complete negative closure and actual-V29 attribution, prospectively extend only "
    "the training budget from64 to256 attempts. Keep the first five boundaries0/10/25/50/64 and "
    "add128/192/256, strict minimax/earlier ties/ineligible0 and one selected ordinary checkpoint. "
    "Repeat the original64-step specification schedule four times with global attempt indices; "
    "replace every data/backward seed through the original replicated-batch constructor before "
    "sampling. Preserve all28 tasks, both native draws, all structured families, both ratings, "
    "requested shapes, balanced components/orientations, domain masses and observed-only "
    "normalization. Use actual V29 initialization and fresh AdamW, never a V55 warm start or "
    "a resumed closed optimizer. Keep the V55 standardized teacher Huber, NLL/CDF/point/rating "
    "losses, frozen reference targets, quarter displacement, inward margin, norm cap and exact "
    "rollback. Rejected attempts consume budget; do not replace them or adapt the stopping time. "
    "Require new disjoint training/confirmation streams and complete replay of all7168 tasks, "
    "up to6656 imputation targets/12800 retention components and all original endpoint audits. "
    "Preserve all22 development guards,284 fresh confirmation cases/all32 comparisons and every "
    "frozen real suite/full review. Five inherited deficits shrinking does not prove longer "
    "training will qualify or generalize; retain all adverse changes and failed trials. No "
    "case/family-specific change, guard relaxation, inference ensemble, real-data selection "
    "or historical full-autograd claim. This helper itself registers no seed or scientific run."
)


def extend_base_batches(batches):
    if (len(batches) != TEMPLATE_STEPS
            or [batch["step"] for batch in batches] != list(range(1, TEMPLATE_STEPS + 1))
            or any(len(batch["tasks"]) != 14 for batch in batches)):
        raise ValueError("Extend only the complete ordered original64-step fourteen-cohort schedule")
    result = []
    for step in range(1, STEPS + 1):
        batch = deepcopy(batches[(step - 1) % TEMPLATE_STEPS])
        batch["step"] = step
        result.append(batch)
    return result


def replicated_batches(training, data_seed=DATA_SEED, gradient_seed=GRADIENT_SEED):
    return [training.mixture.reweight(training.rehearsal.stratify(
        training.replicated.replicate_batch(batch, data_seed, gradient_seed)))
        for batch in extend_base_batches(training.baseline.previous.batches())]


def batches(training, data_seed=DATA_SEED, gradient_seed=GRADIENT_SEED):
    result = [training.native.balance(batch) for batch in replicated_batches(training, data_seed, gradient_seed)]
    for batch in result:
        training.retention.validate_task_mixture(batch["tasks"])
    return result


def adapt_retention(source):
    source = replace_once(source, "step not in range(1, 65)", "step not in range(1, 257)")
    return replace_once(source, "Keep the original64 registered optimizer attempts",
        "Keep the prospectively registered256 optimizer attempts")


def adapt_blend_replay(source):
    source = replace_once(source, "if len(batches) != 64:", "if len(batches) != 256:")
    source = replace_once(source, "Budget the original64 attempts before training",
        "Budget every registered256 attempt before training")
    source = replace_once(source, "step not in range(1, 65)", "step not in range(1, 257)")
    return replace_once(source, '"serialization_and_manifest_allowance": 8 * 1024**2',
        '"serialization_and_manifest_allowance": 32 * 1024**2')


def adapt_component_storage(source):
    source = replace_once(source, "if len(batches) != 64 or components != 3200:",
        "if len(batches) != 256 or components != 12800:")
    source = replace_once(source, "Budget all64 attempted batches without assuming any rejections",
        "Budget all256 attempted batches without assuming any rejections")
    return replace_once(source, '"serialization_and_manifest_allowance": 64 * 1024**2',
        '"serialization_and_manifest_allowance": 256 * 1024**2')


def adapt_reference_storage(source):
    source = replace_once(source, "if len(batches) != 64 or len(areas) != 1664:",
        "if len(batches) != 256 or len(areas) != 6656:")
    source = replace_once(source, "Account for every original imputation task in all64 fresh attempts",
        "Account for every original imputation task in all256 fresh attempts")
    return replace_once(source, '"serialization_and_manifest_allowance": 64 * 1024**2',
        '"serialization_and_manifest_allowance": 256 * 1024**2')
