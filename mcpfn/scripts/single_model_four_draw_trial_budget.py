"""Prospective equal-task-budget four-draw trial schedule; no scientific execution."""

import single_model_balanced_native_long_budget as original_budget
import single_model_four_draw_batching as batching
import single_model_four_draw_retention as retention


STEPS = 128
BOUNDARIES = (0, 5, 13, 25, 32, 64, 96, 128)
DATA_SEED, GRADIENT_SEED = 1511100901, 1512100901
CONFIRMATION_SEEDS = (1521100901, 1522100901)
SPECIFICATION = {
    "attempts": STEPS,
    "boundaries": list(BOUNDARIES),
    "reference_source_boundaries": list(original_budget.BOUNDARIES),
    "paired_source_boundaries": [2 * step for step in BOUNDARIES],
    "boundary_rounding": "ceil each original positive source boundary divided by two; original25 maps to paired13/source26",
    "tasks_per_attempt": 56,
    "total_scheduled_training_tasks": 7168,
    "native_tasks": 512,
    "structured_tasks": 6144,
    "rating_tasks": 512,
    "imputation_teacher_targets": 6656,
    "imputation_teacher_components": 13312,
    "retention_targets": 6656,
    "retention_components": 12800,
    "retention_component_roles": {"v8": 6144, "v29": 6656},
    "rejected_attempts_consume_budget": True,
    "fresh_seed_admission_required": True,
    "probe_native_shape_override_applied": False,
    "initializer": "unchanged actual V29, never a probe endpoint",
    "adaptive_stopping": False,
}
POLICY = (
    "Prospective equal-task-budget four-draw full-trial plan, pending complete independent closure "
    "of the original feasibility probe. Pair all256 fresh V56-style metadata batches into128 "
    "attempts, retaining every original requested shape, orientation, component and task exactly "
    "once with half weights. Preserve7168 scheduled slots, not a guarantee of7168 admitted tasks. "
    "Whole-batch admission rejection can reduce coverage; no replacement or partial salvage. "
    "Do not propagate the feasibility probe's four-native maximum-shape stress override into "
    "this full schedule. Use eight prospectively fixed validation boundaries0/5/13/25/32/64/96/128: "
    "ceil each original source boundary divided by two, explicitly mapping source25 to26. "
    "No extra validation boundary, favorable stopping or consumed checkpoint reevaluation. "
    "Keep every original22 development guard, strict minimax selection, earlier ties and "
    "ineligible step0; one284-case fresh confirmation/all32 comparisons and the full480-comparison "
    "real review remain required. Preserve actual V29 initialization, quarter-protected CUDA "
    "arithmetic, exact FP32 gradient staging, all losses/references/teachers and single-network "
    "inference. The declared streams are prospective metadata only, neither reserved nor sampled. "
    "Require separately admitted fresh training/confirmation seeds, complete storage/host/GPU "
    "capacity and trainer/audit integration before registration or execution. Probe success "
    "does not authorize training, establish validation gains or fulfill the model goal."
)


def batches(training, data_seed=DATA_SEED, gradient_seed=GRADIENT_SEED):
    result = batching.pair_batches(original_budget.batches(training, data_seed, gradient_seed))
    for batch in result:
        retention.validate_task_mixture(batch["tasks"])
    return result
