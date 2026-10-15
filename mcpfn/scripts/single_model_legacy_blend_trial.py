"""Bind legacy blend distillation and complete component replay to one fresh full trial."""

import single_model_all_imputation_trial as previous
import single_model_legacy_blend_full_replay as blend_replay
from single_model_registered_adaptation import replace_once


reference_bytes = previous.reference_bytes
POLICY = ("V41 initializes only from the unchanged V29 step25/98-update checkpoint after the complete "
    "accepted legacy-blend probe and independent storage preservation. Change only legacy-imputation "
    "training targets to the original observed-source-error blend. Retain original general targets for "
    "structured imputation and specialist targets for ratings, all priors and.5/.25/.25 masses, observed-only "
    "normalization, Huber/NLL/CDF/query/point losses, all-imputation weight10 count-shrunk V29 retention, "
    "fresh5e-6 AdamW, fixed-half proposal displacement,.1 projection margin,84 raw gradients,30 point "
    "protections,2x norm cap and exact rollback. Attempt64 fresh batches, preserving data-only rejections "
    "without teacher/student work or replacement draws. Keep all22 development guards, boundaries0/10/25/"
    "50/64, strict minimax/earlier ties/ineligible0, sole selection and284 fresh confirmation cases with32 "
    "comparisons, then every frozen real benchmark. The ordinary inference network is unchanged and has "
    "no ensemble. Preserve18/6/50 GiB reserves and2432 MiB/16 MiB/40 GiB training budgets. Explicitly budget "
    "all four archived legacy grids at the actual full-trial shapes, not the small probe dimensions. "
    + blend_replay.POLICY + " Scratch is not provider-persistent backup. No probe/V40 warm start, hidden-truth "
    "routing, benchmark selection, relaxed guard, consumed-seed reuse, scientific retry or accuracy claim "
    "from feasibility. Full benchmark success remains unproven.")


def adapt_training(source):
    return replace_once(source, 'or config["teacher_policy"] != teacher_policy.POLICY',
        'or config["teacher_policy"] != teacher_policy.POLICY\n'
        '            or config["legacy_blend_replay"] != blend_replay.specification(SCRATCH, config["batches"])')


def adapt_registration(source):
    source = replace_once(source, 'config = {"policy": POLICY,',
        'config = {"policy": POLICY, "legacy_blend_replay": blend_replay.specification(SCRATCH, batches()),')
    return replace_once(source, '    sources.update(Path(module.__file__).resolve()',
        '    sources.update({REPO / "tests/test_single_model_legacy_blend_full_replay.py", '
        'REPO / "tests/test_single_model_legacy_blend_teacher_trial.py", '
        'REPO / "tests/test_single_model_legacy_blend_teacher_audit.py", '
        'REPO / "tests/test_single_model_legacy_blend_teacher_pipeline.py"})\n'
        '    sources.update(Path(module.__file__).resolve()')


def adapt_recipe(source):
    source = previous.adapt_recipe(source)
    source = replace_once(source, 'import single_model_full_network_training as full_network',
        'import single_model_full_network_training as full_network\n'
        'import single_model_legacy_blend_teacher as teacher_policy\n'
        'import single_model_legacy_blend_full_replay as blend_replay')
    source = replace_once(source,
        'STORAGE_ACCOUNTING["scratch"]["all_imputation_reference_allowance"] = all_trial.reference_bytes(previous_training.batches())',
        'STORAGE_ACCOUNTING["scratch"]["all_imputation_reference_allowance"] = all_trial.reference_bytes(previous_training.batches())\n'
        'STORAGE_ACCOUNTING["scratch"]["legacy_blend_component_allowance"] = blend_replay.storage_bytes(previous_training.batches())')
    source = replace_once(source, 'ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()',
        'ADAPTED_SOURCE = all_trial.adapt_training(ADAPTED_SOURCE)\n'
        'ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()')
    return replace_once(source, 'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())',
        'REGISTER_SOURCE = all_trial.adapt_registration(REGISTER_SOURCE)\n'
        'exec(compile(REGISTER_SOURCE, __file__, "exec"), globals())')
