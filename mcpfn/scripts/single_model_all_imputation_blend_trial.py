"""Bind uniform blend distillation to the unchanged full V44 qualification recipe."""

import single_model_uniform_v8_trial as blueprint
from single_model_registered_adaptation import replace_once


ARITHMETIC = blueprint.ARITHMETIC
reference_bytes = blueprint.reference_bytes
wrap_cli = blueprint.wrap_cli
POLICY = (
    "A separately registered V45 trial initializes only from unchanged V29 step25/98 inherited updates. "
    "After the complete negative V44 real review, accepted all-imputation-blend probe, independent "
    "original/component/reference audits and six-file exact-byte storage preservation, change only "
    "structured-imputation teacher targets from general to the original observed-only blend used "
    "on legacy tasks. Ratings retain the specialist callback. Preserve adverse synthetic teacher "
    "evidence: structured blend common NRMSE .6764002341944885 versus general .6487101563731997 on "
    "V44's consumed bank, including roughly28.60 percent worse censoring. The feasibility probe has "
    "three adverse individual diagnostic slots and16/26 increased retention penalties despite an "
    "accepted grouped update. Neither result proves held-out accuracy. Keep all priors, supervised "
    ".5/.25/.25 masses, observed-only normalization, Huber/NLL/CDF/query/point losses, uniform V8 "
    "weight10 count-shrunk equal-column retention, fresh5e-6 AdamW,fixed-half displacement,.1 margin, "
    "84 raw gradients,30 guards,2x norm bound and exact rollback. Attempt64 fresh batches without "
    "replacing rejected draws. Keep boundaries0/10/25/50/64, all22 development guards, strict minimax, "
    "earlier ties and ineligible0; confirm only the sole selection on284 fresh cases and32 unchanged "
    "comparisons before every frozen real suite and review. Replay all1,664 possible imputation "
    "targets and3,328 raw teacher components,26 V8 references per admitted batch and unchanged "
    "rating targets. Budget41,224,310,912 scratch bytes within40 GiB, with original2432 MiB/16 MiB "
    "checkpoint/root allowances and18/6/50 GiB reserves. Keep24 BLAS/4 Torch threads and.9 allocator "
    "cap. Scratch is machine-local, not external backup. No inference ensemble, probe warm start, "
    "real-benchmark checkpoint selection, seed reuse, scientific retry or relaxed qualification.")


def adapt_training(source):
    return replace_once(blueprint.adapt_training(source), 'config["legacy_blend_replay"]',
        'config["all_imputation_blend_replay"]')


def adapt_registration(source):
    source = replace_once(blueprint.adapt_registration(source), '"legacy_blend_replay":',
        '"all_imputation_blend_replay":')
    return replace_once(source, '    sources.update(Path(module.__file__).resolve()',
        '    sources.update({REPO / "tests/test_single_model_all_imputation_blend_trial.py", '
        'REPO / "tests/test_single_model_all_imputation_blend_audit.py", '
        'REPO / "tests/test_single_model_all_imputation_blend_pipeline.py", '
        'REPO / "tests/test_single_model_all_imputation_blend_teacher.py", '
        'REPO / "tests/test_single_model_all_imputation_blend_full_replay.py"})\n'
        '    sources.update(Path(module.__file__).resolve()')


def adapt_recipe(source):
    source = blueprint.adapt_recipe(source)
    for old, new in (
        ("import single_model_legacy_blend_teacher as teacher_policy", "import single_model_all_imputation_blend_teacher as teacher_policy"),
        ("import single_model_legacy_blend_full_replay as blend_replay", "import single_model_all_imputation_blend_full_replay as blend_replay"),
        ('["legacy_blend_component_allowance"]', '["all_imputation_blend_component_allowance"]'),
    ):
        source = replace_once(source, old, new)
    return source
