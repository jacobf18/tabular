"""Prospective cohort-reference full-trial recipe with uniform exact BLAS arithmetic."""

import single_model_legacy_blend_trial as previous
import single_model_cohort_anchor_retention as retention
from single_model_registered_adaptation import replace_once


ARITHMETIC = {"blas_threads": 24, "torch_threads": 4}
reference_bytes = previous.reference_bytes
POLICY = ("A separately registered V42 trial initializes one ordinary student only from unchanged V29 "
    "step25/98 inherited updates after the complete accepted cohort-reference probe and independent storage "
    "preservation. Change only the training retention reference assignment. " + retention.POLICY
    + " The accepted probe proves feasibility only: every soft retention penalty increased, despite all30 "
    "original point protections passing. Preserve this adverse result. Keep legacy observed-blend, structured "
    "general and rating-specialist targets, all priors/.5/.25/.25 masses, observed-only normalization, "
    "Huber/NLL/CDF/query/point losses, fresh5e-6 AdamW,fixed-half displacement,.1 margin,84 gradients,30 "
    "protections,2x norm bound and exact rollback. Attempt64 fresh batches with unchanged data-only rejection "
    "and no replacement draws. Keep boundaries0/10/25/50/64, all22 development guards, strict minimax/earlier "
    "ties/ineligible0, sole selection and284 fresh confirmation cases with32 original comparisons, followed "
    "by every frozen real suite and per-comparison review. Pin24 BLAS/4 Torch threads in training, replay, "
    "audits and qualification. Keep the.9 allocator cap,18/6/50 GiB reserves and2432 MiB/16 MiB/40 GiB "
    "training budgets, complete original teacher/component replay and one complete retention grid per "
    "imputation task. Both references are training-only: no additional inference model or ensemble. No probe "
    "warm start, consumed-seed reuse, scientific retry, relaxed guard, benchmark selection or success claim "
    "from synthetic feasibility. This recipe helper alone registers no trial or seed.")


def wrap_cli(source):
    marker = '\nif __name__ == "__main__":\n'
    if source.count(marker) != 1:
        raise ValueError("Wrap exactly one original CLI without replacing its stages, locks or failure handling")
    prefix, body = source.split(marker)
    return (prefix + marker + "    with arithmetic_recovery.previous.original_arithmetic():\n"
        + "".join("    " + line if line.strip() else line for line in body.splitlines(keepends=True)))


def adapt_training(source):
    source = previous.adapt_training(source)
    return replace_once(source, 'or config["teacher_policy"] != teacher_policy.POLICY',
        'or config["teacher_policy"] != teacher_policy.POLICY\n'
        '            or config["arithmetic"] != all_trial.ARITHMETIC')


def adapt_registration(source):
    source = previous.adapt_registration(source)
    source = replace_once(source, 'config = {"policy": POLICY,',
        'config = {"policy": POLICY, "arithmetic": all_trial.ARITHMETIC,')
    return replace_once(source, '    sources.update(Path(module.__file__).resolve()',
        '    sources.update({REPO / "tests/test_single_model_cohort_anchor_trial.py", '
        'REPO / "tests/test_single_model_cohort_anchor_retention_training.py", '
        'REPO / "tests/test_single_model_cohort_anchor_retention_trial.py", '
        'REPO / "tests/test_single_model_cohort_anchor_retention_audit.py", '
        'REPO / "tests/test_single_model_cohort_anchor_retention_pipeline.py"})\n'
        '    sources.update(Path(module.__file__).resolve()')


def adapt_recipe(source):
    source = previous.adapt_recipe(source)
    source = replace_once(source, 'import single_model_full_network_training as full_network',
        'import single_model_full_network_training as full_network\n'
        'import recover_single_model_legacy_blend_blas_stable as arithmetic_recovery')
    return wrap_cli(source)
