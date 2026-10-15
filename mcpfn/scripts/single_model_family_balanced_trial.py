"""Prospective V43 recipe preserving the full original qualification protocol."""

import inspect

import single_model_cohort_anchor_trial as blueprint
from single_model_registered_adaptation import replace_assignment, replace_once


POLICY_TEXT = ('A separately registered V43 trial initializes one ordinary student only from unchanged '
    'V29 step25/98 inherited updates after the accepted family-balanced probe and independent three-file '
    'storage preservation. Change only the additional structured retention weighting, not priors, '
    'supervised weights or reference assignment. Preserve the adverse25/26 retention-penalty increases '
    'in the feasibility probe; local acceptance does not prove retention or accuracy. Keep legacy '
    'observed-blend, structured general and rating-specialist targets, all priors/.5/.25/.25 masses, '
    'observed-only normalization, Huber/NLL/CDF/query/point losses, fresh5e-6 AdamW, fixed-half displacement, '
    '.1 margin,84 gradients,30 protections,2x norm bound and exact rollback. Attempt64 fresh batches '
    'with unchanged data-only rejection and no replacement draws. Keep boundaries0/10/25/50/64, all22 '
    'development guards, strict minimax/earlier ties/ineligible0, sole selection and284 fresh confirmation '
    'cases with32 original comparisons, followed by every frozen real suite and per-comparison review. '
    'Pin24 BLAS/4 Torch threads in training, replay, audits and qualification children. Keep the.9 '
    'allocator cap,18/6/50 GiB reserves and2432 MiB/16 MiB/40 GiB training budgets. Independently '
    'replay every original teacher, legacy component and full V8/V29 reference grid, binding all '
    'retention weighting decisions to original task weights and raw gradient scale1. Both references '
    'are training-only: no additional inference model or ensemble. No probe warm start, consumed-seed '
    'reuse, scientific retry, relaxed guard, benchmark selection or synthetic-feasibility success claim. '
    'This recipe helper alone registers no trial or seed.')

SOURCE = inspect.getsource(blueprint)
SOURCE = SOURCE.replace('single_model_cohort_anchor_retention as retention',
    'single_model_family_balanced_retention as retention')
SOURCE = SOURCE.replace('test_single_model_cohort_anchor', 'test_single_model_family_balanced')
SOURCE = replace_assignment(SOURCE, 'POLICY', POLICY_TEXT)
exec(compile(SOURCE, __file__, 'exec'), globals())
