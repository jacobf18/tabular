"""Prospective V44 recipe preserving the original full qualification protocol."""

import single_model_family_balanced_trial as blueprint
from single_model_registered_adaptation import replace_assignment


POLICY_TEXT = ('A separately registered V44 trial initializes one ordinary student only from unchanged '
    'V29 step25/98 inherited updates after the accepted uniform-V8 probe and independently rechecked '
    'three-file storage preservation. Change only the retention target assignment from V43: every '
    'imputation task uses frozen V8, with no seed, component or family exception. Keep V29 identity '
    'verification but never choose it as a retention target. Ratings have no retention inference. '
    'Preserve the adverse13/26 probe penalty increases and weighted sum .06122912268815583 to '
    '.07817616306016777; feasibility does not prove retention or accuracy. Keep legacy observed-blend, '
    'structured general and rating-specialist targets, all priors/.5/.25/.25 masses, observed-only '
    'normalization, Huber/NLL/CDF/query/point losses, family-balanced weight10 count-shrunk retention, '
    'fresh5e-6 AdamW, fixed-half displacement,.1 margin,84 raw gradients,30 protections,2x norm bound '
    'and exact rollback. Attempt64 fresh batches without replacement of data-only rejections. Keep '
    'boundaries0/10/25/50/64, all22 development guards, strict minimax/earlier ties/ineligible0, sole '
    'selection and284 fresh confirmation cases with32 original comparisons, then every frozen real '
    'suite and per-comparison review. Pin24 BLAS/4 Torch threads throughout. Keep the.9 allocator cap, '
    '18/6/50 GiB reserves and2432 MiB/16 MiB/40 GiB training budgets. Independently replay every original '
    'teacher, legacy component and V8 reference grid, binding original task weights and gradient scale1. '
    'No inference ensemble, probe/failed-trial warm start, consumed-seed reuse, scientific retry, relaxed '
    'guard, benchmark selection or synthetic-feasibility success claim. This helper registers no seed.')

SOURCE = blueprint.SOURCE.replace('single_model_family_balanced_retention as retention',
    'single_model_uniform_v8_retention as retention')
SOURCE = SOURCE.replace('test_single_model_family_balanced', 'test_single_model_uniform_v8')
SOURCE = replace_assignment(SOURCE, 'POLICY', POLICY_TEXT)
exec(compile(SOURCE, __file__, 'exec'), globals())
