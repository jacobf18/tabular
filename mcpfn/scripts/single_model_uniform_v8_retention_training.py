"""Bind uniform V8 retention to full training and independent grid replay."""

from collections import Counter
import types

import single_model_family_balanced_retention_training as previous
import single_model_uniform_v8_retention as cohort_retention
from single_model_registered_adaptation import replace_assignment


POLICY_TEXT = ('Full-trial integration of uniform V8 training-only retention. ' + cohort_retention.POLICY
    + ' Bind every original task, supervised weight, retention multiplier and raw gradient scale1. '
    'Verify both frozen checkpoint identities but replay only V8 on all26 imputation tasks per batch; '
    'both rating tasks retain their original callback without retention inference. Preserve every '
    'original teacher/component grid, full source/mask, finite protection and qualification guard. '
    'No inference ensemble or historical full-autograd/optimizer-trajectory replay claim.')

for name in ('reference_identities', 'verify_references', 'original_identity', 'cohort_identity'):
    function = getattr(previous, name)
    if function.__closure__ is not None:
        raise ValueError('Uniform-reference integration must not retain predecessor closures')
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned

SOURCE = previous.SOURCE.replace('single_model_family_balanced_retention as retention',
    'single_model_uniform_v8_retention as retention')
SOURCE = replace_assignment(SOURCE, 'POLICY', POLICY_TEXT)
exec(compile(SOURCE, __file__, 'exec'), globals())
