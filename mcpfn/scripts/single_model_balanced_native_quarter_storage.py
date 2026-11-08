"""Preserve complete stage allowances for the native-balanced quarter-step trial."""

from math import prod
from pathlib import Path
import shutil
import types

import single_model_standardized_huber_storage as template
from single_model_standardized_huber_storage import storage, sync_directory


RESERVES = dict(template.RESERVES)
WRITE_BUDGETS = {stage: dict(amounts) for stage, amounts in template.WRITE_BUDGETS.items()}
POLICY = template.POLICY.replace('V54', 'V55') + (
    ' Native component/orientation balancing preserves every requested matrix area. '
    'Reserve all original/replayed teacher and retention components over64 attempts; '
    'do not assume intrinsic smaller shapes, compression or rejected updates to fit.'
)
for name in ('component_storage', 'component_bytes', 'layout', 'require_plan', 'require_filesystems', 'prepare',
        'verify', 'require_empty_validation', 'require_capacity'):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError('Native-quarter storage callbacks must bind the complete unchanged budget')
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned
