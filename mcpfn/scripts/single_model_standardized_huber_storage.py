"""Keep original reserves and complete stage allowances for the V54 trial."""

from math import prod
from pathlib import Path
import shutil
import types

import single_model_balanced_structured_storage as template
from single_model_balanced_structured_storage import storage, sync_directory


RESERVES = dict(template.RESERVES)
WRITE_BUDGETS = {stage: dict(amounts) for stage, amounts in template.WRITE_BUDGETS.items()}
POLICY = template.POLICY.replace('V53', 'V54')
for name in ('component_storage', 'component_bytes', 'layout', 'require_plan', 'require_filesystems', 'prepare',
        'verify', 'require_empty_validation', 'require_capacity'):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError('Standardized-Huber storage must not retain predecessor closures')
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned
