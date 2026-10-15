"""Budget all imputation reference grids without mutating earlier storage plans."""

from pathlib import Path
import shutil
import types

import single_model_full_network_storage as template
from single_model_full_network_storage import storage, sync_directory


RESERVES = dict(template.RESERVES)
WRITE_BUDGETS = {stage: dict(amounts) for stage, amounts in template.WRITE_BUDGETS.items()}
WRITE_BUDGETS["training"].update(checkpoint=2432 * 1024**2, scratch=40 * 1024**3)
POLICY = ("Prospectively allow2432 MiB checkpoint,16 MiB root and40 GiB scratch for the new all-imputation "
    "trial only. Include every reference prediction, original FP32 source and bool mask, independent FP32 "
    "replay, serialization overhead and one atomic reference replacement. Keep18/6/50 GiB reserves and "
    "every later-stage allowance unchanged. This is not provider-persistent scratch storage or backup. "
    + template.POLICY)


for name in ("layout", "require_plan", "require_filesystems", "prepare", "verify", "require_empty_validation", "require_capacity"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Never retain a preceding storage closure")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned
