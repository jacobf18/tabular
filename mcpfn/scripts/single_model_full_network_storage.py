"""Prospectively budget full-network training without changing earlier storage plans or reserves."""

from pathlib import Path
import shutil
import types

import single_model_scratch_diagnostics as template
from single_model_scratch_diagnostics import storage, sync_directory


RESERVES = dict(template.RESERVES)
WRITE_BUDGETS = {stage: dict(amounts) for stage, amounts in template.WRITE_BUDGETS.items()}
WRITE_BUDGETS["training"].update(checkpoint=2304 * 1024**2, scratch=32768 * 1024**2)
POLICY = ("For a new full-network trial only, prospectively expand the training allowance to2304 MiB checkpoint "
          "and32768 MiB scratch to cover all238 optimizer coordinates,84 raw gradients and atomic checkpoint "
          "replacement. Keep every original reserve, root allowance and later qualification/publication/real-"
          "evaluation stage budget unchanged. Never mutate an earlier module, plan, source or artifact. " + template.POLICY)


for name in ("layout", "require_plan", "require_filesystems", "prepare", "verify", "require_empty_validation", "require_capacity"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Never retain an earlier storage-plan closure")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned
