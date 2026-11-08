"""Budget the complete256-attempt trial without modifying any previous storage plan."""

import inspect
from math import prod
from pathlib import Path
import shutil
import types

import single_model_balanced_native_quarter_storage as template
import single_model_balanced_native_long_budget as budget
from single_model_balanced_native_quarter_storage import storage, sync_directory


RESERVES = dict(template.RESERVES)
WRITE_BUDGETS = {stage: dict(amounts) for stage, amounts in template.WRITE_BUDGETS.items()}
WRITE_BUDGETS["training"].update(checkpoint=4096 * 1024**2, root=64 * 1024**2, scratch=192 * 1024**3)
POLICY = (
    "Reserve4096 MiB checkpoint,64 MiB root and192 GiB machine-local scratch for the256-attempt "
    "trial, including all6656 reference targets and12800 retention components, original/replayed "
    "teacher components, metadata, full84-gradient terminal evidence and atomic replacement. "
    "Preserve18/6/50 GiB reserves and every later-stage allowance. Compute complete requested "
    "areas before admission; never assume compression, intrinsic smaller shapes or rejected "
    "batches. Preserve all original storage plans. Scratch is not external or persistent backup."
)
COMPONENT_SOURCE = budget.adapt_component_storage(inspect.getsource(template.component_storage))
exec(compile(COMPONENT_SOURCE, __file__, "exec"), globals())
for name in ("component_bytes", "layout", "require_plan", "require_filesystems", "prepare", "verify",
        "require_empty_validation", "require_capacity"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Long-budget storage must not retain a historical callback closure")
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned
