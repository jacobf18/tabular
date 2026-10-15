"""Keep the complete fixed-mean component storage allowance for the V51 trial."""

from math import prod
from pathlib import Path
import shutil
import types

import single_model_dual_parent_storage as template
from single_model_dual_parent_storage import storage, sync_directory


RESERVES = dict(template.RESERVES)
WRITE_BUDGETS = {stage: dict(amounts) for stage, amounts in template.WRITE_BUDGETS.items()}
POLICY = (
    "Prospectively retain the complete48 GiB training scratch allowance for V51, including "
    "every original and independently replayed FP32 retention component, serialization and "
    "atomic replacement. Native V29 is one component and structured V8/V29 remains two; "
    "all64 batches still reserve3200 components without compression or rejection assumptions. "
    "Keep2432 MiB checkpoint/16 MiB root allowances,18/6/50 GiB reserves and every later-stage "
    "allowance unchanged. Do not alter any prior plan. Scratch is machine-local, not external "
    "or provider-persistent backup."
)


for name in ("component_storage", "component_bytes", "layout", "require_plan", "require_filesystems", "prepare",
        "verify", "require_empty_validation", "require_capacity"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Native-parent storage must not retain predecessor closures")
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned
