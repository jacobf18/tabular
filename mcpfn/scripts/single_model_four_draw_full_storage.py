"""Apply the complete prospective four-draw storage plan to a separately registered run."""

from pathlib import Path
import shutil
import types

import single_model_balanced_native_long_storage as template
import single_model_four_draw_trial_storage as accounting
from single_model_balanced_native_long_storage import storage, sync_directory


RESERVES = dict(accounting.RESERVES)
WRITE_BUDGETS = {stage: dict(values) for stage, values in accounting.WRITE_BUDGETS.items()}
POLICY = (
    "Runtime storage for one separately registered four-draw trial. Retain the complete "
    "4 GiB checkpoint/64 MiB root/192 GiB scratch training allowances, every original later "
    "stage allowance and18/6/50 GiB reserves. Keep ordinary checkpoints and real evaluations "
    "on checkpoint storage, with only the declared synthetic diagnostics on UUID-bound "
    "machine-local scratch. Require the complete requested-area and168-row/atomic-replacement "
    "accounting before registration. Prepare only new empty diagnostic paths; no historical "
    "source, model, validation or scientific retry is authorized. Scratch is not external backup."
)
component_storage = accounting.component_storage
for name in ("component_bytes", "layout", "require_plan", "require_filesystems", "prepare", "verify",
        "require_empty_validation", "require_capacity"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Bind every four-draw storage callback to its own complete plan")
    cloned = types.FunctionType(function.__code__, globals(), name, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned
