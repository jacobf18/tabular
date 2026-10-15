"""Budget every dual-parent component without changing registered predecessor plans."""

from math import prod
from pathlib import Path
import shutil
import types

import single_model_all_imputation_storage as template
from single_model_all_imputation_storage import storage, sync_directory


RESERVES = dict(template.RESERVES)
WRITE_BUDGETS = {stage: dict(amounts) for stage, amounts in template.WRITE_BUDGETS.items()}
WRITE_BUDGETS["training"]["scratch"] = 48 * 1024**3
POLICY = (
    "Prospectively allow48 GiB scratch for V50 only, explicitly including every additional "
    "FP32 retention component in both the original archive and independent replay, serialization "
    "and atomic replacement. Do not rely on compression, shared tensor storage or rejected batches "
    "to fit. Preserve2432 MiB checkpoint/16 MiB root allowances,18/6/50 GiB reserves and every "
    "later-stage allowance. No prior plan or scientific recipe changes. Scratch is machine-local, "
    "not an external or provider-persistent backup."
)


def component_storage(batches):
    areas = []
    components = 0
    for batch in batches:
        batch_components = 0
        batch_tasks = 0
        for task in batch["tasks"]:
            specification = task["specification"]
            if specification["domain"] != "imputation":
                continue
            shape = specification.get("structured", specification)["shape"]
            if len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape):
                raise ValueError("Budget positive exact full-matrix shapes for every component")
            count = 2 if "structured" in specification else 1
            areas.append(count * prod(shape))
            batch_components += count
            batch_tasks += 1
        if batch_tasks != 26 or batch_components != 50:
            raise ValueError("Budget all26 retention tasks and50 components of each admitted batch")
        components += batch_components
    if len(batches) != 64 or components != 3200:
        raise ValueError("Budget all64 attempted batches without assuming any rejections")
    return {"tasks": len(areas), "components": components, "component_cells": sum(areas),
        "archived_component_bytes": 4 * sum(areas), "replay_component_bytes": 4 * sum(areas),
        "atomic_component_bytes": 4 * max(areas), "serialization_and_manifest_allowance": 64 * 1024**2}


def component_bytes(batches):
    accounting = component_storage(batches)
    return sum(value for key, value in accounting.items() if key not in {"tasks", "components", "component_cells"})


for name in ("layout", "require_plan", "require_filesystems", "prepare", "verify", "require_empty_validation", "require_capacity"):
    function = getattr(template, name)
    if function.__closure__ is not None:
        raise ValueError("Never retain predecessor storage closures")
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned
