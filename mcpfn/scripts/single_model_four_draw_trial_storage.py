"""Prospective complete four-draw storage accounting without allocation or training."""

from collections import Counter
from math import prod

import single_model_balanced_native_long_storage as predecessor
import single_model_four_draw_trial_budget as budget
import single_model_four_draw_gradient_staging as staging


PARAMETERS = 77998496
RESERVES = dict(predecessor.RESERVES)
WRITE_BUDGETS = {stage: dict(amounts) for stage, amounts in predecessor.WRITE_BUDGETS.items()}
POLICY = (
    "Metadata-only four-draw full-trial capacity plan. Account for all128 scheduled56-task "
    "attempts, including every requested native/structured/rating shape, all6656 imputation "
    "targets,13312 teacher components and12800 retention components. Never reduce storage "
    "based on expected rejection, intrinsic smaller shapes, compression or aliasing. Preserve "
    "the complete192 GiB scratch/4 GiB checkpoint/64 MiB root training allowances and all "
    "later-stage allowances, plus18/6/50 GiB reserves. Double the terminal gradient capacity "
    "to168 complete FP32 rows and separately account for simultaneous atomic raw replacement, "
    "twelve snapshot vectors, ordinary checkpoint files and every original/replayed target. "
    "Retain176 GiB host working plus32 GiB reserve; a storage plan is not a measured runtime "
    "bound. No allocation, relocation, seed registration, inference, training or scientific "
    "authorization. Full trainer/audit integration and byte-preserving capacity admission "
    "remain prerequisites. Scratch is machine-local, not external backup."
)


def areas(batches):
    if (len(batches) != budget.STEPS
            or [batch["step"] for batch in batches] != list(range(1, budget.STEPS + 1))):
        raise ValueError("Budget all128 complete ordered attempts")
    imputation, components, all_tasks = [], [], []
    for batch in batches:
        tasks = batch["tasks"]
        budget.retention.validate_task_mixture(tasks)
        if ([task["index"] for task in tasks] != list(range(56))
                or batch["source_steps"] != [2 * batch["step"] - 1, 2 * batch["step"]]):
            raise ValueError("Retain the original two source batches and every ordered task")
        for task in tasks:
            specification = task["specification"]
            shape = specification.get("structured", specification)["shape"]
            if len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape):
                raise ValueError("Budget exact positive requested matrix dimensions")
            area = prod(shape)
            all_tasks.append(area)
            if specification["domain"] == "imputation":
                imputation.append(area)
                components.append((2 if "structured" in specification else 1, area))
            elif specification["domain"] != "recsys":
                raise ValueError("Never add or reinterpret a training domain")
    if (len(all_tasks) != 7168 or len(imputation) != 6656 or sum(count for count, area in components) != 12800
            or Counter(count for count, area in components) != {1: 512, 2: 6144}):
        raise ValueError("Preserve every requested target and both structured reference components")
    return all_tasks, imputation, components


def reference_storage(batches):
    all_tasks, imputation, components = areas(batches)
    return {"tasks": len(imputation), "cells": sum(imputation), "archived_bytes": 9 * sum(imputation),
        "replay_bytes": 4 * sum(imputation), "atomic_reference_bytes": 9 * max(imputation),
        "serialization_and_manifest_allowance": 256 * 1024**2}


def component_storage(batches):
    all_tasks, imputation, components = areas(batches)
    sizes = [count * area for count, area in components]
    return {"tasks": len(imputation), "components": sum(count for count, area in components),
        "component_cells": sum(sizes), "archived_component_bytes": 4 * sum(sizes),
        "replay_component_bytes": 4 * sum(sizes), "atomic_component_bytes": 4 * max(sizes),
        "serialization_and_manifest_allowance": 256 * 1024**2}


def teacher_storage(batches):
    all_tasks, imputation, components = areas(batches)
    return {"blend_tasks": len(imputation), "raw_component_grids": 2 * len(imputation), "cells": sum(imputation),
        "array_bytes": 16 * sum(imputation), "atomic_archive_bytes": 16 * max(imputation),
        "serialization_and_manifest_allowance": 32 * 1024**2}


def accounting(batches):
    references, components, teachers = reference_storage(batches), component_storage(batches), teacher_storage(batches)
    vector = 4 * PARAMETERS
    result = {
        "checkpoint": {"two_atomic_optimizer_checkpoints": 6 * vector, "selected_checkpoint": vector,
            "metadata_and_audit_allowance": 1024**3},
        "root": {"logs_and_registration_allowance": 64 * 1024**2},
        "scratch": {"raw_gradient_bytes": staging.CAPACITY * vector, "snapshot_vector_allowance": 12 * vector,
            "atomic_raw_allowance": (staging.CAPACITY + 12) * vector,
            "ordinary_model_allowance": 8 * vector, "prediction_task_allowance": 8 * 1024**3,
            "all_imputation_reference_allowance": sum(value for key, value in references.items() if key not in {"tasks", "cells"}),
            "dual_parent_component_allowance": sum(value for key, value in components.items() if key not in {"tasks", "components", "component_cells"}),
            "all_imputation_blend_component_allowance": sum(value for key, value in teachers.items() if key not in {"blend_tasks", "raw_component_grids", "cells"})},
    }
    for volume, fields in result.items():
        if sum(fields.values()) > WRITE_BUDGETS["training"][volume]:
            raise ValueError(f"Complete uncompressed and atomic-write evidence exceeds {volume} allowance")
    return result
