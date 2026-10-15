"""Bounded new-trial artifacts without modifying earlier queues or their guard."""
import math
import os
from pathlib import Path
import shutil
import time

import torch

from run_dwip_budgeted_queue_v1 import check_storage, regular_bytes, RESERVE_BYTES
from run_rowcol_controlled_experiments import BENCH, write_json


GIB = 1024**3
MIB = 1024**2
MODEL_FILES = ("best_validation_checkpoint.pth", "checkpoint_64000.pth")
ROLLING = "latest_checkpoint.pth"


def compact_budget(reference_work):
    model_paths = [path for path in reference_work.glob("*checkpoint*.pth") if path.name != ROLLING]
    if not model_paths or not (reference_work / ROLLING).is_file():
        raise ValueError("Require completed reference model and optimizer files")
    measured_model = max(path.stat().st_size for path in model_paths)
    measured_latest = (reference_work / ROLLING).stat().st_size
    model = math.ceil(measured_model * 1.25 / MIB) * MIB
    latest = max(3 * model + MIB, math.ceil(measured_latest * 1.25 / MIB) * MIB)
    other = 3 * GIB // 4
    persistent = len(MODEL_FILES) * model + latest
    peak = persistent + latest + other
    return {"version": "dwip_teacher_compact_v1", "reference_work": str(reference_work),
            "reference_model_bytes": measured_model, "reference_latest_bytes": measured_latest,
            "model_only_files": list(MODEL_FILES), "model_file_ceiling_bytes": model,
            "latest_file_ceiling_bytes": latest, "persistent_checkpoint_bytes": persistent,
            "atomic_temporary_bytes": latest, "other_artifacts_bytes": other,
            "peak_additional_bytes": peak, "reserve_bytes": RESERVE_BYTES,
            "required_free_bytes_at_launch": peak + RESERVE_BYTES}


class CompactStorageGuard:
    def __init__(self, root, work, budget):
        self.root, self.work, self.budget = Path(root), Path(work), budget
        if self.root == self.work or self.root in self.work.parents or self.work in self.root.parents:
            raise ValueError("Queue and work directories must be disjoint")
        if any(path.is_symlink() for path in (self.root, self.work)):
            raise ValueError("Owned output roots cannot be symlinks")
        if any(path.stat().st_dev != self.work.stat().st_dev for path in (self.root, BENCH)):
            raise ValueError("All output locations must share the budgeted filesystem")

    def __call__(self, launch=False):
        external = list((BENCH / "logs").glob(f"tabimpute_{self.root.name}_*.npz"))
        allocated = sum(regular_bytes(path) for path in (self.root, self.work, *external))
        status = check_storage(self.budget, shutil.disk_usage(self.work).free, allocated, launch=launch)
        write_json(self.root / "storage_status.json", {**status, "updated_unix": time.time(),
                   "owned_external_paths": [str(path) for path in external]})
        return status


class CompactCheckpointer:
    def __init__(self, work, budget, guard):
        self.work, self.budget, self.guard = Path(work), budget, guard

    def save(self, payload, name):
        if name not in (*MODEL_FILES, ROLLING):
            raise ValueError("Unregistered checkpoint filename; preserve compact policy")
        path = self.work / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        if temporary.exists() or path.is_symlink():
            raise ValueError("Preserve unexpected existing temporary or symlink checkpoint")
        self.guard()
        torch.save(payload, temporary)
        ceiling = self.budget["latest_file_ceiling_bytes" if name == ROLLING else "model_file_ceiling_bytes"]
        if temporary.stat().st_size > ceiling:
            raise RuntimeError("Checkpoint exceeds its reviewed ceiling; preserve the temporary and stop")
        os.replace(temporary, path)
        self.guard()


def fixed_improves(metrics, best_score):
    score = metrics["macro_rmse"]
    if not math.isfinite(score) or score < 0:
        raise ValueError("Require a finite nonnegative fixed macro RMSE")
    return score < best_score
