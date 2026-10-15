"""Two-file checkpoints with measured ceilings and no redundant terminal copy."""
import io
import os
from pathlib import Path
import shutil

import torch

from dwip_teacher_compact_artifacts import CompactStorageGuard
from run_dwip_budgeted_queue_v1 import regular_bytes, RESERVE_BYTES


MIB = 1024**2
BEST = "best_validation_checkpoint.pth"
LATEST = "latest_checkpoint.pth"


def distribution_budget(model_bytes, optimizer_bytes, observed_other_bytes):
    if any(type(value) is not int for value in (model_bytes, optimizer_bytes, observed_other_bytes)) or min(model_bytes, optimizer_bytes) <= 0 or observed_other_bytes < 0:
        raise ValueError("Require positive measured checkpoint sizes and nonnegative other artifacts")
    model = ((model_bytes * 105 + 100 * MIB - 1) // (100 * MIB)) * MIB
    latest = max(3 * model + MIB, ((optimizer_bytes * 105 + 100 * MIB - 1) // (100 * MIB)) * MIB)
    other = max(128 * MIB, ((5 * observed_other_bytes + MIB - 1) // MIB) * MIB)
    persistent = model + latest
    peak = persistent + latest + other
    return {"version": "dwip_distribution_two_file_v1", "reference_model_bytes": model_bytes,
            "reference_optimizer_bytes": optimizer_bytes, "observed_other_artifacts_bytes": observed_other_bytes,
            "model_only_files": [BEST], "rolling_optimizer_file": LATEST, "terminal_checkpoint": LATEST,
            "model_file_ceiling_bytes": model, "latest_file_ceiling_bytes": latest,
            "persistent_checkpoint_bytes": persistent, "atomic_temporary_bytes": latest,
            "other_artifacts_bytes": other, "peak_additional_bytes": peak,
            "reserve_bytes": RESERVE_BYTES, "required_free_bytes_at_launch": RESERVE_BYTES + peak,
            "basis": "Measured unchanged FP32 model and full Adam/scheduler payloads,5% padding rounded upward to MiB; latest also covers3 padded model sizes plus1MiB. Keep one best model and one rolling complete optimizer checkpoint, which is also the terminal checkpoint. Count one largest atomic temporary and max(128MiB,5x measured completed same-pipeline noncheckpoint artifacts). Serialize in memory and reject oversize before writing; no previous artifact is deleted or rewritten by this policy."}


def measured_budget(reference_root, reference_work, preflight_sizes, external_paths=()):
    reference_root, reference_work = Path(reference_root), Path(reference_work)
    expected = {BEST, LATEST, "checkpoint_64000.pth"}
    if {path.name for path in reference_work.glob("*checkpoint*.pth*")} != expected:
        raise ValueError("Require the complete known three-file reference, without failed temporaries")
    model = max(preflight_sizes["model"], *(int((reference_work / name).stat().st_size)
                for name in (BEST, "checkpoint_64000.pth")))
    latest = max(preflight_sizes["optimizer"], (reference_work / LATEST).stat().st_size)
    other = regular_bytes(reference_root) + sum(regular_bytes(path) for path in reference_work.iterdir() if path.name not in expected)
    other += sum(regular_bytes(path) for path in external_paths)
    return distribution_budget(model, latest, other)


class DistributionCheckpointer:
    def __init__(self, work, budget, guard):
        self.work, self.budget, self.guard = Path(work), budget, guard

    def save(self, payload, name):
        if name not in (BEST, LATEST):
            raise ValueError("Only best-model and rolling-optimizer files are registered")
        if (name == LATEST) != ("optimizer" in payload):
            raise ValueError("Checkpoint payload does not match its registered model/optimizer role")
        path = self.work / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        if path.is_symlink() or temporary.exists() or temporary.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("Preserve unexpected checkpoint paths or failed temporaries")
        self.guard()
        ceiling = self.budget["latest_file_ceiling_bytes" if name == LATEST else "model_file_ceiling_bytes"]
        with io.BytesIO() as buffer:
            torch.save(payload, buffer)
            if buffer.tell() > ceiling:
                raise RuntimeError("Serialized checkpoint exceeds its ceiling; no artifact was written")
            self.guard()
            buffer.seek(0)
            with temporary.open("xb") as stream:
                shutil.copyfileobj(buffer, stream, length=8 * MIB)
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary, path)
        self.guard()


__all__ = ["BEST", "LATEST", "DistributionCheckpointer", "CompactStorageGuard", "distribution_budget", "measured_budget"]
