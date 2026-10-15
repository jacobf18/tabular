"""Replay only cached rating-normalization vectors before the original scalar transform."""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import inspect

import torch

import single_model_distillation as distillation
from single_model_registered_adaptation import replace_once


POLICY = ("Regenerate ratings and context selection with the unchanged original sampler. Substitute only "
    "the four saved observed-data normalization vectors after context cropping and before the original "
    "location and sequential normalization arithmetic. Do not restore source/target/mask grids, modify "
    "the frozen prior file or change teacher data. Reject missing, extra, nonfinite, wrongly shaped or "
    "nonpositive scale vectors and repeated/missing application. Restore the original sampler after each "
    "context, including errors. This is compact-state replay, not seed-only regeneration.")
NAMES = ("means", "stds", "row_offsets", "row_scales")
ORIGINAL_SAMPLE = distillation.sample_rating_episode
ACTIVE = ContextVar("single_model_compact_rating_replay", default=None)


def validate_scaling(saved):
    if not isinstance(saved, dict) or set(saved) != set(NAMES):
        raise ValueError("Replay exactly the four normalization vectors, never matrix data")
    for name in NAMES:
        value = saved[name]
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32
                or value.layout != torch.strided or value.ndim != 1 or value.numel() == 0
                or value.requires_grad or not torch.isfinite(value).all()):
            raise ValueError("Require nonempty finite detached CPU float32 normalization vectors")
        if name in ("stds", "row_scales") and not (value > 0).all():
            raise ValueError("Both saved normalization scales must be strictly positive")
    if saved["means"].shape != saved["stds"].shape or saved["row_offsets"].shape != saved["row_scales"].shape:
        raise ValueError("Require aligned column and row normalization vectors")


def restore_scaling(means, stds, row_offsets, row_scales):
    active = ACTIVE.get()
    if active is None or active["calls"] != 0:
        raise ValueError("Use each compact rating state exactly once in its declared context")
    values = (means, stds, row_offsets, row_scales)
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
           or value.shape != active["vectors"][name].shape
           or value.device != means.device for name, value in zip(NAMES, values)):
        raise ValueError("Saved vectors must match the actual cropped two-way user-scale coordinates")
    active["calls"] += 1
    return tuple(active["vectors"][name].to(device=value.device).clone() for name, value in zip(NAMES, values))


ADAPTED_SOURCE = replace_once(inspect.getsource(ORIGINAL_SAMPLE),
    '    location = means if row_offsets is None else means + row_offsets[:, None]',
    '    means, stds, row_offsets, row_scales = restore_scaling(means, stds, row_offsets, row_scales)\n'
    '    location = means if row_offsets is None else means + row_offsets[:, None]')
ADAPTED_SOURCE_SHA256 = hashlib.sha256(ADAPTED_SOURCE.encode()).hexdigest()
_namespace = {**ORIGINAL_SAMPLE.__globals__, "restore_scaling": restore_scaling}
exec(compile(ADAPTED_SOURCE, __file__, "exec"), _namespace)
sample_with_saved_scaling = _namespace["sample_rating_episode"]


@contextmanager
def runtime(saved):
    if ACTIVE.get() is not None or distillation.sample_rating_episode is not ORIGINAL_SAMPLE:
        raise ValueError("Do not nest or replace an already adapted rating sampler")
    receipt = {"calls": 0, "vector_bytes": 0}
    if saved is None:
        yield receipt
        return
    validate_scaling(saved)
    active = {**receipt, "vectors": {name: saved[name].clone() for name in NAMES}}
    active["vector_bytes"] = sum(value.numel() * value.element_size() for value in active["vectors"].values())
    token = ACTIVE.set(active)
    distillation.sample_rating_episode = sample_with_saved_scaling
    try:
        yield active
        if active["calls"] != 1:
            raise ValueError("The saved rating state was not applied exactly once")
    finally:
        distillation.sample_rating_episode = ORIGINAL_SAMPLE
        ACTIVE.reset(token)
