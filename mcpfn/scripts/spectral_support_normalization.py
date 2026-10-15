"""Observed-only variance shrinkage for synthetic sparse-support augmentation."""
import math

import torch

from rowcol_fixed_validation import apply_support_mask


def normalize_remaining_support(source, target, keep, variance_df=4.):
    if not math.isfinite(variance_df) or variance_df <= 0:
        raise ValueError("variance_df must be positive and finite")
    if source.ndim != 3 or source.shape != target.shape or keep.shape != source.shape:
        raise ValueError("Expected matching B x R x C tensors")
    keep = keep.bool() & torch.isfinite(source)
    values = torch.where(keep, source.float(), 0.)
    counts = keep.sum(1, keepdim=True)
    mean = values.sum(1, keepdim=True) / counts.clamp_min(1)
    centered = torch.where(keep, values - mean, 0.)
    ss = centered.square().sum(1, keepdim=True)
    # Pool only within-column variation; column offsets must not inflate the prior.
    pool = ss.sum(2, keepdim=True) / counts.sum(2, keepdim=True).clamp_min(1)
    pool = torch.where(pool > 1e-6, pool, 1.)
    variance = (ss + variance_df * pool) / (counts + variance_df)
    scale = variance.clamp_min(1e-6).sqrt()
    masked = source.float().masked_fill(~keep, float("nan"))
    return (masked - mean) / scale, (target.float() - mean) / scale, mean, scale


def apply_shrunk_support_mask(source, target, mode="mixed", seed=None, variance_df=4.):
    masked, normalized, mask, stats = apply_support_mask(source, target, mode, seed)
    if not stats["support_episode"]:
        return masked, normalized, mask, stats
    x, y, _, scale = normalize_remaining_support(source, target, torch.isfinite(masked), variance_df)
    stats = {**stats, "support_variance_df": variance_df,
             "support_scale_min": float(scale.min()),
             "support_target_rms": float(y[mask].square().mean().sqrt()),
             "support_original_target_rms": float(target[mask].square().mean().sqrt())}
    return x, y, mask, stats
