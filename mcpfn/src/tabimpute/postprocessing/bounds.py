"""Lower and upper bound constraints for imputation logits.

Masks out-of-range buckets by setting their logits to a large negative value
before softmax, so the resulting distribution assigns zero probability to
values outside the allowed range.
"""

from __future__ import annotations

import warnings
from typing import Union

import torch


# Large negative value to effectively zero out logits before softmax.
# Using a finite value instead of -inf for numerical stability.
MASK_LOGIT_VALUE = -1e10


def apply_bounds_constraint(
    logits: torch.Tensor,
    borders: torch.Tensor,
    lower_bound: Union[float, torch.Tensor, None] = None,
    upper_bound: Union[float, torch.Tensor, None] = None,
    mask_value: float = MASK_LOGIT_VALUE,
) -> torch.Tensor:
    """Apply lower and upper bound constraints to imputation logits.

    Runs the imputation through the model (logits are assumed to already be
    the model output). Any logits corresponding to buckets outside the allowed
    range [lower_bound, upper_bound] are set to a large negative value before
    softmax, so those buckets receive effectively zero probability.

    Bucket i is defined by [borders[i], borders[i+1]]. A bucket is masked
    (set to mask_value) if it does not overlap with [lower_bound, upper_bound]:
    - borders[i+1] <= lower_bound  (bucket entirely below range)
    - borders[i] >= upper_bound    (bucket entirely above range)

    Args:
        logits: Model output of shape (..., num_bars) where the last dim
            indexes over buckets. Typically (B, T, num_bars).
        borders: Tensor of shape (num_bars + 1) defining bucket boundaries.
            borders[i] and borders[i+1] define the range of bucket i.
        lower_bound: Minimum allowed value. If None, no lower constraint.
            Can be scalar or broadcastable to logits.shape[:-1].
        upper_bound: Maximum allowed value. If None, no upper constraint.
            Can be scalar or broadcastable to logits.shape[:-1].
        mask_value: Value to assign to out-of-range logits. Default -1e10.

    Returns:
        Constrained logits of the same shape as input, with out-of-range
        buckets masked. Safe to pass to softmax / bar_distribution.
    """
    if lower_bound is None and upper_bound is None:
        return logits

    device = logits.device
    dtype = logits.dtype

    # Ensure borders on same device
    borders = borders.to(device=device, dtype=dtype)

    num_bars = borders.shape[0] - 1
    left_bounds = borders[:-1]   # (num_bars,) - left edge of each bucket
    right_bounds = borders[1:]   # (num_bars,) - right edge of each bucket

    # Build invalid mask: bucket i is invalid if entirely outside [lower, upper]
    # right_bounds, left_bounds: (num_bars,)
    # We need invalid_mask: same shape as logits (..., num_bars)
    logits_shape = logits.shape
    batch_dims = logits_shape[:-1]

    def _to_tensor(x: Union[float, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, (int, float)):
            return torch.tensor(x, device=device, dtype=dtype)
        return x.to(device=device, dtype=dtype)

    invalid_mask = torch.zeros(logits_shape, dtype=torch.bool, device=device)

    if lower_bound is not None:
        lower = _to_tensor(lower_bound)
        # Bucket entirely below range: right_bounds[i] <= lower
        # right_bounds: (num_bars,) -> (1,...,1,num_bars); lower: (...) -> (...,1)
        right_expanded = right_bounds.view((1,) * len(batch_dims) + (num_bars,))
        lower_expanded = lower.unsqueeze(-1) if lower.dim() > 0 else lower
        invalid_mask = invalid_mask | (right_expanded <= lower_expanded)

    if upper_bound is not None:
        upper = _to_tensor(upper_bound)
        # Bucket entirely above range: left_bounds[i] >= upper
        left_expanded = left_bounds.view((1,) * len(batch_dims) + (num_bars,))
        upper_expanded = upper.unsqueeze(-1) if upper.dim() > 0 else upper
        invalid_mask = invalid_mask | (left_expanded >= upper_expanded)

    # Clone to avoid in-place modification
    result = logits.clone()

    # Set out-of-range logits to mask_value
    result = torch.where(invalid_mask, torch.full_like(result, mask_value), result)

    # Warn if any position would have all buckets masked (would cause softmax issues)
    all_masked = invalid_mask.all(dim=-1)
    if all_masked.any():
        warnings.warn(
            f"Bounds constraint masked all buckets for {all_masked.sum().item()} "
            "positions; softmax may produce NaN. Consider widening the bounds.",
            UserWarning,
            stacklevel=2,
        )

    return result
