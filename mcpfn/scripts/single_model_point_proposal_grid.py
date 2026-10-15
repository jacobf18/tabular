"""Prospective bounded point-protection proposals and exact finite-step feasibility."""

import math

import numpy as np
import torch


MARGINS = (1e-3, 1e-2, 1e-1)
SCALES = (1., .5, .25)
MAXIMUM_NORM_RATIO = 2.
REPEATS = 2


def specifications():
    return [{"name": f"margin_{margin:g}_scale_{scale:g}", "margin": margin, "scale": scale}
            for margin in MARGINS for scale in SCALES]


@torch.no_grad()
def scaled_parameters(before, projected, scale):
    if (before.ndim != 1 or not before.numel() or before.shape != projected.shape
            or before.dtype != torch.float32 or projected.dtype != torch.float32
            or before.device != projected.device or not torch.isfinite(before).all()
            or not torch.isfinite(projected).all() or scale not in SCALES):
        raise ValueError("Require aligned finite FP32 vectors and a registered contraction")
    if scale == 1.:
        return projected.clone()
    return (before.double() + scale * (projected.double() - before.double())).float()


@torch.no_grad()
def direction_products(gradients, direction, chunk_elements=1_000_000):
    if (gradients.ndim != 2 or direction.ndim != 1 or not direction.numel() or gradients.shape[0] < 1
            or gradients.shape[1] != direction.numel() or gradients.device != direction.device or chunk_elements < 1):
        raise ValueError("Require aligned full gradients and a nonempty actual displacement")
    products = torch.zeros(gradients.shape[0], device=direction.device, dtype=torch.float64)
    square = torch.zeros((), device=direction.device, dtype=torch.float64)
    for start in range(0, direction.numel(), chunk_elements):
        current = direction[start:start + chunk_elements].double()
        tasks = gradients[:, start:start + chunk_elements].double()
        if not torch.isfinite(tasks).all() or not torch.isfinite(current).all():
            raise ValueError("Every gradient and displacement coordinate must be finite")
        products.add_(tasks @ current)
        square.add_(current @ current)
    return products.cpu().numpy(), math.sqrt(max(0., float(square)))


def displacement_decision(products, norm, original_norm):
    products = np.asarray(products, dtype=np.float64)
    if (products.ndim != 1 or not products.size or not np.isfinite(products).all()
            or not math.isfinite(norm) or norm < 0 or not math.isfinite(original_norm) or original_norm <= 0):
        raise ValueError("Require finite measured products and nonzero original Adam proposal")
    protected = bool(np.all(products >= 0) and np.any(products > 0))
    ratio = norm / original_norm
    return {"protected": protected, "adverse_directions": int(np.sum(products < 0)), "norm_ratio": ratio,
            "within_norm_bound": ratio <= MAXIMUM_NORM_RATIO,
            "admissible": protected and 0 < ratio <= MAXIMUM_NORM_RATIO}


def finite_decision(before, after):
    if not before or set(before) != set(after):
        raise ValueError("Retain every task/metric slot")
    changes = []
    worsened, improved, undefined = 0, 0, 0
    all_repeats_retained = True
    for key in before:
        initial, current = before[key], after[key]
        if len(initial) != REPEATS or len(current) != REPEATS:
            raise ValueError("Retain every registered forward repetition")
        if any(value is None for value in (*initial, *current)):
            if not all(value is None for value in (*initial, *current)):
                raise ValueError("Truth-defined metric coverage changed")
            undefined += 1
            changes.append({"slot": key, "defined": False})
            continue
        if not all(math.isfinite(value) and value >= 0 for value in (*initial, *current)):
            raise ValueError("Require finite nonnegative point metrics")
        differences = [right - left for left, right in zip(initial, current)]
        initial_mean, current_mean = float(np.mean(initial)), float(np.mean(current))
        difference = current_mean - initial_mean
        retained = all(value <= 0 for value in differences)
        all_repeats_retained &= retained
        worsened += int(difference > 0)
        improved += int(difference < 0)
        changes.append({"slot": key, "defined": True, "before": initial_mean, "after": current_mean,
                        "difference": difference, "relative_difference": difference / initial_mean if initial_mean else None,
                        "before_repeats": initial, "after_repeats": current, "differences": differences,
                        "every_repeat_retained": retained})
    defined = len(before) - undefined
    return {"defined_slots": defined, "undefined_slots": undefined, "improved_slots": improved,
            "worsened_slots": worsened, "equal_slots": defined - improved - worsened,
            "all_repeats_retained": bool(all_repeats_retained),
            "finite_feasible": bool(defined and all_repeats_retained and improved > 0), "changes": changes}
