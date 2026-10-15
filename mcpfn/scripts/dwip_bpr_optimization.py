"""First-order protection of a synthetic rating objective during optimizer steps."""
import math

import torch


EXPOSURE_PARAMETER_NAMES = frozenset({
    "feature_encoder.mask_token",
    "feature_encoder.missingness_embedding.weight",
    "feature_encoder.row_cls_embedding.weight",
    "feature_encoder.col_cls_embedding.weight",
})


def configure_trainable_parameters(model, scope="full"):
    if scope not in ("full", "exposure_embeddings"):
        raise ValueError("Unknown synthetic optimization parameter scope")
    named = dict(model.named_parameters())
    selected = set(named) if scope == "full" else set(EXPOSURE_PARAMETER_NAMES)
    if not selected or not selected.issubset(named):
        raise ValueError("Optimization scope does not match the DWIP parameter names")
    for name, parameter in named.items():
        parameter.requires_grad_(name in selected)
        parameter.grad = None
    return {"scope": scope, "parameter_names": sorted(selected),
            "parameter_tensors": len(selected),
            "parameter_elements": sum(named[name].numel() for name in selected),
            "frozen_parameter_tensors": len(named) - len(selected)}


def verify_frozen_parameter_state(current, initial, scope="full"):
    if scope not in ("full", "exposure_embeddings"):
        raise ValueError("Unknown synthetic optimization parameter scope")
    if current.keys() != initial.keys():
        raise ValueError("Optimization changed the model state keys")
    if scope == "full":
        return 0
    if not EXPOSURE_PARAMETER_NAMES.issubset(current):
        raise ValueError("Optimization scope does not match the DWIP parameter names")
    frozen = set(current) - EXPOSURE_PARAMETER_NAMES
    if any(not torch.equal(current[name], initial[name]) for name in frozen):
        raise ValueError("Frozen model state changed during exposure-only optimization")
    return len(frozen)


@torch.no_grad()
def protected_optimizer_step(optimizer, parameters, protected_gradients, tolerance=1e-7):
    if len(parameters) != len(protected_gradients) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Require aligned protected gradients and a positive tolerance")
    if any(gradient is not None and (gradient.shape != parameter.shape or gradient.device != parameter.device)
           for parameter, gradient in zip(parameters, protected_gradients)):
        raise ValueError("Protected gradient shape/device mismatch")
    selected = [(parameter, gradient, parameter.detach().clone()) for parameter, gradient in zip(parameters, protected_gradients)
                if gradient is not None]
    if not selected:
        optimizer.step()
        return {"protected_step_projected": False, "protected_dot_before": 0., "protected_dot_after": 0.,
                "protected_step_rolled_back": False, "protected_first_order_passed": True}
    norm_squared = sum(gradient.double().square().sum() for _, gradient, _ in selected)
    if not torch.isfinite(norm_squared):
        raise ValueError("Nonfinite protected synthetic gradient")

    def directional_derivative():
        return sum(((parameter.double() - original.double()) * gradient.double()).sum()
                   for parameter, gradient, original in selected)

    optimizer.step()
    before = directional_derivative()
    if not torch.isfinite(before):
        raise ValueError("Nonfinite optimizer update")
    after = before
    projected = False
    for _ in range(3):
        if after.item() <= tolerance or norm_squared.item() == 0:
            break
        coefficient = 1.01 * (after / norm_squared).item()
        for parameter, gradient, _ in selected:
            parameter.add_(gradient, alpha=-coefficient)
        projected = True
        after = directional_derivative()
    projection_dot = after.item()
    rolled_back = bool(torch.isfinite(after)) and projection_dot > tolerance
    if rolled_back:
        for parameter, _, original in selected:
            parameter.copy_(original)
        after = directional_derivative()
    passed = bool(torch.isfinite(after)) and after.item() <= tolerance
    if not passed:
        raise ValueError("Synthetic first-order update protection failed")
    return {"protected_step_projected": projected, "protected_dot_before": before.item(),
            "protected_dot_after": after.item(), "protected_gradient_norm": norm_squared.sqrt().item(),
            "protected_step_rolled_back": rolled_back, "protected_dot_after_projection": projection_dot,
            "protected_first_order_passed": passed}
