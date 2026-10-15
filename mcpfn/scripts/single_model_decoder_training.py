"""Train the existing decoder while preserving the ordinary network and frozen backbone."""

from contextlib import contextmanager

import torch

import preflight_single_model_family_projection as coordinates


class DecoderParameterView(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        if model.decoder is None:
            raise ValueError("Require the original decoder")
        model.requires_grad_(False)
        model.zero_grad(set_to_none=True)
        model.decoder.requires_grad_(True)
        self.decoder = model.decoder
        expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        if not expected or expected != {name for name, parameter in self.named_parameters()}:
            raise ValueError("Only original decoder parameters may be trainable")


@contextmanager
def decoder_capture_coordinates(model, view):
    original = coordinates.flatten

    def flatten(current, gradients=False):
        return original(view if current is model else current, gradients=gradients)

    coordinates.flatten = flatten
    try:
        yield
    finally:
        coordinates.flatten = original


def frozen_state_equal(initial, current, mutable_names):
    if set(initial) != set(current) or not set(mutable_names).issubset(initial):
        raise ValueError("The original state inventory must remain intact")
    changed, frozen = [], 0
    for name, before in initial.items():
        after = current[name]
        if before.shape != after.shape or before.dtype != after.dtype or not torch.isfinite(after).all():
            raise ValueError("Every model tensor must retain its finite original shape and dtype")
        equal = torch.equal(before.cpu(), after.detach().cpu())
        if name not in mutable_names:
            frozen += 1
            if not equal:
                raise ValueError(f"Frozen model state changed: {name}")
        elif not equal:
            changed.append(name)
    return {"frozen_tensors": frozen, "changed_decoder_tensors": changed, "backbone_and_buffers_bitwise_unchanged": True}
