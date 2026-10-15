"""Prepare last-block plus decoder training without changing the inference network."""

from contextlib import contextmanager

import torch

import preflight_single_model_family_projection as coordinates
from single_model_composite_gradient_storage import byte_equal


POLICY = ("An optimizer-only parameter view of the existing last transformer block and complete decoder. "
          "Share original parameter objects and names in their original order; introduce no new inference "
          "module, parameter, branch or ensemble. Freeze all other parameters and preserve every buffer. "
          "Keep streamed forward/loss/backward implementations unchanged and restrict only captured optimizer "
          "coordinates. This helper and CPU derivative tests are preparation, not a registered full-size GPU "
          "preflight, new training recipe, validation result or change to the ongoing V23 decoder-only run.")


class TopBlockParameterView(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        blocks, decoder = getattr(model, "transformer_blocks", None), getattr(model, "decoder", None)
        if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) < 2 or not isinstance(decoder, torch.nn.Module):
            raise ValueError("Require the original ordered transformer blocks and complete decoder")
        self.block_index = len(blocks) - 1
        self.prefixes = (f"transformer_blocks.{self.block_index}.", "decoder.")
        selected = [(name, parameter) for name, parameter in model.named_parameters() if name.startswith(self.prefixes)]
        if any(not any(name.startswith(prefix) for name, parameter in selected) for prefix in self.prefixes):
            raise ValueError("Both the original last block and decoder must contain distinct named parameters")
        self.transformer_blocks = torch.nn.ModuleDict({str(self.block_index): blocks[self.block_index]})
        self.decoder = decoder
        actual = list(self.named_parameters())
        if (len(actual) != len(selected) or any(name != original_name or parameter is not original
                for (name, parameter), (original_name, original) in zip(actual, selected))):
            raise ValueError("Retain the exact original selected parameter identities and coordinate order")
        self.mutable_names = tuple(name for name, parameter in selected)
        model.requires_grad_(False)
        model.zero_grad(set_to_none=True)
        self.requires_grad_(True)
        check_layout(model, self)


def check_layout(model, view):
    expected = [(name, parameter) for name, parameter in model.named_parameters() if name.startswith(view.prefixes)]
    actual = list(view.named_parameters())
    trainable = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if (tuple(name for name, parameter in expected) != view.mutable_names or trainable != view.mutable_names
            or len(actual) != len(expected) or any(name != original_name or parameter is not original
                for (name, parameter), (original_name, original) in zip(actual, expected))):
        raise ValueError("Only the original last-block/decoder coordinates may be trainable or captured")


@contextmanager
def capture_coordinates(model, view):
    check_layout(model, view)
    original = coordinates.flatten

    def flatten(current, gradients=False):
        return original(view if current is model else current, gradients=gradients)

    coordinates.flatten = flatten
    try:
        yield
    finally:
        coordinates.flatten = original


def frozen_state_equal(initial, model, view):
    check_layout(model, view)
    current = model.state_dict()
    if set(initial) != set(current) or not set(view.mutable_names).issubset(initial):
        raise ValueError("Preserve the ordinary model's complete original state inventory")
    changed, frozen = [], 0
    for name, before in initial.items():
        after = current[name]
        if before.shape != after.shape or before.dtype != after.dtype or not torch.isfinite(after).all():
            raise ValueError("Every original tensor must retain its finite shape and dtype")
        equal = byte_equal(before.detach().cpu(), after.detach().cpu())
        if name not in view.mutable_names:
            frozen += 1
            if not equal:
                raise ValueError(f"Frozen parameter or buffer changed: {name}")
        elif not equal:
            changed.append(name)
    return {"frozen_tensors": frozen, "changed_trainable_tensors": changed,
            "trainable_parameters": sum(parameter.numel() for parameter in view.parameters()),
            "unselected_parameters_and_all_buffers_bitwise_unchanged": True}
