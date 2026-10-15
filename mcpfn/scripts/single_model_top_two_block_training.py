"""Expose two existing final blocks and decoder to an optimizer, not inference."""

from contextlib import contextmanager

import torch

import single_model_top_block_training as original


POLICY = ("An optimizer-only view of the existing final two transformer blocks and complete decoder. "
          "Keep the ordinary network, forward, state keys and parameter identities unchanged. Freeze every "
          "other parameter and all buffers. This helper and its small CPU tests do not register a training "
          "recipe, consume synthetic seed streams, qualify a model or establish full-size GPU feasibility.")


class TopTwoBlockParameterView(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        blocks, decoder = getattr(model, "transformer_blocks", None), getattr(model, "decoder", None)
        if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) < 3 or not isinstance(decoder, torch.nn.Module):
            raise ValueError("Require at least three original ordered transformer blocks and complete decoder")
        self.block_indices = (len(blocks) - 2, len(blocks) - 1)
        self.prefixes = tuple(f"transformer_blocks.{index}." for index in self.block_indices) + ("decoder.",)
        selected = [(name, parameter) for name, parameter in model.named_parameters() if name.startswith(self.prefixes)]
        if any(not any(name.startswith(prefix) for name, parameter in selected) for prefix in self.prefixes):
            raise ValueError("Each selected original block and decoder must contain distinct named parameters")
        self.transformer_blocks = torch.nn.ModuleDict({str(index): blocks[index] for index in self.block_indices})
        self.decoder = decoder
        actual = list(self.named_parameters())
        if (len(actual) != len(selected) or any(name != original_name or parameter is not source
                for (name, parameter), (original_name, source) in zip(actual, selected))):
            raise ValueError("Retain exact original parameter identities and coordinate order")
        self.mutable_names = tuple(name for name, parameter in selected)
        model.requires_grad_(False)
        model.zero_grad(set_to_none=True)
        self.requires_grad_(True)
        check_layout(model, self)


def check_layout(model, view):
    blocks = getattr(model, "transformer_blocks", None)
    if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) < 3:
        raise ValueError("Preserve the original ordered transformer block inventory")
    indices = (len(blocks) - 2, len(blocks) - 1)
    prefixes = tuple(f"transformer_blocks.{index}." for index in indices) + ("decoder.",)
    if view.block_indices != indices or view.prefixes != prefixes:
        raise ValueError("Only the final two original blocks and decoder may be captured")
    original.check_layout(model, view)


@contextmanager
def capture_coordinates(model, view):
    check_layout(model, view)
    with original.capture_coordinates(model, view):
        yield


def frozen_state_equal(initial, model, view):
    check_layout(model, view)
    return original.frozen_state_equal(initial, model, view)
