"""Expose three existing final blocks and decoder to training, without changing inference."""

from contextlib import contextmanager

import torch

import single_model_top_block_training as original


POLICY = ("An optimizer-only view of the existing final three transformer blocks and complete decoder. "
          "Preserve the ordinary network, forward, state keys and exact parameter identities. Freeze every "
          "other parameter and every buffer, including RoPE buffers inside selected block prefixes. This "
          "helper does not register training, consume synthetic seeds or establish GPU feasibility or accuracy.")


class TopThreeBlockParameterView(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        blocks, decoder = getattr(model, "transformer_blocks", None), getattr(model, "decoder", None)
        if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) < 4 or not isinstance(decoder, torch.nn.Module):
            raise ValueError("Require at least four original ordered transformer blocks and complete decoder")
        self.block_indices = tuple(range(len(blocks) - 3, len(blocks)))
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
    if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) < 4:
        raise ValueError("Preserve the original ordered transformer block inventory")
    indices = tuple(range(len(blocks) - 3, len(blocks)))
    prefixes = tuple(f"transformer_blocks.{index}." for index in indices) + ("decoder.",)
    if view.block_indices != indices or view.prefixes != prefixes:
        raise ValueError("Only the final three original blocks and decoder may be captured")
    original.check_layout(model, view)


@contextmanager
def capture_coordinates(model, view):
    check_layout(model, view)
    with original.capture_coordinates(model, view):
        yield


def frozen_state_equal(initial, model, view):
    check_layout(model, view)
    return original.frozen_state_equal(initial, model, view)
