"""Expose all existing model parameters to training without changing ordinary inference."""

from contextlib import contextmanager

import torch

import single_model_top_block_training as original


PREFIXES = ("feature_encoder.", "transformer_blocks.", "decoder.")
POLICY = ("An optimizer-only view of the original feature encoder, every transformer block and complete decoder. "
          "Preserve every original parameter object, name, order, buffer and inference operation; introduce no "
          "ensemble, router or inference module. Train all original parameters while preserving every buffer "
          "bitwise, including RoPE buffers. This helper and CPU tests neither register an experiment nor consume "
          "synthetic seeds, initialize a new trial, establish full-size GPU feasibility or change any live run.")


class FullNetworkParameterView(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        encoder = getattr(model, "feature_encoder", None)
        blocks, decoder = getattr(model, "transformer_blocks", None), getattr(model, "decoder", None)
        if (not isinstance(encoder, torch.nn.Module) or not isinstance(blocks, torch.nn.ModuleList)
                or len(blocks) < 2 or not isinstance(decoder, torch.nn.Module)):
            raise ValueError("Require the original feature encoder, ordered transformer blocks and complete decoder")
        self.block_indices = tuple(range(len(blocks)))
        self.prefixes = PREFIXES
        selected = list(model.named_parameters())
        required = ("feature_encoder.", *(f"transformer_blocks.{index}." for index in self.block_indices), "decoder.")
        if (not selected or any(not name.startswith(PREFIXES) for name, parameter in selected)
                or any(not any(name.startswith(prefix) for name, parameter in selected) for prefix in required)):
            raise ValueError("Every original encoder, distinct block and decoder must retain its own named parameters")
        self.feature_encoder = encoder
        self.transformer_blocks = torch.nn.ModuleList(list(blocks))
        self.decoder = decoder
        actual = list(self.named_parameters())
        if (len(actual) != len(selected) or any(name != source_name or parameter is not source
                for (name, parameter), (source_name, source) in zip(actual, selected))):
            raise ValueError("Retain every exact original parameter identity and coordinate order")
        self.mutable_names = tuple(name for name, parameter in selected)
        model.zero_grad(set_to_none=True)
        model.requires_grad_(True)
        check_layout(model, self)


def check_layout(model, view):
    blocks = getattr(model, "transformer_blocks", None)
    if (not isinstance(blocks, torch.nn.ModuleList) or len(blocks) < 2
            or view.block_indices != tuple(range(len(blocks))) or view.prefixes != PREFIXES
            or tuple(name for name, parameter in model.named_parameters()) != view.mutable_names
            or not all(parameter.requires_grad for parameter in model.parameters())):
        raise ValueError("Only the complete original encoder, ordered blocks and decoder may supply coordinates")
    original.check_layout(model, view)


@contextmanager
def capture_coordinates(model, view):
    check_layout(model, view)
    with original.capture_coordinates(model, view):
        yield


def frozen_state_equal(initial, model, view):
    check_layout(model, view)
    return original.frozen_state_equal(initial, model, view)
