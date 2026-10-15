"""Restrict a saved parameter displacement without changing any frozen coordinates."""

import math

import torch


PREFIXES = {"decoder": "decoder.", "output_layer": "decoder.layers.4."}


def inventory_ranges(inventory, scope):
    if scope not in PREFIXES:
        raise ValueError("Use a declared decoder coordinate subset")
    ranges, names, offset = [], set(), 0
    for name, shape, elements in inventory:
        if name in names or not shape or elements != math.prod(shape) or elements <= 0:
            raise ValueError("Require a unique complete parameter inventory")
        names.add(name)
        if name.startswith(PREFIXES[scope]):
            ranges.append({"name": name, "shape": shape, "start": offset, "stop": offset + elements})
        offset += elements
    if not ranges:
        raise ValueError("The requested parameter subset is empty")
    return ranges, offset


def extract(vector, ranges, full_elements):
    if vector.ndim != 1 or vector.numel() != full_elements or vector.dtype != torch.float32:
        raise ValueError("Require the complete original FP32 coordinate vector")
    result = torch.cat([vector[row["start"]:row["stop"]] for row in ranges])
    if not torch.isfinite(result).all():
        raise ValueError("Selected coordinates must be finite")
    return result


def replace(vector, selected, ranges, full_elements):
    expected = extract(vector, ranges, full_elements)
    if selected.shape != expected.shape or selected.dtype != vector.dtype or not torch.isfinite(selected).all():
        raise ValueError("Replace exactly the declared finite coordinate subset")
    result, offset = vector.clone(), 0
    for row in ranges:
        elements = row["stop"] - row["start"]
        result[row["start"]:row["stop"]].copy_(selected[offset:offset + elements])
        offset += elements
    return result
