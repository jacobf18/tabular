"""Measure checkpoint update directions without data, inference, or weight changes."""

import argparse
from collections import defaultdict
import math
from pathlib import Path

import torch

from recsys_evaluation_common import CHECKPOINT as SHARED, REPO, sha256, write_json
from train_single_matrixformer import GENERAL, SPECIALIST


def compare_states(candidate, general, specialist, shared):
    if not tuple(candidate) == tuple(general) == tuple(specialist) == tuple(shared):
        raise ValueError("State keys must match")
    groups = defaultdict(lambda: defaultdict(float))
    for name, value in candidate.items():
        relatives = (general[name], specialist[name], shared[name])
        if any(other.shape != value.shape or other.dtype != value.dtype for other in relatives):
            raise ValueError("State shapes/dtypes differ")
        if not all(torch.isfinite(other).all() for other in (value, *relatives)):
            raise ValueError("Nonfinite checkpoint state")
        current, parent, recsys, anchor = (other.double() for other in (value, *relatives))
        update = current - parent
        directions = {"interpolation": recsys - parent, "recsys_task": recsys - anchor,
                      "large_matrix_task": parent - anchor}
        prefix = ".".join(name.split(".")[:2]) if name.startswith("transformer_blocks.") else name.split(".")[0]
        for group in ("all", prefix):
            totals = groups[group]
            totals["elements"] += value.numel()
            totals["student_update_squared"] += float(update.square().sum())
            for label, direction in directions.items():
                totals[label + "_squared"] += float(direction.square().sum())
                totals[label + "_dot_update"] += float((direction * update).sum())
    result = {}
    for group, totals in groups.items():
        update_norm = math.sqrt(totals["student_update_squared"])
        row = {"state_elements": int(totals["elements"]), "student_update_l2": update_norm}
        for label in ("interpolation", "recsys_task", "large_matrix_task"):
            direction_norm = math.sqrt(totals[label + "_squared"])
            dot = totals[label + "_dot_update"]
            row[label] = {"direction_l2": direction_norm,
                          "projection_coefficient": dot / direction_norm**2 if direction_norm else None,
                          "cosine_with_update": dot / (direction_norm * update_norm) if direction_norm and update_norm else None}
        result[group] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    paths = {"student": arguments.checkpoint.resolve(), "general": GENERAL, "specialist": SPECIALIST, "shared": SHARED}
    payloads = {name: torch.load(path, map_location="cpu", weights_only=True) for name, path in paths.items()}
    geometry = compare_states(*(payloads[role]["model"] for role in ("student", "general", "specialist", "shared")))
    result = {"inputs": {role: {"path": str(path), "sha256": sha256(path)} for role, path in paths.items()},
              "source_sha256": sha256(Path(__file__)), "selected_step": payloads["student"]["step"],
              "definitions": {"student_update": "student - general", "interpolation": "specialist - general",
                              "recsys_task": "specialist - shared98k", "large_matrix_task": "general - shared98k"},
              "data_used": False, "inference_performed": False, "checkpoints_changed": False,
              "limitation": "Weight-space geometry is not a predictor of benchmark performance or evidence that any merge is safe",
              "groups": geometry}
    if arguments.output.exists():
        raise ValueError("Preserve existing geometry reports")
    write_json(arguments.output, result)
    print(geometry["all"])


if __name__ == "__main__":
    main()
