"""Check conservative V57 lifecycle headroom without changing its registered trial."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time


ROOT = Path("/mnt/checkpoints_200/single_model_goal_20261005")
RUN = ROOT / "four_draw_v57"
PIPELINE = ROOT / "four_draw_v57_pipeline"
SCRATCH = Path("/mnt/matrixformer_scratch")
BANK = ROOT / "domain_clipping_v10/synthetic_bank.pt"
OUTPUT = ROOT / "four_draw_v57_lifecycle_capacity_v1.json"
TRAINING_SHA256 = "bbf0d361f6a9492ec3cc2c4e8ce0ff80c40eb56aeffdde511ef4d8a8e41ba6d3"
PIPELINE_SHA256 = "09c5d7ebb9a400b91b4f915a8c8a2abbac0481f96b01203da6443ca230c2dfad"
PARAMETERS = 77998496
SUPPLEMENTAL_REPLAY_BYTES = 2 * 1024**3
RECEIPT_ALLOWANCE = 256 * 1024
STAGES = {"replay", "audit-replay", "confirm", "audit-confirmation", "benchmarks", "review"}
POLICY = (
    "Read-only prospective lifecycle-capacity calculation for the already registered V57 trial. "
    "Preserve the original768 MiB replay admission and every registered source, budget, seed, "
    "model, metric, guard and output. Explicitly report that the old replay admission is not "
    "a complete worst-case size bound after doubling endpoint tasks. Compute every requested "
    "endpoint shape, all152 selected prediction grids, full FP32 candidate coordinates, "
    "conservative64-bit prediction arrays, scaling vectors, metadata and atomic-write overlap. "
    "Separately require2 GiB replay scratch headroom and conservatively admit the complete "
    "training accounting plus every later stage from currently free space, crediting no "
    "already-written bytes or deletions. Keep18/6/50 GiB reserves. This report does not alter "
    "the running coordinator, add a scientific attempt, reserve statistical seeds, sample, "
    "infer, train, select, qualify, relocate or delete anything. It is not an actual replay "
    "footprint, hard filesystem reservation or guarantee against unrelated future writers. "
    "Retain the original per-stage checks and measure the actual later footprint."
)


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hashes(mapping):
    for filename, digest in mapping.items():
        if sha256(filename) != digest:
            raise ValueError(f"Registered artifact changed: {filename}")


def matrix_shape(shape):
    if len(shape) not in (2, 3) or (len(shape) == 3 and shape[0] != 1):
        raise ValueError("Require one complete positive matrix, not a cropped or batched substitute")
    if any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("Require exact positive requested dimensions")
    return shape[-2:]


def replay_accounting(batches, validation_shapes):
    if len(batches) != 128 or [batch["step"] for batch in batches] != list(range(1, 129)):
        raise ValueError("Retain every registered possible terminal attempt")
    areas, scaling, individual = [], [], []
    for batch in batches:
        if len(batch["tasks"]) != 56 or [task["index"] for task in batch["tasks"]] != list(range(56)):
            raise ValueError("Account for all56 endpoint tasks, including every domain")
        shapes = [matrix_shape(task["specification"].get("structured", task["specification"])["shape"])
            for task in batch["tasks"]]
        current = [math.prod(shape) for shape in shapes]
        areas.append(sum(current))
        scaling.append(sum(8 * sum(shape) for shape in shapes))
        individual.extend(current)
    if len(validation_shapes) != 152:
        raise ValueError("Preserve all152 sole-selected development prediction grids")
    validation_areas = [math.prod(matrix_shape(shape)) for shape in validation_shapes]
    maximum, largest, selected = max(areas), max(individual), sum(validation_areas)
    arrays = {"materialized_episode_and_teacher_bytes": 13 * maximum,
        "scaling_vector_allowance": max(scaling), "two_endpoint_prediction_grids": 16 * maximum,
        "selected_prediction_grids": 8 * selected, "candidate_parameter_vector": 4 * PARAMETERS,
        "atomic_candidate_allowance": 4 * PARAMETERS,
        "atomic_materialized_task_allowance": 13 * largest + max(scaling),
        "atomic_prediction_allowance": 8 * max(largest, max(validation_areas)),
        "serialization_and_metadata_allowance": 64 * 1024**2}
    total = sum(arrays.values())
    if total > SUPPLEMENTAL_REPLAY_BYTES:
        raise ValueError("The conservative complete replay estimate exceeds supplemental headroom")
    return {"maximum_requested_endpoint_cells": maximum, "selected_prediction_cells": selected,
        "float32_arrays_before_metadata": 21 * maximum + 4 * PARAMETERS + 4 * selected,
        "conservative_replay_fields": arrays, "conservative_replay_bytes": total,
        "supplemental_replay_scratch_bytes": SUPPLEMENTAL_REPLAY_BYTES,
        "compression_assumed": False, "intrinsic_smaller_shapes_assumed": False,
        "favorable_terminal_attempt_assumed": False, "actual_replay_usage_measured": False}


def lifecycle_accounting(config, pipeline, replay):
    original = pipeline["stage_write_budget_bytes"]
    if set(original) != STAGES or original["replay"]["scratch"] != 768 * 1024**2:
        raise ValueError("Bind every unchanged original qualification stage and its original replay admission")
    if any(original[name] != config["diagnostic_storage"]["stage_write_budget_bytes"][name] for name in STAGES):
        raise ValueError("Keep both frozen declarations identical")
    supplemental = deepcopy(original)
    supplemental["replay"]["scratch"] = replay["supplemental_replay_scratch_bytes"]
    result = {volume: sum(config["storage_accounting"][volume].values())
        + sum(stage[volume] for stage in supplemental.values()) for volume in ("checkpoint", "root", "scratch")}
    result["checkpoint"] += RECEIPT_ALLOWANCE
    return {"original_stage_admission_bytes": original, "supplemental_stage_accounting_bytes": supplemental,
        "whole_lifecycle_additional_byte_bound": result, "already_written_bytes_credited": 0,
        "original_replay_admission_covers_float32_array_bound":
            original["replay"]["scratch"] >= replay["float32_arrays_before_metadata"]}


def admit_space(bounds, reserves, volumes):
    if set(bounds) != set(reserves) or set(volumes) != set(reserves):
        raise ValueError("Never omit a writing filesystem from supplemental admission")
    result = {}
    for volume, directory in volumes.items():
        free = shutil.disk_usage(directory).free
        required = bounds[volume] + reserves[volume]
        if free < required:
            raise OSError(f"Insufficient conservative lifecycle headroom on {volume}: {free} < {required}")
        result[volume] = {"path": str(directory), "device": Path(directory).stat().st_dev,
            "free_bytes": free, "required_bytes": required, "surplus_bytes": free - required}
    return result


def check():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("This supplementary calculation must be CPU-only")
    import torch

    if torch.cuda.is_initialized():
        raise ValueError("Never initialize CUDA for metadata/capacity inspection")
    identities = {str(RUN / "config.json"): TRAINING_SHA256, str(PIPELINE / "config.json"): PIPELINE_SHA256}
    verify_hashes(identities)
    config, pipeline = read(RUN / "config.json"), read(PIPELINE / "config.json")
    for directory in (RUN, PIPELINE):
        if read(directory / "registration.json")["config_sha256"] != sha256(directory / "config.json"):
            raise ValueError("Bind both unchanged original registrations")
        identities[str(directory / "registration.json")] = sha256(directory / "registration.json")
    for mapping in (config["source_sha256"], pipeline["source_sha256"]):
        verify_hashes(mapping)
    endpoint = Path(pipeline["endpoint_replay"])
    if read(endpoint / "status.json")["state"] != "registered" or (endpoint / "result.json").exists():
        raise ValueError("Perform the additional accounting before any endpoint replay")
    identities[str(BANK)] = config["input_sha256"][str(BANK)]
    verify_hashes({str(BANK): identities[str(BANK)]})
    bank = torch.load(BANK, map_location="cpu", weights_only=True, mmap=True)
    shapes = []
    for episode in bank:
        source, target, mask = (episode[name] for name in ("source", "target", "mask"))
        if (source.dtype != torch.float32 or target.dtype != torch.float32 or mask.dtype != torch.bool
                or source.shape != target.shape or source.shape != mask.shape):
            raise ValueError("Require the original complete FP32/bool development bank schema")
        shapes.append(list(source.shape))
    replay = replay_accounting(config["batches"], shapes)
    lifecycle = lifecycle_accounting(config, pipeline, replay)
    reserves = config["diagnostic_storage"]["minimum_free_bytes"]
    if reserves != {"checkpoint": 18 * 1024**3, "root": 6 * 1024**3, "scratch": 50 * 1024**3}:
        raise ValueError("Preserve all original reserves")
    volumes = {"checkpoint": ROOT, "root": Path("/"), "scratch": SCRATCH}
    if len({directory.stat().st_dev for directory in volumes.values()}) != 3:
        raise ValueError("Keep checkpoint, root and scratch on their original distinct filesystems")
    space = admit_space(lifecycle["whole_lifecycle_additional_byte_bound"], reserves, volumes)
    sources = (Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "tests/test_single_model_four_draw_lifecycle_capacity.py")
    return {"complete": True, "policy": POLICY, "checked_unix": time.time(), "input_sha256": identities,
        "source_sha256": {str(path): sha256(path) for path in sources}, "torch_version": torch.__version__, "replay": replay,
        "lifecycle": lifecycle, "reserves": reserves, "measured_capacity": space,
        "registered_artifacts_modified": False, "registered_source_hashes_checked": True,
        "scientific_execution_performed": False, "scientific_retry_authorized": False,
        "model_qualification_established": False, "cuda_initialized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.parent != ROOT or output.exists() or output.is_symlink():
        raise ValueError("Write one new supplementary receipt, never replace a registered artifact")
    report = check()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if len(encoded.encode()) > RECEIPT_ALLOWANCE:
        raise ValueError("Keep the receipt within its separately accounted metadata allowance")
    with output.open("x") as destination:
        destination.write(encoded)
        destination.flush()
        os.fsync(destination.fileno())
    print(json.dumps({"receipt": str(output), "sha256": sha256(output),
        "conservative_replay_bytes": report["replay"]["conservative_replay_bytes"],
        "original_replay_admission_complete_bound": report["lifecycle"]["original_replay_admission_covers_float32_array_bound"],
        "measured_capacity": report["measured_capacity"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
