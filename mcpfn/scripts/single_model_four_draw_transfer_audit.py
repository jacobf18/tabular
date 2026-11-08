"""Audit every actual gradient-transfer receipt without inventing rejected-batch evidence."""

import math

import single_model_four_draw_gradient_staging as staging
from recsys_evaluation_common import sha256


def expected_transfer(rows, parameters):
    if type(rows) is not int or not 0 < rows <= 168 or type(parameters) is not int or parameters <= 0:
        raise ValueError("Require complete positive defined-gradient and parameter counts")
    return {"policy": staging.POLICY, "capacity_rows": 168, "defined_rows": rows, "parameters": parameters,
        "capture_bytes": 168 * 4 * parameters, "transferred_bytes": rows * 4 * parameters,
        "source_device": "cpu", "destination_device": "cuda:0", "verified_elements": rows * parameters,
        "exact_fp32_transfer": True, "unused_capacity_rows": 168 - rows,
        "peak_memory_measured": False, "scientific_feasibility_established": False}


def check_memory(record, working_bytes):
    fields = ("peak_gib", "peak_reserved_gib", "peak_host_rss_bytes", "capture_peak_allocated_bytes",
        "capture_peak_reserved_bytes", "cuda_total_bytes")
    if any(isinstance(record[name], bool) or not isinstance(record[name], (int, float))
            or not math.isfinite(record[name]) or record[name] <= 0 for name in fields):
        raise ValueError("Retain finite positive measured GPU and host memory quantities")
    allocated, reserved = record["peak_gib"] * 1024**3, record["peak_reserved_gib"] * 1024**3
    if (record["allocator_fraction"] != .9 or not allocated <= reserved <= .9 * record["cuda_total_bytes"]
            or not record["capture_peak_allocated_bytes"] <= record["capture_peak_reserved_bytes"] <= reserved
            or record["capture_peak_allocated_bytes"] > allocated or record["peak_host_rss_bytes"] > working_bytes):
        raise ValueError("Never exceed the original CUDA allocator cap or host working allowance")


def audit(training, config, records):
    directory = training.OUTPUT / "gradient_transfers"
    expected, hashes, slots = set(), {}, 0
    for record in records:
        if record["attempt_kind"] == "data_admission_rejected":
            if any(name in record for name in ("gradient_transfer", "gradient_transfer_sha256")):
                raise ValueError("Data-only rejection must not fabricate a gradient transfer")
            continue
        if record["attempt_kind"] != "optimizer_attempt":
            raise ValueError("Keep only original declared attempt kinds")
        path = directory / f'step_{record["step"]:03d}.json'
        if (path in expected or record["gradient_transfer"] != str(path)
                or record["gradient_transfer_sha256"] != sha256(path)
                or training.read(path) != expected_transfer(record["protected_tasks"], training.TRAINABLE_PARAMETERS)):
            raise ValueError("Bind each actual complete defined-prefix transfer to its original recorded attempt")
        check_memory(record, training.preflight.HOST_WORKING_BYTES)
        expected.add(path)
        hashes[str(path)] = record["gradient_transfer_sha256"]
        slots += record["protected_tasks"]
    actual = set(directory.iterdir()) if directory.exists() else set()
    if actual != expected or config["gradient_staging_policy"] != staging.POLICY:
        raise ValueError("Never omit, duplicate or add unregistered gradient-transfer artifacts")
    return {"complete": True, "optimizer_attempts": len(expected), "defined_gradient_rows": slots,
        "all_defined_fp32_elements_checked_at_transfer": True, "recorded_memory_within_allowances": True,
        "independent_neural_gradient_replay": False, "artifact_sha256": hashes}
