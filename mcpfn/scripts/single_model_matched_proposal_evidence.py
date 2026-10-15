"""Losslessly reference shared raw gradients and initial parameters in matched controls."""

from copy import deepcopy

import torch

from single_model_finite_guarded_audit import optimizer_equal


SHARED_FIELDS = {"before", "after", "task_gradients"}
SNAPSHOT_FIELDS = {"before", "after", "task_gradients", "tentative", "combined_gradient", "applied_gradient",
                   "optimizer_before", "optimizer_after", "record", "rollback_rates", "protected_record"}


def identical(actual, expected):
    return (actual.device.type == expected.device.type == "cpu" and actual.dtype == expected.dtype
            and actual.shape == expected.shape
            and torch.equal(actual.detach().contiguous().view(torch.uint8), expected.detach().contiguous().view(torch.uint8)))


def proposal_gradients(reference, method, count):
    if method not in {"composite", "point_inclusive"} or count < 1 or len(reference["gradients"]) <= count:
        raise ValueError("Require the original complete composite/point gradient inventory")
    return reference["gradients"][:count] if method == "composite" else reference["gradients"]


def expected_optimizer(snapshot, record):
    if record["accepted"]:
        return snapshot["optimizer_after"]
    result = deepcopy(snapshot["optimizer_before"])
    if len(result["param_groups"]) != len(snapshot["rollback_rates"]):
        raise ValueError("Preserve all preceding optimizer rates")
    for group, rate in zip(result["param_groups"], snapshot["rollback_rates"]):
        group["lr"] = rate
    return result


def compact(snapshot, record, reference, identity, method, candidate, actual_optimizer):
    if set(snapshot) != SNAPSHOT_FIELDS:
        raise ValueError("Never silently omit an unknown optimizer snapshot field")
    if (not identical(snapshot["before"], reference["snapshot"]["before"])
            or not identical(snapshot["task_gradients"], proposal_gradients(reference, method, record["composite_tasks"]))):
        raise ValueError("Referenced initialization and raw gradients must match bitwise")
    after = candidate if record["accepted"] else reference["snapshot"]["before"]
    if after is None or not identical(snapshot["after"], after):
        raise ValueError("Actual final parameters must equal the declared candidate or exact rollback")
    optimizer_equal(actual_optimizer, expected_optimizer(snapshot, record))
    return {"reference": identity, "method": method, "record": record,
            "snapshot": {name: value for name, value in snapshot.items() if name not in SHARED_FIELDS},
            "after_source": "candidate" if record["accepted"] else "initialization",
            "final_optimizer_bitwise_verified": True}


def expand(payload, reference, identity, candidate):
    record, snapshot = payload["record"], payload["snapshot"]
    if (payload["reference"] != identity or set(snapshot) != SNAPSHOT_FIELDS - SHARED_FIELDS
            or not payload["final_optimizer_bitwise_verified"]
            or payload["after_source"] != ("candidate" if record["accepted"] else "initialization")):
        raise ValueError("Preserve complete compact evidence and its original external references")
    after = candidate if record["accepted"] else reference["snapshot"]["before"]
    if after is None:
        raise ValueError("Never omit an accepted candidate")
    restored = {**snapshot, "before": reference["snapshot"]["before"], "after": after,
                "task_gradients": proposal_gradients(reference, payload["method"], record["composite_tasks"])}
    return {"record": record, "snapshot": restored, "gradients": reference["gradients"]}


def storage_allowance(parameters, prediction_bytes, cases=3):
    if type(parameters) is not int or parameters <= 0 or type(prediction_bytes) is not int or prediction_bytes < 0 or cases != 3:
        raise ValueError("Require the fixed three-control evidence inventory")
    return cases * (6 * parameters * 4 + 2 * prediction_bytes + 3 * 4 * 1024**2) + 16 * 1024**2
