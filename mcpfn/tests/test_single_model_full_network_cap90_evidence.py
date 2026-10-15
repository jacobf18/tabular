from copy import deepcopy

import pytest

import single_model_full_network_cap90_evidence as evidence


def fixture():
    result = {"complete": True, "config_sha256": "config", "memory_fraction": .9,
        "parent_sha256": evidence.memory.training.PARENT_SHA256, "historical_task_and_teacher_digest_equal": True,
        "reconstructed_step8_state": False, "gpu_gradient_rows": 84, "gpu_gradient_bytes": 26207494656,
        "headroom_bytes": 1247975936, "gradient_records": [{"kind": name, "parameters": 77998496}
            for name in ("composite", "rmse", "common_nrmse")],
        "frozen": {"changed_trainable_tensors": [], "frozen_tensors": 16, "unselected_parameters_and_all_buffers_bitwise_unchanged": True},
        "gpu_total_gib": 140., "peak_allocated_gib": 113., "peak_reserved_gib": 122.,
        "training_updates_performed": 0, "optimizer_stepped": False, "checkpoint_written": False,
        "validation_evaluated": False, "real_benchmarks_evaluated": False, "all_training_tasks_memory_qualified": False}
    return result, {"state": "complete", "result_sha256": "result"}, "config", "result"


def test_completed_measurement_has_narrow_scope():
    evidence.validate_probe(*fixture())


@pytest.mark.parametrize("path,value", [(("memory_fraction",), .8), (("gpu_gradient_rows",), 3),
    (("gpu_gradient_bytes",), 0), (("headroom_bytes",), 0), (("peak_reserved_gib",), 130.),
    (("peak_allocated_gib",), 80.), (("training_updates_performed",), 1), (("optimizer_stepped",), True),
    (("checkpoint_written",), True), (("historical_task_and_teacher_digest_equal",), False),
    (("reconstructed_step8_state",), True), (("all_training_tasks_memory_qualified",), True),
    (("frozen", "frozen_tensors"), 0), (("frozen", "changed_trainable_tensors"), ["encoder"]),
    (("config_sha256",), "changed"), (("parent_sha256",), "changed")])
def test_underallocated_or_overstated_evidence_is_rejected(path, value):
    result, status, config_digest, result_digest = fixture()
    current = result
    for field in path[:-1]:
        current = current[field]
    current[path[-1]] = deepcopy(value)
    with pytest.raises(ValueError, match="resident-memory"):
        evidence.validate_probe(result, status, config_digest, result_digest)
