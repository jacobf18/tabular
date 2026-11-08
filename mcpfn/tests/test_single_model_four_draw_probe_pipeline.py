from copy import deepcopy

import pytest

import run_single_model_four_draw_probe_pipeline as pipeline


def reports(accepted=True, measured=True):
    result = {"data_admission_only": False, "record": {"point_after": {} if measured else None,
        "accepted": accepted, "accepted_updates": int(accepted), "reason": "accepted" if accepted else "projection_infeasible"},
        "gradient_transfer_sha256": "transfer", "raw_sha256": "raw", "peak_gib": 110., "peak_reserved_gib": 120.,
        "cuda_total_bytes": 140 * 1024**3, "allocator_fraction": .9, "peak_host_rss_bytes": 160 * 1024**3}
    audit = {"complete": True, "training_tasks": 56, "prediction_grids": 112 if measured else 56}
    retention = {"complete": True, "eligibility_decisions": 56, "reference_grids": 52, "component_grids": 100,
        "component_role_counts": {"v8": 48, "v29": 52}, "reference_role_counts": {"v29": 4, "structured_mean": 48},
        "endpoint_roles": ["before", "candidate"] if measured else ["before"]}
    transfer = {"complete": True, "transfer_sha256": "transfer", "raw_sha256": "raw"}
    mean = {"complete": True, "tasks": 56, "imputation_reweighted_tasks": 52, "native_reweighted_tasks": 4,
        "structured_reweighted_tasks": 48, "unchanged_rating_tasks": 4, "cuda_initialized": False} if measured else None
    return result, audit, retention, transfer, mean


@pytest.mark.parametrize("accepted,measured", [(True, True), (False, True), (False, False)])
def test_complete_feasibility_outcomes_preserve_all_measured_and_unmeasured_endpoints(accepted, measured):
    result = pipeline.summarize(*reports(accepted, measured))
    assert result["feasibility_passed"] is accepted
    assert result["mean_audit_evaluated"] is measured
    assert result["prediction_grids"] == (112 if measured else 56)


@pytest.mark.parametrize("location,key,value", [(1, "training_tasks", 28), (1, "prediction_grids", 56),
    (2, "reference_grids", 26), (2, "component_grids", 50), (2, "eligibility_decisions", 28),
    (3, "raw_sha256", "changed"), (4, "imputation_reweighted_tasks", 26), (4, "unchanged_rating_tasks", 2)])
def test_incomplete_or_changed_evidence_cannot_qualify(location, key, value):
    current = deepcopy(reports())
    current[location][key] = value
    with pytest.raises(ValueError):
        pipeline.summarize(*current)


@pytest.mark.parametrize("field,value", [("peak_gib", 130.), ("peak_reserved_gib", 127.),
    ("peak_host_rss_bytes", 177 * 1024**3), ("allocator_fraction", .95)])
def test_memory_admission_is_not_inferred_from_an_accepted_optimizer_step(field, value):
    current = reports()
    current[0][field] = value
    outcome = pipeline.summarize(*current)
    assert not outcome["feasibility_passed"] and not outcome["memory_within_registered_allowances"]


def test_data_only_noop_has_no_fabricated_training_or_mean_result():
    result = {"data_admission_only": True, "accepted_updates": 0, "optimizer_proposal_performed": False}
    audit = {"complete": True, "data_admission_only": True, "data_admission": {"eligible": False,
        "structured_draws": 48, "deferred_task_indices": [0, 13, 14, 27, 28, 41, 42, 55]}}
    retention = {"complete": True, "data_admission_only": True}
    assert pipeline.summarize(result, audit, retention, None, None) == {
        "feasibility_passed": False, "reason": "data_admission_rejected", "mean_audit_evaluated": False}
    result["accepted_updates"] = 1
    with pytest.raises(ValueError):
        pipeline.summarize(result, audit, retention, None, None)


@pytest.mark.parametrize("record,exists", [({"point_after": {}, "accepted": True}, False),
    ({"point_after": None, "accepted": False}, True), ({"point_after": None, "accepted": True}, False)])
def test_missing_or_fabricated_candidate_is_rejected(record, exists):
    with pytest.raises(ValueError):
        pipeline.probe.endpoint_roles(record, exists)
