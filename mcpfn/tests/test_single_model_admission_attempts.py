from copy import deepcopy

import pytest

import single_model_admission_attempts as attempts


def schedule(eligibility=(True, False, True, False), accept=(True, False, False, False)):
    batches, records, admissions = [], [], []
    accepted = 0
    for step, (eligible, updated) in enumerate(zip(eligibility, accept), 1):
        tasks = [{"index": index, "specification": {"domain": domain}} for index, domain in enumerate(("imputation", "recsys"))]
        batches.append({"step": step, "tasks": tasks})
        accepted += int(updated)
        kind = "optimizer_attempt" if eligible else "data_admission_rejected"
        record = {"step": step, "attempt_kind": kind, "accepted": updated, "accepted_updates": accepted,
                  "reason": "accepted" if updated else "finite_metrics_rejected" if eligible else kind,
                  "admission_manifest": f"step_{step}.json", "admission_sha256": "a" * 64, "seconds": 1., "peak_gib": 2.}
        if eligible:
            record["rate"] = 5e-6
        records.append(record)
        admissions.append({"step": step, "eligible": eligible, "task_count": 2, "tasks": [{"task": task} for task in tasks]})
    return batches, records, admissions


def test_exact_counts_include_both_noops_and_finite_optimizer_rejection():
    counts = attempts.inventory(*schedule())
    assert counts == {"scheduled_attempts": 4, "optimizer_attempts": 2, "admitted_steps": [1, 3],
        "data_rejected_steps": [2, 4], "training_tasks": 4, "rating_contexts": 2,
        "accepted_updates": 1, "last_optimizer_attempt": 3}
    descriptor = {"terminal_step": 4, "optimizer_attempt_step": 3, "accepted_updates": 1,
                  "raw_endpoint_sha256": "f" * 64, "config_sha256": "config"}
    attempts.check_endpoint_descriptor(descriptor, counts, "config")
    descriptor["optimizer_attempt_step"] = 4
    with pytest.raises(ValueError):
        attempts.check_endpoint_descriptor(descriptor, counts, "config")


def test_all_data_rejected_has_no_training_or_invented_endpoint():
    counts = attempts.inventory(*schedule((False,) * 4, (False,) * 4))
    assert counts["training_tasks"] == counts["rating_contexts"] == counts["accepted_updates"] == 0
    assert counts["last_optimizer_attempt"] is None
    descriptor = {"terminal_step": 4, "optimizer_attempt_step": None, "accepted_updates": 0,
                  "raw_endpoint_sha256": None, "config_sha256": "config"}
    attempts.check_endpoint_descriptor(descriptor, counts, "config")
    descriptor["raw_endpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        attempts.check_endpoint_descriptor(descriptor, counts, "config")


@pytest.mark.parametrize("mutation", ["missing", "ordinal", "task", "accepted", "count", "rate", "metric", "manifest", "nan"])
def test_no_missing_reweighted_or_fabricated_attempt_evidence(mutation):
    batches, records, admissions = deepcopy(schedule())
    if mutation == "missing":
        records.pop()
    elif mutation == "ordinal":
        records[1]["step"] = 3
    elif mutation == "task":
        admissions[1]["task_count"] = 1
    elif mutation == "accepted":
        records[1]["accepted"] = True
        records[1]["accepted_updates"] += 1
    elif mutation == "count":
        records[1]["accepted_updates"] += 1
    elif mutation == "rate":
        records[1]["rate"] = 5e-6
    elif mutation == "metric":
        records[1]["point_after"] = {}
    elif mutation == "manifest":
        records[1]["task_manifest"] = "fabricated.json"
    else:
        records[1]["seconds"] = float("nan")
    with pytest.raises(ValueError):
        attempts.inventory(batches, records, admissions)
