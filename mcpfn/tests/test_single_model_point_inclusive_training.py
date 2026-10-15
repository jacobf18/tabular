from copy import deepcopy

import pytest
import torch

import audit_single_model_point_inclusive_training as audit
import train_single_matrixformer_point_inclusive as training
from test_single_model_structured_teacher_training import remove_seeds


def test_original_priors_teachers_shapes_and_qualification_are_unchanged():
    batches = training.batches()
    assert len(batches) == 64 and all(len(batch["tasks"]) == 28 for batch in batches)
    assert remove_seeds(batches) == remove_seeds(training.baseline.batches())
    assert remove_seeds(training.confirmation_cases()) == remove_seeds(training.baseline.confirmation_cases())
    assert training.PARENT == training.original.alignment.fixed.PARENT
    assert training.INITIAL_RUN == training.original.alignment.fixed.PRODUCER
    assert training.BOUNDARIES == (0, 10, 25, 50, 64)
    assert training.QUALIFICATION == training.baseline.QUALIFICATION
    fresh = training.original.alignment.fixed.seeds_in([batches, training.confirmation_cases()])
    earlier = training.original.alignment.fixed.seeds_in([training.baseline.batches(), training.baseline.confirmation_cases()])
    assert len(fresh) == 3868 and not fresh.intersection(earlier)
    assert training.teacher_policy is training.baseline.teacher_policy


def test_all_actual_attempts_use_half_rate():
    assert all(training.constant_rate(step) == 1e-5 for step in range(1, 65))


@pytest.mark.parametrize("step", [0, 65, -1, 1.0, True])
def test_unregistered_rate_requests_are_rejected(step):
    with pytest.raises(ValueError):
        training.constant_rate(step)


def test_full_proposal_and_raw_evidence_are_not_composite_only():
    source = training.ADAPTED_SOURCE
    assert 'inclusive.protected_update(' in source and 'paired.protected_update(' not in source
    assert 'paired.validate_layout(entries, batch["tasks"], before)' in source
    assert 'shared_storage.save(payload, output / "raw_final_update.pt")' in source
    assert 'shared_gradients(gradients[:len(entries)], snapshot)' in source
    assert 'shared_gradients(gradients[:len(entries)], error.snapshot)' in source
    assert 'with teacher_policy.runtime(config["base"], output, config["teacher_identities"]):' in source
    assert 'with original.alignment.mixture.runtime(config["base"], directory):' in source
    assert 'shutil.disk_usage(ROOT).free < 31 * 1024**3' in source
    assert 'require_matched()' in source and 'preflight.require_predecessor()' not in source
    assert 'not gate["passed"]' not in source
    assert audit.checks is training.inclusive
    assert 'shared_storage.audit(payload, raw_path)' in audit.ADAPTED_SOURCE


def matched_report():
    return {"complete": True, "matched_cells": 4, "new_controls": 3, "prediction_grids": 224,
            "selection_performed": False, "cuda_initialized": False,
            "rows": [{**row, "accepted": row["rate"] == 1e-5, "reason": "accepted",
                      "paired_improved": 28, "paired_worsened": 0} for row in training.matched.registered_cases()]}


def test_feasibility_uses_the_successful_half_rate_control():
    assert training.matched_acceptance(matched_report())["name"] == "point_1e-5"


@pytest.mark.parametrize("field,value", [("complete", False), ("prediction_grids", 223), ("matched_cells", 3),
                                       ("new_controls", 2), ("selection_performed", True), ("cuda_initialized", True)])
def test_incomplete_or_selected_feasibility_is_rejected(field, value):
    report = matched_report()
    report[field] = value
    with pytest.raises(ValueError, match="224-grid"):
        training.matched_acceptance(report)


@pytest.mark.parametrize("field,value", [("accepted", False), ("reason", "finite_metrics_rejected"),
                                       ("paired_improved", 27), ("paired_worsened", 1), ("rate", 2e-5)])
def test_failed_or_changed_point_control_cannot_authorize_training(field, value):
    report = matched_report()
    report["rows"][2][field] = value
    with pytest.raises(ValueError):
        training.matched_acceptance(report)


def test_gradient_alias_requires_the_complete_identical_storage():
    gradients = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    payload = {"gradients": gradients, "snapshot": {"task_gradients": gradients}}
    audit.check_gradient_alias(payload)
    payload["snapshot"]["task_gradients"] = gradients.clone()
    with pytest.raises(ValueError, match="shared tensor"):
        audit.check_gradient_alias(payload)
    payload["snapshot"]["task_gradients"] = gradients[:1]
    with pytest.raises(ValueError, match="shared tensor"):
        audit.check_gradient_alias(payload)


def test_failed_development_guard_still_prevents_selection():
    best = {"step": 25, "score": 1.1}
    assert training.selected_boundary(best, 50, 47, {"passed": False, "score": 1.0}) == deepcopy(best)
