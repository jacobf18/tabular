from copy import deepcopy
import math

import pytest

import train_single_matrixformer_point_only as training
import single_model_legacy_heavy_mixture as original_mixture
import test_single_model_legacy_strata_training as inherited
from test_single_model_top_two_blocks_trial import valid_preflight as base_preflight


def test_fresh_schedule_preserves_priors_parent_objectives_and_full_raw_proposals():
    assert training.PARENT == training.template.PARENT == training.preflight.PARENT
    assert training.PARENT_UPDATES == 98 and training.RATE == 5e-6
    assert training.STEPS == 64 and training.BOUNDARIES == (0, 10, 25, 50, 64)
    assert training.paired is training.preflight.paired and training.paired is not training.template.paired
    assert training.composite is training.preflight.composite is training.template.composite
    assert training.teacher_policy is training.template.teacher_policy
    batches = training.batches()
    assert len(batches) == 64 and sum(len(batch["tasks"]) for batch in batches) == 1792
    for actual, reference in zip(batches, training.template.batches()):
        for target, previous in zip(actual["tasks"], reference["tasks"]):
            before, after = deepcopy(previous), deepcopy(target)
            before["gradient_seed"] = after["gradient_seed"]
            before["specification"].get("structured", before["specification"])["seed"] = after["specification"].get(
                "structured", after["specification"])["seed"]
            assert before == after
        for group, mass in training.mixture.MASSES.items():
            assert math.isclose(math.fsum(task["weight"] for task in actual["tasks"]
                if original_mixture.group(task) == group), mass, rel_tol=0, abs_tol=1e-15)
    cases = training.confirmation_cases()
    fresh = training.original.alignment.fixed.seeds_in([batches, cases])
    old = training.original.alignment.fixed.seeds_in([training.template.batches(), training.template.confirmation_cases(),
                                                    training.preflight.batch()])
    assert len(cases) == 284 and len(fresh) == 3868 and not fresh.intersection(old)
    assert training.CONFIRMATION_SEEDS == (761100901, 762100901)
    for name in ("save_endpoint", "audit_endpoint"):
        assert getattr(training, name).__globals__["composite"] is training.composite
    assert training.train.__globals__["paired"] is training.paired
    assert 'INITIAL_RUN / "validation_25.json"' in training.REGISTER_SOURCE
    assert '"projection_scope": paired.SCOPE' in training.REGISTER_SOURCE


@pytest.mark.parametrize("eligible_steps,accepted_steps", [({1, 3}, {1}), ({1}, {1}), (set(), set()), ({4}, {4})])
def test_actual_loop_preserves_last_optimizer_endpoint_across_noops(tmp_path, monkeypatch, eligible_steps, accepted_steps):
    monkeypatch.setattr(inherited, "training", training)
    inherited.test_actual_training_loop_keeps_last_real_endpoint_across_noops(tmp_path, monkeypatch, eligible_steps, accepted_steps)


def valid_preflight():
    records = base_preflight()
    result, audit, gate, current = records
    weights = [task["weight"] for task in training.preflight.batch()["tasks"]]
    before = {f'{task["name"]}/{kind}': 1. for task in training.preflight.batch()["tasks"]
              for kind in ("rmse", "mae" if task["specification"]["domain"] == "recsys" else "common_nrmse")}
    result["data_admission_only"] = False
    result["record"].update(rate=training.RATE, point_before=before, proposal_policy=training.paired.POLICY,
        proposal_weights=training.paired.proposal_weights(before, weights),
        projection={"scope": training.paired.SCOPE, "groups": list(range(30))})
    audit["data_admission"] = {"complete": True, "eligible": True, "structured_draws": 24}
    audit["raw_endpoint"]["adam"] = {"gradient_vectors": 84}
    return records


@pytest.mark.parametrize("entry,path,value", [
    (0, ("trainable_parameters",), 19496843),
    (0, ("record", "rate"), 1e-5),
    (0, ("record", "accepted"), False),
    (0, ("record", "finite", "all_retained"), False),
    (0, ("frozen", "frozen_tensors"), 220),
    (0, ("frozen", "unselected_parameters_and_all_buffers_bitwise_unchanged"), False),
    (1, ("training_tasks",), 27),
    (1, ("prediction_grids",), 55),
    (1, ("teacher_replay", "full_grids"), 27),
    (1, ("teacher_replay", "complete"), False),
    (1, ("teacher_replay", "bitwise_equal"), False),
    (1, ("raw_endpoint", "gradient_vectors"), 28),
    (1, ("raw_endpoint", "accepted_updates"), 0),
    (1, ("raw_endpoint", "accepted"), False),
    (1, ("autograd_replayed",), True),
    (1, ("cuda_initialized",), True),
    (1, ("frozen", "changed_trainable_tensors"), []),
    (2, ("passed",), False),
    (2, ("audit_sha256",), "substituted"),
    (3, ("state",), "running"),
    (0, ("data_admission_only",), True),
    (0, ("record", "projection", "scope"), "legacy_strata_other_paired_means"),
    (0, ("record", "projection", "groups"), list(range(45))),
    (0, ("record", "proposal_policy"), "composite only"),
    (0, ("record", "proposal_weights"), [1.]),
    (1, ("data_admission", "complete"), False),
    (1, ("data_admission", "eligible"), False),
    (1, ("data_admission", "structured_draws"), 23),
    (1, ("raw_endpoint", "adam", "gradient_vectors"), 28),
])
def test_preflight_requires_original_frozen_teacher_admission_and_full_optimizer_checks(entry, path, value):
    records = valid_preflight()
    training.validate_preflight(*records, "digest")
    current = records[entry]
    for field in path[:-1]:
        current = current[field]
    current[path[-1]] = value
    with pytest.raises(ValueError, match="independently audited"):
        training.validate_preflight(*records, "digest")
