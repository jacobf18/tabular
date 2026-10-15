from copy import deepcopy
import math

import pytest

import train_single_matrixformer_legacy_point as training
import test_single_model_legacy_strata_training as inherited


def test_schedule_preserves_v30_except_fresh_seeds_and_point_proposal():
    assert training.PARENT == training.template.PARENT == training.preflight.PARENT
    assert training.PARENT_UPDATES == 98 and training.RATE == 5e-6
    assert training.STEPS == 64 and training.BOUNDARIES == (0, 10, 25, 50, 64)
    assert training.paired is training.preflight.paired and training.paired is not training.template.paired
    assert training.composite is training.preflight.composite and training.composite is not training.template.composite
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
                if training.template.mixture.group(task) == group), mass, rel_tol=0, abs_tol=1e-15)
    cases = training.confirmation_cases()
    fresh = training.original.alignment.fixed.seeds_in([batches, cases])
    old = training.original.alignment.fixed.seeds_in([training.template.batches(), training.template.confirmation_cases(),
                                                    training.preflight.batch()])
    assert len(cases) == 284 and len(fresh) == 3868 and not fresh.intersection(old)
    assert training.CONFIRMATION_SEEDS == (731100901, 732100901)
    assert training.save_endpoint.__globals__["composite"] is training.composite
    assert training.audit_endpoint.__globals__["composite"] is training.composite
    assert training.train.__globals__["paired"] is training.paired
    assert 'INITIAL_RUN / "validation_25.json"' in training.REGISTER_SOURCE
    assert '"proposal_policy": paired.POLICY' in training.REGISTER_SOURCE


@pytest.mark.parametrize("eligible_steps,accepted_steps", [({1, 3}, {1}), ({1}, {1}), (set(), set()), ({4}, {4})])
def test_actual_training_loop_preserves_last_endpoint_across_noops(tmp_path, monkeypatch, eligible_steps, accepted_steps):
    monkeypatch.setattr(inherited, "training", training)
    inherited.test_actual_training_loop_keeps_last_real_endpoint_across_noops(tmp_path, monkeypatch, eligible_steps, accepted_steps)


@pytest.mark.parametrize("mutation", [None, "prefix", "policy", "weights"])
def test_additional_preflight_check_requires_real_point_inclusive_evidence(monkeypatch, mutation):
    called = []
    monkeypatch.setattr(training.template, "validate_preflight", lambda *args: called.append(True))
    weights = [task["weight"] for task in training.preflight.batch()["tasks"]]
    before = {f'{task["name"]}/{kind}': 1. for task in training.preflight.batch()["tasks"]
              for kind in ("rmse", "mae" if task["specification"]["domain"] == "recsys" else "common_nrmse")}
    result = {"record": {"proposal_policy": training.paired.POLICY, "point_before": before,
                         "proposal_weights": training.paired.proposal_weights(before, weights)}}
    audit = {"raw_endpoint": {"adam": {"gradient_vectors": 84}}}
    if mutation == "prefix":
        audit["raw_endpoint"]["adam"]["gradient_vectors"] = 28
    elif mutation == "policy":
        result["record"]["proposal_policy"] = "composite only"
    elif mutation == "weights":
        result["record"]["proposal_weights"] = weights
    if mutation:
        with pytest.raises(ValueError):
            training.validate_preflight(result, audit, {}, {}, "digest")
    else:
        training.validate_preflight(result, audit, {}, {}, "digest")
    assert called == [True]
