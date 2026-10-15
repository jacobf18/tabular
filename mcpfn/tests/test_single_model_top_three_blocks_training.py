from copy import deepcopy
import math

import pytest

import train_single_matrixformer_top_three_blocks as training
import single_model_legacy_heavy_mixture as original_mixture
import test_single_model_legacy_strata_training as inherited
from test_single_model_point_only_training import valid_preflight as old_preflight


def test_fresh_schedule_preserves_priors_parent_and_proposal_with_expanded_depth():
    assert training.PARENT == training.template.PARENT == training.preflight.PARENT
    assert training.PARENT_SHA256 == training.preflight.PARENT_SHA256
    assert training.PARENT_UPDATES == 98 and training.RATE == 5e-6
    assert training.STEPS == 64 and training.BOUNDARIES == (0, 10, 25, 50, 64)
    assert training.PREFIXES == training.preflight.PREFIXES
    assert training.paired is training.preflight.paired is training.template.paired
    assert training.composite is training.template.composite
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
    assert (training.DATA_SEED, training.GRADIENT_SEED) == (781100901, 782100901)
    assert training.CONFIRMATION_SEEDS == (791100901, 792100901)
    assert "top_two." not in training.ADAPTED_SOURCE
    assert training.setup.__globals__["top_three"] is training.top_three
    assert training.train.__globals__["paired"] is training.paired
    assert 'INITIAL_RUN / "validation_25.json"' in training.REGISTER_SOURCE
    assert '"projection_scope": paired.SCOPE' in training.REGISTER_SOURCE


@pytest.mark.parametrize("eligible_steps,accepted_steps", [({1, 3}, {1}), ({1}, {1}), (set(), set()), ({4}, {4})])
def test_actual_three_block_loop_keeps_last_real_endpoint_across_noops(tmp_path, monkeypatch, eligible_steps, accepted_steps):
    monkeypatch.setattr(inherited, "training", training)
    monkeypatch.setattr(training, "top_two", training.top_three, raising=False)
    inherited.test_actual_training_loop_keeps_last_real_endpoint_across_noops(tmp_path, monkeypatch, eligible_steps, accepted_steps)


def valid_preflight():
    records = old_preflight()
    result, audit, gate, current = records
    result.update(trainable_parameters=36036497, backbone_parameters=41961999)
    result["frozen"].update(frozen_tensors=164, changed_trainable_tensors=[str(index) for index in range(90)])
    audit["frozen"] = deepcopy(result["frozen"])
    return records


@pytest.mark.parametrize("entry,path,value", [
    (0, ("trainable_parameters",), 27766670),
    (0, ("backbone_parameters",), 50231826),
    (0, ("record", "rate"), 1e-5),
    (0, ("record", "accepted"), False),
    (0, ("record", "finite", "all_retained"), False),
    (0, ("frozen", "frozen_tensors"), 192),
    (0, ("frozen", "changed_trainable_tensors"), list(range(96))),
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
def test_original_teacher_frozen_admission_and_full_optimizer_checks_remain(entry, path, value):
    records = valid_preflight()
    training.validate_preflight(*records, "digest")
    current = records[entry]
    for field in path[:-1]:
        current = current[field]
    current[path[-1]] = value
    with pytest.raises(ValueError, match="independently audited"):
        training.validate_preflight(*records, "digest")


def test_checkpoint_and_scratch_budgets_include_expanded_atomic_artifacts(monkeypatch):
    assert training.STORAGE_ACCOUNTING["scratch"]["raw_gradient_bytes"] == 12108262992
    assert training.STORAGE_ACCOUNTING["checkpoint"]["two_atomic_optimizer_checkpoints"] == 1200571920
    assert sum(training.STORAGE_ACCOUNTING["checkpoint"].values()) == 1579674768
    assert sum(training.STORAGE_ACCOUNTING["scratch"].values()) == 14959160688
    for volume, amounts in training.STORAGE_ACCOUNTING.items():
        assert sum(amounts.values()) <= training.diagnostics.WRITE_BUDGETS["training"][volume]
    monkeypatch.setattr(training.diagnostics.storage, "require_mount", lambda: None)
    monkeypatch.setattr(training.diagnostics, "require_capacity", lambda stage: None)
    training.require_capacity()
    bad = deepcopy(training.STORAGE_ACCOUNTING)
    bad["checkpoint"]["unexpected"] = 1024**3
    monkeypatch.setattr(training, "STORAGE_ACCOUNTING", bad)
    with pytest.raises(ValueError, match="atomic replacements"):
        training.require_capacity()
