"""Exercise the full V30 schedule and actual no-op-aware training loop."""

from copy import deepcopy
import math

import pytest

import train_single_matrixformer_legacy_heavy as training
import test_single_model_legacy_strata_training as inherited


def test_full_schedule_preserves_every_task_except_declared_weights_and_new_seeds():
    assert training.PARENT == training.preflight.PARENT == training.INITIAL_RUN / "selected_checkpoint.pth"
    assert training.PARENT != training.preflight.OUTPUT / "preflight_checkpoint.pth"
    assert training.PARENT_UPDATES == 98 and training.RATE == 5e-6
    assert training.STEPS == 64 and training.BOUNDARIES == (0, 10, 25, 50, 64)
    batches = training.batches()
    assert len(batches) == 64 and sum(len(batch["tasks"]) for batch in batches) == 1792
    for actual, reference in zip(batches, training.template.batches()):
        restored = training.mixture.validate(actual)
        assert actual["step"] == reference["step"]
        for target, previous in zip(restored["tasks"], reference["tasks"]):
            before, after = deepcopy(previous), deepcopy(target)
            before["gradient_seed"] = after["gradient_seed"]
            before["specification"].get("structured", before["specification"])["seed"] = after["specification"].get(
                "structured", after["specification"])["seed"]
            assert before == after
        for group, mass in training.mixture.MASSES.items():
            assert math.isclose(math.fsum(task["weight"] for task in actual["tasks"] if training.mixture.group(task) == group),
                                mass, rel_tol=0, abs_tol=1e-15)
    cases = training.confirmation_cases()
    assert len(cases) == 284
    fresh = training.original.alignment.fixed.seeds_in([batches, cases])
    old = training.original.alignment.fixed.seeds_in([training.template.batches(), training.template.confirmation_cases(),
                                                    training.preflight.batch()])
    assert len(fresh) == 3868 and not fresh.intersection(old)
    assert training.CONFIRMATION_SEEDS == (701100901, 702100901)
    assert 'INITIAL_RUN / "validation_25.json"' in training.REGISTER_SOURCE
    assert '"mixture_masses": mixture.MASSES' in training.REGISTER_SOURCE
    assert '"parent_training_updates": PARENT_UPDATES' in training.REGISTER_SOURCE
    assert training.paired is training.template.paired and training.teacher_policy is training.template.teacher_policy


@pytest.mark.parametrize("eligible_steps,accepted_steps", [({1, 3}, {1}), ({1}, {1}), (set(), set()), ({4}, {4})])
def test_actual_v30_training_loop_preserves_last_endpoint_across_data_only_noops(tmp_path, monkeypatch, eligible_steps, accepted_steps):
    monkeypatch.setattr(inherited, "training", training)
    inherited.test_actual_training_loop_keeps_last_real_endpoint_across_noops(tmp_path, monkeypatch, eligible_steps, accepted_steps)
