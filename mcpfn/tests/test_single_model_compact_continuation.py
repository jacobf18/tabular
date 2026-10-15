from copy import deepcopy
import math

import pytest
import torch

import train_single_matrixformer_compact_continuation as training


def test_complete_fresh_96_attempt_mixture_and_seed_streams():
    batches = training.batches()
    assert len(batches) == 96 and training.BOUNDARIES == (0, 16, 32, 64, 96)
    assert [batch["step"] for batch in batches] == list(range(1, 97))
    for batch in batches:
        assert len(batch["tasks"]) == 28
        assert math.isclose(math.fsum(task["weight"] for task in batch["tasks"]), 1., abs_tol=1e-15)
        assert training.constant_rate(batch["step"]) == 2e-5
    seeds = training.original.alignment.fixed.seeds_in
    assert len(seeds([batches, training.confirmation_cases()])) == 5660
    assert seeds([batches, training.confirmation_cases()]).isdisjoint(
        seeds([training.baseline.batches(), training.baseline.confirmation_cases()]))


def test_first64_priors_and_weights_unchanged_except_seeds():
    for current, parent in zip(training.batches(), training.baseline.batches()):
        stripped = deepcopy(current)
        for task, old in zip(stripped["tasks"], parent["tasks"]):
            task["gradient_seed"] = old["gradient_seed"]
            task["specification"].get("structured", task["specification"])["seed"] = old["specification"].get("structured", old["specification"])["seed"]
        assert stripped == parent


def test_training_uses_new_cache_and_correct_prefix_writer_without_old_storage_helper():
    source = training.ADAPTED_SOURCE
    assert "compact_cache.runtime()" in source and "compact_cache.capture_batch(" in source
    assert "compact_cache.evaluate_training(model, criterion, tasks)" in source
    assert "shared_gradients(" not in source and "shared_storage.save(" not in source
    assert 'record["raw_endpoint_sha256"] = save_endpoint(payload' in source
    assert 'teacher_policy.identities(base, original.alignment.fixed.PARENT)' in source
    assert '"parent_training_updates": 115' in source
    assert training.PARENT.name == "selected_checkpoint.pth"
    assert training.PARENT.parent.name == "teacher_continuation_v22_cached_recovery_v1"


def test_full_and_prefix_gradient_storage_is_preserved(monkeypatch, tmp_path):
    gradients = torch.arange(84 * 3, dtype=torch.float32).reshape(84, 3)
    payload = {"gradients": gradients, "snapshot": {"task_gradients": gradients[:28].clone()},
               "record": {"composite_tasks": 28, "protected_tasks": 84}}
    def save(shared, path):
        training.composite_storage.check_alias(shared)
        assert torch.equal(shared["gradients"], gradients)
        assert shared["snapshot"]["task_gradients"].shape == (28, 3)
        return "saved"
    monkeypatch.setattr(training.compressed_endpoint, "save", save)
    assert training.save_endpoint(payload, tmp_path / "endpoint.pt") == "saved"


def test_step_zero_never_promotes_unqualified_parent():
    anchor = {"step": 0, "score": 2.}
    assert training.selected_boundary(anchor, 0, 0, {}) == anchor
    assert training.selected_boundary(anchor, 16, 0, {}) == anchor


@pytest.mark.parametrize("step", [0, 97, True, 1.0])
def test_rate_rejects_undeclared_steps(step):
    with pytest.raises(ValueError, match="registered attempted batch"):
        training.constant_rate(step)
