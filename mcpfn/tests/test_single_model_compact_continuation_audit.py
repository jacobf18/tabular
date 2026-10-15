from types import SimpleNamespace

import pytest
import torch

import audit_single_model_compact_continuation as audit


def test_preserves_endpoint_and_selection_checks_but_discloses_gpu_replay():
    source = audit.ADAPTED_SOURCE
    assert "all22" in source.lower()
    assert 'decision != report["decision"]' in source
    assert 'len(decision["comparisons"]) != 22' in source
    assert 'payload = load_regenerated_task(path)' in source
    assert 'hashes.update(compressed_endpoint.audit(payload, raw_path))' in source
    assert 'check_gradient_alias(payload)' in source
    assert '"cuda_initialized": torch.cuda.is_initialized()' in source
    assert '"autograd_replayed": False' in source
    assert '"full_optimizer_trajectory_replayed": False' in source
    assert 'saved["parent_training_updates"] != 115' in source


def test_audit_load_regenerates_and_rejects_repeat(tmp_path):
    original = {"episode": {"scaling": None, "source": torch.tensor([1., float("nan")])},
        "teacher": torch.tensor([-0., 2.]), "routing": {}, "task": {"specification": {"seed": 123}}}
    capsule = audit.compact_cache.capsule_for(original)
    path = tmp_path / "task.pt"
    torch.save(capsule, path)
    backend = SimpleNamespace(teacher_target=lambda *args: (original["teacher"], original["routing"]),
                              cache_episode=lambda episode: episode)
    active = {"output": tmp_path, "sampler": SimpleNamespace(sample=lambda **args: original["episode"]),
              "backend": backend, "teachers": None, "criterion": None, "seen": set(), "rating_contexts": 0}
    token = audit.REPLAY.set(active)
    try:
        restored = audit.load_regenerated_task(path)
        assert audit.compact_cache.content_digest(restored) == capsule["content_sha256"]
        assert active["seen"] == {str(path)} and active["rating_contexts"] == 0
        row = audit.json.loads((tmp_path / "tasks.jsonl").read_text())
        assert row["all_episode_and_teacher_bytes_match"] and row["sha256"] == audit.sha256(path)
        with pytest.raises(ValueError, match="once"):
            audit.load_regenerated_task(path)
    finally:
        audit.REPLAY.reset(token)


def test_regeneration_rejects_missing_cases(tmp_path):
    active = {"output": tmp_path, "seen": set(), "rating_contexts": 0}
    token = audit.REPLAY.set(active)
    try:
        with pytest.raises(ValueError, match="every new"):
            audit.regeneration_evidence(1)
    finally:
        audit.REPLAY.reset(token)


def test_never_substitutes_new_training_parent_for_retention_baseline():
    assert audit.training.PARENT != audit.training.original.alignment.fixed.PARENT
    assert 'initial["student"], reference["references"], config["base"]' in audit.ADAPTED_SOURCE
    assert audit.check_gradient_alias is audit.composite_storage.check_alias
