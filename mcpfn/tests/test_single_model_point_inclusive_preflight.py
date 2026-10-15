import hashlib
from types import SimpleNamespace

import pytest
import torch

import preflight_single_model_point_inclusive as preflight
import preflight_single_model_structured_teacher as previous


def test_fresh_v8_preflight_keeps_priors_teachers_and_heldout_separation():
    assert preflight.PARENT == previous.PARENT
    assert preflight.preceding.OUTPUT.name == "structured_teacher_v20_pipeline"
    assert preflight.OUTPUT.name == "point_inclusive_preflight_v1"
    assert preflight.RATE == previous.RATE == 2e-5
    assert preflight.DATA_SEED == 401100901 and preflight.GRADIENT_SEED == 402100901
    assert preflight.teacher_policy.POLICY == previous.teacher_policy.POLICY
    actual, old = preflight.batch(), previous.batch()
    assert len(actual["tasks"]) == 28
    for task, earlier in zip(actual["tasks"], old["tasks"]):
        assert task["weight"] == earlier["weight"]
        assert task["name"] == earlier["name"]
        task["gradient_seed"] = earlier["gradient_seed"]
        task["specification"].get("structured", task["specification"])["seed"] = earlier["specification"].get(
            "structured", earlier["specification"])["seed"]
    assert actual == old
    seeds = preflight.training.alignment.fixed.seeds_in(preflight.batch())
    assert len(seeds) == 56
    assert not seeds.intersection(preflight.training.alignment.fixed.seeds_in(previous.batch()))


def test_shared_gradients_preserve_all_bytes_and_serialization_aliases(tmp_path):
    gradients = torch.arange(6 * 2048, dtype=torch.float32).reshape(6, 2048)
    snapshot = {"task_gradients": gradients.clone()}
    shared = preflight.shared_gradients(gradients, snapshot)
    assert shared is snapshot["task_gradients"]
    path = tmp_path / "shared.pt"
    torch.save({"gradients": shared, "snapshot": snapshot}, path)
    loaded = torch.load(path, weights_only=True)
    assert loaded["gradients"].data_ptr() == loaded["snapshot"]["task_gradients"].data_ptr()
    torch.testing.assert_close(loaded["gradients"], gradients, rtol=0, atol=0)
    assert path.stat().st_size < gradients.numel() * gradients.element_size() + 4096
    snapshot["task_gradients"][0, 0] += 1
    with pytest.raises(ValueError, match="complete original"):
        preflight.shared_gradients(gradients, snapshot)


def test_fixed_space_allowance_never_spends_eighteen_gib_reserve(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda path: SimpleNamespace(free=23 * 1024**3 - 1))
    with pytest.raises(OSError, match="unchanged18 GiB"):
        preflight.require_space()
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda path: SimpleNamespace(free=23 * 1024**3))
    preflight.require_space()


def test_registration_binds_actual_proposal_and_storage_recipe():
    source = preflight.ADAPTED_SOURCE
    assert 'inclusive.protected_update(' in source
    assert 'inclusive.check_endpoint(' in source
    assert 'inclusive.check_record(' in source
    assert '"proposal_policy": inclusive.POLICY' in source
    assert '"raw_gradient_storage": RAW_GRADIENT_STORAGE' in source
    assert 'config["proposal_policy"] != inclusive.POLICY' in source
    assert 'audit_teacher_replay(output, config)' in source
    assert 'teacher_policy.runtime(config["base"], output, config["teacher_identities"])' in source
    assert 'paired.protected_update(' in previous.ADAPTED_SOURCE
    assert preflight.ADAPTED_SOURCE_SHA256 == hashlib.sha256(source.encode()).hexdigest()
