from copy import deepcopy
import inspect
from types import SimpleNamespace

import pytest
import torch

import preflight_single_model_anchor_rehearsal as preflight


def test_fresh_stratified_batch_has_correct_parent_teachers_seeds_and_unchanged_references():
    batch = preflight.batch()
    previous = preflight.template.batch()
    fresh = preflight.training.alignment.fixed.seeds_in(batch)
    assert len(fresh) == 56 and len(batch["tasks"]) == 28
    assert not fresh.intersection(preflight.training.alignment.fixed.seeds_in(previous))
    assert not fresh.intersection(preflight.training.alignment.fixed.seeds_in(preflight.preceding.training.confirmation_cases()))
    assert preflight.PARENT == preflight.preceding.training.OUTPUT / "selected_checkpoint.pth"
    assert preflight.PARENT_SHA256 == "ef775cef8dee57338132f1756ad39042995950976d5df15c18445b205b99002d"
    assert preflight.PARENT != preflight.training.alignment.fixed.PARENT
    assert preflight.RATE == 1e-5
    assert [preflight.expected_role(task) for task in batch["tasks"]].count("general") == 26
    assert [preflight.expected_role(task) for task in batch["tasks"]].count("specialist") == 2
    assert preflight.teacher_policy.POLICY == {"legacy": "general", "structured": "general", "recsys": "specialist"}
    for task, source in zip(batch["tasks"], previous["tasks"]):
        restored = deepcopy(task)
        restored["gradient_seed"] = source["gradient_seed"]
        current_specification = restored["specification"].get("structured", restored["specification"])
        source_specification = source["specification"].get("structured", source["specification"])
        current_specification["seed"] = source_specification["seed"]
        if task["replicate"] == 1 and task["specification"]["domain"] == "imputation":
            current_specification["component"] = source_specification["component"]
        assert restored == source


def test_adapted_execution_and_independent_audits_bind_new_namespace_without_source_mutation():
    source = preflight.ADAPTED_SOURCE
    assert 'view = top_two.TopTwoBlockParameterView(model)' in source
    assert 'with top_two.capture_coordinates(model, view):' in source
    assert 'rehearsal.stratify(replicated.replicate_batch(' in source
    assert 'paired.protected_update(' in source and 'paired.check_endpoint(' in source
    assert 'teacher_evidence = audit_teacher_replay(output, config)' in source
    assert 'hashes.update(audit_raw(payload, raw_path))' in source
    assert '"selection_qualified": False' in source
    assert preflight.PREFIXES == ('transformer_blocks.6.', 'transformer_blocks.7.', 'decoder.')
    for name in ("require_space", "frozen_inventory", "save_raw", "audit_raw", "check_routing", "replay_targets", "audit_teacher_replay"):
        assert inspect.unwrap(getattr(preflight, name)).__globals__ is preflight.__dict__
    assert "rehearsal.stratify" not in preflight.template.ADAPTED_SOURCE
    assert preflight.template.PARENT_SHA256 != preflight.PARENT_SHA256
    assert "test_single_model_anchor_rehearsal.py" in preflight.REGISTER_SOURCE
    assert "test_single_model_anchor_preflight.py" in preflight.REGISTER_SOURCE


def predecessor():
    result = {"complete": True, "gate": {"passed": False, "confirmation_evaluated": True}, "campaign": None, "review": None}
    audit = {"complete": True, "complete_training_audit": True, "through_step": 64, "accepted_updates": 63,
             "best": {"step": 10}, "training_episodes": 1792, "prediction_grids": 1368, "top_two_block_training": True,
             "independently_regenerated_training_tasks": 1792, "independently_regenerated_rating_contexts": 128,
             "artifact_sha256": {str(preflight.PARENT): preflight.PARENT_SHA256}}
    replay = {"complete": True, "endpoint_grids": 56, "selected_grids": 152}
    confirmation = {"complete": True, "cases": 284, "prediction_grids": 1136}
    return result, audit, replay, confirmation, preflight.PARENT_SHA256


def test_complete_unqualified_original_selection_is_admitted():
    preflight.validate_predecessor(*predecessor())


@pytest.mark.parametrize("mutation", ["published", "campaign", "review", "incomplete", "late_selection", "updates",
    "task_replay", "rating_replay", "grids", "selected_replay", "confirmation", "digest", "inventory"])
def test_any_incomplete_or_reselected_predecessor_is_rejected(mutation):
    result, audit, replay, confirmation, digest = predecessor()
    if mutation == "published":
        result["gate"]["passed"] = True
    elif mutation in ("campaign", "review"):
        result[mutation] = "new"
    elif mutation == "incomplete":
        audit["complete_training_audit"] = False
    elif mutation == "late_selection":
        audit["best"]["step"] = 64
    elif mutation == "updates":
        audit["accepted_updates"] = 64
    elif mutation == "task_replay":
        audit["independently_regenerated_training_tasks"] -= 1
    elif mutation == "rating_replay":
        audit["independently_regenerated_rating_contexts"] -= 1
    elif mutation == "grids":
        audit["prediction_grids"] -= 1
    elif mutation == "selected_replay":
        replay["selected_grids"] -= 1
    elif mutation == "confirmation":
        confirmation["cases"] -= 1
    elif mutation == "inventory":
        audit["top_two_block_training"] = False
    else:
        digest = "changed"
    with pytest.raises(ValueError):
        preflight.validate_predecessor(result, audit, replay, confirmation, digest)


def test_full_gradient_alias_and_checkpoint_scratch_evidence_round_trip(tmp_path, monkeypatch):
    output, scratch = tmp_path / "canonical", tmp_path / "scratch"
    output.mkdir()
    scratch.mkdir()
    monkeypatch.setattr(preflight, "OUTPUT", output)
    monkeypatch.setattr(preflight, "SCRATCH", scratch)
    gradients = torch.arange(84 * 8, dtype=torch.float32).reshape(84, 8)
    gradients[0, 0] = -0.
    payload = {"gradients": gradients, "snapshot": {"task_gradients": gradients[:28].clone()},
               "record": {"composite_tasks": 28, "protected_tasks": 84}, "batch": {"tasks": [None] * 28}}
    path = output / "raw_update.pt"
    digest = preflight.save_raw(payload, path)
    restored = torch.load(path, weights_only=True, mmap=True)
    assert digest == preflight.sha256(path)
    assert preflight.composite.byte_equal(restored["gradients"], gradients)
    assert path.is_symlink() and path.resolve() == scratch / path.name
    assert preflight.audit_raw(restored, path)


@pytest.mark.parametrize("low_volume", [None, "checkpoint", "root", "scratch"])
def test_write_budgets_preserve_all_original_volume_reserves(monkeypatch, low_volume):
    available = {str(preflight.ROOT): 19 * 1024**3, "/": 6 * 1024**3, str(preflight.SCRATCH.parent): 66 * 1024**3}
    names = {"checkpoint": str(preflight.ROOT), "root": "/", "scratch": str(preflight.SCRATCH.parent)}
    if low_volume is not None:
        available[names[low_volume]] -= 1
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda path: SimpleNamespace(free=available[str(path)]))
    if low_volume is None:
        preflight.require_space()
    else:
        with pytest.raises(OSError):
            preflight.require_space()
