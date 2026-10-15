"""Verify prospective V29 continuation scope, dependencies and nonselectable storage."""

import builtins
from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import preflight_single_model_legacy_heavy as preflight
from recover_single_model_legacy_strata_replay import code_globals


def test_new_preflight_has_fresh_scope_and_all_runtime_dependencies():
    assert preflight.PARENT == preflight.preceding.training.OUTPUT / "selected_checkpoint.pth"
    assert preflight.PARENT != preflight.template.PARENT
    assert preflight.RATE == 5e-6
    assert (preflight.DATA_SEED, preflight.GRADIENT_SEED) == (681100901, 682100901)
    assert preflight.paired is preflight.template.paired
    assert preflight.teacher_policy is preflight.template.teacher_policy
    assert preflight.PREFIXES == preflight.template.PREFIXES
    assert preflight.mixture.validate(preflight.batch())["step"] == 1
    seeds = preflight.training.alignment.fixed.seeds_in(preflight.batch())
    assert len(seeds) == 56 and not seeds.intersection(preflight.training.alignment.fixed.seeds_in(preflight.template.batch()))
    for name, function in vars(preflight).items():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(preflight):
            assert not code_globals(inspect.unwrap(function).__code__) - vars(preflight).keys() - vars(builtins).keys(), name
    for name in ("frozen_inventory", "save_raw", "audit_raw", "expected_role", "check_routing", "audit_teacher_replay",
                 "admit_batch", "audit_admission"):
        assert inspect.unwrap(getattr(preflight, name)).__code__ is inspect.unwrap(getattr(preflight.template, name)).__code__
    assert inspect.unwrap(preflight.replay_admitted).__code__ is inspect.unwrap(preflight.template.replay_admitted).__code__
    assert preflight.replay_admitted is not inspect.unwrap(preflight.replay_admitted)
    assert preflight.audit_admitted.__globals__ is vars(preflight)
    assert "save_probe_checkpoint(" in preflight.ADAPTED_SOURCE
    assert 'hashes.update(audit_probe_checkpoint(checkpoint_path))' in preflight.ADAPTED_SOURCE


def predecessor():
    return ({"complete": True, "gate": {"passed": False, "confirmation_evaluated": True, "selected_step": 25,
             "checkpoint_sha256": preflight.PARENT_SHA256}, "campaign": None, "review": None,
             "scientific_stages_repeated": False, "no_inference_ensemble": True},
            {"complete": True, "complete_training_audit": True, "through_step": 64, "accepted_updates": 63,
             "best": {"step": 25}, "prediction_grids": 1368},
            {"complete": True, "endpoint_grids": 56, "selected_grids": 152},
            {"complete": True, "prediction_grids": 1136, "cases": 284}, preflight.PARENT_SHA256)


@pytest.mark.parametrize("mutation", [None, "selection", "unfinished", "replay", "confirmation", "published", "digest"])
def test_only_fully_audited_unqualified_v29_can_initialize(mutation):
    result, training, replay, confirmation, digest = predecessor()
    if mutation == "selection":
        training["best"]["step"] = 64
    elif mutation == "unfinished":
        training["complete_training_audit"] = False
    elif mutation == "replay":
        replay["selected_grids"] = 151
    elif mutation == "confirmation":
        confirmation["prediction_grids"] = 1135
    elif mutation == "published":
        result["campaign"] = "other"
    elif mutation == "digest":
        digest = "changed"
    if mutation:
        with pytest.raises(ValueError):
            preflight.validate_predecessor(result, training, replay, confirmation, digest)
    else:
        preflight.validate_predecessor(result, training, replay, confirmation, digest)


@pytest.fixture
def placement(tmp_path, monkeypatch):
    output, scratch = tmp_path / "output", tmp_path / "scratch"
    output.mkdir()
    scratch.mkdir()
    monkeypatch.setattr(preflight, "OUTPUT", output)
    monkeypatch.setattr(preflight, "SCRATCH", scratch)
    return output, scratch


def test_diagnostic_checkpoint_saved_once_and_independently_binds_exact_scratch_link(placement):
    output, scratch = placement
    path = output / "preflight_checkpoint.pth"
    payload = {"model": {"test": torch.ones(2)}, "preflight_only": True, "selection_qualified": False}
    digest = preflight.save_probe_checkpoint(payload, path)
    assert path.is_symlink() and path.readlink() == scratch / path.name
    assert preflight.sha256(path) == digest
    assert len(preflight.audit_probe_checkpoint(path)) == 1
    assert torch.equal(torch.load(path, weights_only=True)["model"]["test"], torch.ones(2))
    with pytest.raises(ValueError):
        preflight.save_probe_checkpoint(payload, path)
    path.unlink()
    torch.save(payload, path)
    with pytest.raises(ValueError):
        preflight.audit_probe_checkpoint(path)


@pytest.mark.parametrize("mutation", ["qualified", "promotable", "wrong_name"])
def test_no_selected_or_promotable_model_may_use_diagnostic_storage(placement, mutation):
    output, scratch = placement
    path = output / "preflight_checkpoint.pth"
    payload = {"preflight_only": True, "selection_qualified": False}
    if mutation == "qualified":
        payload["selection_qualified"] = True
    elif mutation == "promotable":
        payload["preflight_only"] = False
    else:
        path = output / "selected_checkpoint.pth"
    with pytest.raises(ValueError):
        preflight.save_probe_checkpoint(payload, path)
    assert not list(output.iterdir()) and not list(scratch.iterdir())


def test_data_admission_rejection_still_does_no_model_or_optimizer_work(placement, monkeypatch):
    output, scratch = placement
    config = {"base": {}, "batch": {}}
    record = {"eligible": False, "task_count": 28, "structured_draws": 24,
              "deferred_task_indices": [0, 13, 14, 27], "rejected_task_indices": [23]}
    consumed = []
    checker = SimpleNamespace(eligible=False, record=record, assert_consumed=lambda: consumed.append(True))
    monkeypatch.setattr(preflight, "StructuredSampler", lambda *args, **kwargs: object())
    monkeypatch.setattr(preflight.admission, "TrainingBatchAdmission", lambda *args: checker)
    monkeypatch.setattr(preflight, "verify", lambda output: config)
    monkeypatch.setattr(preflight, "require_space", lambda: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(preflight.torch.cuda, "is_initialized", lambda: False)

    def forbidden(*args, **kwargs):
        pytest.fail("A rejected admission cannot initialize or update a model")

    monkeypatch.setattr(preflight, "wait_for_gpu", forbidden)
    monkeypatch.setattr(preflight.training.alignment.mixture.backend, "setup", forbidden)
    monkeypatch.setattr(preflight.paired, "protected_update", forbidden)
    for name in ("config.json", "registration.json"):
        preflight.write_json(output / name, {})
    preflight.status(output, "registered")
    preflight.run(output)
    result = preflight.read(output / "result.json")
    result["training_pid"] += 1
    preflight.write_json(output / "result.json", result)
    preflight.replay_targets(output)
    preflight.audit(output)
    audit = preflight.read(output / "audit.json")
    assert audit["complete"] and audit["data_admission_only"]
    assert audit["training_tasks"] == audit["prediction_grids"] == audit["accepted_updates"] == 0
    assert not preflight.read(output / "gate.json")["passed"]
    assert not list(scratch.iterdir()) and consumed == [True, True]


@pytest.mark.parametrize("volume", ["checkpoint", "root", "scratch"])
def test_budget_checks_preserve_all_three_reserves(monkeypatch, volume):
    monkeypatch.setattr(preflight.preceding.diagnostics.storage, "require_mount", lambda: None)
    paths = {preflight.ROOT: "checkpoint", Path("/"): "root", preflight.SCRATCH.parent: "scratch"}
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda path: SimpleNamespace(
        free=preflight.RESERVES[paths[path]] + preflight.WRITE_BUDGETS[paths[path]] - int(paths[path] == volume)))
    with pytest.raises(OSError):
        preflight.require_space()
