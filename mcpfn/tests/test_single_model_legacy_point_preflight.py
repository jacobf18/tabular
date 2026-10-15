import builtins
import inspect
from types import SimpleNamespace

import pytest
import torch

import preflight_single_model_legacy_point as preflight
from recover_single_model_legacy_strata_replay import code_globals


def test_fresh_probe_preserves_parent_recipe_and_complete_runtime_namespace():
    assert preflight.PARENT == preflight.template.PARENT
    assert preflight.PARENT_SHA256 == preflight.template.PARENT_SHA256
    assert preflight.RATE == 5e-6
    assert preflight.PREFIXES == preflight.template.PREFIXES
    assert preflight.teacher_policy is preflight.template.teacher_policy
    assert preflight.mixture.MASSES == {"legacy": .5, "structured": .25, "recsys": .25}
    assert (preflight.DATA_SEED, preflight.GRADIENT_SEED) == (711100901, 712100901)
    assert len(preflight.training.alignment.fixed.seeds_in(preflight.batch())) == 56
    assert preflight.paired is not preflight.template.paired
    assert preflight.composite is not preflight.template.composite
    for name, function in vars(preflight).items():
        if inspect.isfunction(function) and inspect.unwrap(function).__globals__ is vars(preflight):
            missing = code_globals(inspect.unwrap(function).__code__) - vars(preflight).keys() - vars(builtins).keys()
            assert not missing, (name, missing)
    for name in ("require_space", "save_raw", "audit_raw", "admit_batch", "audit_admission", "save_probe_checkpoint"):
        assert getattr(preflight, name).__code__ is getattr(preflight.template, name).__code__
        assert getattr(preflight, name).__globals__ is vars(preflight)
    assert preflight.run.__globals__["paired"] is preflight.paired
    assert preflight.audit_admitted.__globals__["paired"] is preflight.paired
    assert preflight.save_raw.__globals__["composite"] is preflight.composite
    assert preflight.audit_raw.__globals__["composite"] is preflight.composite
    assert inspect.unwrap(preflight.replay_admitted).__code__ is inspect.unwrap(preflight.template.replay_admitted).__code__
    assert inspect.unwrap(preflight.replay_admitted).__globals__ is vars(preflight)


def latest():
    gate = {"passed": False, "confirmation_evaluated": False, "selected_step": 0}
    return ({"complete": True, "gate": gate, "campaign": None, "review": None, "no_inference_ensemble": True},
            {"complete": True, "complete_training_audit": True, "through_step": 64, "accepted_updates": 64,
             "best": {"step": 0}, "prediction_grids": 1368, "independently_regenerated_training_tasks": 1792,
             "independently_regenerated_rating_contexts": 128, "independently_regenerated_structured_draws": 1536},
            {"complete": True, "endpoint_grids": 56, "selected_grids": 0}, gate)


@pytest.mark.parametrize("mutation", [None, "unfinished", "selected", "confirmation", "replay", "tasks", "published"])
def test_requires_completed_v30_nonselection_without_confirmation(mutation):
    result, training, replay, gate = latest()
    if mutation == "unfinished":
        training["complete_training_audit"] = False
    elif mutation == "selected":
        training["best"]["step"] = 64
    elif mutation == "confirmation":
        gate["confirmation_evaluated"] = True
    elif mutation == "replay":
        replay["endpoint_grids"] = 55
    elif mutation == "tasks":
        training["independently_regenerated_training_tasks"] = 1791
    elif mutation == "published":
        result["campaign"] = "other"
    if mutation:
        with pytest.raises(ValueError):
            preflight.validate_latest(result, training, replay, gate)
    else:
        preflight.validate_latest(result, training, replay, gate)


def test_rejected_admission_still_performs_no_model_or_optimizer_work(tmp_path, monkeypatch):
    output, scratch = tmp_path / "output", tmp_path / "scratch"
    output.mkdir()
    scratch.mkdir()
    monkeypatch.setattr(preflight, "OUTPUT", output)
    monkeypatch.setattr(preflight, "SCRATCH", scratch)
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
        pytest.fail("Rejected admission cannot initialize a model or consume an optimizer proposal")

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


def test_actual_proposal_endpoint_uses_full_alias_through_preflight_storage(tmp_path, monkeypatch):
    output, scratch = tmp_path / "output", tmp_path / "scratch"
    output.mkdir()
    scratch.mkdir()
    monkeypatch.setattr(preflight, "OUTPUT", output)
    monkeypatch.setattr(preflight, "SCRATCH", scratch)
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.02)
    gradients = torch.ones((6, 2))
    before = {f"legacy_draw_{draw}/{kind}": 1. for draw in range(2) for kind in ("rmse", "common_nrmse")}
    record, snapshot = preflight.paired.protected_update(model, optimizer, gradients, [.5, .5], .01, before,
                                                        lambda current: {name: .9 for name in before})
    record.update(step=1, accepted_updates=int(record["accepted"]))
    endpoint = {"record": record, "snapshot": snapshot, "gradients": gradients, "batch": {"tasks": [{}, {}]}}
    path = output / "endpoint.pt"
    preflight.save_raw(endpoint, path)
    restored = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert path.is_symlink() and path.resolve() == scratch / path.name
    preflight.audit_raw(restored, path)
    preflight.paired.check_endpoint(restored, record, [2], .02, .003, optimizer.state_dict())
    assert restored["snapshot"]["task_gradients"].shape == (6, 2)
    assert restored["snapshot"]["task_gradients"].data_ptr() == restored["gradients"].data_ptr()
