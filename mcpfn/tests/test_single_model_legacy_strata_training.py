from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import train_single_matrixformer_legacy_strata as training
import audit_single_model_legacy_strata as audit


def test_schedule_changes_only_fresh_seed_streams_not_prior_scope_or_gates():
    assert training.PARENT == training.template.PARENT
    assert training.PARENT_SHA256 == training.template.PARENT_SHA256
    assert training.PARENT_UPDATES == 73
    assert training.BOUNDARIES == training.template.BOUNDARIES and training.STEPS == 64
    assert training.PREFIXES == training.template.PREFIXES and training.RATE == 5e-6
    assert training.paired.__name__ == "single_model_legacy_stratum_protection"
    assert audit.checks is training.paired
    assert training.train.__globals__["paired"] is training.paired
    assert audit.audit.__globals__["checks"] is training.paired
    actual, previous = training.batches(), training.template.batches()
    for current, old in zip(actual, previous):
        for task, original in zip(current["tasks"], old["tasks"]):
            comparable = deepcopy(task)
            comparable["gradient_seed"] = original["gradient_seed"]
            comparable["specification"].get("structured", comparable["specification"])["seed"] = (
                original["specification"].get("structured", original["specification"])["seed"])
            assert comparable == original
    seeds = training.original.alignment.fixed.seeds_in([actual, training.confirmation_cases()])
    assert len(seeds) == 3868
    assert not seeds.intersection(training.original.alignment.fixed.seeds_in([previous, training.template.confirmation_cases()]))


@pytest.mark.parametrize("eligible_steps,accepted_steps", [({1, 3}, {1}), ({1}, {1}), (set(), set()), ({4}, {4})])
def test_actual_training_loop_keeps_last_real_endpoint_across_noops(tmp_path, monkeypatch, eligible_steps, accepted_steps):
    model = torch.nn.Module()
    model.decoder = torch.nn.Linear(1, 1, bias=False)
    model.decoder.weight.data.zero_()
    inventory = [[name, list(parameter.shape), parameter.numel()] for name, parameter in model.named_parameters()]
    batches = [{"step": step, "tasks": [{"index": 0, "name": f"task_{step}", "weight": 1., "specification": {"domain": "imputation"}}]}
               for step in range(1, 5)]
    config = {"batches": batches, "base": {}, "teacher_identities": {}}
    captured, validations, checkpoints, raw_endpoints = [], [], [], []
    training.write_json(tmp_path / "config.json", config)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "verify_registration", lambda output: config)
    monkeypatch.setattr(training, "require_capacity", lambda: None)
    monkeypatch.setattr(training, "wait_for_gpu", lambda *args: None)
    monkeypatch.setattr(training, "BOUNDARIES", (0, 1, 2, 3, 4))
    monkeypatch.setattr(training.signal, "signal", lambda *args: None)
    original_read = training.read
    monkeypatch.setattr(training, "read", lambda path: inventory if path.name == "parameter_inventory.json" else
                        {"student": {}, "references": {}} if path.name in {"reference.json", "validation_0.json"} else original_read(path))
    monkeypatch.setattr(training.torch, "load", lambda *args, **kwargs: [])
    original_empty = torch.empty
    monkeypatch.setattr(training.torch, "empty", lambda shape, **kwargs: original_empty(shape, dtype=kwargs["dtype"]))
    for name in ("reset_peak_memory_stats",):
        monkeypatch.setattr(training.torch.cuda, name, lambda: None)
    monkeypatch.setattr(training.torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(training.original.alignment.fixed, "selection_decision", lambda *args: {"score": 1.})
    monkeypatch.setattr(training.teacher_policy, "runtime", lambda *args: nullcontext())
    monkeypatch.setattr(training.compact_cache, "runtime", lambda: nullcontext())
    monkeypatch.setattr(training.top_two, "capture_coordinates", lambda *args: nullcontext())
    monkeypatch.setattr(training, "setup", lambda *args: (model, model, object(), object(), object(), object(), {}))

    def validate(output, current, optimizer, step, accepted, best, *args):
        validations.append((step, accepted))
        return best

    def checkpoint(output, current, optimizer, step, accepted, best, reference):
        checkpoints.append((step, accepted, float(current.decoder.weight.item())))

    def checker(sampler, batch):
        consumed = []
        eligible = batch["step"] in eligible_steps

        def assert_consumed():
            assert bool(consumed) is eligible

        return SimpleNamespace(eligible=eligible, record={"step": batch["step"], "eligible": eligible,
            "task_count": 1, "tasks": [{"task": task} for task in batch["tasks"]]},
            assert_consumed=assert_consumed, consumed=consumed)

    def capture(output, directory, batch, current, optimizer, criterion, teachers, checker, gradients):
        captured.append(batch["step"])
        checker.consumed.append(True)
        gradients.fill_(batch["step"])
        training.write_json(directory / "manifest.json", {"step": batch["step"]})
        return [batch["tasks"][0]], batch["tasks"], {"step": batch["step"]}

    def update(view, optimizer, gradients, weights, rate, before, evaluate):
        step = before["step"]
        accepted = step in accepted_steps
        if accepted:
            view.decoder.weight.data.add_(1)
        return {"accepted": accepted, "reason": "accepted" if accepted else "finite_metrics_rejected", "rate": rate}, {
            "after": view.decoder.weight.detach().flatten().clone()}

    def save_endpoint(payload, path):
        raw_endpoints.append(deepcopy(payload))
        return "a" * 64

    monkeypatch.setattr(training, "validate", validate)
    monkeypatch.setattr(training, "checkpoint", checkpoint)
    monkeypatch.setattr(training.admission, "TrainingBatchAdmission", checker)
    monkeypatch.setattr(training.compact_cache, "capture_batch", capture)
    monkeypatch.setattr(training.paired, "validate_layout", lambda *args: None)
    monkeypatch.setattr(training.paired, "protected_update", update)
    monkeypatch.setattr(training, "save_endpoint", save_endpoint)
    training.train(tmp_path)
    assert captured == sorted(eligible_steps)
    assert [step for step, count in validations] == [0, 1, 2, 3, 4]
    assert checkpoints[-1] == (4, len(accepted_steps), float(len(accepted_steps)))
    descriptor = training.read(tmp_path / "endpoint.json")
    assert descriptor["terminal_step"] == 4
    assert descriptor["optimizer_attempt_step"] == (max(eligible_steps) if eligible_steps else None)
    assert descriptor["accepted_updates"] == len(accepted_steps)
    if eligible_steps:
        assert len(raw_endpoints) == 1 and raw_endpoints[0]["batch"]["step"] == max(eligible_steps)
        assert torch.equal(raw_endpoints[0]["gradients"], torch.full((1, 1), float(max(eligible_steps))))
    else:
        assert raw_endpoints == [] and descriptor["raw_endpoint_sha256"] is None
    for step in set(range(1, 5)) - eligible_steps:
        assert not (tmp_path / "training_batches" / f"step_{step:03d}").exists()


@pytest.mark.parametrize("mutation", ["state", "optimizer", "accepted", "signed_zero"])
def test_all_noop_endpoint_audit_checks_full_initializer_and_optimizer(mutation):
    parent = {"weight": torch.ones(2)}
    latest = {"model": deepcopy(parent), "optimizer": {"state": {}}, "gradient_updates_performed": 0}
    audit.check_empty_optimizer_endpoint(latest, parent)
    if mutation == "state":
        latest["model"]["weight"][0] += 1
    elif mutation == "optimizer":
        latest["optimizer"]["state"][0] = {}
    elif mutation == "accepted":
        latest["gradient_updates_performed"] = 1
    else:
        parent["weight"].zero_()
        latest["model"]["weight"].zero_().neg_()
    with pytest.raises(ValueError):
        audit.check_empty_optimizer_endpoint(latest, parent)
