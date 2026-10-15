from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import train_single_matrixformer_teacher_continuation as training
import audit_single_model_teacher_continuation as audit
from test_single_model_structured_teacher_training import remove_seeds


def test_continuation_preserves_priors_and_changes_only_initialization_and_fresh_streams():
    batches = training.batches()
    assert len(batches) == 64 and all(len(batch["tasks"]) == 28 for batch in batches)
    assert remove_seeds(batches) == remove_seeds(training.baseline.batches())
    assert remove_seeds(training.confirmation_cases()) == remove_seeds(training.baseline.confirmation_cases())
    assert training.PARENT == training.baseline.OUTPUT / "selected_checkpoint.pth"
    assert training.PARENT != training.original.alignment.fixed.PARENT
    assert training.INITIAL_RUN == training.baseline.OUTPUT
    assert training.BOUNDARIES == (0, 10, 25, 50, 64)
    assert all(training.constant_rate(step) == 2e-5 for step in range(1, 65))
    fresh = training.original.alignment.fixed.seeds_in([batches, training.confirmation_cases()])
    used = training.original.alignment.fixed.seeds_in([
        training.baseline.batches(), training.baseline.confirmation_cases(),
        training.recovery.pipeline.training.batches(), training.recovery.pipeline.training.confirmation_cases()])
    assert len(fresh) == 3868 and not fresh.intersection(used)


def test_warm_start_never_changes_teachers_or_retention_baseline():
    source = training.ADAPTED_SOURCE
    assert 'teacher_policy.identities(config["base"], original.alignment.fixed.PARENT)' in source
    assert 'teacher_policy.identities(base, original.alignment.fixed.PARENT)' in source
    assert 'teacher_policy.identities(base, PARENT)' not in source
    assert '"parent_selection_qualified": False' in source
    assert '"parent_training_updates": 56' in source
    assert 'read(original.alignment.fixed.PRODUCER / "validation_0.json")["student"]' in source
    assert 'read(INITIAL_RUN / "validation_64.json")["student"]' in source
    assert 'require_history()' in source
    assert 'preflight.require_predecessor()' not in source
    assert 'shared_storage.save(payload, output / "raw_final_update.pt")' in source
    assert 'paired.protected_update(' in source
    assert 'shutil.disk_usage(ROOT).free < 31 * 1024**3' in source
    assert training.baseline.QUALIFICATION in training.QUALIFICATION


def history():
    return [{"complete": True, "gate": {"passed": False, "confirmation_evaluated": True}, "campaign": None, "review": None},
            {"complete": True, "gate": {"passed": False, "confirmation_evaluated": False, "selected_step": 0},
             "campaign": None, "review": None}, {"complete": True, "endpoint_grids": 56}, training.PARENT_SHA256]


def test_requires_complete_negative_qualification_and_exact_parent():
    training.validate_history(*history())
    records = history()
    records[-1] = "unreviewed checkpoint"
    with pytest.raises(ValueError, match="audited unqualified"):
        training.validate_history(*records)


@pytest.mark.parametrize("index,field,value", [(0, "complete", False), (0, "campaign", "real"),
    (0, "gate", {"passed": True, "confirmation_evaluated": True}), (1, "complete", False),
    (1, "gate", {"passed": False, "confirmation_evaluated": True, "selected_step": 0}),
    (1, "review", "promoted"), (2, "complete", False), (2, "endpoint_grids", 55)])
def test_rejects_incomplete_or_changed_predecessors(index, field, value):
    records = history()
    records[index][field] = value
    with pytest.raises(ValueError, match="audited unqualified"):
        training.validate_history(*records)


def test_setup_loads_v20_student_with_fresh_moments_without_overwriting_base(tmp_path, monkeypatch):
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(2, 2)
    model.decoder = torch.nn.Linear(2, 1)
    parent = {name: torch.ones_like(tensor) for name, tensor in model.state_dict().items()}
    path = tmp_path / "parent.pth"
    torch.save({"model": parent}, path)
    original_optimizer = torch.optim.AdamW(model.parameters())
    teachers, sampler, criterion = object(), object(), object()
    config = {"base": {"peak_lr": 2e-5, "weight_decay": .01, "parent_checkpoint": "original_v8"}}
    unchanged = deepcopy(config)
    monkeypatch.setattr(training, "PARENT", path)
    monkeypatch.setattr(training.original.alignment.mixture.backend, "setup",
                        lambda base: (model, original_optimizer, criterion, teachers, sampler))
    actual, view, optimizer, actual_criterion, actual_teachers, actual_sampler, state = training.setup(config)
    assert config == unchanged
    assert actual is model and actual_criterion is criterion and actual_teachers is teachers and actual_sampler is sampler
    assert optimizer is not original_optimizer and not optimizer.state
    assert all(name.startswith("decoder.") for name, parameter in view.named_parameters())
    assert all(torch.equal(actual.state_dict()[name], tensor) for name, tensor in parent.items())
    assert all(torch.equal(state[name], tensor) for name, tensor in parent.items())
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())


def test_parent_development_parity_is_additional_to_original_baseline_audit():
    source = audit.ADAPTED_SOURCE
    assert 'parent_report_path = training.INITIAL_RUN / "validation_64.json"' in source
    assert '(parent_report, ("student",))' in source
    assert '152 * (4 + len(boundaries))' in source
    assert 'saved["parent_training_updates"] != 56' in source
    assert 'initial["student"], reference["references"], config["base"]' in source
    assert 'shared_storage.audit(payload, raw_path)' in source
    assert audit.checks is training.paired


def test_real_runtime_keeps_frozen_v8_teacher_after_loading_v20_student(tmp_path, monkeypatch):
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(2, 2)
    model.decoder = torch.nn.Linear(2, 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    teachers = [deepcopy(model).eval().requires_grad_(False) for index in range(2)]
    identities = {}
    for role, current in zip(("initialization", "general", "specialist"), [model, *teachers]):
        path = tmp_path / f"{role}.pth"
        torch.save({"model": current.state_dict()}, path)
        identities[role] = {"checkpoint": str(path), "sha256": training.sha256(path)}
    parent = tmp_path / "v20.pth"
    torch.save({"model": {name: torch.ones_like(tensor) for name, tensor in model.state_dict().items()}}, parent)
    mixture = training.original.alignment.mixture
    monkeypatch.setattr(training, "PARENT", parent)
    monkeypatch.setattr(mixture, "original_setup", lambda config: (model, None, None, teachers, None))
    monkeypatch.setattr(mixture, "StructuredSampler", lambda config: object())
    called = []

    def capture(chosen, original_teachers, criterion, episode, options):
        called.append(chosen)
        assert all(torch.count_nonzero(tensor) == 0 for tensor in chosen.state_dict().values())
        assert all(torch.equal(tensor, torch.ones_like(tensor)) for tensor in model.state_dict().values())
        return torch.zeros(1), {"specialist_weight": 0.0}

    monkeypatch.setattr(training.teacher_policy, "parent_teacher_target", capture)
    base = {"peak_lr": 2e-5, "weight_decay": .01, "query_seed_offset": 0,
            "parent_checkpoint": identities["initialization"]["checkpoint"]}
    episode = SimpleNamespace(metadata={"gradient_data": "synthetic", "domain": "imputation", "structured": False})
    with training.teacher_policy.runtime(base, tmp_path, identities):
        student, view, optimizer, criterion, original_teachers, sampler, state = training.setup({"base": base})
        prediction, routing = mixture.backend.teacher_target(original_teachers, criterion, episode)
        assert student is model and routing["teacher_identity"] == identities["initialization"]
        assert routing["target"] == "initialization"
    assert len(called) == 1 and called[0] is not model


def test_step_zero_and_any_failed_guard_cannot_select():
    original = {"step": 0, "score": 1.15}
    assert training.selected_boundary(original, 0, 0, {"passed": True, "score": 1.0}) == original
    assert training.selected_boundary(original, 10, 10, {"passed": False, "score": 1.0}) == original
