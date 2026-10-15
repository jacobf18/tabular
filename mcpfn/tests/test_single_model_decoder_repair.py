from copy import deepcopy

import torch

import train_single_matrixformer_decoder_repair as training
from test_single_model_streamed_point_gradients import fixture


def test_decoder_trial_preserves_schedule_but_reserves_all_fresh_seeds():
    current = training.batches()
    previous = training.original.original.training_batches()[:50]
    assert len(current) == 50
    for batch, original in zip(current, previous):
        assert batch["step"] == original["step"] and len(batch["tasks"]) == 14
        for task, old in zip(batch["tasks"], original["tasks"]):
            restored = deepcopy(task)
            restored["gradient_seed"] = old["gradient_seed"]
            specification = restored["specification"].get("structured", restored["specification"])
            specification["seed"] = old["specification"].get("structured", old["specification"])["seed"]
            assert restored == old
    seeds = training.original.alignment.fixed.seeds_in([current, training.confirmation_cases()])
    assert len(seeds) == 1684
    assert not seeds.intersection(training.original.alignment.fixed.seeds_in(training.original.batches()))


def test_confirmation_keeps_the_complete_original_scope():
    cases = training.confirmation_cases()
    assert len(cases) == 284
    assert sum("structured" in row for row in cases) == 192
    assert training.BOUNDARIES == (0, 10, 25, 50)
    assert training.PARENT != training.original.alignment.fixed.PARENT
    assert "V8" in training.QUALIFICATION


def test_unqualified_training_parent_or_zero_updates_cannot_be_selected():
    best = {"step": 0, "score": 1.2}
    decision = {"passed": True, "score": 1.1}
    assert training.selected_boundary(best, 0, 46, decision) == best
    assert training.selected_boundary(best, 10, 0, decision) == best


def test_eligible_trained_boundary_and_earlier_tie_rule():
    best = {"step": 0, "score": 1.2}
    failed = {"passed": False, "score": 1.05}
    assert training.selected_boundary(best, 10, 9, failed) == best
    selected = training.selected_boundary(best, 10, 9, {"passed": True, "score": 1.1})
    assert selected == {"step": 10, "score": 1.1}
    assert training.selected_boundary(selected, 25, 23, {"passed": True, "score": 1.1}) == selected


def test_new_training_parent_does_not_replace_original_teachers(monkeypatch):
    model, criterion, _ = fixture("imputation")
    teacher = deepcopy(model).eval().requires_grad_(False)
    original_teacher = deepcopy(teacher.state_dict())
    parent = deepcopy(model.state_dict())
    for name, value in parent.items():
        if name.endswith("weight"):
            value.add_(.01)
    old_optimizer = torch.optim.AdamW(model.parameters())
    sampler = object()
    monkeypatch.setattr(training.original.alignment.mixture.backend, "setup", lambda config: (model, old_optimizer, criterion, [teacher], sampler))
    monkeypatch.setattr(training.torch, "load", lambda *args, **kwargs: {"model": parent})
    result, view, optimizer, returned_criterion, teachers, returned_sampler, parent_state = training.setup({"base": {"peak_lr": 1e-5, "weight_decay": .02}})
    assert returned_criterion is criterion and returned_sampler is sampler and parent_state is parent
    for name, value in result.state_dict().items():
        assert torch.equal(value, parent[name])
        assert torch.equal(teachers[0].state_dict()[name], original_teacher[name])
    assert {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} == {id(parameter) for parameter in view.parameters()}
    assert all(parameter.requires_grad == name.startswith("decoder.") for name, parameter in model.named_parameters())
