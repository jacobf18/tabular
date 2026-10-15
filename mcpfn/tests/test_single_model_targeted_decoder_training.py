from copy import deepcopy

import torch

import train_single_matrixformer_targeted_decoder as training
import math
from itertools import product
from test_single_model_streamed_point_gradients import fixture


def test_confirmation_keeps_the_complete_original_scope():
    cases = training.confirmation_cases()
    assert len(cases) == 284
    assert sum("structured" in row for row in cases) == 192
    assert training.BOUNDARIES == (0, 10, 25, 50, 100)
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


def test_every_batch_retains_all_families_and_original_domain_mass():
    batches = training.batches()
    assert len(batches) == 100
    previous = training.original.original.training_batches()
    for batch, old in zip(batches, previous):
        assert batch["step"] == old["step"]
        assert len(batch["tasks"]) == 14
        assert [task["name"] for task in batch["tasks"]] == [task["name"] for task in old["tasks"]]
        weights = {task["name"]: task["weight"] for task in batch["tasks"]}
        assert weights["legacy"] == .25 and weights["recsys"] == .5
        assert weights["mar"] == weights["mnar_skip_logic"] == .1
        assert math.fsum(weights.values()) == 1.
        assert all(weight > 0 for weight in weights.values())
        for task, original in zip(batch["tasks"], old["tasks"]):
            if task["name"] in training.COMPONENT_CYCLES:
                continue
            restored = deepcopy(task)
            restored["weight"] = original["weight"]
            restored["gradient_seed"] = original["gradient_seed"]
            restored["specification"].get("structured", restored["specification"])["seed"] = original["specification"].get("structured", original["specification"])["seed"]
            assert restored == original


def test_all_weak_family_components_cross_every_context_regime_and_rate():
    expected = set(product(((240, 80), (1024, 256)), ("legacy", "area_scaled"), (.2, .6)))
    observed = {family: {component: set() for component in set(cycle)} for family, cycle in training.COMPONENT_CYCLES.items()}
    for batch in training.batches()[:64]:
        for task in batch["tasks"]:
            if task["name"] in observed:
                specification = task["specification"]["structured"]
                observed[task["name"]][specification["component"]].add(
                    (tuple(specification["shape"]), specification["regime"], specification["requested_missingness"]))
    assert all(coverage == expected for components in observed.values() for coverage in components.values())


def test_first_emphasized_draws_match_successful_preflight_scope_except_seed():
    trial = {task["name"]: task for task in training.batches()[0]["tasks"]}
    preflight = {task["name"]: task for task in training.preflight.batch()["tasks"]}
    for family in training.COMPONENT_CYCLES:
        current = deepcopy(trial[family]["specification"])
        current["structured"]["seed"] = preflight[family]["specification"]["structured"]["seed"]
        assert current == preflight[family]["specification"]
        assert trial[family]["weight"] == preflight[family]["weight"]


def test_all_training_and_confirmation_seeds_are_fresh_and_separate():
    reserved = training.original.alignment.fixed.seeds_in([training.batches(), training.confirmation_cases()])
    assert len(reserved) == 3084
    earlier = training.original.alignment.fixed.seeds_in([training.preflight.batch(), training.original.batches()])
    assert reserved.isdisjoint(earlier)
    assert training.PARENT == training.ROOT / "decoder_repair_v14/latest_checkpoint.pth"
    assert training.INITIAL_RUN == training.ROOT / "decoder_repair_v14"
