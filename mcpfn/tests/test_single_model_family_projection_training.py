from copy import deepcopy

import numpy as np
import pytest
import torch

import train_single_matrixformer_family_projection as training


def test_training_schedule_has_fresh_matched_full_family_batches():
    batches = training.training_batches()
    assert [batch["step"] for batch in batches] == list(range(1, 101))
    assert len(training.fixed.seeds_in(batches)) == 2800
    for batch in batches:
        tasks = batch["tasks"]
        assert [task["name"] for task in tasks] == ["legacy", *training.mixture.FAMILIES, "recsys"]
        assert [task["index"] for task in tasks] == list(range(14))
        assert [task["weight"] for task in tasks] == [.25, *([.25 / 12] * 12), .5]
        assert sum(task["weight"] for task in tasks) == pytest.approx(1.)
    for index, family in enumerate(training.mixture.FAMILIES, start=1):
        cases = [batch["tasks"][index]["specification"]["structured"] for batch in batches]
        assert {case["family"] for case in cases} == {family}
        assert {case["component"] for case in cases} == set(training.mixture.COMPONENTS)
        assert {case["regime"] for case in cases} == {"legacy", "area_scaled"}
        assert {case["requested_missingness"] for case in cases} == {.2, .6}
        shapes = training.mixture.COMPACT_SHAPES if family in training.mixture.NEURAL_FAMILIES else training.mixture.SPECTRAL_SHAPES
        assert {tuple(case["shape"]) for case in cases} == set(shapes)
    legacy = [batch["tasks"][0]["specification"] for batch in batches]
    assert {case["component"] for case in legacy} == set(range(8))
    assert {tuple(case["shape"]) for case in legacy} == {(256, 128), (2000, 1000)}
    assert {case["regime"] for case in legacy} == {"legacy", "area_scaled"}
    assert {tuple(batch["tasks"][-1]["specification"]["shape"]) for batch in batches} == {(1000, 1000), (2000, 1000)}


def test_confirmation_seeds_are_disjoint_and_reserved_before_training():
    cases = {name: training.confirmation_cases(name) for name in training.ARMS}
    assert all(len(bank) == 284 for bank in cases.values())
    assert len(training.fixed.seeds_in([training.training_batches(), *cases.values()])) == 3368


@pytest.mark.parametrize("step,passed,score,expected", [
    (0, True, 1., 25), (50, False, 1., 25), (50, True, 1.2, 25),
    (50, True, 1.3, 25), (50, True, 1.1, 50),
])
def test_only_strict_guard_retained_improvements_replace_selection(step, passed, score, expected):
    best = {"step": 25, "score": 1.2}
    selected = training.selected_boundary(best, step, {"passed": passed, "score": score})
    assert selected["step"] == expected
    assert best == {"step": 25, "score": 1.2}


@pytest.mark.parametrize("projected", [False, True])
def test_actual_cpu_adamw_update_and_immutable_raw_snapshot(projected):
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=.02)
    gradients = torch.tensor([[1., 0.], [-10., 1.]])
    record, snapshot = training.update_model(model, optimizer, gradients, [.95, .05], projected, .01, raw=True)
    assert snapshot["optimizer_before"]["param_groups"][0]["lr"] == .01
    assert snapshot["optimizer_before"]["state"] == {}
    assert all(parameter.grad is None for parameter in model.parameters())
    assert torch.equal(snapshot["after"], training.preflight.flatten(model))
    assert record["adverse_tasks_before"] == 1
    if projected:
        assert record["adverse_tasks_after"] == 0
        assert np.asarray(record["actual_products_after"]).min() > 0
        expected = training.projection.projected_parameters(snapshot["tentative"], gradients, record["projection"]["raw_gradient_coefficients"])
        assert torch.equal(snapshot["after"], expected)
    else:
        assert record["projection"] is None
        assert record["adverse_tasks_after"] == 1
        assert torch.equal(snapshot["after"], snapshot["tentative"])
    saved = deepcopy(snapshot)
    gradients.fill_(.1)
    training.update_model(model, optimizer, gradients, [.95, .05], projected, .001)
    assert torch.equal(snapshot["task_gradients"], saved["task_gradients"])
    for name in ("before", "tentative", "after", "combined_gradient", "applied_gradient"):
        assert torch.equal(snapshot[name], saved[name])
    for state, original in zip(snapshot["optimizer_after"]["state"].values(), saved["optimizer_after"]["state"].values()):
        for key in original:
            assert torch.equal(state[key], original[key])


def test_projection_failure_is_not_silently_used(monkeypatch):
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)

    def fail(*args, **kwargs):
        raise ValueError("projection failed")

    monkeypatch.setattr(training.projection, "solve_projection", fail)
    with pytest.raises(ValueError, match="projection failed"):
        training.update_model(model, optimizer, torch.ones((2, 2)), [.5, .5], True, .01)


def test_checkpoint_does_not_publish_premature_best_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "storage_guard", lambda path: None)
    training.write_json(tmp_path / "config.json", {"arm": "projected"})
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    training.checkpoint(model, optimizer, tmp_path, 25, {"step": 25, "score": 1.1}, {"bank_sha256": "fixture"}, selected=True)
    state = torch.load(tmp_path / "selected_checkpoint.pth", weights_only=True)
    assert state["gradient_updates_performed"] == state["step"] == 25
    assert not state["independent_confirmation_passed"] and not state["preflight_only"]
    assert not (tmp_path / "best_validation_checkpoint.pth").exists()
    assert "optimizer" not in state


def test_registered_destination_is_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        training.register(tmp_path)
