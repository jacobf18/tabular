from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
import inspect
import math
from types import SimpleNamespace

import pytest
import torch

import single_model_replicated_decoder_batch as replicated
import train_single_matrixformer_v8_decoder as previous


def fixture():
    return replicated.replicate_batch(previous.batches()[0], 311100901, 312100901)


def test_each_cohort_gets_two_independent_draws_without_changing_prior_or_weight():
    original = previous.batches()[0]
    before = deepcopy(original)
    batch = replicated.replicate_batch(original, 311100901, 312100901)
    assert original == before
    assert len(batch["tasks"]) == 28
    assert [task["index"] for task in batch["tasks"]] == list(range(28))
    assert len({task["name"] for task in batch["tasks"]}) == 28
    cohorts = defaultdict(list)
    for task in batch["tasks"]:
        cohorts[task["cohort"]].append(task)
    for source in original["tasks"]:
        draws = cohorts[source["name"]]
        assert {task["replicate"] for task in draws} == {0, 1}
        assert math.fsum(task["weight"] for task in draws) == source["weight"]
        for task in draws:
            specification = deepcopy(task["specification"])
            specification.get("structured", specification)["seed"] = source["specification"].get("structured", source["specification"])["seed"]
            assert specification == source["specification"]
    assert math.fsum(task["weight"] for task in batch["tasks"]) == 1.
    seeds = previous.original.alignment.fixed.seeds_in
    assert len(seeds(batch)) == 56
    assert seeds(batch).isdisjoint(seeds([previous.batches(), previous.confirmation_cases()]))


def test_replicated_steps_never_reuse_data_or_backward_seeds():
    batches = [replicated.replicate_batch(batch, 321100901, 322100901) for batch in previous.batches()]
    assert len(previous.original.alignment.fixed.seeds_in(batches)) == 64 * 56


@pytest.mark.parametrize("mutation", ["drop", "index", "name", "weight", "negative", "step", "seeds"])
def test_invalid_replication_inputs_are_rejected(mutation):
    batch = previous.batches()[0]
    data_seed, gradient_seed = 311100901, 312100901
    if mutation == "drop":
        batch["tasks"].pop()
    elif mutation == "index":
        batch["tasks"][0]["index"] = 1
    elif mutation == "name":
        batch["tasks"][0]["name"] = batch["tasks"][1]["name"]
    elif mutation == "weight":
        batch["tasks"][0]["weight"] += .1
    elif mutation == "negative":
        batch["tasks"][0]["weight"] = -1.
    elif mutation == "step":
        batch["step"] = 0
    else:
        gradient_seed = data_seed
    with pytest.raises(ValueError):
        replicated.replicate_batch(batch, data_seed, gradient_seed)


def test_capture_changes_only_the_initial_slot_count_and_keeps_original_immutable():
    source = inspect.getsource(replicated.original.capture_batch)
    assert 'entries = [None] * 14' in source
    assert replicated.CAPTURE_SOURCE == source.replace('entries = [None] * 14', 'entries = [None] * len(batch["tasks"])')
    assert replicated._capture is not replicated.original.capture_batch


def test_actual_capture_preserves_all_28_composites_and_56_point_slots(tmp_path, monkeypatch):
    batch = fixture()
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)

    def sample(step, **specification):
        seed = specification.get("structured", specification)["seed"]
        return SimpleNamespace(metadata={"seed": seed, "gradient_data": "synthetic"}, source=torch.zeros(2, 2))

    def backward(model, criterion, episode, teacher, weight):
        loss = model.weight.sum()
        loss.backward()
        return float(loss.detach())

    def point_losses(model, criterion, episode, **options):
        return torch.zeros(2, 2), {}, {"rmse": 2 * model.weight.sum(), "common_nrmse": 3 * model.weight.sum()}

    backend = SimpleNamespace(teacher_target=lambda teachers, criterion, episode: (torch.zeros(2, 2), {}),
                              cache_episode=lambda episode: {"metadata": episode.metadata}, backward_episode=backward)
    alignment = SimpleNamespace(mixture=SimpleNamespace(backend=backend),
                                preflight=SimpleNamespace(flatten=lambda model, gradients: model.weight.grad.flatten()),
                                metric_summary=lambda prediction, episode, terms: {"rmse": {"value": 1.}, "common_nrmse": {"value": 2.}})
    for key, value in {"alignment": alignment, "streamed_point_metric_losses": point_losses,
                       "compute_context": lambda device: nullcontext(), "TRAIN_OPTIONS": {}}.items():
        monkeypatch.setitem(replicated._capture.__globals__, key, value)
    directory = tmp_path / "batch"
    directory.mkdir()
    gradients = torch.full((84, 2), float("nan"))
    entries, records, values = replicated.capture_batch(tmp_path, directory, batch, model, optimizer, None, None,
                                                        SimpleNamespace(sample=sample), gradients)
    assert len(entries) == 84 and len(records) == 28 and len(values) == 56
    assert [entry["index"] for entry in entries] == list(range(84))
    assert all(entry["kind"] == "composite" for entry in entries[:28])
    assert torch.equal(gradients[:28], torch.ones(28, 2))
    assert torch.equal(gradients[28::2], torch.full((28, 2), 2.))
    assert torch.equal(gradients[29::2], torch.full((28, 2), 3.))
    assert len(list(directory.glob("task_*.pt"))) == 28


def test_capture_rejects_the_old_42_gradient_allocation(tmp_path):
    with pytest.raises(ValueError, match="every possible"):
        replicated.capture_batch(tmp_path, tmp_path, fixture(), None, None, None, None, None, torch.empty(42, 2))
