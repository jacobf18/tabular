from contextlib import nullcontext
from copy import deepcopy
import inspect
from types import SimpleNamespace

import pytest
import torch

import single_model_compact_training_cache as cache
import single_model_replicated_decoder_batch as replicated
from single_model_registered_adaptation import replace_once
import train_single_matrixformer_v8_decoder as previous


def payload():
    return {"episode": {"source": torch.tensor([[1., float("nan")], [-0., 2.]]), "scaling": None},
            "teacher": torch.ones(2, 2), "routing": {"teacher": "fixed"},
            "task": {"specification": {"seed": 123}}}


def test_only_storage_and_read_expressions_change():
    assert cache.CAPTURE_SOURCE == replace_once(replicated.CAPTURE_SOURCE,
        '        digest = save_tensor({"episode":', '        digest = persist_task({"episode":')
    assert cache.EVALUATE_SOURCE == replace_once(inspect.getsource(replicated.original.evaluate_training),
        'cached = torch.load(record["artifact"], map_location="cpu", weights_only=False)',
        'cached = load_current(record)')


def test_capsule_retains_no_matrix_grid():
    capsule = cache.capsule_for(payload())
    assert set(capsule) == {"format", "task", "content_sha256", "scaling"}
    assert capsule["scaling"] is None
    cache.validate_capsule(capsule)
    capsule["source"] = torch.ones(2, 2)
    with pytest.raises(ValueError, match="compact-only"):
        cache.validate_capsule(capsule)


def test_current_payload_exact_bytes_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(replicated.original, "save_tensor", lambda value, path: (torch.save(value, path), cache.sha256(path))[1])
    current = payload()
    path = tmp_path / "task.pt"
    with cache.runtime() as memory:
        digest = cache.persist_task(current, path)
        record = {"artifact": str(path), "sha256": digest}
        assert cache.load_current(record) is current
        with pytest.raises(FileExistsError):
            cache.persist_task(current, path)
        with pytest.raises(ValueError, match="nest"):
            with cache.runtime():
                pass
        current["episode"]["source"][1, 0] = 0.
        with pytest.raises(ValueError, match="unchanged"):
            cache.load_current(record)
    assert memory == {} and cache.ACTIVE.get() is None
    with pytest.raises(ValueError, match="current batch"):
        cache.load_current(record)


def test_runtime_restores_after_error():
    with pytest.raises(RuntimeError):
        with cache.runtime() as memory:
            memory["temporary"] = payload()
            raise RuntimeError("stop")
    assert not memory and cache.ACTIVE.get() is None


def test_independent_regeneration_checks_full_teacher_and_metadata():
    original = payload()
    capsule = cache.capsule_for(original)
    sampler = SimpleNamespace(sample=lambda **specification: original["episode"])
    backend = SimpleNamespace(teacher_target=lambda *args: (original["teacher"], original["routing"]),
                              cache_episode=lambda episode: episode)
    assert cache.content_digest(cache.regenerate(capsule, sampler, None, None, backend)) == capsule["content_sha256"]
    original["teacher"][0, 0] = 2.
    with pytest.raises(ValueError, match="every typed byte"):
        cache.regenerate(capsule, sampler, None, None, backend)


def test_rating_capsule_keeps_only_four_cloned_vectors():
    original = payload()
    original["episode"]["scaling"] = {name: torch.ones(2) for name in cache.compact.NAMES}
    capsule = cache.capsule_for(original)
    cache.validate_capsule(capsule)
    original["episode"]["scaling"]["means"][0] = 99.
    assert capsule["scaling"]["means"][0] == 1.


def test_capture_keeps_all84_gradients_and_original_backward_seeds(tmp_path, monkeypatch):
    batch = replicated.replicate_batch(previous.batches()[0], 311100901, 312100901)
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    seeds = []

    def sample(step, **specification):
        seed = specification.get("structured", specification)["seed"]
        return SimpleNamespace(metadata={"seed": seed, "gradient_data": "synthetic"}, source=torch.zeros(2, 2))

    def backward(model, criterion, episode, teacher, weight):
        seeds.append(torch.initial_seed())
        loss = model.weight.sum()
        loss.backward()
        return float(loss.detach())

    def point_losses(model, criterion, episode, **options):
        seeds.append(torch.initial_seed())
        return torch.zeros(2, 2), {}, {"rmse": 2 * model.weight.sum(), "common_nrmse": 3 * model.weight.sum()}

    backend = SimpleNamespace(teacher_target=lambda *args: (torch.zeros(2, 2), {}),
        cache_episode=lambda episode: {"metadata": deepcopy(episode.metadata), "scaling": None}, backward_episode=backward)
    alignment = SimpleNamespace(mixture=SimpleNamespace(backend=backend),
        preflight=SimpleNamespace(flatten=lambda model, gradients: model.weight.grad.flatten()),
        metric_summary=lambda *args: {"rmse": {"value": 1.}, "common_nrmse": {"value": 2.}})
    for key, value in {"alignment": alignment, "streamed_point_metric_losses": point_losses,
                      "compute_context": lambda device: nullcontext(), "TRAIN_OPTIONS": {}}.items():
        monkeypatch.setitem(cache._capture.__globals__, key, value)
    monkeypatch.setattr(replicated.original, "save_tensor", lambda value, path: (torch.save(value, path), cache.sha256(path))[1])
    directory = tmp_path / "batch"
    directory.mkdir()
    gradients = torch.full((84, 2), float("nan"))
    with cache.runtime() as memory:
        memory["old_batch"] = payload()
        entries, records, values = cache.capture_batch(tmp_path, directory, batch, model, optimizer, None, None,
            SimpleNamespace(sample=sample), gradients)
        assert len(memory) == 28 and "old_batch" not in memory
        assert len(entries) == 84 and len(records) == 28 and len(values) == 56
        assert torch.equal(gradients[:28], torch.ones(28, 2))
        assert torch.equal(gradients[28::2], torch.full((28, 2), 2.))
        assert torch.equal(gradients[29::2], torch.full((28, 2), 3.))
        assert seeds == [task["gradient_seed"] for task in batch["tasks"] for repeat in range(2)]
        for record in records:
            capsule = torch.load(record["artifact"], weights_only=True)
            assert capsule["content_sha256"] == cache.content_digest(cache.load_current(record))
    assert not memory
