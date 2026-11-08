from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import single_model_four_draw_batching as batching
import single_model_four_draw_gradient_staging as staging
import single_model_four_draw_protection as protection
import single_model_four_draw_retention as retention
import single_model_point_gradient_storage as gradient_storage
import single_model_replicated_decoder_batch as replicated
import single_model_training_admission as admission
from test_single_model_four_draw_batching import source_batches
from test_single_model_training_admission import sampler


@pytest.fixture(scope="module")
def batch(source_batches):
    return batching.pair_batches(source_batches)[0]


@pytest.mark.parametrize("task_index", [1, 26, 29, 54])
@pytest.mark.parametrize("mode", ["no_queries", "no_observations"])
def test_bad_draw_in_either_original_half_consumes_the_entire_attempt(batch, task_index, mode):
    seed = batch["tasks"][task_index]["specification"]["structured"]["seed"]
    original, calls = sampler({seed: mode})
    saved = deepcopy(batch)
    checker = admission.TrainingBatchAdmission(original, batch)
    assert not checker.eligible and checker.record["task_count"] == 56
    assert checker.record["structured_draws"] == len(original.structured_prior.calls) == 48
    assert checker.record["deferred_task_indices"] == [0, 13, 14, 27, 28, 41, 42, 55]
    assert checker.record["rejected_task_indices"] == [task_index]
    for task in batch["tasks"]:
        with pytest.raises(ValueError, match="Rejected data-admission"):
            checker.sample(step=1, **task["specification"])
    checker.assert_consumed()
    assert not calls and not checker._episodes and batch == saved
    for field in ("teacher_inference_performed", "student_inference_performed", "optimizer_proposal_performed",
            "masks_changed", "draws_replaced", "qualification_changed"):
        assert checker.record[field] is False


def test_valid_admission_consumes_all48_original_draws_once_and_defers_all8_other_tasks(batch):
    retention.validate_task_mixture(batch["tasks"])
    original, calls = sampler()
    checker = admission.TrainingBatchAdmission(original, batch)
    assert checker.eligible and not calls
    for task in batch["tasks"]:
        episode = checker.sample(step=1, **task["specification"])
        if "structured" in task["specification"]:
            assert admission.episode_digest(episode) == checker.record["tasks"][task["index"]]["episode_content_sha256"]
    checker.assert_consumed()
    assert len(original.structured_prior.calls) == 48 and len(calls) == 8


@pytest.mark.parametrize("undefined", [False, True])
def test_actual_capture_update_serialization_and_independent_audit_support_four_draws(batch, tmp_path, monkeypatch, undefined):
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.02)
    undefined_seed = batch["tasks"][42]["specification"]["seed"] if undefined else None
    seen = []
    backward_seeds = []

    def sample(step, **specification):
        seed = specification.get("structured", specification)["seed"]
        seen.append(seed)
        return SimpleNamespace(metadata={"seed": seed, "domain": specification["domain"], "gradient_data": "synthetic"},
            source=torch.zeros(2, 2))

    def backward(current, criterion, episode, teacher, weight):
        assert weight == 1.
        loss = current.weight.sum()
        loss.backward()
        return float(loss.detach())

    def point_losses(current, criterion, episode, **options):
        second = "mae" if episode.metadata["domain"] == "recsys" else "common_nrmse"
        losses = {"rmse": 2 * current.weight.sum(), second: 3 * current.weight.sum()}
        if episode.metadata["seed"] == undefined_seed:
            losses[second] = None
        return torch.zeros(2, 2), losses, losses

    def metric_summary(prediction, episode, terms):
        return {name: None if value is None else {"value": float(value.detach())} for name, value in terms.items()}

    backend = SimpleNamespace(teacher_target=lambda teachers, criterion, episode: (torch.zeros(2, 2), {}),
        cache_episode=lambda episode: {"metadata": episode.metadata}, backward_episode=backward)
    alignment = SimpleNamespace(mixture=SimpleNamespace(backend=backend),
        preflight=SimpleNamespace(flatten=lambda current, gradients: current.weight.grad.flatten()),
        metric_summary=metric_summary)
    for key, value in {"alignment": alignment, "streamed_point_metric_losses": point_losses,
            "compute_context": lambda device: nullcontext(), "TRAIN_OPTIONS": {},
            "torch": SimpleNamespace(manual_seed=backward_seeds.append)}.items():
        monkeypatch.setitem(replicated._capture.__globals__, key, value)
    directory = tmp_path / "batch"
    directory.mkdir()
    gradients = torch.full((168, 2), float("nan"))
    entries, records, before = replicated.capture_batch(tmp_path, directory, batch, model, optimizer, None, None,
        SimpleNamespace(sample=sample), gradients)
    assert len(entries) == (167 if undefined else 168)
    assert len(records) == len(seen) == len(set(seen)) == len(list(directory.glob("task_*.pt"))) == 56
    assert backward_seeds == [task["gradient_seed"] for task in batch["tasks"] for invocation in range(2)]
    assert len(before) == 112 and sum(value is None for value in before.values()) == int(undefined)
    assert torch.equal(gradients[:56], torch.ones(56, 2))
    assert torch.isfinite(gradients[:len(entries)]).all()
    if undefined:
        assert torch.isnan(gradients[len(entries):]).all()
    for entry in entries[56:]:
        assert torch.equal(gradients[entry["index"]], torch.full((2,), 2. if entry["kind"] == "rmse" else 3.))
    protection.validate_layout(entries, batch["tasks"], before)
    transferred, transfer_report = staging.stage_for_update(gradients, entries, batch["tasks"], before, "cpu", chunk_elements=17)
    del gradients
    assert transfer_report["exact_fp32_transfer"]

    def evaluate(current):
        return {name: None if value is None else float(current.weight.detach().sum()) *
            (2. if name.endswith("/rmse") else 3.) for name, value in before.items()}

    weights = [task["weight"] for task in batch["tasks"]]
    record, snapshot = protection.protected_update(model, optimizer, transferred, weights, .01, before, evaluate)
    record.update(step=1, accepted_updates=int(record["accepted"]))
    assert record["accepted"]
    complete, shared = gradient_storage.share(transferred, snapshot, 56)
    payload = {"record": record, "snapshot": shared, "gradients": complete}
    gradient_storage.check_alias(payload)
    path = tmp_path / "raw_update.pt"
    torch.save(payload, path)
    restored = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    gradient_storage.check_alias(restored)
    protection.check_record(record, weights, before, len(entries))
    report = protection.check_endpoint(restored, record, [2], .02, .003, optimizer.state_dict())
    assert report["adam"]["gradient_vectors"] == len(entries)
    assert report["fixed_quarter_step"]["actual_saved_displacements_checked"]


def test_capture_rejects_two_draw_gradient_capacity_before_sampling(batch, tmp_path):
    with pytest.raises(ValueError, match="every possible"):
        replicated.capture_batch(tmp_path, tmp_path, batch, None, None, None, None, None, torch.empty(84, 2))
