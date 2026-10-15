from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
import torch

from dwip_teacher_mean_training import FrozenMeanTeachers, objective_terms
from streamed_large_matrix_training import (
    DEFAULT_PHASES, CurriculumSampler, atomic_torch_save, backward_microstep,
    learning_rate, parse_curriculum, per_cell_objective, phase_at, restore_rng,
    rng_state, selection_score, teacher_mean_grid,
)
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.streamed_training import streamed_training_loss
from test_streamed_training import assert_gradients_match, example


OPTIONS = dict(attention_batch_size=2, cell_batch_size=13, encoder_row_batch_size=3,
               decoder_cell_batch_size=7)


def test_streamed_full_objective_matches_dense_gradients_and_teacher_alignment():
    reference, source, target, mask = example()
    actual = deepcopy(reference)
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    auxiliary = torch.randn_like(target, requires_grad=True)
    expected = objective_terms(reference, criterion, reference(source, decode_mask=mask), target[mask], auxiliary[mask])["loss"]
    observed = streamed_training_loss(
        actual, source, target, mask, lambda logits, truth, teacher: per_cell_objective(criterion, logits, truth, teacher),
        auxiliary_targets=auxiliary, **OPTIONS,
    )
    torch.testing.assert_close(observed, expected)
    expected.backward()
    observed.backward()
    assert_gradients_match(reference, actual)
    assert auxiliary.grad is None


@pytest.mark.parametrize("invalid", ("shape", "nonfinite", "decoder_chunk"))
def test_invalid_auxiliary_inputs_fail(invalid):
    model, source, target, mask = example()
    auxiliary = target.clone()
    options = dict(OPTIONS)
    if invalid == "shape":
        auxiliary = auxiliary[..., :-1]
    elif invalid == "nonfinite":
        auxiliary[mask] = torch.nan
    else:
        options["decoder_cell_batch_size"] = 0
    with pytest.raises(ValueError):
        streamed_training_loss(model, source, target, mask, lambda logits, truth, teacher: truth,
                               auxiliary_targets=auxiliary, **options)


def test_streamed_teachers_match_dense_and_remain_external():
    model, source, target, mask = example()
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    teachers = FrozenMeanTeachers([deepcopy(model), deepcopy(model)], criterion)
    streamed = teacher_mean_grid(teachers, source, mask, {key: value for key, value in OPTIONS.items() if key != "decoder_cell_batch_size"})
    dense = teachers.target(source, mask)
    torch.testing.assert_close(streamed[mask], dense, rtol=1e-5, atol=1e-6)
    assert not streamed.requires_grad
    before = tuple(model.state_dict())
    backward_microstep(model, teachers, criterion, source, target, mask, options=OPTIONS)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(parameter.grad is None for teacher in teachers.models for parameter in teacher.parameters())
    assert before == tuple(model.state_dict())


def test_accumulation_matches_mean_of_different_matrix_losses():
    reference, source, target, mask = example(shape=(1, 9, 7))
    actual = deepcopy(reference)
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    teachers = FrozenMeanTeachers([deepcopy(reference), deepcopy(reference)], criterion)
    sources = [(source, target, mask), (source[:, :5, :3], target[:, :5, :3], mask[:, :5, :3])]
    expected = []
    for values, truth, selected in sources:
        teacher = teachers.target(values, selected)
        expected.append(objective_terms(reference, criterion, reference(values, decode_mask=selected), truth[selected], teacher)["loss"])
        backward_microstep(actual, teachers, criterion, values, truth, selected, options=OPTIONS)
    (sum(expected) / 2).backward()
    assert_gradients_match(reference, actual)


def test_curriculum_boundaries_and_schedule():
    phases = parse_curriculum(DEFAULT_PHASES)
    assert phase_at(phases, 500).name == "bridge"
    assert phase_at(phases, 501).name == "large"
    assert phase_at(phases, 5000).rows_high == 2000
    assert learning_rate(200, 5000) == 5e-6
    assert learning_rate(5000, 5000) == pytest.approx(2.5e-7)
    assert learning_rate(1, 5000) < learning_rate(200, 5000)
    with pytest.raises(ValueError):
        phase_at(phases, 0)
    with pytest.raises(ValueError):
        phase_at(phases, 5001)


@pytest.mark.parametrize("change", ({"end_step": 0}, {"batch_size": 2}, {"rows_low": 513}, {"cols_high": 1}))
def test_invalid_curricula(change):
    with pytest.raises(ValueError):
        parse_curriculum([{**DEFAULT_PHASES[0], **change}])


def test_sampler_is_seeded_and_preserves_small_latent_overrides():
    phases = parse_curriculum(DEFAULT_PHASES)
    sampler = CurriculumSampler(phases)
    first = sampler.sample(2, 0, forced_shape=(24, 16), component=0, transpose=False)
    sampler.sample(3, 0, forced_shape=(20, 15), component=1)
    second = sampler.sample(2, 0, forced_shape=(24, 16), component=0, transpose=False)
    for expected, actual in zip(first[:3], second[:3]):
        torch.testing.assert_close(actual, expected, equal_nan=True)
    assert first[3] == second[3]
    small = sampler.sample(4999, 0, component=6, transpose=False)
    assert 10 <= small[0].shape[1] <= 50 and 5 <= small[0].shape[2] <= 50
    assert torch.isnan(small[0][small[2]]).all()
    assert torch.isfinite(small[1][small[2]]).all()
    next_sample = sampler.sample(2, 1, forced_shape=(24, 16), component=0, transpose=False)
    assert first[3]["seed"] != next_sample[3]["seed"]


def test_replay_uses_old_spectral_shape_bounds():
    sampler = CurriculumSampler(parse_curriculum(DEFAULT_PHASES), spectral_replay=1.)
    source, _, _, stats = sampler.sample(4000, 0, component=0, transpose=False)
    assert 80 <= source.shape[1] <= 320 and 35 <= source.shape[2] <= 100
    assert stats["spectral_replay"]


def test_selection_enforces_retention_and_baseline():
    baseline = {"retention": 1., "large": 2.}
    assert selection_score(baseline, baseline) == (1., True)
    score, eligible = selection_score({"retention": 1.02, "large": 1.8}, baseline)
    assert score < 1 and eligible
    assert not selection_score({"retention": 1.04, "large": 1.}, baseline)[1]


def test_atomic_save_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "latest.pth"
    atomic_torch_save({"step": 1}, path)
    def failure(payload, stream):
        stream.write(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(torch, "save", failure)
    with pytest.raises(OSError):
        atomic_torch_save({"step": 2}, path)
    assert torch.load(path, weights_only=True)["step"] == 1
    assert not path.with_suffix(".pth.tmp").exists()


def test_adam_and_rng_resume_reproduce_next_streamed_update(tmp_path):
    model, source, target, mask = example(shape=(1, 9, 7), dropout=.1)
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    teachers = FrozenMeanTeachers([deepcopy(model), deepcopy(model)], criterion)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    def update(current, current_optimizer):
        current_optimizer.zero_grad(set_to_none=True)
        loss = backward_microstep(current, teachers, criterion, source, target, mask, accumulation=1, options=OPTIONS)
        current_optimizer.step()
        return loss
    update(model, optimizer)
    path = tmp_path / "latest.pth"
    atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state()}, path)
    expected_loss = update(model, optimizer)
    resumed = deepcopy(model)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
    state = torch.load(path, weights_only=True)
    resumed.load_state_dict(state["model"])
    resumed_optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    assert update(resumed, resumed_optimizer) == expected_loss
    for expected, actual in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_trainer_interrupt_resume_matches_uninterrupted_updates(tmp_path, monkeypatch):
    import train_streamed_large_matrices as trainer

    model, source, target, mask = example(shape=(1, 9, 7), dropout=.1)
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    config = {
        "phases": [{**DEFAULT_PHASES[0], "end_step": 4}], "parent_epoch": 98000,
        "peak_lr": 1e-4, "warmup_steps": 1, "min_lr_ratio": .05, "grad_clip_norm": .5,
        "accumulation": 2, "stream_options": OPTIONS, "validation_seed": 912,
        "validation_every": 2, "checkpoint_every": 2, "checkpoint_seconds": 600,
        "input_sha256": {},
    }
    phases = parse_curriculum(config["phases"])
    def setup(settings):
        torch.manual_seed(11)
        student = deepcopy(model)
        teachers = FrozenMeanTeachers([deepcopy(model), deepcopy(model)], criterion)
        sampler = SimpleNamespace(phases=phases, sample=lambda step, microstep, device:
                                  (source, target, mask, {"seed": 16 * step + microstep}))
        return student, torch.optim.AdamW(student.parameters(), lr=1e-4), criterion, teachers, sampler
    monkeypatch.setattr(trainer, "setup", setup)
    monkeypatch.setattr(trainer, "validate", lambda *args, **kwargs: {"retention": 1., "large": 1.})
    monkeypatch.setattr(trainer, "storage_guard", lambda *args, **kwargs: None)
    for function in ("reset_peak_memory_stats", "synchronize"):
        monkeypatch.setattr(torch.cuda, function, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    outputs = [tmp_path / name for name in ("complete", "resumed")]
    for output in outputs:
        output.mkdir()
        (output / "config.json").write_text(json.dumps(config))
        (output / "preflight.json").write_text(json.dumps({"passed": True, "config_sha256": trainer.sha256(output / "config.json")}))
    trainer.train(config, outputs[0])
    trainer.train(config, outputs[1], max_updates=2)
    assert json.loads((outputs[1] / "status.json").read_text())["state"] == "paused"
    trainer.train(config, outputs[1], resume=True)
    complete, resumed = [torch.load(output / "latest_checkpoint.pth", weights_only=True) for output in outputs]
    assert complete["step"] == resumed["step"] == 4
    for name in complete["model"]:
        torch.testing.assert_close(complete["model"][name], resumed["model"][name], rtol=0, atol=0)
    assert complete["best_step"] == resumed["best_step"] == 0
    with pytest.raises(ValueError, match="resume"):
        trainer.train(config, outputs[1])
