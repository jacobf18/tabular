from copy import deepcopy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import single_model_area_prior as area
import single_model_column_objective as column
from general_spectral_geometry import GeometrySpectrumGenerator, install_geometry_generator
from rowcol_fixed_validation import preserved_rng
from streamed_large_matrix_training import CurriculumSampler, parse_curriculum
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from test_streamed_training import example


def config():
    return {"data_seed": 44100501, "regime_seed": 48100501,
            "phases": [{"name": "area_mixture", "end_step": 400, "rows_low": 256, "rows_high": 2000,
                        "cols_low": 128, "cols_high": 1000, "batch_size": 1}],
            "rating_training_shapes": [[1000, 1000], [2000, 1000]]}


def assert_episode_equal(first, second):
    for before, after in zip(first[:3], second[:3]):
        torch.testing.assert_close(before, after, rtol=0, atol=0, equal_nan=True)
    assert first[3] == {key: value for key, value in second[3].items() if key != "area_prior"}


@pytest.mark.parametrize("shape,expected", [((80, 35), 1.), ((320, 35), 1.), ((240, 80), 1.),
                                          ((2000, 1000), (2_000_000 / 19200)**.5),
                                          ((1000, 2000), (2_000_000 / 19200)**.5)])
def test_area_gain_preserves_small_contexts_and_both_large_orientations(shape, expected):
    assert area.signal_gain(*shape) == expected


@pytest.mark.parametrize("shape", [(1, 10), (100., 20), (True, 50), (32, -1)])
def test_invalid_dimensions_are_rejected(shape):
    with pytest.raises(ValueError):
        area.signal_gain(*shape)


def test_spectral_scaling_changes_only_clean_amplitude_with_matched_noise_and_rng():
    install_geometry_generator()
    specification = {"num_rows_low": 200, "num_rows_high": 200, "num_cols_low": 128, "num_cols_high": 128,
                     "spectrum_rank_low": 8, "spectrum_rank_high": 8, "spectrum_target_rank": 10000,
                     "spectrum_basis_kind": "mixed", "spectrum_noise_std_low": .001, "spectrum_noise_std_high": .001}
    with preserved_rng(717):
        before = GeometrySpectrumGenerator(specification).generate_matrix_bundle()
        before_rng = np.random.get_state()
    with preserved_rng(717), area.spectral_amplitude(True) as records:
        after = area.spectrum_module.DecayingSpectrumGenerator(specification).generate_matrix_bundle()
        after_rng = np.random.get_state()
    gain = area.signal_gain(200, 128)
    assert len(records) == 1 and records[0]["signal_gain"] == gain
    assert before["rank"] == after["rank"] == before["target_rank"] == after["target_rank"] == 8
    assert before["noise_std"] == after["noise_std"]
    np.testing.assert_array_equal(before_rng[1], after_rng[1])
    assert before_rng[2:] == after_rng[2:]
    torch.testing.assert_close(after["clean_signal"], gain * before["clean_signal"], rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(after["target_matrix"], gain * before["target_matrix"], rtol=2e-5, atol=1e-7)
    torch.testing.assert_close(after["full_matrix"] - after["clean_signal"],
                               before["full_matrix"] - before["clean_signal"], rtol=1e-3, atol=2e-7)
    assert area.spectrum_module.DecayingSpectrumGenerator is GeometrySpectrumGenerator


@pytest.mark.parametrize("component,shape,transpose", [(0, (240, 100), False), (4, (240, 100), True),
                                                      (1, (240, 100), False), (2, (240, 100), True),
                                                      (3, (240, 100), False), (5, (240, 100), True),
                                                      (4, (2000, 1000), False), (2, (2000, 1000), True),
                                                      (6, (240, 80), False), (7, (240, 80), True)])
def test_legacy_sampler_is_tensor_identical_to_registered_original(component, shape, transpose):
    phases = parse_curriculum(config()["phases"])
    original = CurriculumSampler(phases, data_seed=config()["data_seed"])
    revised = area.AreaCurriculumSampler(phases, data_seed=config()["data_seed"])
    arguments = dict(seed=1318, forced_shape=shape, component=component, transpose=transpose)
    before = original.sample(1, 0, **arguments)
    after = revised.sample(1, 0, **arguments)
    assert_episode_equal(before, after)
    assert after[3]["area_prior"]["requested_regime"] == "legacy" and not after[3]["area_prior"]["applied"]


@pytest.mark.parametrize("component,replay,shape", [(0, 1., None), (4, 1., None), (6, 0., (240, 100)),
                                                  (7, 0., (240, 100)), (0, 0., (80, 35))])
def test_area_draw_preserves_replay_anchors_and_small_spectra(component, replay, shape):
    phases = parse_curriculum(config()["phases"])
    original = CurriculumSampler(phases, data_seed=91, spectral_replay=replay)
    revised = area.AreaCurriculumSampler(phases, data_seed=91, spectral_replay=replay)
    arguments = dict(seed=2445, forced_shape=shape, component=component, transpose=False)
    before = original.sample(1, 0, **arguments)
    after = revised.sample(1, 0, regime="area_scaled", **arguments)
    assert_episode_equal(before, after)
    assert not after[3]["area_prior"]["applied"]


@pytest.mark.parametrize("transpose", [False, True])
def test_training_amplitude_is_applied_before_noise_mask_normalization_and_transpose(transpose):
    sampler = area.AreaCurriculumSampler(parse_curriculum(config()["phases"]), data_seed=83)
    arguments = dict(seed=615, forced_shape=(240, 100), component=4, transpose=transpose, support_mode="mixed")
    legacy = sampler.sample(1, 0, **arguments)
    scaled = sampler.sample(1, 0, regime="area_scaled", **arguments)
    info = scaled[3]["area_prior"]
    assert info["applied"] and info["reason"] == "dimension_scaled"
    assert info["spectral_shape"] == [240, 100] and info["signal_gain"] == area.signal_gain(240, 100)
    assert list(scaled[0].shape[1:]) == ([100, 240] if transpose else [240, 100])
    assert torch.equal(legacy[2], scaled[2])
    assert not torch.equal(legacy[1], scaled[1])
    assert not torch.allclose(scaled[1], legacy[1] * info["signal_gain"])
    assert torch.isfinite(scaled[1][scaled[2]]).all()


def test_mixture_coin_is_reproducible_independent_and_not_used_by_default_evaluation():
    random.seed(516)
    before = random.getstate()
    modes = [area.regime_for_seed(config()["data_seed"] + 16 * step, config()["regime_seed"]) for step in range(1, 401)]
    assert random.getstate() == before and .35 < modes.count("area_scaled") / 400 < .65
    assert modes == [area.regime_for_seed(config()["data_seed"] + 16 * step, config()["regime_seed"]) for step in range(1, 401)]
    sampler = area.AreaMixedSampler(config(), device="cpu")
    episode = sampler.sample("imputation", 1, seed=1171, shape=(240, 100), component=0)
    assert episode.metadata["area_prior"]["requested_regime"] == "legacy"
    forced = area.AreaMixedSampler(config(), device="cpu", force_area=True)
    episode = forced.sample("imputation", 1, seed=1171, shape=(240, 100), component=0)
    assert episode.metadata["area_prior"]["applied"]


def test_rating_sampler_is_forwarded_unchanged_and_rejects_scaling(monkeypatch):
    calls = []
    monkeypatch.setattr(area.MixedSampler, "sample", lambda *args, **kwargs: calls.append((args[1:], kwargs)) or "original rating")
    sampler = area.AreaMixedSampler(config(), device="cpu", force_area=True)
    assert sampler.sample("recsys", 4, seed=55, shape=(2000, 1000)) == "original rating"
    assert calls == [(("recsys", 4), {"seed": 55, "shape": (2000, 1000), "component": None, "transpose": None})]
    with pytest.raises(ValueError):
        sampler.sample("recsys", 4, regime="area_scaled")


def test_sampler_callbacks_restore_after_error_and_keep_training_and_evaluation_separate(monkeypatch):
    runner = area.continuation
    monkeypatch.setattr(runner, "setup", lambda config: (1, 2, 3, 4, 5))
    monkeypatch.setattr(area.balanced, "setup", lambda config: (1, 2, 3, 4, 5))
    original = (runner.setup, area.balanced.setup, runner.training_episode)
    with pytest.raises(RuntimeError, match="stop"), area.training_prior():
        training_sampler = runner.setup(config())[-1]
        preflight_sampler = area.balanced.setup(config())[-1]
        assert not training_sampler.force_area and preflight_sampler.force_area
        calls = []
        sampler = SimpleNamespace(sample=lambda *args, **kwargs: calls.append((args, kwargs)))
        runner.training_episode(sampler, config(), "imputation", 3)
        assert calls[-1][1]["regime"] == area.regime_for_seed(config()["data_seed"] + 48, config()["regime_seed"])
        runner.training_episode(sampler, config(), "recsys", 4)
        assert calls[-1][1] == {"shape": (2000, 1000)}
        raise RuntimeError("stop")
    assert (runner.setup, area.balanced.setup, runner.training_episode) == original


def test_generator_restores_after_error_and_rejects_unresolved_shape_or_changed_target():
    install_geometry_generator()
    specification = {"num_rows_low": 100, "num_rows_high": 101, "num_cols_low": 100, "num_cols_high": 100,
                     "spectrum_target_rank": 10000}
    with pytest.raises(ValueError), area.spectral_amplitude(True):
        area.spectrum_module.DecayingSpectrumGenerator(specification).sample_spectrum(4)
    assert area.spectrum_module.DecayingSpectrumGenerator is GeometrySpectrumGenerator
    specification.update(num_rows_high=100, spectrum_target_rank=2)
    with pytest.raises(ValueError), area.spectral_amplitude(True):
        area.spectrum_module.DecayingSpectrumGenerator(specification).sample_spectrum(4)
    assert area.spectrum_module.DecayingSpectrumGenerator is GeometrySpectrumGenerator


def test_scaled_synthetic_episode_supports_real_streamed_backward_with_frozen_teacher(monkeypatch):
    student, _, _, _ = example(shape=(1, 9, 7))
    initial = deepcopy(student.state_dict())
    sampler = area.AreaMixedSampler(config(), device="cpu")
    episode = sampler.sample("imputation", 1, seed=953, shape=(160, 128), component=0, transpose=False, regime="area_scaled")
    criterion = FullSupportBarDistribution(torch.linspace(-5, 5, 21))
    teacher = torch.zeros_like(episode.target, requires_grad=True)
    monkeypatch.setattr(column, "TRAIN_OPTIONS", {"attention_batch_size": 16, "cell_batch_size": 1024,
                                                 "encoder_row_batch_size": 32, "decoder_cell_batch_size": 256})
    optimizer = torch.optim.SGD(student.parameters(), lr=1e-4)
    loss = column.backward_episode(student, criterion, episode, teacher, .5)
    assert np.isfinite(loss) and episode.metadata["area_prior"]["applied"]
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in student.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in student.parameters())
    optimizer.step()
    assert teacher.grad is None and any(not torch.equal(value, initial[name]) for name, value in student.state_dict().items())
    assert all(torch.isfinite(value).all() for value in student.state_dict().values())
