"""Training-only spectral-amplitude mixture with unchanged legacy evaluation inputs."""

from contextlib import contextmanager
from dataclasses import asdict
import math
import random

from general_spectral_geometry import GeometrySpectrumGenerator
from rowcol_fixed_validation import preserved_rng
from single_model_distillation import Episode, MixedSampler
from spectral_support_normalization import apply_shrunk_support_mask
from streamed_large_matrix_training import CurriculumSampler, base, phase_at
import tabimpute.prior.spectrum_training_set_generation as spectrum_module
from train_single_matrixformer import setup as original_setup
import train_single_matrixformer_probability_balance as balanced


continuation = balanced.continuation
SPECIFICATION = {"scaled_probability": .5, "reference_area": 240 * 80, "minimum_gain": 1.,
                 "preserve_spectral_replay": True, "preserve_latent_anchors": True,
                 "validation_regime": "legacy", "application": "clean singular values before noise and normalization"}


def signal_gain(rows, columns):
    if type(rows) is not int or type(columns) is not int or min(rows, columns) < 2:
        raise ValueError("Require integer matrix dimensions of at least two")
    return max(1., math.sqrt(rows * columns / SPECIFICATION["reference_area"]))


def regime_for_seed(seed, regime_seed):
    return "area_scaled" if random.Random(seed + regime_seed).random() < SPECIFICATION["scaled_probability"] else "legacy"


@contextmanager
def spectral_amplitude(enabled):
    original = spectrum_module.DecayingSpectrumGenerator
    records = []

    class AreaSpectrumGenerator(GeometrySpectrumGenerator):
        def sample_spectrum(self, rank):
            if (self.config["num_rows_low"] != self.config["num_rows_high"]
                    or self.config["num_cols_low"] != self.config["num_cols_high"]
                    or self.config.get("spectrum_target_rank") != 10000):
                raise ValueError("Require resolved dimensions and the original full-clean-signal target")
            rows, columns = self.config["num_rows_low"], self.config["num_cols_low"]
            values, parameters = super().sample_spectrum(rank)
            gain = signal_gain(rows, columns)
            records.append({"rows": rows, "columns": columns, "signal_gain": gain})
            return values * gain, parameters

    if enabled:
        if original is not GeometrySpectrumGenerator:
            raise ValueError("Require the original installed geometry generator")
        spectrum_module.DecayingSpectrumGenerator = AreaSpectrumGenerator
    try:
        yield records
    finally:
        spectrum_module.DecayingSpectrumGenerator = original


class AreaCurriculumSampler(CurriculumSampler):
    def sample(self, step, microstep, *, device="cpu", forced_shape=None, component=None,
               seed=None, support_mode="broad_support", transpose=None, regime="legacy"):
        if regime not in ("legacy", "area_scaled"):
            raise ValueError("Unknown spectral-amplitude regime")
        seed = self.data_seed + 16 * step + microstep if seed is None else seed
        phase = phase_at(self.phases, step)
        with preserved_rng(seed):
            replay = random.random() < self.spectral_replay
            chosen = self.original_phase if replay else phase
            dimensions = asdict(chosen)
            dimensions["batch_size"] = 1
            if forced_shape is not None:
                rows, columns = forced_shape
                dimensions.update(rows_low=rows, rows_high=rows, cols_low=columns, cols_high=columns)
                replay = False
            prior = self.prior(base.CurriculumPhase(**dimensions))
            index = random.choices(range(len(prior.children)), weights=prior.weights)[0] if component is None else component
            enabled = regime == "area_scaled" and index < 6 and not replay
            with spectral_amplitude(enabled) as records:
                (source, target, _, _, _), _ = prior.children[index].get_batch()
            if enabled and len(records) != 1:
                raise ValueError("Require one spectral matrix per episode")
            gain = records[0]["signal_gain"] if records else 1.
            raw_shape = list(source.shape[1:])
            if records and raw_shape != [records[0]["rows"], records[0]["columns"]]:
                raise ValueError("Spectral amplitude must use actual pre-transpose dimensions")
            transposed = random.random() < .5 if transpose is None else transpose
            if transposed:
                source, target = source.transpose(1, 2).contiguous(), target.transpose(1, 2).contiguous()
        source, target, mask, stats = apply_shrunk_support_mask(
            source.float(), target.float(), support_mode, seed=seed + 10000000, variance_df=1.)
        if not mask.any() or not target[mask].isfinite().all():
            raise ValueError("Require nonempty finite synthetic targets")
        reason = ("latent_anchor" if index >= 6 else "spectral_replay" if replay else
                  "legacy_draw" if regime == "legacy" else "dimension_scaled" if gain > 1 else "small_context")
        return source.to(device), target.to(device), mask.to(device), {
            **stats, "phase": phase.name, "prior": prior.names[index], "component": index,
            "spectral_replay": replay and index < 6, "transposed": transposed, "seed": seed,
            "rows": source.shape[1], "columns": source.shape[2], "targets": int(mask.sum()),
            "area_prior": {"requested_regime": regime, "applied": gain > 1, "signal_gain": gain,
                           "reference_area": SPECIFICATION["reference_area"], "spectral_shape": raw_shape, "reason": reason}}


class AreaMixedSampler(MixedSampler):
    def __init__(self, config, *, force_area=False, device="cuda"):
        super().__init__(config)
        self.imputation = AreaCurriculumSampler(self.imputation.phases, data_seed=config["data_seed"], spectral_replay=.2)
        self.force_area, self.device = force_area, device

    def sample(self, domain, step, *, seed=None, shape=None, component=None, transpose=None, regime=None):
        if domain != "imputation":
            if regime is not None:
                raise ValueError("Ratings must not receive a spectral-amplitude regime")
            return super().sample(domain, step, seed=seed, shape=shape, component=component, transpose=transpose)
        seed = self.config["data_seed"] + 16 * step if seed is None else seed
        regime = ("area_scaled" if self.force_area else "legacy") if regime is None else regime
        source, target, mask, metadata = self.imputation.sample(step, 0, device=self.device, seed=seed,
            forced_shape=shape, component=component, transpose=transpose, regime=regime)
        return Episode(source, target, mask, {**metadata, "domain": domain, "gradient_data": "synthetic"})


def evaluation_setup(config):
    result = original_setup(config)
    return (*result[:-1], AreaMixedSampler(config))


@contextmanager
def training_prior():
    previous_setup, previous_preflight = continuation.setup, balanced.setup
    previous_episode = continuation.training_episode

    def setup(config):
        result = previous_setup(config)
        return (*result[:-1], AreaMixedSampler(config))

    def preflight_setup(config):
        result = previous_preflight(config)
        return (*result[:-1], AreaMixedSampler(config, force_area=True))

    def episode(sampler, config, domain, step):
        if domain != "imputation":
            return previous_episode(sampler, config, domain, step)
        seed = config["data_seed"] + 16 * step
        return sampler.sample(domain, step, regime=regime_for_seed(seed, config["regime_seed"]))

    continuation.setup, balanced.setup, continuation.training_episode = setup, preflight_setup, episode
    try:
        yield
    finally:
        continuation.setup, balanced.setup, continuation.training_episode = previous_setup, previous_preflight, previous_episode
