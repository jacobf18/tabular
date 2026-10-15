"""Additive structured imputation episodes; existing rating and legacy samplers stay untouched."""

from copy import deepcopy

import torch

import diagnose_single_model_missingness_coverage as coverage
from general_spectral_geometry import GeometrySpectrumGenerator
from rowcol_fixed_validation import preserved_rng
from single_model_area_prior import signal_gain
from single_model_distillation import Episode
from tabimpute.model.encoders import normalize_data
from tabimpute.prior.training_set_generation import LatentFactorPrior


COMPONENTS = ("low_rank_mcar", "slow_tail_mcar", "steep_tail_mcar", "anchor_latent_mcar")
NEURAL_FAMILIES = ("mar_neural", "mar_block_neural")
COMPACT_SHAPES = ((24, 12), (48, 24))
SPECTRAL_SHAPES = ((240, 80), (1024, 256))
SPECIFICATION = {
    "families": list(coverage.FAMILIES), "components": list(COMPONENTS),
    "neural_shapes": [list(shape) for shape in COMPACT_SHAPES],
    "nonneural_shapes": [list(shape) for shape in SPECTRAL_SHAPES],
    "regimes": ["legacy", "area_scaled"], "requested_missingness": [.2, .6],
    "normalization": "observed-only raw-source per-column mean, DF1 variance shrinkage, no extra support augmentation",
    "targets": "full clean spectral signal with original noise retained only in source; original latent targets",
    "neural_configuration": "unchanged canonical dictionaries and their internal seed42",
    "latent_shapes": "preserve original intrinsic overrides; never relabel small anchors as large contexts",
    "application": "optional additional imputation episode; this module never replaces or modifies rating/legacy samplers",
    "selection": "no model, loss, training weight, initialization or checkpoint selection is chosen by this module",
}


class ScaledCoverageGenerator(GeometrySpectrumGenerator):
    def __init__(self, config, gain):
        super().__init__(config)
        self.gain = gain

    def sample_spectrum(self, rank):
        values, parameters = super().sample_spectrum(rank)
        return values * self.gain, parameters


class CompleteMissingnessSampler:
    def __init__(self):
        self.args = coverage.control_recipe()
        self.configurations = {}

    def configuration(self, component, shape):
        if component not in COMPONENTS:
            raise ValueError("Unknown original generator component")
        shape = tuple(shape)
        if shape not in COMPACT_SHAPES + SPECTRAL_SHAPES:
            raise ValueError("Require an explicit supported shape; never silently resize a matrix")
        if shape not in self.configurations:
            rows, columns = shape
            phase = coverage.base.CurriculumPhase(name="complete_missingness", end_step=1,
                rows_low=rows, rows_high=rows, cols_low=columns, cols_high=columns, batch_size=1)
            prior = coverage.build_training_prior(self.args, phase=phase, device="cpu")
            self.configurations[shape] = {name: deepcopy(child.config) for name, child in zip(prior.names, prior.children)
                                          if name in COMPONENTS}
            if set(self.configurations[shape]) != set(COMPONENTS):
                raise ValueError("The original four generator components changed")
        return deepcopy(self.configurations[shape][component])

    def draw(self, *, family, component, shape, regime, requested_missingness, seed):
        if family not in coverage.FAMILIES or regime not in SPECIFICATION["regimes"]:
            raise ValueError("Require a known mechanism and amplitude regime")
        allowed = COMPACT_SHAPES if family in NEURAL_FAMILIES else SPECTRAL_SHAPES
        if tuple(shape) not in allowed:
            raise ValueError("Neural masks require explicitly compact contexts; other masks require the declared spectral shapes")
        if type(seed) is not int or not 0 <= seed < 2**32 - 1 or requested_missingness not in (.2, .6):
            raise ValueError("Invalid registered draw seed or requested missingness")
        config = self.configuration(component, shape)
        spectral = component != "anchor_latent_mcar"
        gain = signal_gain(*shape) if spectral and regime == "area_scaled" else 1.
        with preserved_rng(seed):
            if spectral:
                bundle = ScaledCoverageGenerator(config, gain).generate_matrix_bundle()
                source, target = bundle["full_matrix"], bundle["target_matrix"]
            else:
                source = torch.as_tensor(LatentFactorPrior(config).generate_complete_matrix().to_numpy(), dtype=torch.float32)
                target = source.clone()
        if not torch.isfinite(source).all() or not torch.isfinite(target).all():
            raise ValueError("Nonfinite raw pair; never silently redraw or substitute noisy truth")
        mechanism_source = normalize_data(source.unsqueeze(1)).squeeze(1)
        mask_config = coverage.pattern_configuration(config, family, requested_missingness)
        query = coverage.apply_family(mechanism_source, mask_config, family, seed + 1)
        normalized, truth, mean, scale = coverage.prepare_input(source, target, query)
        metadata = {"domain": "imputation", "gradient_data": "synthetic", "family": family,
                    "component": component, "seed": seed, "requested_shape": list(shape),
                    "requested_missingness": requested_missingness, "regime": regime,
                    "signal_gain": gain, "spectral": spectral, "normalization": SPECIFICATION["normalization"],
                    "usable_for_training": bool(query.any() and (~query).any()),
                    **coverage.describe_mask(source, target, query),
                    "maximum_absolute_normalized_target": float(truth.abs().max())}
        arrays = {"raw_source": source, "raw_target": target, "query": query,
                  "model_source": normalized, "model_target": truth, "observed_mean": mean, "observed_scale": scale}
        return Episode(normalized.unsqueeze(0), truth.unsqueeze(0), query.unsqueeze(0), metadata), arrays

    def sample(self, *, device="cpu", **specification):
        episode, _ = self.draw(**specification)
        if not episode.metadata["usable_for_training"]:
            raise ValueError("Keep and report degenerate episodes; this sampler never silently resamples")
        return Episode(episode.source.to(device), episode.target.to(device), episode.mask.to(device), episode.metadata)


def preflight_cases(seed=65100601):
    cases = []
    for family in coverage.FAMILIES:
        shapes = COMPACT_SHAPES if family in NEURAL_FAMILIES else SPECTRAL_SHAPES
        for component in COMPONENTS:
            for shape_index, shape in enumerate(shapes):
                for regime in SPECIFICATION["regimes"]:
                    index = len(cases)
                    cases.append({"family": family, "component": component, "shape": list(shape),
                                  "regime": regime, "requested_missingness": (.2, .6)[shape_index],
                                  "seed": seed + 100 * index})
    return cases
