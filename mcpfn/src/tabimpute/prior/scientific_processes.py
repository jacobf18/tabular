"""Additional process, profile, local-smoothness, and score-panel generators."""

import numpy as np
from scipy.interpolate import CubicSpline

from tabimpute.prior.scientific_panels import (
    ScientificEpisode, observed_standardize, structured_missingness,
)


PROCESS_FAMILIES = ('treatment_process', 'assay_covariates', 'smooth_local', 'spatial_profiles')


def _standardize(values):
    return (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-6)


def sample_process_episode(seed, *, family, rows=128, columns=24, pattern='bursts'):
    if family not in PROCESS_FAMILIES or rows < 16 or columns < 4:
        raise ValueError('Invalid process family or dimensions')
    random = np.random.default_rng(seed)
    time = np.linspace(0., 1., rows)
    latent = random.normal(size=(rows, 6))
    if random.random() < .5:
        latent = _standardize(CubicSpline(np.linspace(0., 1., 12), random.normal(size=(12, 6)))(time))
    if family == 'treatment_process':
        flow = np.exp(.4 * latent[:, 0])
        load = np.exp(.5 * latent[:, 1:4] + .15 * latent[:, :1])
        concentration = load / flow[:, None]
        residence = np.exp(.25 * latent[:, 4]) / flow
        oxygen = np.logaddexp(0., latent[:, 5])
        reaction = random.uniform(.15, 2., 3)[None, :] * residence[:, None]
        efficiency = np.exp(-reaction * (oxygen[:, None] + .4))
        outlet = concentration * efficiency
        saturation = concentration / (random.uniform(.4, 1.5, 3) + concentration)
        features = np.column_stack((flow, concentration, residence, oxygen, outlet, saturation,
                                    concentration[:, 0] * concentration[:, 2]))
        choices = np.arange(columns) % features.shape[1]
        random.shuffle(choices)
        values = features[:, choices]
        values += random.uniform(0., .15) * features @ random.normal(size=(features.shape[1], columns)) / np.sqrt(features.shape[1])
    elif family == 'assay_covariates':
        linear = latent @ random.normal(size=(6, columns)) / np.sqrt(6)
        transforms = np.stack((linear, np.tanh(linear), np.maximum(linear, 0.),
                               linear ** 2, np.sin(linear), np.logaddexp(0., linear)), axis=0)
        choices = random.integers(len(transforms), size=columns)
        values = transforms[choices, :, np.arange(columns)].T
        regimes = latent[:, 0] > random.uniform(-1., 1.)
        values += regimes[:, None] * random.normal(0., .3, columns)
    elif family == 'smooth_local':
        knots = int(random.integers(4, 15))
        states = CubicSpline(np.linspace(0., 1., knots), random.normal(size=(knots, 4)))(time)
        values = states @ random.normal(size=(4, columns)) / 2.
        individual = CubicSpline(np.linspace(0., 1., 6), random.normal(size=(6, columns)))(time)
        values += random.uniform(.05, .4) * individual
    else:
        casts = int(random.integers(2, 6))
        groups = np.minimum(np.arange(rows) * casts // rows, casts - 1)
        depth = np.zeros(rows)
        for group in range(casts):
            depth[groups == group] = np.linspace(0., 1., (groups == group).sum())
        offset = random.normal(0., .2, casts)[groups]
        transition = .5 * (1 + np.tanh((depth - random.uniform(.2, .7) - offset) / random.uniform(.02, .2)))
        features = np.column_stack((depth, transition, np.exp(-3 * depth), offset,
                                    depth * offset, transition * depth))
        values = features @ random.normal(size=(features.shape[1], columns))
        values[:, 0] = depth
    values = _standardize(values)
    noise = np.exp(random.uniform(np.log(.001 if family == 'smooth_local' else .01), np.log(.04 if family == 'smooth_local' else .15)))
    values += noise * random.lognormal(0., .4, columns) * random.normal(size=values.shape)
    values = values * random.lognormal(0., .8, columns) + random.normal(0., 2., columns)
    missing = structured_missingness(values, pattern, random)
    if family == 'spatial_profiles':
        missing[:, 0] = False
    source, target, center, scale = observed_standardize(values, missing)
    return ScientificEpisode(source, target, missing, center, scale, dict(
        family=family, seed=int(seed), pattern=pattern, rows=rows, columns=columns,
        gradient_data='synthetic', noise_scale=float(noise),
    ))


def sample_score_panel(seed, rows=160, columns=48):
    if rows < 16 or columns < 4:
        raise ValueError('Invalid score-panel dimensions')
    random = np.random.default_rng(seed)
    rank = int(random.integers(2, min(columns, 12)))
    abilities = random.normal(size=(rows, rank))
    loadings = random.normal(size=(rank, columns)) / np.sqrt(rank)
    logits = abilities @ loadings + random.normal(0., .7, columns)
    logits += random.normal(0., .6, (rows, 1))
    values = 1 / (1 + np.exp(-np.clip(logits, -20., 20.)))
    values += random.uniform(.005, .08) * random.normal(size=values.shape)
    missing = random.random(values.shape) < .1
    boundary = int(rows * random.uniform(.6, .85))
    missing[boundary:] = True
    selected = random.choice(columns, int(random.integers(1, min(16, columns))), replace=False)
    missing[boundary:, selected] = False
    source, target, center, scale = observed_standardize(values, missing)
    return ScientificEpisode(source, target, missing, center, scale, dict(
        family='score_panel_replay', seed=int(seed), rows=rows, columns=columns,
        gradient_data='synthetic', context_rows=boundary, observed_test_columns=selected.tolist(),
    ))
