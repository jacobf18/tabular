"""Synthetic scientific panels with observed-only scaling and structured outages."""

from dataclasses import dataclass

import numpy as np


TRAIN_FAMILIES = (
    'smooth_sensors', 'dynamic_factors', 'transport', 'nonlinear_calibration',
    'regime_panel', 'chemical_mixture',
)
TRANSFER_FAMILIES = ('damped_oscillators', 'spatial_field')
MASK_PATTERNS = ('isolated', 'bursts', 'network_outage', 'scheduled_assays', 'self_mnar', 'covariate_mar')


@dataclass
class ScientificEpisode:
    source: np.ndarray
    target: np.ndarray
    query: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    metadata: dict


def observed_standardize(truth, missing):
    truth = np.asarray(truth, dtype=np.float64)
    missing = np.asarray(missing, dtype=bool)
    if truth.ndim != 2 or truth.shape != missing.shape or not np.isfinite(truth).all():
        raise ValueError('Require matching finite truth and missingness matrices')
    counts = (~missing).sum(axis=0, keepdims=True)
    if (counts < 2).any():
        raise ValueError('Each generated channel needs observed support')
    center = np.where(missing, 0., truth).sum(axis=0, keepdims=True) / counts
    scale = np.sqrt(np.where(missing, 0., (truth - center) ** 2).sum(axis=0, keepdims=True) / counts)
    scale = np.where(scale > 1e-6, scale, 1.)
    target = ((truth - center) / scale).astype(np.float32)
    source = np.where(missing, np.nan, target)
    return source, target, center, scale


def _unit_columns(values):
    return (values - values.mean(axis=0, keepdims=True)) / np.maximum(values.std(axis=0, keepdims=True), 1e-6)


def _smooth_latents(random, times, rank):
    frequencies = np.exp(random.uniform(np.log(.15), np.log(8.), size=(rank, 6)))
    phases = random.uniform(-np.pi, np.pi, size=(rank, 6))
    amplitudes = random.normal(size=(rank, 6)) / np.maximum(frequencies, 1.) ** 1.5
    waves = np.sin(2 * np.pi * times[:, None, None] * frequencies + phases)
    values = (waves * amplitudes).sum(axis=-1)
    values += times[:, None] * random.normal(size=rank)
    return _unit_columns(values)


def _truth(random, family, rows, columns):
    intervals = random.lognormal(0., random.choice([0., .3, .8]), rows)
    times = np.cumsum(intervals)
    times = (times - times[0]) / (times[-1] - times[0])
    rank = int(random.integers(2, min(7, columns) + 1))
    latent = _smooth_latents(random, times, rank)
    loading = random.normal(size=(rank, columns)) / np.sqrt(rank)
    noise = float(np.exp(random.uniform(np.log(.002), np.log(.2))))
    values = latent @ loading
    if family == 'smooth_sensors':
        values += .25 * _smooth_latents(random, times, columns)
    elif family == 'dynamic_factors':
        persistence = random.uniform(.6, .995, rank)
        states = random.normal(size=(rows, rank))
        for row in range(1, rows):
            states[row] = persistence * states[row - 1] + np.sqrt(1 - persistence ** 2) * states[row]
        values = (.5 * latent + _unit_columns(states)) @ loading
        values += random.uniform(0., .4) * np.tanh(values) ** 2
    elif family == 'transport':
        values = np.zeros((rows, columns))
        for _ in range(int(random.integers(2, 7))):
            center = random.uniform(-.15, 1.15)
            width = random.uniform(.02, .2)
            delays = random.uniform(-.08, .08, columns)
            amplitudes = random.lognormal(0., .4, columns)
            values += amplitudes * np.exp(-.5 * ((times[:, None] - center - delays) / width) ** 2)
        values += .15 * latent @ loading
        values = np.log1p(np.maximum(values, 0.))
    elif family == 'nonlinear_calibration':
        if random.random() < .5:
            latent = random.normal(size=(rows, rank))
        values = latent @ loading
        nonlinear = np.tanh(values * random.uniform(.3, 1.5, columns))
        nonlinear += .2 * np.square(values) + .15 * np.sin(values * 2.)
        linear_channels = random.random(columns) < .35
        values = np.where(linear_channels, values, nonlinear)
        values += times[:, None] * random.normal(0., .25, columns)
    elif family == 'regime_panel':
        groups = random.integers(0, rank, columns)
        values = latent[:, groups] + .2 * values
        for _ in range(int(random.integers(1, 4))):
            change = random.uniform(.15, .85)
            width = random.uniform(.001, .04)
            transition = .5 * (1 + np.tanh((times - change) / width))
            values += transition[:, None] * random.normal(0., .6, rank)[groups]
    elif family == 'chemical_mixture':
        if random.random() < .5:
            latent = random.normal(size=(rows, rank))
        proportions = np.exp(latent - latent.max(axis=1, keepdims=True))
        proportions /= proportions.sum(axis=1, keepdims=True)
        endmembers = random.lognormal(0., .7, size=(rank, columns))
        values = proportions @ endmembers
        interaction = proportions[:, 0] * proportions[:, 1]
        values += random.uniform(0., .3) * interaction[:, None] * random.normal(size=columns)
    elif family == 'damped_oscillators':
        frequencies = random.uniform(.5, 12., rank)
        decay = random.uniform(.1, 3., rank)
        states = np.exp(-times[:, None] * decay) * np.sin(2 * np.pi * times[:, None] * frequencies)
        values = states @ loading
    elif family == 'spatial_field':
        locations = random.uniform(-1., 1., (rows, 2))
        centers = random.uniform(-1., 1., (rank * 2, 2))
        distances = ((locations[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        features = np.exp(-distances / random.uniform(.03, 1.))
        values = features @ random.normal(size=(rank * 2, columns))
    else:
        raise ValueError(f'Unknown scientific family: {family}')
    values = _unit_columns(values)
    heterogeneity = random.lognormal(0., .4, columns)
    values += noise * heterogeneity * random.normal(size=values.shape)
    if random.random() < .2:
        outliers = random.random(values.shape) < .005
        values += outliers * random.normal(0., .5, values.shape)
    values = values * random.lognormal(0., 1., columns) + random.normal(0., 3., columns)
    return values, times, noise


def structured_missingness(truth, pattern, random):
    rows, columns = truth.shape
    if pattern not in MASK_PATTERNS:
        raise ValueError(pattern)
    rate = random.uniform(.02, .3)
    missing = random.random(truth.shape) < rate
    if pattern == 'bursts':
        for column in range(columns):
            for _ in range(int(random.integers(1, 4))):
                start = int(random.integers(rows))
                length = int(random.integers(2, max(3, rows // 3)))
                missing[start:start + length, column] = True
    elif pattern == 'network_outage':
        for _ in range(int(random.integers(1, 4))):
            start = int(random.integers(rows))
            length = int(random.integers(2, max(3, rows // 3)))
            affected = random.random(columns) < random.uniform(.2, .9)
            missing[start:start + length, affected] = True
    elif pattern == 'scheduled_assays':
        for column in random.permutation(columns)[:max(1, columns // 3)]:
            period = int(random.integers(2, min(rows // 3, 25)))
            phase = int(random.integers(period))
            missing[:, column] |= np.arange(rows) % period != phase
    elif pattern == 'self_mnar':
        values = _unit_columns(truth)
        slope = random.uniform(-3., 3., columns)
        probability = 1 / (1 + np.exp(-np.clip(values * slope - .5, -30, 30)))
        missing |= random.random(truth.shape) < probability
    elif pattern == 'covariate_mar':
        covariate = _unit_columns(truth[:, :1])
        probability = 1 / (1 + np.exp(-np.clip(covariate * random.normal(size=columns), -30, 30)))
        missing |= random.random(truth.shape) < probability
        missing[:, 0] = False
    for column in range(columns):
        if (~missing[:, column]).sum() < 4:
            missing[random.choice(rows, 4, replace=False), column] = False
    if not missing.any():
        missing[0, -1] = True
    return missing


def sample_scientific_episode(seed, *, family=None, rows=128, columns=24,
                              pattern=None, allow_transfer=False, time_features=None):
    if rows < 16 or columns < 4:
        raise ValueError('Require at least 16 rows and four measurement columns')
    random = np.random.default_rng(seed)
    family = str(random.choice(TRAIN_FAMILIES)) if family is None else family
    allowed = TRAIN_FAMILIES + TRANSFER_FAMILIES if allow_transfer else TRAIN_FAMILIES
    if family not in allowed:
        raise ValueError('Transfer-only or unknown family cannot enter training')
    pattern = str(random.choice(MASK_PATTERNS)) if pattern is None else pattern
    truth, times, noise = _truth(random, family, rows, columns)
    missing = structured_missingness(truth, pattern, random)
    use_time = bool(random.random() < .5) if time_features is None else bool(time_features)
    if use_time and family != 'spatial_field':
        truth = np.column_stack((truth, times, np.sin(2 * np.pi * times), np.cos(2 * np.pi * times)))
        missing = np.column_stack((missing, np.zeros((rows, 3), dtype=bool)))
    source, target, center, scale = observed_standardize(truth, missing)
    return ScientificEpisode(source, target, missing, center, scale, dict(
        family=family, pattern=pattern, seed=int(seed), rows=rows, measurement_columns=columns,
        columns=truth.shape[1], time_features=truth.shape[1] != columns, noise_scale=noise,
        missing_fraction=float(missing[:, :columns].mean()), gradient_data='synthetic',
    ))
