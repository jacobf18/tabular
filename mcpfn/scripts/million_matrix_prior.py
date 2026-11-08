"""Prospective, domain-balanced synthetic draws for a million-matrix continuation."""

import numpy as np
import torch

from diagnose_single_model_missingness_coverage import FAMILIES as MASK_FAMILIES
from single_model_area_prior import AreaMixedSampler
from single_model_complete_missingness_prior import COMPONENTS, NEURAL_FAMILIES, CompleteMissingnessSampler
from single_model_distillation import Episode
from tabimpute.prior.scientific_panels import TRAIN_FAMILIES, MASK_PATTERNS, sample_scientific_episode
from tabimpute.prior.scientific_processes import PROCESS_FAMILIES, sample_process_episode, sample_score_panel


DOMAINS = ('science', 'imputation', 'recsys', 'score_panel')
SCIENCE_FAMILIES = TRAIN_FAMILIES + PROCESS_FAMILIES
WEIGHTS = dict(science=.35, imputation=.35, recsys=.20, score_panel=.10)
TRAIN_SEED = 1_810_000_000
VALIDATION_SEED = 2_310_000_000
PREFLIGHT_SEED = 2_410_000_000
SMALL = ((32, 8), (64, 16), (96, 24), (128, 32), (192, 48), (256, 64))
MEDIUM = ((384, 96), (512, 128), (768, 128))


def shape_draw(seed):
    random = np.random.default_rng(seed)
    bucket = int(random.integers(1000))
    if bucket < 800:
        return list(SMALL[int(random.integers(len(SMALL)))]), 'small'
    if bucket < 980:
        return list(MEDIUM[int(random.integers(len(MEDIUM)))]), 'medium'
    return list((1024, 256) if bucket < 998 else ((1536, 512) if bucket == 998 else (2000, 1000))), 'large'


def specification(domain, ordinal, seed):
    shape, size = shape_draw(seed)
    random = np.random.default_rng(seed + 1)
    result = dict(domain=domain, seed=seed, shape=shape, requested_size=size)
    if domain == 'science':
        result.update(family=SCIENCE_FAMILIES[ordinal % len(SCIENCE_FAMILIES)],
                      pattern=MASK_PATTERNS[(ordinal // len(SCIENCE_FAMILIES)) % len(MASK_PATTERNS)])
    elif domain == 'imputation':
        if ordinal % 2 == 0:
            result.update(kind='native', component=(ordinal // 2) % 8,
                          regime='area_scaled' if random.random() < .5 else 'legacy')
        else:
            pair = (ordinal // 2) % (len(MASK_FAMILIES) * len(COMPONENTS))
            family, component = MASK_FAMILIES[pair // len(COMPONENTS)], COMPONENTS[pair % len(COMPONENTS)]
            shapes = ((24, 12), (48, 24)) if family in NEURAL_FAMILIES else ((240, 80), (1024, 256))
            result.update(kind='structured', family=family, component=component,
                          shape=list(shapes[int(random.random() >= .85)]),
                          regime='area_scaled' if random.random() < .5 else 'legacy',
                          requested_missingness=.2 if random.random() < .5 else .6)
    elif domain == 'recsys':
        result['shape'] = [max(128, shape[0]), max(48, shape[1])]
    elif domain == 'score_panel':
        result['shape'] = [max(96, min(512, shape[0])), max(16, min(128, shape[1]))]
    else:
        raise ValueError('Unknown domain')
    return result


def training_specs(step):
    if type(step) is not int or not 1 <= step <= 250_000:
        raise ValueError('Require an update in the million-matrix horizon')
    return [specification(domain, step - 1, TRAIN_SEED + 16 * (4 * (step - 1) + index))
            for index, domain in enumerate(DOMAINS)]


def validation_specs():
    result = []
    for family in SCIENCE_FAMILIES:
        for pattern in ('bursts', 'self_mnar', 'covariate_mar'):
            result.append(dict(domain='science', family=family, pattern=pattern, shape=[192, 32]))
    for component in range(8):
        for regime in ('legacy', 'area_scaled'):
            result.append(dict(domain='imputation', kind='native', component=component, regime=regime, shape=[256, 64]))
    for family in MASK_FAMILIES:
        for component in ('slow_tail_mcar', 'anchor_latent_mcar'):
            shape = [48, 24] if family in NEURAL_FAMILIES else [240, 80]
            result.append(dict(domain='imputation', kind='structured', family=family, component=component,
                               regime='area_scaled', requested_missingness=.6, shape=shape))
    for domain in ('recsys', 'score_panel'):
        for shape in ((128, 48), (256, 96), (512, 128)):
            for repeat in range(2):
                result.append(dict(domain=domain, shape=list(shape)))
    for domain in ('science', 'imputation'):
        entry = dict(domain=domain, shape=[1024, 256])
        entry.update(dict(family='smooth_local', pattern='bursts') if domain == 'science'
                     else dict(kind='native', component=2, regime='area_scaled'))
        result.append(entry)
    return [dict(spec, seed=VALIDATION_SEED + 16 * index) for index, spec in enumerate(result)]


class MillionSampler:
    def __init__(self, config):
        self.mixed = AreaMixedSampler(config)
        self.structured = CompleteMissingnessSampler()

    def draw(self, spec, step=1):
        spec = dict(spec)
        requested_seed = spec['seed']
        rejected = []
        for retry in range(16):
            spec['seed'] = requested_seed + retry
            domain, shape = spec['domain'], spec['shape']
            if domain == 'science' or domain == 'score_panel':
                if domain == 'score_panel':
                    generated = sample_score_panel(spec['seed'], rows=shape[0], columns=shape[1])
                else:
                    generator = sample_process_episode if spec['family'] in PROCESS_FAMILIES else sample_scientific_episode
                    generated = generator(spec['seed'], family=spec['family'], pattern=spec['pattern'], rows=shape[0], columns=shape[1])
                episode = Episode(torch.from_numpy(generated.source)[None].cuda(), torch.from_numpy(generated.target)[None].cuda(),
                                  torch.from_numpy(generated.query)[None].cuda(), dict(generated.metadata, domain=domain))
            elif spec.get('kind') == 'structured':
                fields = {key: spec[key] for key in ('family', 'component', 'shape', 'regime', 'requested_missingness', 'seed')}
                episode, arrays = self.structured.draw(**fields)
                del arrays
                if not episode.metadata['usable_for_training']:
                    rejected.append(dict(seed=spec['seed'], reason='structured_mask_has_no_queries_or_context'))
                    continue
                episode = Episode(episode.source.cuda(), episode.target.cuda(), episode.mask.cuda(), episode.metadata)
            else:
                fields = dict(seed=spec['seed'], shape=shape)
                if domain == 'imputation':
                    fields.update(component=spec['component'], regime=spec['regime'])
                episode = self.mixed.sample(domain, step, **fields)
            if episode.source.shape[0] != 1 or not episode.mask.any() or not torch.isfinite(episode.target[episode.mask]).all():
                raise ValueError('Require exactly one usable finite synthetic matrix per draw')
            metadata = dict(episode.metadata, domain=domain, specification=spec, requested_seed=requested_seed,
                            rejected_draws=rejected, matrix_count=1, rows=episode.source.shape[1], columns=episode.source.shape[2],
                            observed_cells=int(torch.isfinite(episode.source).sum()), query_cells=int(episode.mask.sum()))
            return Episode(episode.source, episode.target, episode.mask, metadata, episode.ratings)
        raise ValueError(f'Exhausted registered degenerate-mask retries: {rejected}')


def teacher_weight(domain, matrices):
    progress = min(1., max(0., matrices / 250_000))
    if domain == 'science':
        return 0.
    initial, final = (2., .5) if domain == 'imputation' else (8., 4.)
    return initial + (final - initial) * progress
