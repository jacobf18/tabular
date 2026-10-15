"""Balance structured retention across families without changing supervised task weights."""

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import hashlib
import inspect
import math
from pathlib import Path
import types

import numpy as np
import torch

import single_model_cohort_anchor_retention as predecessor
from single_model_cohort_anchor_retention import (
    validate_columns, compute_context, query, column, query_budget_majorizer, TRAIN_OPTIONS,
    streamed_training_loss, verify_model, STREAM_OPTIONS, atomic_torch_save,
    streamed_point_predictions, sha256, write_json, FAMILIES,
)
from single_model_registered_adaptation import replace_once


REFERENCE_SHA256S = dict(predecessor.REFERENCE_SHA256S)
STRUCTURED_MASS = .25
REPLICAS = 2
STRUCTURED_TASK_MASS = STRUCTURED_MASS / (len(FAMILIES) * REPLICAS)
SPECIFICATION = {**predecessor.SPECIFICATION,
    'structured_retention_mass': STRUCTURED_MASS,
    'structured_retention_family_weighting': 'equal over all twelve original families and both replicas',
    'structured_retention_task_mass': STRUCTURED_TASK_MASS,
    'legacy_retention_weighting': 'unchanged original supervised task weight',
    'supervised_task_weighting': 'unchanged original weights for every teacher and true-target term',
    'retention_multiplier_inputs': 'original synthetic cohort metadata and declared supervised task weight only'}
POLICY = ('Prospective retention-weighting-only experiment after V42 completed64 attempts/58 updates '
    'and all independent audits without a qualifying boundary. Preserve its negative MAR block-neural '
    'result and every earlier adverse outcome. Keep unchanged V29 initialization and V42 metadata-only '
    'V8 legacy/two-phase and V29 other-structured reference assignment. Keep weight10, count-shrunk '
    'equal-column penalty, all original queries and both frozen training-only references. Allocate only '
    'the additional structured retention term equally across all12 original families and both replicas, '
    'with total mass.25. Legacy retains its original.5 mass; ratings retain their exact original callback '
    'without retention. Keep original supervised mixture weights, priors, teachers, Huber/NLL/CDF/query/'
    'point losses, normalizations and optimizer protections unchanged. The multiplier is fixed from '
    'synthetic metadata and declared task weight, never seeds, errors, validation failures or columns. '
    'Reduced MAR/skip-logic retention is an explicit risk; endpoint loss magnitudes do not prove causation '
    'or expected validation gains. No inference ensemble, guard relaxation or benchmark selection. '
    'This implementation alone authorizes no new seeds, probe, training or qualification.')


for name in ('reference_role', 'eligible', 'validate_identities', 'make_references', 'validate_target',
        'shrunk_variance', 'column_coefficients', 'independent_penalty'):
    function = getattr(predecessor, name)
    if function.__closure__ is not None:
        raise ValueError('Do not retain predecessor loss closures')
    cloned = types.FunctionType(function.__code__, globals(), function.__name__, function.__defaults__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    globals()[name] = cloned


def retention_weighting(metadata):
    role = reference_role(metadata)
    if role is None:
        weight, multiplier, effective = .125, 0., 0.
    elif metadata.get('structured', False):
        weight = .05 if metadata['family'] in ('mar', 'mnar_skip_logic') else .0025
        multiplier, effective = STRUCTURED_TASK_MASS / weight, STRUCTURED_TASK_MASS
    else:
        weight = .25
        multiplier, effective = 1., float(weight)
    return {'task_weight': float(weight), 'multiplier': multiplier, 'effective_task_weight': effective}


def validate_task_mixture(tasks):
    expected = {'legacy': REPLICAS, 'recsys': REPLICAS, **{family: REPLICAS for family in FAMILIES}}
    if Counter(task['cohort'] for task in tasks) != expected:
        raise ValueError('Require both replicas of every original cohort without omissions')
    for task in tasks:
        cohort = task['cohort']
        expected_weight = .25 if cohort == 'legacy' else .125 if cohort == 'recsys' else (
            .05 if cohort in ('mar', 'mnar_skip_logic') else .0025)
        if task['weight'] != expected_weight:
            raise ValueError('Never change the original supervised task mixture')


BACKWARD_SOURCE = predecessor.BACKWARD_SOURCE
BACKWARD_SOURCE = replace_once(BACKWARD_SOURCE, '    validate_target(episode, retention_target)',
    '    validate_target(episode, retention_target)\n'
    '    retention_multiplier = retention_weighting(episode.metadata)["multiplier"]')
BACKWARD_SOURCE = replace_once(BACKWARD_SOURCE,
    'SPECIFICATION["weight"] * retention_weights[lookup]',
    'SPECIFICATION["weight"] * retention_multiplier * retention_weights[lookup]')
BACKWARD_SOURCE_SHA256 = hashlib.sha256(BACKWARD_SOURCE.encode()).hexdigest()
exec(compile(BACKWARD_SOURCE, __file__, 'exec'), globals())

RUNTIME_SOURCE = inspect.getsource(predecessor.runtime)
RUNTIME_SOURCE = replace_once(RUNTIME_SOURCE, '        seed = episode.metadata["seed"]',
    '        seed = episode.metadata["seed"]\n'
    '        weighting = retention_weighting(episode.metadata)')
RUNTIME_SOURCE = replace_once(RUNTIME_SOURCE,
    '"queries": int(episode.mask.sum()), "metadata": dict(episode.metadata)}',
    '"queries": int(episode.mask.sum()), "metadata": dict(episode.metadata), "weighting": weighting,\n'
    '            "gradient_scale": float(weight)}')
RUNTIME_SOURCE = replace_once(RUNTIME_SOURCE,
    '"specification": SPECIFICATION, "metadata": dict(episode.metadata)}, path)',
    '"specification": SPECIFICATION, "metadata": dict(episode.metadata), "weighting": weighting}, path)')
exec(compile(RUNTIME_SOURCE, __file__, 'exec'), globals())
