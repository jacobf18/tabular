import math

import pytest
import torch

import analyze_single_model_cohort_retention_endpoint as analysis


def test_bitwise_source_check_preserves_nans_and_signed_zero():
    source = torch.tensor([float('nan'), 0.])
    assert analysis.same_tensor(source, source.clone())
    assert not analysis.same_tensor(source, torch.tensor([float('nan'), -0.]))
    assert not analysis.same_tensor(source, source.double())


def test_summary_keeps_increases_decreases_ties_and_original_mixture_weights():
    rows = [
        {'reference_role': 'v8', 'cohort': 'legacy', 'weight': .25, 'before': 2., 'after': 3.},
        {'reference_role': 'v29', 'cohort': 'mar', 'weight': .01, 'before': 5., 'after': 4.},
        {'reference_role': 'v29', 'cohort': 'mar', 'weight': .01, 'before': 6., 'after': 6.},
        {'reference_role': None, 'cohort': 'recsys', 'weight': .125},
    ]
    summaries = analysis.summarize(rows)
    overall = next(row for row in summaries if row['grouping'] == 'all')
    assert overall['tasks'] == 3
    assert overall['increased'] == overall['decreased'] == overall['unchanged'] == 1
    assert overall['mean_before'] == overall['mean_after'] == 13. / 3.
    assert math.isclose(overall['mixture_weighted_before'], .61)
    assert math.isclose(overall['mixture_weighted_after'], .85)
    assert {row['name'] for row in summaries if row['grouping'] == 'cohort'} == {'legacy', 'mar'}


def test_hash_mismatch_cannot_enter_evidence(tmp_path):
    path = tmp_path / 'evidence.json'
    path.write_text('{}')
    identities = {}
    with pytest.raises(ValueError, match='changed'):
        analysis.bind(path, '0' * 64, identities)
    assert not identities
    analysis.bind(path, analysis.digest(path), identities)
    assert identities[str(path)] == analysis.digest(path)


def test_existing_diagnostic_cannot_be_overwritten(tmp_path):
    with pytest.raises(FileExistsError, match='Never replace'):
        analysis.run(tmp_path)


def test_scoring_keeps_constant_single_query_and_empty_columns():
    target = torch.zeros(1, 3, 3)
    mask = torch.tensor([[[True, True, False], [True, False, False], [False, False, False]]])
    penalty = analysis.retention.independent_penalty(target, mask, torch.ones_like(target), target)
    assert penalty == 25.
