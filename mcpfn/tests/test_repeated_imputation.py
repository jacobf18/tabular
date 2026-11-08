import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from repeated_imputation_masks import (
    PATTERNS, calibrated_probabilities, family_state, mask_at_rate, seed_for,
)
from repeated_imputation_methods import impute, metrics, observed_standardize
from repeated_imputation_statistics import dataset_scores, interval, summarize
from run_repeated_imputation import affine_equivalent, load_mask, prepare, verify_hashes
from recsys_evaluation_common import atomic_npz, sha256


@pytest.fixture
def truth():
    return np.random.default_rng(10).normal(size=(30, 5))


@pytest.mark.parametrize('pattern', list(PATTERNS))
def test_reproducible_valid_masks(pattern, truth):
    before = truth.copy()
    first = family_state(truth, pattern, 18)
    second = family_state(truth, pattern, 18)
    for rate in (.1, .4, .5):
        mask, details = mask_at_rate(first, rate)
        repeated, unused = mask_at_rate(second, rate)
        np.testing.assert_array_equal(mask, repeated)
        assert mask.dtype == bool and mask.shape == truth.shape
        assert mask.any() and not mask.all()
        assert details['realized_rate'] == mask.mean()
        if details['expected_rate'] is not None:
            assert details['expected_rate'] == pytest.approx(rate, abs=1e-12)
    np.testing.assert_array_equal(truth, before)


@pytest.mark.parametrize('pattern', ['MCAR', 'MNARLatentFactorPattern', 'MNARClusterLevelPattern'])
def test_mcar_families_do_not_depend_on_values(pattern, truth):
    first = mask_at_rate(family_state(truth, pattern, 20), .4)[0]
    second = mask_at_rate(family_state(truth**3 + 90, pattern, 20), .4)[0]
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize('pattern', ['MAR', 'MNARTwoPhaseSubsetPattern'])
def test_mar_depends_only_on_protected_columns(pattern, truth):
    state = family_state(truth, pattern, 34)
    protected = state['protected_columns']
    other = [column for column in range(truth.shape[1]) if column not in protected]
    changed = truth.copy()
    changed[:, other] = np.random.default_rng(20).normal(size=(30, len(other))) * 100
    mask = mask_at_rate(state, .4)[0]
    np.testing.assert_array_equal(mask, mask_at_rate(family_state(changed, pattern, 34), .4)[0])
    assert not mask[:, protected].any()
    if pattern == 'MNARTwoPhaseSubsetPattern':
        assert np.all(mask[:, other] == mask[:, other[0], None])


def test_deterministic_polarization_is_not_five_independent_masks(truth):
    masks = [mask_at_rate(family_state(truth, 'MNARPolarizationPattern', seed), .4)[0] for seed in range(5)]
    assert all(np.array_equal(masks[0], mask) for mask in masks)


def test_censoring_reports_ties_instead_of_manufacturing_target_rate():
    truth = np.tile(np.array([0, 0, 0, 0, 1]), (30, 1)).T
    mask, details = mask_at_rate(family_state(truth, 'MNARCensoringPattern', 11), .2)
    assert details['realized_rate'] == mask.mean()
    assert details['expected_rate'] is None


def test_calibration_rejects_infeasible_protected_fraction():
    with pytest.raises(ValueError, match='infeasible'):
        calibrated_probabilities(np.zeros((20, 2)), np.array([True, False]), .6)


def test_seed_stability_and_namespaces():
    assert seed_for('uci', 'iris', 0) == seed_for('uci', 'iris', 0)
    assert seed_for('uci', 'iris', 0) != seed_for('missbench', 'iris', 0)


def test_affine_dataset_audit_and_duplicate_identity(truth):
    assert affine_equivalent(truth, truth * np.arange(1, 6) + np.arange(10, 15))
    assert not affine_equivalent(truth, truth[::-1])
    assert not affine_equivalent(truth, -truth)


def test_mask_roundtrip(tmp_path, truth):
    mask = truth > 0
    path = tmp_path / 'mask.npz'
    atomic_npz(path, packed=np.packbits(mask.reshape(-1)), shape=np.array(mask.shape))
    np.testing.assert_array_equal(mask, load_mask(path))
    verify_hashes({str(path): sha256(path)})
    with pytest.raises(ValueError, match='changed'):
        verify_hashes({str(path): 'wrong'})


def test_prepare_audit_and_resume(tmp_path, truth):
    path = tmp_path / 'truth.npy'
    np.save(path, truth)
    case = dict(case_id='uci/test/MCAR/rate_0.4/seed_0', suite='uci', dataset='test', pattern='MCAR',
                repeat=0, mask_seed=123, missingness=.4, deterministic=False, truth_path=str(path), truth_sha256=sha256(path))
    manifest = dict(cases=[case])
    prepare(tmp_path, manifest)
    prepare(tmp_path, manifest)
    audit = json.loads((tmp_path / 'mask_audit.json').read_text())
    assert audit['passed'] and audit['cases'] == 1


def test_observed_only_normalization_and_empty_columns():
    missing = np.array([[1., np.nan, 2.], [3., np.nan, 2.], [np.nan, np.nan, 2.]])
    normalized, center, scale = observed_standardize(missing)
    np.testing.assert_allclose(center, [2., 0., 2.])
    np.testing.assert_allclose(scale, [1., 1., 1.])
    np.testing.assert_allclose(normalized[:, 0], [-1., 1., np.nan], equal_nan=True)


@pytest.mark.parametrize('method', ['column_mean', 'knn'])
def test_small_cpu_adapters_and_scoring(method, truth):
    missing = truth.copy()
    missing[::3, 0:3] = np.nan
    prediction = impute(method, missing, 42)
    result = metrics(truth, missing, prediction, 'uci')
    assert np.isfinite(result['nrmse']) and result['scored_columns'] == 3
    assert np.isfinite(result['nrmse_observed_sd'])
    np.testing.assert_array_equal(prediction[np.isfinite(missing)], missing[np.isfinite(missing)])


def example_results(repeats=5):
    cases, results = [], []
    for dataset in range(4):
        for pattern in ('MCAR', 'MNAR'):
            for rate in (.1, .4):
                for seed in range(repeats):
                    case = dict(case_id=f'{dataset}/{pattern}/{rate}/{seed}', suite='uci', dataset=str(dataset),
                                pattern=pattern, family=pattern, missingness=rate, repeat=seed)
                    cases.append(case)
                    for method, offset in [('matrixformer', 0.), ('missforest', .5)]:
                        results.append(dict(**case, method=method, nrmse=float(dataset + offset), status='ok'))
    return cases, results


def test_intervals_do_not_treat_masks_as_datasets():
    for repeats in (1, 5):
        cases, results = example_results(repeats)
        scores, coverage, summaries, tests = summarize(cases, results, ['matrixformer', 'missforest'])
        result = next(row for row in summaries if row['method'] == 'matrixformer' and row['panel'] == 'Overall')
        assert result['datasets'] == 4
        assert result['mean'] == 1.5
        assert result['ci95_half_width'] == interval([0., 1., 2., 3.])
        assert all(row['complete'] for row in coverage)
        assert tests[0]['datasets'] == 4 and tests[0]['alternative'] == 'two-sided'


def test_failure_invalidates_dataset_panel_not_only_one_seed():
    cases, results = example_results()
    results[0]['status'] = 'error'
    scores, coverage, summaries, tests = summarize(cases, results, ['matrixformer', 'missforest'])
    result = next(row for row in summaries if row['method'] == 'matrixformer' and row['panel'] == 'Overall')
    assert result['datasets'] == 3 and not result['complete']
    assert not tests
    assert next(row for row in coverage if row['method'] == 'matrixformer')['failed'] == 1


def test_no_results_have_zero_scores():
    cases, unused = example_results()
    scores, coverage = dataset_scores(cases, [], ['matrixformer'])
    assert scores == [] and coverage[0]['successful'] == 0
    assert coverage[0]['pending'] == len(cases)


def test_duplicate_result_rejected():
    cases, results = example_results()
    with pytest.raises(ValueError, match='Duplicate'):
        dataset_scores(cases, results + [results[0]], ['matrixformer', 'missforest'])
