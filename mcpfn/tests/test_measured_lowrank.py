import numpy as np
import pytest

import evaluate_measured_physics as measured
import refine_measured_softimpute as refinement
import refine_sdss_pca as pca


@pytest.mark.parametrize('shape', [(9, 5), (5, 9), (8, 8)])
def test_gram_prox_matches_full_svd(shape):
    values = np.random.default_rng(4).normal(size=shape)
    np.testing.assert_allclose(refinement.gram_prox(values, .1), measured.nuclear_prox(values, .1), atol=1e-12)


def test_converged_solver_recovers_constant_and_small_residual():
    random = np.random.default_rng(5)
    matrix = np.full((30, 20), 2.)
    missing = random.random(matrix.shape) < .8
    result, details = refinement.complete(np.where(missing, np.nan, matrix), .001)
    np.testing.assert_array_equal(result[~missing], matrix[~missing])
    assert np.sqrt(np.mean((result[missing] - matrix[missing]) ** 2)) < .05
    assert details['fixed_point_residual'] < 1e-6


def test_exact_pca_preserves_observations_and_rank_one_recovery():
    random = np.random.default_rng(11)
    matrix = np.full((20, 30), 2.)
    missing = random.random(matrix.shape) < .5
    result, details = pca.complete(np.where(missing, np.nan, matrix), rank=1)
    np.testing.assert_array_equal(result[~missing], matrix[~missing])
    assert np.max(np.abs(result[missing] - matrix[missing])) < .001
    assert details['exact_projection']
