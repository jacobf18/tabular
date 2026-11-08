import numpy as np
import pytest
from qiskit.quantum_info import DensityMatrix, Pauli, random_density_matrix

import evaluate_quantum_tomography as evaluation


def test_pauli_basis_transform_and_complex_y_convention():
    labels, operators = evaluation.pauli_basis(2)
    density = random_density_matrix(4, seed=17).data
    coefficients = evaluation.pauli_coefficients(density, operators)
    np.testing.assert_allclose(evaluation.density_from_coefficients(coefficients, operators), density, atol=1e-14)
    assert labels[0] == "II" and labels[-1] == "ZZ"
    assert coefficients.flat[0] == pytest.approx(1)
    for index, label in enumerate(labels):
        assert coefficients.flat[index] == pytest.approx(DensityMatrix(density).expectation_value(Pauli(label)).real)
    gram = np.einsum("pij,qji->pq", operators, operators)
    np.testing.assert_allclose(gram, 4 * np.eye(16), atol=1e-14)


def test_density_projection_is_simplex_projection_not_renormalization():
    density = np.diag([1.2, .4, -.3, -.3]).astype(complex)
    projected = evaluation.project_density(density)
    np.testing.assert_allclose(np.diag(projected), [.9, .1, 0, 0], atol=1e-14)
    np.testing.assert_allclose(evaluation.project_density(projected), projected, atol=1e-14)


@pytest.mark.parametrize("family", evaluation.FAMILIES)
def test_state_families_are_physical_and_reproducible(family):
    density, details = evaluation.generate_state(family, 4, 1)
    repeat, _ = evaluation.generate_state(family, 4, 1)
    np.testing.assert_array_equal(density, repeat)
    np.testing.assert_allclose(np.trace(density), 1)
    assert np.linalg.eigvalsh(density).min() > -1e-12
    if family == "Rank-two mixed":
        assert np.count_nonzero(np.linalg.eigvalsh(density) > 1e-10) == 2
    if family == "Ising ground":
        assert details["field"] == 1


def test_measurement_mask_is_nested_and_independent_of_hidden_truth():
    _, operators = evaluation.pauli_basis(4)
    density, _ = evaluation.generate_state("Haar pure", 4, 0)
    truth = evaluation.pauli_coefficients(density, operators)
    matrix, counts = evaluation.measure_paulis(truth, .5, 2000, 3)
    sparse, sparse_counts = evaluation.measure_paulis(truth, .8, 2000, 3)
    assert np.isfinite(matrix).sum() == 129
    assert np.isfinite(sparse).sum() == 52
    assert matrix.flat[0] == sparse.flat[0] == 1
    assert counts.flat[0] == -1
    assert np.all(counts[np.isnan(matrix)] == -1)
    observed = np.isfinite(sparse)
    np.testing.assert_array_equal(sparse[observed], matrix[observed])
    np.testing.assert_array_equal(sparse_counts[observed], counts[observed])
    changed_truth = truth.copy()
    changed_truth[np.isnan(matrix)] = .99
    repeated, repeated_counts = evaluation.measure_paulis(changed_truth, .5, 2000, 3)
    np.testing.assert_array_equal(repeated, matrix)
    np.testing.assert_array_equal(repeated_counts, counts)
    measured = counts >= 0
    np.testing.assert_allclose(matrix[measured], 2 * counts[measured] / 2000 - 1)


def test_shot_sampling_deterministic_eigenstates():
    truth = np.ones((16, 16))
    truth.flat[1::2] = -1
    matrix, counts = evaluation.measure_paulis(truth, .5, 100, 0)
    observed = np.isfinite(matrix)
    np.testing.assert_array_equal(matrix[observed], truth[observed])
    assert set(counts[counts >= 0]) == {0, 100}


def test_completion_preserves_measurements_and_clips_only_predictions():
    matrix = np.array([[1, np.nan], [np.nan, -.25]])
    prediction = np.array([[-99., 2.], [-3., 99.]])
    np.testing.assert_array_equal(evaluation.complete_coefficients(matrix, prediction), [[1, 1], [-1, -.25]])


def test_physical_least_squares_recovers_full_noiseless_y_observables():
    _, operators = evaluation.pauli_basis(2)
    density = random_density_matrix(4, seed=23).data
    matrix = evaluation.pauli_coefficients(density, operators)
    reconstructed, raw, diagnostics = evaluation.physical_least_squares(matrix, operators)
    assert diagnostics["status"] == "optimal"
    np.testing.assert_allclose(reconstructed, density, atol=2e-5)
    assert np.linalg.eigvalsh(reconstructed).min() > -1e-10
    np.testing.assert_allclose(np.trace(raw), 1, atol=2e-5)


def test_metrics_known_state_and_hidden_only_error():
    _, operators = evaluation.pauli_basis(2)
    density = random_density_matrix(4, seed=24).data
    truth = evaluation.pauli_coefficients(density, operators)
    matrix = truth.copy()
    matrix.flat[1:4] = np.nan
    matrix.flat[4:] += .1
    metrics = evaluation.reconstruction_metrics(density, truth, density, matrix, operators)
    assert metrics["fidelity"] == pytest.approx(1, abs=1e-8)
    assert metrics["trace_distance"] == 0
    assert metrics["hidden_pauli_rmse"] == 0
    assert metrics["observed_fit_rmse"] == pytest.approx(.1)
