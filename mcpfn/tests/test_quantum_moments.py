import numpy as np

import evaluate_quantum_moments as moments


def test_products_match_explicit_pauli_multiplication():
    _, operators = moments.experiment.quantum.pauli_basis(2)
    products, phases = moments.pauli_products(2)
    for left in range(16):
        for right in range(16):
            np.testing.assert_array_equal(operators[left] @ operators[right], phases[left, right] * operators[products[left, right]])


def test_moment_matrix_is_psd_and_has_physical_rank():
    _, operators = moments.experiment.quantum.pauli_basis(2)
    state = np.array([1, 1j, 2, -1j]) / np.sqrt(7)
    density = np.outer(state, state.conj())
    coefficients = moments.experiment.quantum.pauli_coefficients(density, operators)
    products, phases = moments.pauli_products(2)
    lifted = moments.lift_coefficients(coefficients, products, phases)
    np.testing.assert_allclose(lifted, lifted.T, atol=1e-12)
    assert np.linalg.eigvalsh(lifted).min() >= -1e-12
    assert np.linalg.matrix_rank(lifted, tol=1e-10) == 8
    np.testing.assert_allclose(moments.unlift_coefficients(lifted, products, phases, coefficients.shape), coefficients, atol=1e-12)


def test_all_duplicates_of_hidden_coefficient_are_masked():
    products, phases = moments.pauli_products(2)
    coefficients = np.arange(16, dtype=float).reshape(4, 4)
    full = moments.lift_coefficients(coefficients, products, phases)
    coefficients.flat[5] = np.nan
    masked = moments.lift_coefficients(coefficients, products, phases)
    alternate = coefficients.copy()
    alternate.flat[5] = 999.
    changed = moments.lift_coefficients(alternate, products, phases) != full
    np.testing.assert_array_equal(np.isnan(masked), changed)
    np.testing.assert_array_equal(masked[~changed], full[~changed])
