import numpy as np
import pytest

import evaluate_mri_completion as evaluation


def test_fft_is_centered_unitary_and_invertible():
    random = np.random.default_rng(3)
    image = random.normal(size=(16, 16)) + 1j * random.normal(size=(16, 16))
    kspace = evaluation.fft_image(image)
    np.testing.assert_allclose(evaluation.ifft_kspace(kspace), image, atol=1e-14)
    assert np.linalg.norm(kspace) == pytest.approx(np.linalg.norm(image))
    constant = evaluation.fft_image(np.ones((16, 16)))
    assert constant[8, 8] == 16
    assert np.count_nonzero(constant) == 1


def test_s_lift_roundtrip_and_geometry():
    lift = evaluation.SLift(16)
    random = np.random.default_rng(4)
    kspace = random.normal(size=(16, 16)) + 1j * random.normal(size=(16, 16))
    restored = lift.inverse(lift.forward(kspace))
    covered = lift.counts.reshape(16, 16) > 0
    np.testing.assert_allclose(restored[covered], kspace[covered], atol=1e-14)
    assert np.all(restored[~covered] == 0)
    assert np.all(covered[3:-2, 1:])
    assert not covered[0].any() and not covered[:, 0].any()
    assert (~covered).sum() == 2 * 16 - 1 + 12
    assert evaluation.SLift(64).shape == (6962, 26)


def test_s_block_convention_and_real_linearity():
    lift = evaluation.SLift(16)
    random = np.random.default_rng(5)
    kspace = random.normal(size=(16, 16)) + 1j * random.normal(size=(16, 16))
    matrix = lift.forward(kspace)
    positive = kspace.ravel()[lift.positive]
    negative = kspace.ravel()[lift.negative]
    expected_wide = np.block([[(positive - negative).real.T, (positive + negative).imag.T],
                              [-(positive - negative).imag.T, (positive + negative).real.T]])
    np.testing.assert_array_equal(matrix.T, expected_wide)
    np.testing.assert_allclose(lift.forward(2 * kspace), 2 * matrix)
    np.testing.assert_allclose(np.linalg.svd(matrix, compute_uv=False),
                               np.linalg.svd(expected_wide, compute_uv=False), atol=1e-12)


@pytest.mark.parametrize("pattern", ["points", "lines"])
def test_masks_have_exact_nested_budgets_and_calibration(pattern):
    dense, central = evaluation.acquisition_mask(64, .5, pattern, 6)
    sparse, _ = evaluation.acquisition_mask(64, .75, pattern, 6)
    assert dense.sum() == 2048 and sparse.sum() == 1024
    assert np.all(dense[sparse]) and np.all(sparse[central])
    if pattern == "lines":
        assert np.all(dense == dense[:, :1])
    heldout = evaluation.validation_mask(dense, central, pattern, 6)
    assert np.all(dense[heldout]) and not (central & heldout).any()
    if pattern == "lines":
        assert np.all(heldout == heldout[:, :1])


def test_lift_masks_all_occurrences_of_unmeasured_frequencies():
    lift = evaluation.SLift(16)
    random = np.random.default_rng(7)
    truth = random.normal(size=(16, 16)) + 1j * random.normal(size=(16, 16))
    acquired = random.random((16, 16)) > .5
    measured = np.where(acquired, truth, complex(np.nan, np.nan))
    raw, lifted = evaluation.observed_inputs(measured, lift)
    np.testing.assert_array_equal(np.isfinite(raw), np.tile(acquired, (1, 2)))
    np.testing.assert_array_equal(np.isfinite(lifted), lift.observed(acquired))
    alternate = truth.copy()
    alternate[~acquired] += 1000 + 2000j
    alternate_lift = lift.forward(alternate)
    known = np.isfinite(lifted)
    np.testing.assert_array_equal(alternate_lift[known], lifted[known])
    assert (~known).any()


def test_lifting_averaging_is_a_structured_projection():
    lift = evaluation.SLift(16)
    random = np.random.default_rng(8)
    arbitrary = random.normal(size=lift.shape)
    projected = lift.forward(lift.inverse(arbitrary))
    repeated = lift.forward(lift.inverse(projected))
    np.testing.assert_allclose(repeated, projected, atol=1e-14)


def test_rank_projection_and_lowrank_preserve_measurements():
    random = np.random.default_rng(9)
    matrix = random.normal(size=(30, 12))
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    expected = (left[:, :4] * singular_values[:4]) @ right[:4]
    np.testing.assert_allclose(evaluation.truncated_rank(matrix, 4), expected, atol=1e-13)
    _, kspace = evaluation.make_phantom(16)
    acquired, _ = evaluation.acquisition_mask(16, .5, "points", 9, calibration=4)
    measured = np.where(acquired, kspace, complex(np.nan, np.nan))
    prediction, diagnostics = evaluation.structured_lowrank(measured, evaluation.SLift(16), 4, max_iterations=3)
    np.testing.assert_array_equal(prediction[acquired], kspace[acquired])
    assert diagnostics["iterations"] <= 3


def test_scores_parseval_and_data_consistency():
    image, kspace = evaluation.make_phantom(16)
    acquired, _ = evaluation.acquisition_mask(16, .5, "points", 10, calibration=4)
    prediction = np.where(acquired, kspace, 0)
    scores = evaluation.score_reconstruction(image, kspace, prediction, acquired)
    assert scores["complex_image_nrmse"] == pytest.approx(np.linalg.norm(prediction - kspace) / np.linalg.norm(kspace))
    assert scores["hidden_kspace_relative_l2"] == pytest.approx(1)
    assert scores["max_observed_error"] == 0
    prediction[acquired] += 1
    with pytest.raises(ValueError, match="preserve acquired"):
        evaluation.score_reconstruction(image, kspace, prediction, acquired)


def test_rank_selection_only_hides_observed_samples(monkeypatch):
    image, kspace = evaluation.make_phantom(16)
    acquired, central = evaluation.acquisition_mask(16, .5, "points", 11, calibration=4)
    measured = np.where(acquired, kspace, complex(np.nan, np.nan))
    calls = []

    def fake_solver(source, lift, rank):
        assert np.isnan(source[~acquired]).all()
        calls.append(source.copy())
        return np.nan_to_num(source), {"rank": rank}

    monkeypatch.setattr(evaluation, "structured_lowrank", fake_solver)
    _, heldout, details = evaluation.fit_lowrank(measured, central, "points", 11, evaluation.SLift(16))
    assert not (heldout & ~acquired).any()
    assert len(calls) == len(evaluation.RANKS) + 1
    assert np.isnan(calls[0][heldout]).all()
    np.testing.assert_array_equal(calls[-1], measured)
    assert details["selected_rank"] == evaluation.RANKS[0]
