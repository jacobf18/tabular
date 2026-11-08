import numpy as np
import pytest
import torch

import evaluate_mri_completion as synthetic
import evaluate_real_mri_completion as evaluation
import report_real_mri_completion as reporting


def random_kspace(shape=(16, 20, 3)):
    random = np.random.default_rng(42)
    return random.normal(size=shape) + 1j * random.normal(size=shape)


def test_rectangular_multicoil_lift_roundtrip_and_single_coil_convention():
    truth = random_kspace()
    lift = evaluation.MultiCoilSLift(truth.shape[:2], truth.shape[2])
    reconstructed = lift.inverse(lift.forward(truth))
    covered = lift.counts.reshape(truth.shape[:2]) > 0
    np.testing.assert_allclose(reconstructed[covered], truth[covered], atol=1e-14)
    assert not reconstructed[~covered].any()
    assert lift.shape == (2 * 11 * 15, 2 * 13 * 3)
    square = truth[:, :16, :1]
    np.testing.assert_array_equal(evaluation.MultiCoilSLift((16, 16), 1).forward(square),
                                  synthetic.SLift(16).forward(square[..., 0]))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_torch_operator_and_projection_match_numpy(device):
    truth = random_kspace()
    lift = evaluation.MultiCoilSLift(truth.shape[:2], truth.shape[2])
    operator = evaluation.TorchSLift(lift, device=device)
    matrix = lift.forward(truth)
    np.testing.assert_array_equal(operator.forward(torch.as_tensor(truth, device=device)).cpu().numpy(), matrix)
    np.testing.assert_allclose(operator.inverse(torch.as_tensor(matrix, device=device)).cpu().numpy(),
                               lift.inverse(matrix), atol=1e-14)
    arbitrary = np.random.default_rng(4).normal(size=lift.shape)
    projected = lift.forward(lift.inverse(arbitrary))
    np.testing.assert_allclose(lift.forward(lift.inverse(projected)), projected, atol=1e-14)


@pytest.mark.parametrize("pattern", ["points", "lines"])
def test_rectangular_masks_nested_shared_across_coils_and_no_leakage(pattern):
    shape = (64, 80)
    dense, central = evaluation.acquisition_mask(shape, .5, pattern, 3)
    sparse, _ = evaluation.acquisition_mask(shape, .75, pattern, 3)
    assert dense.sum() == .5 * np.prod(shape) and sparse.sum() == .25 * np.prod(shape)
    assert dense[sparse].all() and sparse[central].all()
    validation = synthetic.validation_mask(dense, central, pattern, 3)
    assert dense[validation].all() and not (validation & central).any()
    if pattern == "lines":
        assert (dense == dense[:, :1]).all()
        assert (validation == validation[:, :1]).all()
    truth = random_kspace((*shape, 2))
    lift = evaluation.MultiCoilSLift(shape, 2)
    measured = np.where(dense[..., None], truth, complex(np.nan, np.nan))
    raw, lifted = evaluation.observed_inputs(measured, dense, lift)
    np.testing.assert_array_equal(np.isfinite(raw), np.tile(np.repeat(dense, 2, axis=1), (1, 2)))
    np.testing.assert_array_equal(np.isfinite(lifted), lift.observed(dense))
    alternate = truth.copy()
    alternate[~dense] += 1000 + 2000j
    known = np.isfinite(lifted)
    np.testing.assert_array_equal(lift.forward(alternate)[known], lifted[known])


@pytest.mark.parametrize("paired", [False, True])
def test_tiles_cover_each_cell_once_and_preserve_pairs(paired):
    tiles = evaluation.axis_tiles(30, 8, paired, 7)
    np.testing.assert_array_equal(np.sort(np.concatenate(tiles)), np.arange(30))
    assert max(map(len, tiles)) <= 8
    for first, second in zip(tiles, evaluation.axis_tiles(30, 8, paired, 7)):
        np.testing.assert_array_equal(first, second)
        if paired:
            half = len(first) // 2
            np.testing.assert_array_equal(first[half:], first[:half] + 15)
    np.testing.assert_array_equal(evaluation.axis_tiles(6, 8, paired, 7)[0], np.arange(6))


def test_observed_normalization_respects_tiny_mri_scale():
    values = np.array([[1e-12, 3e-12], [np.nan, np.nan]])
    mean, scale = evaluation.observed_statistics(values)
    assert mean == pytest.approx(2e-12, abs=1e-25)
    assert scale == pytest.approx(1e-12, abs=1e-25)
    np.testing.assert_allclose((values[np.isfinite(values)] - mean) / scale, [-1, 1])
    with pytest.raises(ValueError, match="No acquired"):
        evaluation.observed_statistics(np.full((2, 2), np.nan))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("lifted", [False, True])
def test_neural_tiles_preserve_observations_and_use_one_global_normalization(lifted):
    class FakeModel:
        def __call__(self, values, return_embeddings):
            assert return_embeddings
            return torch.zeros((*values.shape, 1), device=values.device)

        def decoder(self, features):
            return features

    class FakeDistribution:
        def mean(self, logits):
            return logits[:, 0]

    matrix = np.arange(360, dtype=np.float64).reshape(12, 30) * 1e-12
    matrix[::2, 1::2] = np.nan
    expected = np.where(np.isfinite(matrix), matrix, np.nanmean(matrix))
    prediction, details = evaluation.neural_complete(matrix, FakeModel(), FakeDistribution(), lifted, maximum=8)
    np.testing.assert_allclose(prediction, expected, rtol=1e-15, atol=0)
    np.testing.assert_array_equal(prediction[np.isfinite(matrix)], matrix[np.isfinite(matrix)])
    assert details["mean"] == pytest.approx(float(np.nanmean(matrix)), abs=1e-25)
    assert max(details["maximum_tile_shape"]) <= 8
    assert details["empty_tiles"] == 0


def test_multicoil_scores_fft_axes_parseval_and_measurement_consistency():
    truth = random_kspace()
    acquired, _ = evaluation.acquisition_mask(truth.shape[:2], .5, "points", 1, calibration=4)
    prediction = np.where(acquired[..., None], truth, 0)
    images = evaluation.coil_images(truth)
    for coil in range(truth.shape[2]):
        np.testing.assert_array_equal(images[..., coil], synthetic.ifft_kspace(truth[..., coil]))
    scores = evaluation.score_reconstruction(truth, prediction, acquired)
    assert scores["complex_image_nrmse"] == pytest.approx(np.linalg.norm(prediction - truth) / np.linalg.norm(truth))
    assert scores["hidden_kspace_relative_l2"] == pytest.approx(1)
    assert scores["max_observed_error"] == 0
    prediction[acquired] += 1
    with pytest.raises(ValueError, match="changed measured"):
        evaluation.score_reconstruction(truth, prediction, acquired)


def test_structured_solver_matches_numpy_and_preserves_all_coils():
    truth = random_kspace((16, 16, 1))
    acquired, _ = evaluation.acquisition_mask((16, 16), .5, "points", 2, calibration=4)
    measured = np.where(acquired[..., None], truth, complex(np.nan, np.nan))
    lift = evaluation.TorchSLift(evaluation.MultiCoilSLift((16, 16), 1), device="cpu")
    prediction, details = evaluation.structured_lowrank(measured, lift, 4, max_iterations=3)
    expected, _ = synthetic.structured_lowrank(measured[..., 0], synthetic.SLift(16), 4, max_iterations=3)
    np.testing.assert_allclose(prediction[..., 0], expected, atol=1e-12)
    np.testing.assert_array_equal(prediction[acquired], truth[acquired])
    assert details["iterations"] == 3


def test_rank_selection_never_receives_hidden_truth(monkeypatch):
    truth = random_kspace()
    acquired, central = evaluation.acquisition_mask(truth.shape[:2], .5, "points", 1, calibration=4)
    measured = np.where(acquired[..., None], truth, complex(np.nan, np.nan))
    calls = []

    def fake_solver(source, lift, rank):
        assert np.isnan(source[~acquired]).all()
        calls.append(source.copy())
        return np.nan_to_num(source), {"rank": rank}

    monkeypatch.setattr(evaluation, "structured_lowrank", fake_solver)
    _, heldout, diagnostics = evaluation.fit_lowrank(measured, acquired, central, "points", 1, None)
    assert len(calls) == len(evaluation.RANKS) + 1
    assert np.isnan(calls[0][heldout]).all()
    np.testing.assert_array_equal(calls[-1], measured)
    assert diagnostics["selected_rank"] == evaluation.RANKS[0]


@pytest.mark.parametrize("shape", [(16, 25), (25, 16)])
def test_gram_spectra_match_svd(shape):
    matrix = np.random.default_rng(3).normal(size=shape) + 1j * np.random.default_rng(4).normal(size=shape)
    np.testing.assert_allclose(reporting.singular_values(matrix), np.linalg.svd(matrix, compute_uv=False), atol=1e-13)
