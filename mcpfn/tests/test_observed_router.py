import numpy as np
import pytest

from tabimpute.routing import TwoExpertRouter, fit_routing, observed_holdout, observed_scales


class FakeExpert:
    def __init__(self, raw_value, missing_value):
        self.raw_value = raw_value
        self.missing_value = missing_value
        self.inputs = []

    def impute(self, matrix, return_full=False):
        self.inputs.append(matrix.copy(order="K"))
        raw = np.full(matrix.shape, self.raw_value, dtype=float)
        raw[np.isnan(matrix)] = self.missing_value
        matrix[np.isnan(matrix)] = self.missing_value
        return matrix, raw


def test_uses_raw_not_copied_observed_predictions_and_preserves_input():
    source = np.array([[1., np.nan], [1., 1.]])
    unchanged = source.copy()
    router = TwoExpertRouter(FakeExpert(4, 8), FakeExpert(1, 2))
    result = router.predict_all(source, include_holdout=False)
    assert result.decisions["observed"]["selected"] == "specialist"
    assert result.predictions["observed_hard"][0, 1] == 2
    np.testing.assert_array_equal(source, unchanged)
    for prediction in result.predictions.values():
        np.testing.assert_array_equal(prediction[np.isfinite(source)], source[np.isfinite(source)])


def test_convex_blend_analytic_solution():
    target = np.full((2, 3), 2.)
    result = fit_routing(target, np.zeros_like(target), np.full_like(target, 4),
                         np.ones_like(target, dtype=bool), np.ones(3))
    assert result["specialist_weight"] == .5
    assert result["selected"] == "general"


def test_outer_missing_values_are_not_evidence():
    target = np.array([[1., np.nan]])
    first = fit_routing(target, np.array([[1., 3.]]), np.array([[2., 9.]]), np.isfinite(target), np.ones(2))
    second = fit_routing(target, np.array([[1., 1e9]]), np.array([[2., -1e9]]), np.isfinite(target), np.ones(2))
    assert first == second
    with pytest.raises(ValueError, match="observed targets"):
        fit_routing(target, np.ones_like(target), np.ones_like(target), np.ones_like(target, dtype=bool), np.ones(2))


def test_holdout_deterministic_and_preserves_support():
    matrix = np.arange(100., dtype=float).reshape(10, 10)
    matrix[0, 1:] = np.nan
    heldout = observed_holdout(matrix)
    np.testing.assert_array_equal(heldout, observed_holdout(matrix))
    assert not (heldout & np.isnan(matrix)).any()
    assert heldout.sum() == 9
    assert not heldout[0, 0]
    remaining = np.isfinite(matrix) & ~heldout
    assert (remaining.sum(0) > 0).all() and (remaining.sum(1) > 0).all()


def test_holdout_passes_reduced_context_and_final_uses_full_context():
    matrix = np.ones((10, 10))
    general, specialist = FakeExpert(0, 2), FakeExpert(1, 3)
    result = TwoExpertRouter(general, specialist).predict_all(matrix)
    assert result.decisions["observed"]["selected"] == "specialist"
    assert result.decisions["holdout"]["selected"] == "general"
    assert np.isfinite(general.inputs[0]).sum() == 100
    assert np.isfinite(general.inputs[1]).sum() == 90
    np.testing.assert_array_equal(general.inputs[1], specialist.inputs[1])
    np.testing.assert_array_equal(result.predictions["holdout_hard"], matrix)


@pytest.mark.parametrize("matrix", [np.full((2, 3), np.nan), np.array([[1., np.nan]])])
def test_no_evidence_falls_back_to_general(matrix):
    result = TwoExpertRouter(FakeExpert(0, 2), FakeExpert(1, 3)).predict_all(matrix)
    assert result.decisions["holdout"]["selected"] == "general"
    assert result.decisions["holdout"]["evidence_cells"] == 0


def test_scales_and_loss_invariant_to_column_units():
    matrix = np.array([[1., 3.], [4., 9.]])
    evidence = np.ones_like(matrix, dtype=bool)
    def score(source):
        return fit_routing(source, source * .8, source * 1.3, evidence, observed_scales(source))
    assert score(matrix)["specialist_weight"] == pytest.approx(score(matrix * [1, 1000])["specialist_weight"])
    np.testing.assert_array_equal(observed_scales(np.array([[2., np.nan], [2., np.nan]])), [1, 1])


def test_nonfinite_predictions_fail_loudly():
    with pytest.raises(ValueError, match="Nonfinite expert"):
        TwoExpertRouter(FakeExpert(np.nan, 0), FakeExpert(0, 0)).impute(np.ones((2, 2)))


def test_expert_input_dtype_is_preserved():
    general, specialist = FakeExpert(0, 2), FakeExpert(1, 3)
    TwoExpertRouter(general, specialist).predict_all(np.ones((10, 10), dtype=np.float32, order="F"))
    assert all(matrix.dtype == np.float32 for matrix in general.inputs + specialist.inputs)
    assert all(matrix.flags.f_contiguous for matrix in general.inputs + specialist.inputs)


@pytest.mark.parametrize("matrix", [np.ones(3), np.zeros((0, 2)), np.array([[np.inf]])])
def test_invalid_input_rejected(matrix):
    with pytest.raises(ValueError, match="nonempty 2D"):
        TwoExpertRouter(FakeExpert(0, 0), FakeExpert(0, 0)).impute(matrix)


def test_streamed_return_full_does_not_copy_observed_and_averages_tails():
    import torch
    from tabimpute.streamed_inference import StreamedImputePFN

    imputer = StreamedImputePFN.__new__(StreamedImputePFN)
    imputer.device, imputer.direct_dtype = "cpu", torch.float32
    imputer.max_num_rows, imputer.max_num_cols = 2, 2
    imputer._stream_points = lambda values: torch.full_like(values, 7.)
    matrix = np.array([[1., np.nan, 2.], [3., 4., 5.], [6., np.nan, 8.]])
    completed, raw = imputer.impute(matrix, means=np.zeros(3), stds=np.ones(3), return_full=True)
    np.testing.assert_array_equal(raw, np.full(matrix.shape, 7.))
    np.testing.assert_array_equal(completed[np.isfinite(matrix)], matrix[np.isfinite(matrix)])
    np.testing.assert_array_equal(completed[np.isnan(matrix)], [7., 7.])
    np.testing.assert_array_equal(completed, imputer.impute(matrix, means=np.zeros(3), stds=np.ones(3)))
