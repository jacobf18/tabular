import pytest
import torch

from audit_dwip_spectral_coverage import quantiles, support_statistics


def test_support_statistics_measures_loss_weighted_zero_support_without_imputation():
    source = torch.tensor([[[float("nan"), float("nan")], [1., float("nan")], [1., 2.]]])
    target = torch.zeros_like(source)
    result = support_statistics(source, target, torch.isnan(source))
    assert result["loss_cells_zero_row_support_fraction"] == pytest.approx(2 / 3)
    assert result["loss_cells_at_most_one_row_support_fraction"] == 1
    assert result["loss_cells_zero_column_support_fraction"] == 0
    assert result["loss_cells_at_most_one_column_support_fraction"] == pytest.approx(2 / 3)
    assert result["observed_residual_rms"] == pytest.approx(2 ** .5)


def test_coverage_quantiles_reject_invalid_values():
    assert quantiles([1, 2, 3])["p50"] == 2
    with pytest.raises(ValueError, match="nonempty finite"):
        quantiles([])
    with pytest.raises(ValueError, match="nonempty finite"):
        quantiles([float("nan")])
