from __future__ import annotations

import torch

from tabimpute.prior.training_set_generation import (
    MissingnessPrior,
    RecSysBiasResidualExposedHoldoutPrior,
    RecSysBiasResidualPrior,
)


def _base_config() -> dict:
    return {
        "num_rows_low": 32,
        "num_rows_high": 40,
        "num_cols_low": 48,
        "num_cols_high": 56,
        "latent_rank_low": 10,
        "latent_rank_high": 10,
        "recsys_bias_residual_rank_low": 6,
        "recsys_bias_residual_rank_high": 10,
        "recsys_bias_rating_clip_probability": 1.0,
        "recsys_bias_rating_quantization_probability": 1.0,
        "recsys_exposure_density_low": 0.04,
        "recsys_exposure_density_high": 0.08,
        "recsys_exposure_binary_steps": 20,
    }


def test_rec_sys_bias_residual_generator_quantized_ratings_and_metadata():
    generator = RecSysBiasResidualPrior(
        {
            **_base_config(),
            "num_rows_low": 24,
            "num_rows_high": 24,
            "num_cols_low": 30,
            "num_cols_high": 30,
            "recsys_bias_output_mode": "rating",
        }
    )

    matrix = torch.tensor(generator.generate_complete_matrix().values)

    assert matrix.shape == (24, 30)
    assert torch.isfinite(matrix).all()
    assert float(matrix.min()) >= 1.0
    assert float(matrix.max()) <= 5.0
    assert generator.last_exposure_signal.shape == matrix.shape
    assert generator.last_metadata["recsys_bias_rating_quantized"] == 1.0
    assert generator.last_metadata["recsys_bias_residual_rank"] >= 6


def test_rec_sys_bias_residual_prior_batch_shapes_masks_and_density():
    prior = MissingnessPrior(
        generator_type="rec_sys_bias_residual",
        missingness_type="mnar_rec_sys_exposure",
        config=_base_config(),
        batch_size=3,
        device="cpu",
        entry_wise_features=False,
    )

    (batch_x, batch_target, d, seq_lens, train_sizes), weights = prior.get_batch()

    assert batch_x.ndim == 3
    assert batch_target.shape == batch_x.shape
    assert d.shape == (3,)
    assert seq_lens.shape == (3,)
    assert train_sizes.shape == (3,)
    assert weights.shape == (3,)
    assert torch.isnan(batch_x).any()
    assert torch.isfinite(batch_x[~torch.isnan(batch_x)]).all()
    assert torch.isfinite(batch_target).all()
    realized_density = float((~torch.isnan(batch_x)).float().mean())
    assert 0.02 <= realized_density <= 0.12
    assert "recsys_bias_residual_rank" in prior.last_batch_metrics
    assert "recsys_exposure_density_realized" in prior.last_batch_metrics


def test_rec_sys_bias_residual_prior_residual_mode_uses_exposure_mask():
    prior = MissingnessPrior(
        generator_type="rec_sys_bias_residual",
        missingness_type="mnar_rec_sys_exposure",
        config={
            **_base_config(),
            "num_rows_low": 36,
            "num_rows_high": 36,
            "num_cols_low": 44,
            "num_cols_high": 44,
            "recsys_bias_output_mode": "residual",
            "recsys_bias_rating_quantization_probability": 0.0,
        },
        batch_size=2,
        device="cpu",
        entry_wise_features=False,
    )

    (batch_x, batch_target, _, _, _), _ = prior.get_batch()

    assert batch_x.shape == (2, 36, 44)
    assert torch.isnan(batch_x).any()
    assert torch.isfinite(batch_target).all()
    assert prior.last_batch_metrics["recsys_bias_output_is_residual"] == 1.0
    assert 0.02 <= prior.last_batch_metrics["recsys_exposure_density_realized"] <= 0.12
    assert (
        prior.last_batch_metrics["recsys_exposure_row_count_p99"]
        >= prior.last_batch_metrics["recsys_exposure_row_count_median"]
    )


def test_rec_sys_bias_residual_exposed_holdout_prior_returns_disjoint_masks():
    prior = RecSysBiasResidualExposedHoldoutPrior(
        config={
            **_base_config(),
            "num_rows_low": 32,
            "num_rows_high": 32,
            "num_cols_low": 48,
            "num_cols_high": 48,
            "recsys_holdout_target_mode": "rating",
            "recsys_holdout_target_fraction_low": 0.2,
            "recsys_holdout_target_fraction_high": 0.2,
        },
        batch_size=2,
        device="cpu",
        entry_wise_features=False,
    )

    (batch_x, batch_target, loss_mask, exposure_mask, context_mask), weights = prior.get_batch()

    assert batch_x.shape == (2, 32, 48)
    assert batch_target.shape == batch_x.shape
    assert loss_mask.shape == batch_x.shape
    assert exposure_mask.shape == batch_x.shape
    assert context_mask.shape == batch_x.shape
    assert weights.shape == (2,)
    assert loss_mask.dtype == torch.bool
    assert exposure_mask.dtype == torch.bool
    assert context_mask.dtype == torch.bool
    assert not torch.logical_and(loss_mask, context_mask).any()
    assert torch.equal(torch.logical_or(loss_mask, context_mask), exposure_mask)
    assert torch.isnan(batch_x[loss_mask]).all()
    assert torch.isnan(batch_x[~exposure_mask]).all()
    assert torch.isfinite(batch_x[context_mask]).all()
    assert torch.isfinite(batch_target).all()
    assert 0.005 <= prior.last_batch_metrics["recsys_holdout_target_density"] <= 0.025
    assert 0.04 <= prior.last_batch_metrics["recsys_holdout_exposure_density"] <= 0.08


def test_rec_sys_bias_residual_exposed_holdout_prior_supports_fitted_residuals():
    prior = RecSysBiasResidualExposedHoldoutPrior(
        config={
            **_base_config(),
            "num_rows_low": 28,
            "num_rows_high": 28,
            "num_cols_low": 36,
            "num_cols_high": 36,
            "recsys_holdout_target_mode": "residual_fitted_bias",
            "recsys_holdout_bias_ridge_reg": 2.0,
            "recsys_holdout_bias_ridge_iters": 5,
        },
        batch_size=1,
        device="cpu",
        entry_wise_features=False,
    )

    (batch_x, batch_target, loss_mask, exposure_mask, context_mask), _ = prior.get_batch()

    assert batch_x.shape == (1, 28, 36)
    assert loss_mask.any()
    assert context_mask.any()
    assert exposure_mask.any()
    assert torch.isfinite(batch_target).all()
    assert prior.last_batch_metrics["recsys_holdout_target_mode_id"] == 2.0
    assert "recsys_holdout_bias_only_rmse" in prior.last_batch_metrics
