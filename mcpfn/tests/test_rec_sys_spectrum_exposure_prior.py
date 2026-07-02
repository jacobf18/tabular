from __future__ import annotations

import torch

from tabimpute.prior.rec_sys_spectrum_exposure import RecSysSpectrumExposurePrior


def test_rec_sys_spectrum_exposure_prior_batch_shapes_and_masks():
    prior = RecSysSpectrumExposurePrior(
        config={
            "num_rows_low": 32,
            "num_rows_high": 40,
            "num_cols_low": 48,
            "num_cols_high": 56,
            "recsys_density_low": 0.04,
            "recsys_density_high": 0.08,
            "recsys_spectrum_rank_low": 8,
            "recsys_spectrum_rank_high": 12,
            "recsys_noise_std_low": 0.1,
            "recsys_noise_std_high": 0.2,
        },
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
    assert "recsys_density_realized" in prior.last_batch_metrics
    assert "recsys_rank" in prior.last_batch_metrics


def test_rec_sys_spectrum_exposure_prior_batch_level_shape_randomization_stacks():
    prior = RecSysSpectrumExposurePrior(
        config={
            "num_rows_low": 24,
            "num_rows_high": 36,
            "num_cols_low": 30,
            "num_cols_high": 42,
            "recsys_density_low": 0.05,
            "recsys_density_high": 0.05,
            "recsys_spectrum_rank_low": 6,
            "recsys_spectrum_rank_high": 10,
        },
        batch_size=4,
        device="cpu",
    )

    (batch_x, batch_target, _, _, _), _ = prior.get_batch()

    assert batch_x.shape[0] == 4
    assert batch_target.shape == batch_x.shape


def test_rec_sys_spectrum_exposure_prior_marginal_calibration_tracks_density():
    prior = RecSysSpectrumExposurePrior(
        config={
            "num_rows_low": 40,
            "num_rows_high": 40,
            "num_cols_low": 80,
            "num_cols_high": 80,
            "recsys_density_low": 0.05,
            "recsys_density_high": 0.05,
            "recsys_spectrum_rank_low": 8,
            "recsys_spectrum_rank_high": 8,
            "recsys_exposure_calibration": "marginal",
            "recsys_exposure_uses_full_preference": True,
            "recsys_user_activity_distribution": "lognormal",
            "recsys_item_popularity_distribution": "lognormal",
            "recsys_popularity_quality_corr_low": 0.2,
            "recsys_popularity_quality_corr_high": 0.2,
            "recsys_marginal_calibration_steps": 4,
            "recsys_marginal_binary_steps": 20,
        },
        batch_size=2,
        device="cpu",
    )

    (batch_x, batch_target, _, _, _), _ = prior.get_batch()

    assert batch_x.shape == (2, 40, 80)
    assert torch.isfinite(batch_target).all()
    realized_density = float((~torch.isnan(batch_x)).float().mean())
    assert 0.03 <= realized_density <= 0.07
    assert prior.last_batch_metrics["recsys_row_count_p99"] > prior.last_batch_metrics["recsys_row_count_median"]
    assert prior.last_batch_metrics["recsys_col_count_p99"] > prior.last_batch_metrics["recsys_col_count_median"]


def test_rec_sys_spectrum_exposure_prior_can_clip_and_quantize_ratings():
    prior = RecSysSpectrumExposurePrior(
        config={
            "num_rows_low": 32,
            "num_rows_high": 32,
            "num_cols_low": 48,
            "num_cols_high": 48,
            "recsys_density_low": 0.05,
            "recsys_density_high": 0.05,
            "recsys_spectrum_rank_low": 6,
            "recsys_spectrum_rank_high": 6,
            "recsys_global_mean": 3.5,
            "recsys_rating_clip_probability": 1.0,
            "recsys_rating_quantization_probability": 1.0,
            "recsys_rating_min": 1.0,
            "recsys_rating_max": 5.0,
        },
        batch_size=1,
        device="cpu",
    )

    (_, batch_target, _, _, _), _ = prior.get_batch()

    assert torch.isfinite(batch_target).all()
    assert prior.last_batch_metrics["recsys_rating_clipped"] == 1.0
    assert prior.last_batch_metrics["recsys_rating_quantized"] == 1.0
