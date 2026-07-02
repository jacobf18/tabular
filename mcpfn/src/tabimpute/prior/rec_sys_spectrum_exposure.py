from __future__ import annotations

import copy
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from tabimpute.model.encoders import normalize_data


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


class RecSysSpectrumExposureGenerator:
    """Rec-sys-like value plus exposure generator.

    Generates a dense rating target and an MNAR observed matrix. Exposure is
    coupled to user activity, item popularity, latent residual affinity, and
    coarse user/item cluster compatibility.
    """

    def __init__(self, config: dict):
        self.config = config

    def _sample_int(self, low_key: str, high_key: str) -> int:
        low = int(self.config[low_key])
        high = int(self.config[high_key])
        if high < low:
            raise ValueError(f"Expected {high_key} >= {low_key}, got {high} < {low}")
        return int(np.random.randint(low, high + 1))

    def _sample_float(self, low_key: str, high_key: str, default: float) -> float:
        low = float(self.config.get(low_key, default))
        high = float(self.config.get(high_key, low))
        if high < low:
            raise ValueError(f"Expected {high_key} >= {low_key}, got {high} < {low}")
        return float(np.random.uniform(low, high))

    def _sample_orthonormal_matrix(self, n_rows: int, rank: int) -> np.ndarray:
        gaussian = np.random.standard_normal((n_rows, rank))
        q, r = np.linalg.qr(gaussian, mode="reduced")
        signs = np.sign(np.diag(r))
        signs[signs == 0.0] = 1.0
        return q * signs

    def _sample_spectrum(self, rank: int) -> tuple[np.ndarray, dict[str, float]]:
        alpha = self._sample_float("recsys_spectrum_alpha_low", "recsys_spectrum_alpha_high", 0.30)
        head_weight = self._sample_float(
            "recsys_spectrum_head_weight_low",
            "recsys_spectrum_head_weight_high",
            3.0,
        )
        head_exp_scale = self._sample_float(
            "recsys_spectrum_head_exp_scale_low",
            "recsys_spectrum_head_exp_scale_high",
            2.5,
        )
        head_exp_power = self._sample_float(
            "recsys_spectrum_head_exp_power_low",
            "recsys_spectrum_head_exp_power_high",
            2.0,
        )
        tail_amplitude = self._sample_float(
            "recsys_spectrum_tail_amplitude_low",
            "recsys_spectrum_tail_amplitude_high",
            0.25,
        )
        taper_strength = self._sample_float(
            "recsys_spectrum_taper_strength_low",
            "recsys_spectrum_taper_strength_high",
            1.0,
        )
        taper_power = self._sample_float(
            "recsys_spectrum_taper_power_low",
            "recsys_spectrum_taper_power_high",
            3.0,
        )
        residual_scale = self._sample_float(
            "recsys_residual_scale_low",
            "recsys_residual_scale_high",
            1.0,
        )

        component_idx = np.arange(rank, dtype=np.float64)
        normalized_idx = component_idx / max(rank - 1, 1)
        head_component = np.exp(-np.power(component_idx / head_exp_scale, head_exp_power))
        tail_component = tail_amplitude * np.power(1.0 + component_idx, -alpha)
        taper_component = np.exp(-taper_strength * np.power(normalized_idx, taper_power))
        head_mix = 1.0 - np.exp(-head_weight)
        handoff_scale = max(1.0, float(self.config.get("recsys_handoff_rank_fraction", 0.025)) * rank)
        blend = head_mix * np.exp(-component_idx / handoff_scale)
        spectrum = (blend * head_component + (1.0 - blend) * tail_component) * taper_component
        spectrum = spectrum / max(float(spectrum[0]), 1e-300)
        spectrum = spectrum * residual_scale
        return spectrum.astype(np.float64), {
            "recsys_spectrum_alpha": alpha,
            "recsys_spectrum_head_weight": head_weight,
            "recsys_spectrum_head_exp_scale": head_exp_scale,
            "recsys_spectrum_head_exp_power": head_exp_power,
            "recsys_spectrum_tail_amplitude": tail_amplitude,
            "recsys_spectrum_taper_strength": taper_strength,
            "recsys_spectrum_taper_power": taper_power,
            "recsys_residual_scale": residual_scale,
        }

    def _standardize(self, values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float64, copy=False)
        return (values - values.mean()) / max(float(values.std()), 1e-12)

    def _sample_heavy_tailed_scores(
        self,
        n_values: int,
        sigma: float,
        distribution_key: str,
    ) -> np.ndarray:
        distribution = str(self.config.get(distribution_key, "normal")).lower()
        if distribution == "lognormal":
            values = np.random.lognormal(mean=0.0, sigma=max(sigma, 1e-6), size=n_values)
            values = np.log1p(values)
            return self._standardize(values)
        if distribution == "pareto":
            alpha = max(float(self.config.get(f"{distribution_key}_alpha", 1.8)), 1.05)
            values = np.random.pareto(alpha, size=n_values)
            values = np.log1p(values)
            return self._standardize(values)
        return np.random.normal(0.0, 1.0, size=n_values)

    def _normalize_capped_counts(
        self,
        weights: np.ndarray,
        total_count: float,
        max_count: float,
        min_count: float,
    ) -> np.ndarray:
        weights = np.maximum(weights.astype(np.float64, copy=False), 1e-12)
        available = np.ones_like(weights, dtype=bool)
        counts = np.zeros_like(weights, dtype=np.float64)
        remaining_total = float(total_count)
        for _ in range(16):
            if not available.any():
                break
            candidate = weights[available] / weights[available].sum() * remaining_total
            capped = candidate > max_count
            available_idx = np.flatnonzero(available)
            if not capped.any():
                counts[available_idx] = candidate
                break
            capped_idx = available_idx[capped]
            counts[capped_idx] = max_count
            available[capped_idx] = False
            remaining_total = max(float(total_count - counts[~available].sum()), 0.0)
        counts[available] = np.maximum(counts[available], min_count)
        if counts.sum() > 0:
            counts *= float(total_count) / counts.sum()
        return np.clip(counts, min_count, max_count)

    def _sample_degree_targets(
        self,
        *,
        n_rows: int,
        n_cols: int,
        density: float,
        user_activity_sigma: float,
        item_popularity_sigma: float,
        user_activity_score: np.ndarray | None = None,
        item_popularity_score: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        total_count = float(density * n_rows * n_cols)
        if user_activity_score is None:
            user_activity_score = self._sample_heavy_tailed_scores(
                n_rows,
                user_activity_sigma,
                "recsys_user_activity_distribution",
            )
        if item_popularity_score is None:
            item_popularity_score = self._sample_heavy_tailed_scores(
                n_cols,
                item_popularity_sigma,
                "recsys_item_popularity_distribution",
            )
        row_weights = np.exp(user_activity_sigma * user_activity_score)
        col_weights = np.exp(item_popularity_sigma * item_popularity_score)
        max_row_fraction = float(self.config.get("recsys_max_row_observed_fraction", 0.80))
        max_col_fraction = float(self.config.get("recsys_max_col_observed_fraction", 0.80))
        min_expected_count = float(self.config.get("recsys_min_expected_count", 1e-3))
        row_targets = self._normalize_capped_counts(
            row_weights,
            total_count,
            max_count=max_row_fraction * n_cols,
            min_count=min_expected_count,
        )
        col_targets = self._normalize_capped_counts(
            col_weights,
            total_count,
            max_count=max_col_fraction * n_rows,
            min_count=min_expected_count,
        )
        # Keep row and column totals exactly aligned after clipping.
        common_total = min(float(row_targets.sum()), float(col_targets.sum()))
        row_targets *= common_total / max(float(row_targets.sum()), 1e-12)
        col_targets *= common_total / max(float(col_targets.sum()), 1e-12)
        row_targets = np.clip(row_targets, min_expected_count, n_cols - min_expected_count)
        col_targets = np.clip(col_targets, min_expected_count, n_rows - min_expected_count)
        return row_targets, col_targets

    def _solve_axis_bias(
        self,
        base_logits: np.ndarray,
        target_counts: np.ndarray,
        axis: int,
    ) -> np.ndarray:
        low = np.full(target_counts.shape, -80.0, dtype=np.float64)
        high = np.full(target_counts.shape, 80.0, dtype=np.float64)
        for _ in range(int(self.config.get("recsys_marginal_binary_steps", 32))):
            mid = 0.5 * (low + high)
            if axis == 1:
                expected = _sigmoid(base_logits + mid[:, None]).sum(axis=1)
            else:
                expected = _sigmoid(base_logits + mid[None, :]).sum(axis=0)
            too_low = expected < target_counts
            low[too_low] = mid[too_low]
            high[~too_low] = mid[~too_low]
        return 0.5 * (low + high)

    def _calibrate_marginal_biases(
        self,
        interaction_logits: np.ndarray,
        row_targets: np.ndarray,
        col_targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        row_bias = np.zeros(interaction_logits.shape[0], dtype=np.float64)
        col_bias = np.zeros(interaction_logits.shape[1], dtype=np.float64)
        for _ in range(int(self.config.get("recsys_marginal_calibration_steps", 10))):
            row_bias = self._solve_axis_bias(
                interaction_logits + col_bias[None, :],
                row_targets,
                axis=1,
            )
            col_bias = self._solve_axis_bias(
                interaction_logits + row_bias[:, None],
                col_targets,
                axis=0,
            )
        return row_bias, col_bias

    def _resolve_rank(self, n_rows: int, n_cols: int) -> int:
        rank_low = int(self.config.get("recsys_spectrum_rank_low", 80))
        rank_high = int(self.config.get("recsys_spectrum_rank_high", 128))
        rank_high = min(rank_high, n_rows, n_cols)
        rank_low = min(rank_low, rank_high)
        return int(np.random.randint(rank_low, rank_high + 1))

    def _calibrate_density_bias(
        self,
        logits_without_bias: np.ndarray,
        target_density: float,
    ) -> float:
        low = -80.0
        high = 80.0
        for _ in range(int(self.config.get("recsys_density_calibration_steps", 80))):
            mid = 0.5 * (low + high)
            density = float(_sigmoid(logits_without_bias + mid).mean())
            if density < target_density:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)

    def generate_matrix_bundle(self) -> dict[str, torch.Tensor | float | int]:
        n_rows = self._sample_int("num_rows_low", "num_rows_high")
        n_cols = self._sample_int("num_cols_low", "num_cols_high")
        density = self._sample_float("recsys_density_low", "recsys_density_high", 0.05)
        if not 0.0 < density < 1.0:
            raise ValueError(f"rec-sys density must be in (0, 1), got {density}")

        rank = self._resolve_rank(n_rows, n_cols)
        user_factors = self._sample_orthonormal_matrix(n_rows, rank)
        item_factors = self._sample_orthonormal_matrix(n_cols, rank)
        spectrum, spectrum_params = self._sample_spectrum(rank)
        residual = (user_factors * spectrum) @ item_factors.T
        residual = residual - residual.mean()
        residual = residual / max(float(residual.std()), 1e-12)

        user_bias_std = self._sample_float("recsys_user_bias_std_low", "recsys_user_bias_std_high", 0.30)
        item_bias_std = self._sample_float("recsys_item_bias_std_low", "recsys_item_bias_std_high", 0.50)
        noise_std = self._sample_float("recsys_noise_std_low", "recsys_noise_std_high", 0.20)
        global_mean = float(self.config.get("recsys_global_mean", 0.0))
        user_bias = np.random.normal(0.0, user_bias_std, size=n_rows)
        item_bias = np.random.normal(0.0, item_bias_std, size=n_cols)
        clean_preference = global_mean + user_bias[:, None] + item_bias[None, :] + residual
        dense_target = (
            clean_preference
            + noise_std * np.random.standard_normal((n_rows, n_cols))
        )

        quantization_probability = float(
            self.config.get(
                "recsys_rating_quantization_probability",
                1.0 if bool(self.config.get("recsys_rating_quantization", False)) else 0.0,
            )
        )
        clip_probability = float(self.config.get("recsys_rating_clip_probability", 0.0))
        quantize_ratings = bool(self.config.get("recsys_rating_quantization", False)) or (
            np.random.random() < quantization_probability
        )
        clip_ratings = bool(self.config.get("recsys_rating_clip", False)) or (
            quantize_ratings or np.random.random() < clip_probability
        )
        if clip_ratings:
            rating_min = float(self.config.get("recsys_rating_min", 1.0))
            rating_max = float(self.config.get("recsys_rating_max", 5.0))
            dense_target = np.clip(dense_target, rating_min, rating_max)
            if quantize_ratings:
                dense_target = np.round(dense_target * 2.0) / 2.0

        user_activity_sigma = self._sample_float(
            "recsys_user_activity_sigma_low",
            "recsys_user_activity_sigma_high",
            1.25,
        )
        item_popularity_sigma = self._sample_float(
            "recsys_item_popularity_sigma_low",
            "recsys_item_popularity_sigma_high",
            2.0,
        )
        affinity_strength = self._sample_float(
            "recsys_affinity_strength_low",
            "recsys_affinity_strength_high",
            0.50,
        )
        cluster_bonus_std = self._sample_float(
            "recsys_cluster_bonus_std_low",
            "recsys_cluster_bonus_std_high",
            0.50,
        )
        n_user_clusters = int(self.config.get("recsys_n_user_clusters", 8))
        n_item_clusters = int(self.config.get("recsys_n_item_clusters", 12))

        user_activity_score = self._sample_heavy_tailed_scores(
            n_rows,
            user_activity_sigma,
            "recsys_user_activity_distribution",
        )
        item_popularity_score = self._sample_heavy_tailed_scores(
            n_cols,
            item_popularity_sigma,
            "recsys_item_popularity_distribution",
        )
        popularity_quality_corr = self._sample_float(
            "recsys_popularity_quality_corr_low",
            "recsys_popularity_quality_corr_high",
            0.0,
        )
        popularity_quality_corr = float(np.clip(popularity_quality_corr, -0.95, 0.95))
        if popularity_quality_corr != 0.0:
            item_quality_score = self._standardize(item_bias)
            item_popularity_score = (
                popularity_quality_corr * item_quality_score
                + np.sqrt(max(1.0 - popularity_quality_corr**2, 0.0)) * item_popularity_score
            )
            item_popularity_score = self._standardize(item_popularity_score)
        user_activity = user_activity_sigma * user_activity_score
        item_popularity = item_popularity_sigma * item_popularity_score
        user_cluster = np.random.randint(0, n_user_clusters, size=n_rows)
        item_cluster = np.random.randint(0, n_item_clusters, size=n_cols)
        cluster_matrix = np.random.normal(
            0.0,
            cluster_bonus_std,
            size=(n_user_clusters, n_item_clusters),
        )
        cluster_bonus = cluster_matrix[user_cluster[:, None], item_cluster[None, :]]
        exposure_preference = residual
        if bool(self.config.get("recsys_exposure_uses_full_preference", False)):
            item_quality_weight = float(self.config.get("recsys_exposure_item_quality_weight", 0.50))
            user_bias_weight = float(self.config.get("recsys_exposure_user_bias_weight", 0.10))
            exposure_preference = (
                residual
                + item_quality_weight * self._standardize(item_bias)[None, :]
                + user_bias_weight * self._standardize(user_bias)[:, None]
            )
        preference_temperature = max(float(self.config.get("recsys_preference_temperature", 1.0)), 1e-6)
        exposure_preference = np.tanh(exposure_preference / preference_temperature)

        exposure_calibration = str(self.config.get("recsys_exposure_calibration", "global")).lower()
        if exposure_calibration == "marginal":
            row_targets, col_targets = self._sample_degree_targets(
                n_rows=n_rows,
                n_cols=n_cols,
                density=density,
                user_activity_sigma=user_activity_sigma,
                item_popularity_sigma=item_popularity_sigma,
                user_activity_score=user_activity_score,
                item_popularity_score=item_popularity_score,
            )
            interaction_logits = affinity_strength * exposure_preference + cluster_bonus
            row_bias, col_bias = self._calibrate_marginal_biases(
                interaction_logits,
                row_targets,
                col_targets,
            )
            logits = row_bias[:, None] + col_bias[None, :] + interaction_logits
            density_bias = 0.0
        else:
            row_targets = None
            col_targets = None
            logits_without_bias = (
                user_activity[:, None]
                + item_popularity[None, :]
                + affinity_strength * exposure_preference
                + cluster_bonus
            )
            density_bias = self._calibrate_density_bias(logits_without_bias, density)
            logits = logits_without_bias + density_bias
        p_obs = _sigmoid(logits)
        observed_mask = np.random.random((n_rows, n_cols)) < p_obs
        observed_matrix = dense_target.astype(np.float32, copy=True)
        observed_matrix[~observed_mask] = np.nan
        row_counts = observed_mask.sum(axis=1).astype(np.float64)
        col_counts = observed_mask.sum(axis=0).astype(np.float64)
        observed_values = dense_target[observed_mask]
        unobserved_values = dense_target[~observed_mask]

        result = {
            "observed_matrix": torch.tensor(observed_matrix, dtype=torch.float32),
            "target_matrix": torch.tensor(dense_target, dtype=torch.float32),
            "observed_mask": torch.tensor(observed_mask, dtype=torch.bool),
            "rank": rank,
            "density_target": density,
            "density_realized": float(observed_mask.mean()),
            "density_bias": float(density_bias),
            "noise_std": noise_std,
            "user_activity_sigma": user_activity_sigma,
            "item_popularity_sigma": item_popularity_sigma,
            "affinity_strength": affinity_strength,
            "cluster_bonus_std": cluster_bonus_std,
            "popularity_quality_corr": popularity_quality_corr,
            "row_count_median": float(np.median(row_counts)),
            "row_count_p99": float(np.percentile(row_counts, 99)),
            "col_count_median": float(np.median(col_counts)),
            "col_count_p99": float(np.percentile(col_counts, 99)),
            "observed_value_mean": float(observed_values.mean()) if observed_values.size else 0.0,
            "unobserved_value_mean": float(unobserved_values.mean()) if unobserved_values.size else 0.0,
            "target_value_mean": float(dense_target.mean()),
            "target_value_std": float(dense_target.std()),
            "rating_clipped": float(clip_ratings),
            "rating_quantized": float(quantize_ratings),
            **spectrum_params,
        }
        if row_targets is not None and col_targets is not None:
            result.update(
                {
                    "row_target_median": float(np.median(row_targets)),
                    "row_target_p99": float(np.percentile(row_targets, 99)),
                    "col_target_median": float(np.median(col_targets)),
                    "col_target_p99": float(np.percentile(col_targets, 99)),
                }
            )
        return result


class RecSysSpectrumExposurePrior(IterableDataset):
    """Matrix-to-matrix prior for recommendation-system exposure geometry."""

    def __init__(
        self,
        config: Optional[dict] = None,
        batch_size: int = 10,
        device: str = "cuda",
        verbose: bool = False,
        entry_wise_features: bool = False,
    ):
        if config is None:
            config = {}
        if entry_wise_features:
            raise ValueError(
                "RecSysSpectrumExposurePrior only supports matrix-to-matrix batches. "
                "Set entry_wise_features=False."
            )
        self.config = config
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.entry_wise_features = entry_wise_features
        self.last_batch_metrics: dict[str, float] = {}

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None):
        if batch_size is None:
            batch_size = self.batch_size

        sample_config = copy.deepcopy(self.config)
        n_rows = int(
            np.random.randint(
                int(self.config["num_rows_low"]),
                int(self.config["num_rows_high"]) + 1,
            )
        )
        n_cols = int(
            np.random.randint(
                int(self.config["num_cols_low"]),
                int(self.config["num_cols_high"]) + 1,
            )
        )
        sample_config["num_rows_low"] = n_rows
        sample_config["num_rows_high"] = n_rows
        sample_config["num_cols_low"] = n_cols
        sample_config["num_cols_high"] = n_cols

        batch_x_list = []
        batch_target_list = []
        train_sizes = torch.zeros(batch_size, dtype=torch.long)
        pattern_weights = []
        metadata_rows: list[dict[str, float]] = []

        for step in range(batch_size):
            generator = RecSysSpectrumExposureGenerator(sample_config)
            bundle = generator.generate_matrix_bundle()
            observed_matrix = bundle["observed_matrix"]
            target_matrix = bundle["target_matrix"]

            if not torch.isfinite(target_matrix).all():
                raise ValueError("Rec-sys prior produced non-finite target values")
            if not torch.isfinite(torch.nan_to_num(observed_matrix, nan=0.0)).all():
                raise ValueError("Rec-sys prior produced non-finite observed values")

            observed_normalized, (mean, std) = normalize_data(
                torch.unsqueeze(observed_matrix, 1),
                return_scaling=True,
            )
            target_normalized = normalize_data(
                torch.unsqueeze(target_matrix, 1),
                mean=mean,
                std=std,
            )
            batch_x = torch.squeeze(observed_normalized, 1)
            batch_target = torch.squeeze(target_normalized, 1)

            if not torch.isfinite(batch_target).all():
                raise ValueError("Observed-entry normalization produced non-finite target values")

            batch_x_list.append(batch_x)
            batch_target_list.append(batch_target)
            train_sizes[step] = batch_target.shape[0]
            pattern_weights.append(1.0)
            metadata_rows.append(
                {
                    key: float(value)
                    for key, value in bundle.items()
                    if isinstance(value, (float, int, np.floating, np.integer))
                }
            )

        self.last_batch_metrics = {}
        if metadata_rows:
            keys = sorted({key for row in metadata_rows for key in row.keys()})
            for key in keys:
                values = [row[key] for row in metadata_rows if key in row]
                if values:
                    metric_key = key if key.startswith("recsys_") else f"recsys_{key}"
                    self.last_batch_metrics[metric_key] = float(np.mean(values))

        batch_x = torch.stack(batch_x_list, dim=0)
        batch_target = torch.stack(batch_target_list, dim=0)
        d = torch.tensor(batch_x_list[0].shape[1], device=self.device).repeat(batch_size)
        seq_lens = torch.tensor(
            [sample.shape[0] for sample in batch_target_list],
            device=self.device,
            dtype=torch.long,
        )
        return (
            batch_x,
            batch_target,
            d,
            seq_lens,
            train_sizes,
        ), torch.tensor(pattern_weights, device=self.device)

    def __iter__(self) -> "RecSysSpectrumExposurePrior":
        return self

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.get_batch()
