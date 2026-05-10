import copy
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from tabimpute.model.encoders import normalize_data
from tabimpute.prior.training_set_generation import MissingnessPrior


class DecayingSpectrumGenerator:
    """Generate full-spectrum matrices and truncated top-k targets."""

    def __init__(self, config: dict):
        self.config = config

    def _sample_int(self, low_key: str, high_key: str) -> int:
        low = int(self.config[low_key])
        high = int(self.config[high_key])
        if high < low:
            raise ValueError(f"Expected {high_key} >= {low_key}, got {high} < {low}")
        return int(np.random.randint(low, high + 1))

    def _sample_orthonormal_matrix(self, n_rows: int, rank: int) -> np.ndarray:
        gaussian = np.random.standard_normal((n_rows, rank))
        q, r = np.linalg.qr(gaussian, mode="reduced")
        diag = np.sign(np.diag(r))
        diag[diag == 0] = 1.0
        return q * diag

    def sample_spectrum(self, rank: int) -> tuple[np.ndarray, dict[str, float]]:
        alpha = float(
            np.random.uniform(
                self.config.get("spectrum_alpha_low", 0.7),
                self.config.get("spectrum_alpha_high", 1.0),
            )
        )
        scale = float(
            np.random.uniform(
                self.config.get("spectrum_scale_low", 1.0),
                self.config.get("spectrum_scale_high", 5.0),
            )
        )

        head_weight = float(
            np.random.uniform(
                self.config.get("spectrum_head_weight_low", 5.0),
                self.config.get("spectrum_head_weight_high", 10.0),
            )
        )
        head_exp_scale_low = self.config.get("spectrum_head_exp_scale_low")
        head_exp_scale_high = self.config.get("spectrum_head_exp_scale_high")
        head_handoff_scale_low = self.config.get("spectrum_head_handoff_scale_low")
        head_handoff_scale_high = self.config.get("spectrum_head_handoff_scale_high")
        if (
            head_exp_scale_low is not None
            and head_exp_scale_high is not None
            and head_handoff_scale_low is not None
            and head_handoff_scale_high is not None
        ):
            head_exp_scale = float(
                np.random.uniform(head_exp_scale_low, head_exp_scale_high)
            )
            head_handoff_scale = float(
                np.random.uniform(head_handoff_scale_low, head_handoff_scale_high)
            )
            head_decay = head_handoff_scale / max(rank, 1)
        else:
            head_decay = float(
                np.random.uniform(
                    self.config.get("spectrum_head_decay_low", 0.14),
                    self.config.get("spectrum_head_decay_high", 0.24),
                )
            )
            head_handoff_scale = max(1.0, head_decay * rank)
            head_exp_scale = max(1.0, 0.35 * head_handoff_scale)
        taper_strength = float(
            np.random.uniform(
                self.config.get("spectrum_taper_strength_low", 0.6),
                self.config.get("spectrum_taper_strength_high", 1.8),
            )
        )
        taper_power = float(
            np.random.uniform(
                self.config.get("spectrum_taper_power_low", 3.5),
                self.config.get("spectrum_taper_power_high", 6.0),
            )
        )
        head_exp_power = float(
            np.random.uniform(
                self.config.get("spectrum_head_exp_power_low", 1.8),
                self.config.get("spectrum_head_exp_power_high", 2.4),
            )
        )
        tail_amplitude = float(
            np.random.uniform(
                self.config.get("spectrum_tail_amplitude_low", 0.03),
                self.config.get("spectrum_tail_amplitude_high", 0.12),
            )
        )

        component_idx = np.arange(rank, dtype=np.float32)
        normalized_idx = component_idx / max(rank - 1, 1)
        head_component = np.exp(-np.power(component_idx / head_exp_scale, head_exp_power))
        tail_component = tail_amplitude * np.power(1.0 + component_idx, -alpha)
        taper_component = np.exp(-taper_strength * np.power(normalized_idx, taper_power))
        # Keep the very first ranks almost purely in the head regime.
        head_mix = 1.0 - np.exp(-head_weight)
        blend = head_mix * np.exp(-component_idx / head_handoff_scale)
        spectrum = (blend * head_component + (1.0 - blend) * tail_component) * taper_component
        spectrum /= spectrum[0]
        spectrum *= scale
        return spectrum, {
            "alpha": alpha,
            "scale": scale,
            "head_weight": head_weight,
            "head_decay": head_decay,
            "head_scale": head_handoff_scale,
            "head_exp_scale": head_exp_scale,
            "head_exp_power": head_exp_power,
            "head_mix": head_mix,
            "tail_amplitude": tail_amplitude,
            "taper_strength": taper_strength,
            "taper_power": taper_power,
        }

    def _resolve_rank(self, n_rows: int, n_cols: int) -> int:
        rank_low = int(
            self.config.get("spectrum_rank_low", self.config.get("latent_rank_low", 5))
        )
        rank_high = int(
            self.config.get("spectrum_rank_high", self.config.get("latent_rank_high", 30))
        )
        rank_high = min(rank_high, n_rows, n_cols)
        rank_low = min(rank_low, rank_high)
        return int(np.random.randint(rank_low, rank_high + 1))

    def _resolve_target_rank(self, spectrum: np.ndarray, noise_std: float) -> int:
        explicit_rank = self.config.get("spectrum_target_rank")
        if explicit_rank is not None:
            return max(1, min(int(explicit_rank), len(spectrum)))

        threshold_scale = float(
            self.config.get("spectrum_target_noise_floor_scale", 1.0)
        )
        threshold = threshold_scale * float(noise_std)
        effective_rank = int(np.sum(spectrum > threshold))
        return max(1, min(effective_rank, len(spectrum)))

    def generate_matrix_bundle(self) -> dict[str, torch.Tensor | np.ndarray | float | int]:
        n_rows = self._sample_int("num_rows_low", "num_rows_high")
        n_cols = self._sample_int("num_cols_low", "num_cols_high")
        rank = self._resolve_rank(n_rows, n_cols)

        u = self._sample_orthonormal_matrix(n_rows, rank).astype(np.float32)
        v = self._sample_orthonormal_matrix(n_cols, rank).astype(np.float32)
        spectrum, spectrum_params = self.sample_spectrum(rank)
        noise_std = float(
            np.random.uniform(
                self.config.get("spectrum_noise_std_low", 0.05),
                self.config.get("spectrum_noise_std_high", 0.5),
            )
        )
        target_rank = self._resolve_target_rank(spectrum=spectrum, noise_std=noise_std)

        full_signal = (u * spectrum) @ v.T
        truncated_signal = (u[:, :target_rank] * spectrum[:target_rank]) @ v[:, :target_rank].T
        noisy_observation = full_signal + noise_std * np.random.standard_normal((n_rows, n_cols))

        return {
            "full_matrix": torch.tensor(noisy_observation, dtype=torch.float32),
            "target_matrix": torch.tensor(truncated_signal, dtype=torch.float32),
            "clean_signal": torch.tensor(full_signal, dtype=torch.float32),
            "spectrum": spectrum,
            "rank": rank,
            "target_rank": target_rank,
            **spectrum_params,
            "noise_std": noise_std,
        }

    def generate_matrix_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        bundle = self.generate_matrix_bundle()
        return bundle["full_matrix"], bundle["target_matrix"]


class SpectrumMissingnessPrior(IterableDataset):
    """
    Matrix-to-matrix prior for spectrum pretraining.

    `batch_X` contains the full-spectrum matrix after missingness is applied.
    `batch_target` contains the dense top-k truncated target with no missingness.
    """

    missingness_map = MissingnessPrior.missingness_map

    def __init__(
        self,
        missingness_type: str = "mcar",
        config: Optional[dict] = None,
        batch_size: int = 10,
        device: str = "cuda",
        verbose: bool = False,
        entry_wise_features: bool = False,
    ):
        if config is None:
            config = {}
        if missingness_type not in self.missingness_map:
            raise ValueError(
                f"Unknown missingness type: {missingness_type}. "
                f"Available: {list(self.missingness_map.keys())}"
            )
        if entry_wise_features:
            raise ValueError(
                "SpectrumMissingnessPrior only supports matrix-to-matrix batches. "
                "Set entry_wise_features=False."
            )

        self.config = config
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.entry_wise_features = entry_wise_features
        self.generator = DecayingSpectrumGenerator(config)
        self.missingness_type = missingness_type
        self.missingness_inducer = self.missingness_map[missingness_type](config)

    @torch.no_grad()
    def get_batch(self, batch_size: Optional[int] = None):
        if batch_size is None:
            batch_size = self.batch_size

        batch_X_list = []
        batch_target_list = []
        train_sizes = torch.zeros(batch_size, dtype=torch.long)
        pattern_weights = []

        for step in range(batch_size):
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

            generator = DecayingSpectrumGenerator(sample_config)
            bundle = generator.generate_matrix_bundle()
            batch_X_full = bundle["full_matrix"]
            batch_target = bundle["target_matrix"]

            if not torch.isfinite(batch_X_full).all():
                raise ValueError("Spectrum generator produced non-finite values in batch_X")
            if not torch.isfinite(batch_target).all():
                raise ValueError(
                    "Spectrum generator produced non-finite values in batch_target"
                )

            batch_X_full_normalized, (full_mean, full_std) = normalize_data(
                torch.unsqueeze(batch_X_full, 1),
                return_scaling=True,
            )
            batch_X_full = torch.squeeze(batch_X_full_normalized, 1)
            batch_target = torch.squeeze(
                normalize_data(
                    torch.unsqueeze(batch_target, 1),
                    mean=full_mean,
                    std=full_std,
                ),
                1,
            )

            batch_X_missing = self.missingness_inducer._induce_missingness(batch_X_full.clone())
            batch_X_missing_normalized, (mean, std) = normalize_data(
                torch.unsqueeze(batch_X_missing, 1),
                return_scaling=True,
            )
            batch_X_missing = torch.squeeze(batch_X_missing_normalized, 1)
            batch_target = torch.squeeze(
                normalize_data(torch.unsqueeze(batch_target, 1), mean=mean, std=std),
                1,
            )

            if not torch.isfinite(batch_target).all():
                raise ValueError(
                    "Observed-value normalization produced non-finite target values"
                )

            batch_X_list.append(batch_X_missing)
            batch_target_list.append(batch_target)
            train_sizes[step] = batch_target.shape[0]
            pattern_weights.append(1.0)

        batch_X = torch.stack(batch_X_list, dim=0)
        batch_target = torch.stack(batch_target_list, dim=0)
        d = torch.tensor(batch_X_list[0].shape[1], device=self.device).repeat(batch_size)
        seq_lens = torch.tensor(
            [sample.shape[0] for sample in batch_target_list],
            device=self.device,
            dtype=torch.long,
        )
        return (
            batch_X,
            batch_target,
            d,
            seq_lens,
            train_sizes,
        ), torch.tensor(pattern_weights, device=self.device)

    def __iter__(self) -> "SpectrumMissingnessPrior":
        return self

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return self.get_batch()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plot_config = {
        "num_rows_low": 128,
        "num_rows_high": 128,
        "num_cols_low": 224,
        "num_cols_high": 224,
        "p_missing": 0.94,
        "spectrum_rank_low": 80,
        "spectrum_rank_high": 128,
        "spectrum_alpha_low": 0.30,
        "spectrum_alpha_high": 0.50,
        "spectrum_head_weight_low": 3.0,
        "spectrum_head_weight_high": 6.0,
        "spectrum_head_exp_scale_low": 2.5,
        "spectrum_head_exp_scale_high": 4.5,
        "spectrum_head_exp_power_low": 2.2,
        "spectrum_head_exp_power_high": 3.0,
        "spectrum_head_handoff_scale_low": 1.0,
        "spectrum_head_handoff_scale_high": 4.0,
        "spectrum_tail_amplitude_low": 0.25,
        "spectrum_tail_amplitude_high": 0.45,
        "spectrum_taper_strength_low": 1.2,
        "spectrum_taper_strength_high": 1.8,
        "spectrum_taper_power_low": 3.0,
        "spectrum_taper_power_high": 4.5,
        "spectrum_scale_low": 1.0,
        "spectrum_scale_high": 1.0,
        "spectrum_noise_std_low": 0.001,
        "spectrum_noise_std_high": 0.005,
        "spectrum_target_rank": 40,
    }

    generator = DecayingSpectrumGenerator(plot_config)
    fig, axes = plt.subplots(4, 5, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.ravel()

    for plot_idx, axis in enumerate(axes):
        bundle = generator.generate_matrix_bundle()
        full_matrix = bundle["full_matrix"].detach().cpu().numpy()
        singular_values = np.linalg.svd(full_matrix, compute_uv=False)
        component_idx = np.arange(1, len(singular_values) + 1)

        axis.semilogy(
            component_idx,
            singular_values,
            marker="o",
            markersize=3,
            linewidth=1,
        )
        axis.set_title(
            f"r={bundle['rank']}, k={bundle['target_rank']}",
            fontsize=9,
        )
        annotation = "\n".join(
            [
                f"alpha={bundle['alpha']:.2f}",
                f"scale={bundle['scale']:.2f}",
                f"noise={bundle['noise_std']:.2f}",
                f"head_w={bundle['head_weight']:.2f}",
                f"head_d={bundle['head_decay']:.2f}",
                f"head_s={bundle['head_scale']:.2f}",
                f"head_e={bundle['head_exp_scale']:.2f}",
                f"head_p={bundle['head_exp_power']:.2f}",
                f"head_m={bundle['head_mix']:.2f}",
                f"tail_a={bundle['tail_amplitude']:.3f}",
                f"tap_s={bundle['taper_strength']:.2f}",
                f"tap_p={bundle['taper_power']:.2f}",
            ]
        )
        axis.text(
            0.98,
            0.98,
            annotation,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "lightgray",
            },
        )
        axis.grid(True, which="both", alpha=0.3)

        if plot_idx % 5 == 0:
            axis.set_ylabel("Singular value")
        if plot_idx >= 15:
            axis.set_xlabel("Component")

    fig.suptitle("Power Plots of Full-Rank Sampled Matrices", fontsize=16)
    fig.tight_layout()
    output_path = Path(__file__).parent / "spectrum_power_plots.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Spectrum power plots saved to {output_path}")