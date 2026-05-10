#!/usr/bin/env python3
"""Plot singular value spectra for UCI benchmark datasets in a single figure."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

MCAR_PATTERN = "MCAR"
MCAR_MISSINGNESS = 0.4
METRIC_METHOD_FILES = {
    "TabImpute": "tabimpute_new_stable_round2_trial_2.npy",
    # "TabImpute_50x50_ckpt30000": "tabimpute-new-stable_v2-rank_1_30-50x50.npy",
    "SoftImpute": "softimpute.npy",
    "HyperImpute": "hyperimpute.npy",
    "tabimpute-new-stable_v2-rank_10_10-50x50": "tabimpute-new-stable_v2-rank_10_10-50x50.npy",
    "tabimpute-new-stable_v2-rank_15_15_from_10_10-50x50": "tabimpute-new-stable_v2-rank_15_15_from_10_10-50x50.npy",
    "tabimpute-new-stable_v2-spectrum-decay-topk_p_0.2-50x50": "tabimpute-new-stable_v2-spectrum-decay-topk_p_0.2-50x50.npy",
}
METRIC_METHOD_COLORS = {
    "TabImpute": "#2f88a8",
    # "TabImpute_50x50_ckpt30000": "#2f88a8",
    "SoftImpute": "#c97b35",
    "HyperImpute": "#6c757d",
    "tabimpute-new-stable_v2-rank_10_10-50x50": "#2f88a8",
    "tabimpute-new-stable_v2-rank_15_15_from_10_10-50x50": "#2f88a8",
    "tabimpute-new-stable_v2-spectrum-decay-topk_p_0.2-50x50": "#2f88a8",
}


def load_dataset_matrix(dataset_path: Path) -> np.ndarray:
    """Load a saved UCI benchmark dataframe as a column-standardized dense matrix."""
    df = pd.read_pickle(dataset_path)
    matrix = df.to_numpy(dtype=np.float64, copy=True)

    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError(f"Expected a non-empty 2D matrix, got shape={matrix.shape!r}")

    if not np.isfinite(matrix).all():
        col_means = np.nanmean(matrix, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        missing_rows, missing_cols = np.where(~np.isfinite(matrix))
        matrix[missing_rows, missing_cols] = col_means[missing_cols]

    col_means = np.mean(matrix, axis=0)
    col_stds = np.std(matrix, axis=0)
    col_stds = np.where(np.isfinite(col_stds) & (col_stds > 1e-12), col_stds, 1.0)
    return (matrix - col_means) / col_stds


def compute_singular_values(matrix: np.ndarray) -> np.ndarray:
    """Return the singular values in descending order."""
    return np.linalg.svd(matrix, compute_uv=False, full_matrices=False)


def compute_cumulative_spectral_power(singular_values: np.ndarray) -> np.ndarray:
    """Return cumulative spectral power explained by the singular values."""
    squared_singular_values = singular_values ** 2
    total_power = float(np.sum(squared_singular_values))
    if total_power <= 0:
        return np.zeros_like(singular_values, dtype=np.float64)
    return np.cumsum(squared_singular_values / total_power)


def compute_observed_normalized_column_rmse(
    true_values: np.ndarray,
    imputed_values: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Average per-column RMSE after standardizing by observed column statistics."""
    scores = []

    for col_idx in range(true_values.shape[1]):
        missing_mask = mask[:, col_idx]
        observed_mask = ~missing_mask
        if not np.any(missing_mask) or not np.any(observed_mask):
            continue

        observed_column = true_values[observed_mask, col_idx]
        observed_mean = float(np.mean(observed_column))
        observed_std = float(np.std(observed_column))

        if observed_std <= 1e-12:
            standardized_true = true_values[missing_mask, col_idx] - observed_mean
            standardized_pred = imputed_values[missing_mask, col_idx] - observed_mean
            column_rmse = float(np.sqrt(np.mean((standardized_true - standardized_pred) ** 2)))
            if column_rmse <= 1e-12:
                scores.append(0.0)
            continue

        standardized_true = (true_values[missing_mask, col_idx] - observed_mean) / observed_std
        standardized_pred = (imputed_values[missing_mask, col_idx] - observed_mean) / observed_std
        scores.append(float(np.sqrt(np.mean((standardized_true - standardized_pred) ** 2))))

    if not scores:
        return np.nan

    return float(np.mean(scores))


def collect_dataset_metric(
    dataset_dir: Path,
    pattern: str = MCAR_PATTERN,
    missingness: float = MCAR_MISSINGNESS,
) -> dict[str, float]:
    """Average per-repeat normalized column RMSE for each imputation method."""
    repeat_dir = dataset_dir / pattern / f"missingness-{missingness}"
    repeat_paths = sorted(repeat_dir.glob("repeat-*"))
    metric_by_method = {method_name: np.nan for method_name in METRIC_METHOD_FILES}

    if not repeat_paths:
        return metric_by_method

    for method_name, filename in METRIC_METHOD_FILES.items():
        values = []
        for repeat_path in repeat_paths:
            true_path = repeat_path / "true.npy"
            missing_path = repeat_path / "missing.npy"
            prediction_path = repeat_path / filename
            if not (true_path.exists() and missing_path.exists() and prediction_path.exists()):
                continue

            true_values = np.load(true_path)
            missing_values = np.load(missing_path)
            predictions = np.load(prediction_path)
            mask = np.isnan(missing_values)

            if not np.any(mask):
                continue

            values.append(compute_observed_normalized_column_rmse(true_values, predictions, mask))

        if values:
            metric_by_method[method_name] = float(np.mean(values))

    return metric_by_method


def save_combined_spectrum_plot(
    dataset_spectra: list[tuple[str, np.ndarray, tuple[int, int]]],
    output_path: Path,
) -> None:
    """Save all dataset spectra in a single 3-row grid."""
    num_rows = 3
    num_cols = int(np.ceil(len(dataset_spectra) / num_rows))
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(4.5 * num_cols, 3.6 * num_rows),
        squeeze=False,
    )

    for ax, (dataset_name, singular_values, shape) in zip(axes.flat, dataset_spectra):
        ranks = np.arange(1, len(singular_values) + 1)
        ax.plot(ranks, singular_values, linewidth=1.5)
        ax.set_title(f"{dataset_name} ({shape[0]} x {shape[1]})", fontsize=10)
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular Value")
        ax.set_xlim(1, len(singular_values))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5, alpha=0.6)

        positive_singular_values = singular_values[singular_values > 0]
        if positive_singular_values.size:
            ax.set_yscale("log")
            ax.set_ylim(
                bottom=positive_singular_values.min() * 0.8,
                top=positive_singular_values.max() * 1.05,
            )

    for ax in axes.flat[len(dataset_spectra) :]:
        ax.axis("off")

    fig.suptitle("UCI Dataset Singular Value Spectra", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_combined_spectral_power_plot(
    dataset_spectral_power: list[tuple[str, np.ndarray, tuple[int, int]]],
    output_path: Path,
) -> None:
    """Save cumulative spectral power plots in the same subplot layout."""
    num_rows = 3
    num_cols = int(np.ceil(len(dataset_spectral_power) / num_rows))
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(4.5 * num_cols, 3.6 * num_rows),
        squeeze=False,
    )

    for ax, (dataset_name, cumulative_power, shape) in zip(axes.flat, dataset_spectral_power):
        ranks = np.arange(1, len(cumulative_power) + 1)
        ax.plot(ranks, cumulative_power, linewidth=1.5)
        ax.axhline(0.8, color="gray", linestyle="--", linewidth=1)
        ax.axhline(0.9, color="gray", linestyle="--", linewidth=1)
        ax.axhline(0.95, color="gray", linestyle="--", linewidth=1)
        ax.axhline(0.99, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{dataset_name} ({shape[0]} x {shape[1]})", fontsize=10)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Cumulative Power")
        ax.set_xlim(1, len(cumulative_power))
        ax.set_ylim(0, 1.01)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5, alpha=0.6)

    for ax in axes.flat[len(dataset_spectral_power) :]:
        ax.axis("off")

    fig.suptitle("UCI Dataset Cumulative Spectral Power", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_combined_metric_plot(
    dataset_metrics: list[tuple[str, dict[str, float], tuple[int, int]]],
    output_path: Path,
) -> None:
    """Save per-dataset MCAR 0.4 normalized column RMSE bars in the same subplot layout."""
    num_rows = 3
    num_cols = int(np.ceil(len(dataset_metrics) / num_rows))
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(4.5 * num_cols, 3.6 * num_rows),
        squeeze=False,
    )

    method_names = list(METRIC_METHOD_FILES)
    bar_positions = np.arange(len(method_names))
    bar_colors = [METRIC_METHOD_COLORS[method_name] for method_name in method_names]

    for ax, (dataset_name, metric_by_method, shape) in zip(axes.flat, dataset_metrics):
        values = [metric_by_method[method_name] for method_name in method_names]
        ax.bar(bar_positions, values, color=bar_colors, width=0.7)
        ax.set_title(f"{dataset_name} ({shape[0]} x {shape[1]})", fontsize=10)
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(method_names, rotation=20, ha="right")
        ax.set_ylabel("Observed-Std Column RMSE")
        ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    for ax in axes.flat[len(dataset_metrics) :]:
        ax.axis("off")

    fig.suptitle(
        f"UCI Dataset {MCAR_PATTERN} {MCAR_MISSINGNESS} Observed-Std Column RMSE",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def iter_dataset_pickles(base_path: Path) -> list[Path]:
    """Discover saved UCI dataset pickles."""
    return sorted(base_path.glob("*/dataset.pkl"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot singular value spectra for UCI datasets in a single figure."
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets" / "uci",
        help="Directory containing UCI dataset folders with dataset.pkl files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "uci_singular_value_spectra.png",
        help="PNG path for the combined UCI singular value spectrum figure.",
    )
    parser.add_argument(
        "--power-output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "uci_cumulative_spectral_power.png",
        help="PNG path for the combined UCI cumulative spectral power figure.",
    )
    parser.add_argument(
        "--rmse-output-path",
        type=Path,
        default=Path(__file__).resolve().parent / "figures" / "uci_mcar_0.4_normalized_column_rmse_bars.png",
        help="PNG path for the combined MCAR 0.4 normalized column RMSE bar figure.",
    )
    args = parser.parse_args()

    dataset_paths = iter_dataset_pickles(args.base_path)
    if not dataset_paths:
        raise FileNotFoundError(f"No dataset.pkl files found under {args.base_path}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.power_output_path.parent.mkdir(parents=True, exist_ok=True)
    args.rmse_output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_spectra = []
    dataset_spectral_power = []
    dataset_metrics = []
    for dataset_path in dataset_paths:
        dataset_name = dataset_path.parent.name
        dataset_dir = dataset_path.parent
        matrix = load_dataset_matrix(dataset_path)
        singular_values = compute_singular_values(matrix)
        cumulative_power = compute_cumulative_spectral_power(singular_values)
        dataset_spectra.append((dataset_name, singular_values, matrix.shape))
        dataset_spectral_power.append((dataset_name, cumulative_power, matrix.shape))
        dataset_metrics.append((dataset_name, collect_dataset_metric(dataset_dir), matrix.shape))

    save_combined_spectrum_plot(dataset_spectra, args.output_path)
    save_combined_spectral_power_plot(dataset_spectral_power, args.power_output_path)
    save_combined_metric_plot(dataset_metrics, args.rmse_output_path)
    print(f"Saved combined spectrum figure to {args.output_path}")
    print(f"Saved combined cumulative spectral power figure to {args.power_output_path}")
    print(f"Saved combined normalized column RMSE figure to {args.rmse_output_path}")


if __name__ == "__main__":
    main()
