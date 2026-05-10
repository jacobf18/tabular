"""Visualize examples of mask matrices for each missingness pattern used in
``tabimpute/train.py``.

Generates one figure with subplots, one per pattern, showing a mask where
black = missing and white = observed. Uses the same prior config as ``train.py``.
"""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

# Two-tone palette: light gray = observed, green = missing/unobserved
OBSERVED_COLOR = "#ECECEC"
MISSING_COLOR = "#55AFA9"
MASK_CMAP = ListedColormap([OBSERVED_COLOR, MISSING_COLOR])

# Make ``tabimpute`` importable from the repo's src tree
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_PATH = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

from tabimpute.prior.training_set_generation import MissingnessPrior

np.random.seed(42)
torch.manual_seed(42)

# Same config as in mcpfn/src/tabimpute/train.py
P_MISSING = 0.3
CONFIG = {
    "num_rows_low": 50,
    "num_rows_high": 50,
    "num_cols_low": 50,
    "num_cols_high": 50,
    "p_missing": P_MISSING,
    # MNAR configs
    "threshold_quantile": 0.4,
    "n_core_items": 5,
    "n_genres": 3,
    "n_policies": 4,
    # Latent Factor configs
    "latent_rank_low": 1,
    "latent_rank_high": 15,
    "latent_spike_p": 0.3,
    "latent_slab_sigma": 2.0,
    "apply_feature_warping_prob": 0.0,
    "apply_quantization_prob": 0.0,
    # Non-linear Factor configs
    "spline_knot_k": [3, 5, 7],
    "gp_length_scale_low": 0.3,
    "gp_length_scale_high": 2.0,
    "fourier_dim_low": 100,
    "fourier_dim_high": 501,
    # Robust-PCA configs
    "rpca_beta_a": 2,
    "rpca_beta_b": 30,
    # Soft Polarization configs
    "soft_polarization_alpha": 1.3,
    "soft_polarization_epsilon": 0.45,
    # User Cascade configs
    "cascade_n_genres": 5,
    "cascade_delta": 1.5,
    # Cluster Level configs
    "cluster_level_n_row_clusters": 8,
    "cluster_level_n_col_clusters": 8,
    "cluster_level_tau_r_std": 1.0,
    # Spatial Block configs
    "spatial_block_n_blocks": 5,
    "spatial_block_p_geom": 0.2,
    # Last few ones
    "censor_quantile": 0.25,
    "two_phase_cheap_fraction": 0.4,
    "two_phase_beta": 2.5,
    "skip_logic_p_noise": 0.9,
    "cold_start_fraction": 0.3,
    "cold_start_gamma": 0.15,
    # MAR configs
    'mar_config': {
        "num_layers_upper": 3,
        "hidden_lower": 1,
        "hidden_upper": 100,
        "activation": "relu",
        "N": 100, # Row size of X (reduced for testing)
        "T": 50, # Column size of X (reduced for testing)
        "row_neighbor_upper": 5, # Upper bound of row neighbor (reduced for testing)
        "col_neighbor_upper": 5, # Upper bound of column neighbor (reduced for testing)
        "seed": 42,
        "neighbor_type": "random"
    },
    # MAR Block configs
    'mar_block_config': {
        "N": 100,
        "T": 50,
        "row_blocks": 10,
        "col_blocks": 10,
        "convolution_type": "mean"
    },
    # MAR Sequential Bandit configs
    'mar_sequential_bandit_config': {
        'algorithm': 'epsilon_greedy',
        'pooling': False, 
        'epsilon': 0.4, 
        'epsilon_decay': 0.99, 
        'random_seed': 42
    }
}

# 11 patterns matching the OpenML benchmark plots
PATTERNS = [
    ("mcar", "MCAR"),
    ("mar", "Col-MAR"),
    ("mnar", "Self-MNAR"),
    ("mar_neural", "Neural-Network (NN)-MNAR"),
    ("mar_block_neural", "Block-MNAR"),
    ("mnar_polarization", "Pol-MNAR"),
    ("mnar_soft_polarization", "Soft-Pol-MNAR"),
    ("mnar_latent_factor", "Latent-MNAR"),
    ("mnar_cluster_level", "Clust-MNAR"),
    ("mnar_two_phase_subset", "Phase-MNAR"),
    ("mnar_censoring", "Censor-MNAR"),
]


def sample_mask(missingness_type: str, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    prior = MissingnessPrior(
        generator_type="latent_factor",
        missingness_type=missingness_type,
        config=CONFIG,
        batch_size=1,
        verbose=False,
        entry_wise_features=False,
    )
    (batch_X, _, _, _, _), _ = prior.get_batch()
    X = batch_X[0]
    if X.dim() == 3:
        X = X.squeeze(-1)
    mask = torch.isnan(X).cpu().numpy().astype(np.float32)
    return mask


def main():
    out_dir = os.path.join(_HERE, "figures")
    os.makedirs(out_dir, exist_ok=True)

    n_cols_grid = 4
    n_rows_grid = int(np.ceil(len(PATTERNS) / n_cols_grid))
    # Use a GridSpec with 2 sub-columns per panel so the last (partial) row can be centered.
    gs_cols = n_cols_grid * 2
    fig = plt.figure(figsize=(3.0 * n_cols_grid, 3.4 * n_rows_grid))
    gs = fig.add_gridspec(n_rows_grid, gs_cols)

    n_full_rows = len(PATTERNS) // n_cols_grid
    n_last_row = len(PATTERNS) - n_full_rows * n_cols_grid
    last_row_offset = (gs_cols - n_last_row * 2) // 2 if n_last_row else 0

    axes_list = []
    for i in range(len(PATTERNS)):
        row = i // n_cols_grid
        col_in_row = i % n_cols_grid
        if row < n_full_rows:
            c_start = col_in_row * 2
        else:
            c_start = last_row_offset + col_in_row * 2
        axes_list.append(fig.add_subplot(gs[row, c_start:c_start + 2]))

    for i, ((mtype, display), ax) in enumerate(zip(PATTERNS, axes_list)):
        mask = sample_mask(mtype, seed=42 + i)
        ax.imshow(mask, cmap=MASK_CMAP, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        if mtype == "mar_neural":
            ax.set_title(display, fontsize=16)
        else:
            ax.set_title(display, fontsize=18)
        # ax.set_xlabel(f"cols ({mask.shape[1]})", fontsize=10)
        # ax.set_ylabel(f"rows ({mask.shape[0]})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#888888")
            spine.set_linewidth(0.6)

    # fig.suptitle(
    #     f"Missingness mask examples (p_missing = {P_MISSING}, latent_factor generator)",
    #     fontsize=16,
    #     y=1.0,
    # )
    fig.tight_layout()

    out_pdf = os.path.join(out_dir, "missingness_pattern_masks.pdf")
    # out_png = os.path.join(out_dir, "missingness_pattern_masks.png")
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    # fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")
    # print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
