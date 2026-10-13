"""
The 11 missingness patterns used in MissBench (paper Section 3.1.2, Table 8).

Each pattern has:
  - a CLI key (e.g. "self_mnar"),
  - a directory code used on disk (e.g. "MNAR"). The codes match the folder
    names used when the paper results were produced, so existing result folders
    can be re-plotted without recomputing,
  - a display name used in figures and tables (e.g. "Self-MNAR").
"""
from __future__ import annotations

import copy
import zlib
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Pattern:
    key: str
    code: str
    display: str
    group: str  # "MCAR", "MAR" or "MNAR" (grouping used in the paper tables)
    cls_name: str
    config: dict = field(default_factory=dict)
    # Pattern parameters reported in paper Table 8: (parameter, symbol, value)
    table_params: tuple = ()
    # Whether the pattern's missing probability is controlled by `p`
    uses_p: bool = False

    def make(self, p: float):
        """Instantiate the pattern from tabimpute.prior.training_set_generation."""
        from tabimpute.prior import training_set_generation as tsg

        config = copy.deepcopy(self.config)
        if self.uses_p:
            config["p_missing"] = p
        return getattr(tsg, self.cls_name)(config=config)


PATTERNS: dict[str, Pattern] = {
    p.key: p
    for p in [
        Pattern(
            "mcar", "MCAR", "MCAR", "MCAR", "MCARPattern",
            table_params=(("Missing probability", "$p$", "0.4"),),
            uses_p=True,
        ),
        Pattern(
            "col_mar", "MAR", "Col-MAR", "MAR", "MARPattern",
            table_params=(("Missing probability", "$p$", "0.4"),),
            uses_p=True,
        ),
        Pattern(
            "self_mnar", "MNAR", "Self-MNAR", "MNAR", "MNARPattern",
            table_params=(
                ("Coefficient set", "$\\alpha$", "$\\{-2,-1,1,2\\}$"),
                ("Target missing proportion", "$p_{missing}$", "Variable"),
            ),
            uses_p=True,
        ),
        Pattern(
            "nn_mnar", "MAR_Neural", "NN-MNAR", "MNAR", "MARNeuralNetwork",
            config={
                "mar_config": {
                    "num_layers_upper": 3,
                    "hidden_lower": 1,
                    "hidden_upper": 100,
                    "activation": "relu",
                    "seed": 42,
                    "neighbor_type": "random",
                }
            },
            table_params=(
                ("Neighborhood size", "$|N_{ij}|$", "Variable"),
                ("Network layers", "$L$", "Random"),
                ("Network depth", "$d$", "Random"),
                ("Weight initialization", "$W$", "Random"),
                ("Bias initialization", "$b$", "Random"),
            ),
            uses_p=True,
        ),
        Pattern(
            "block_mnar", "MAR_BlockNeural", "Block-MNAR", "MNAR", "MARBlockNeuralNetwork",
            config={
                "mar_block_config": {
                    "N": 100,
                    "T": 50,
                    "row_blocks": 10,
                    "col_blocks": 10,
                    "convolution_type": "mean",
                }
            },
            table_params=(
                ("Missing probability", "$p$", "0.4"),
                ("Matrix size N", "$N$", "100"),
                ("Matrix size T", "$T$", "50"),
                ("Row blocks", "$B_r$", "10"),
                ("Column blocks", "$B_c$", "10"),
                ("Convolution type", "-", "mean"),
            ),
            uses_p=True,
        ),
        Pattern(
            "pol_mnar", "MNARPolarizationPattern", "Pol-MNAR", "MNAR", "MNARPolarizationPattern",
            config={"threshold_quantile": 0.25},
            table_params=(("Threshold quantile", "$q_{thresh}$", "0.25"),),
        ),
        Pattern(
            "soft_pol_mnar", "MNARSoftPolarizationPattern", "Soft-Pol-MNAR", "MNAR",
            "MNARSoftPolarizationPattern",
            config={"soft_polarization_alpha": 2.5, "soft_polarization_epsilon": 0.05},
            table_params=(
                ("Polarization alpha", "$\\alpha$", "2.5"),
                ("Polarization epsilon", "$\\epsilon$", "0.05"),
            ),
        ),
        Pattern(
            "latent_mnar", "MNARLatentFactorPattern", "Latent-MNAR", "MNAR", "MNARLatentFactorPattern",
            config={"latent_rank_low": 1, "latent_rank_high": 5},
            table_params=(
                ("Latent rank (low)", "$k_{low}$", "1"),
                ("Latent rank (high)", "$k_{high}$", "5"),
            ),
        ),
        Pattern(
            "clust_mnar", "MNARClusterLevelPattern", "Clust-MNAR", "MNAR", "MNARClusterLevelPattern",
            config={"cluster_level_n_row_clusters": 5, "cluster_level_n_col_clusters": 4},
            table_params=(
                ("Number of row clusters", "$K_R$", "5"),
                ("Number of column clusters", "$K_C$", "4"),
            ),
        ),
        Pattern(
            "phase_mnar", "MNARTwoPhaseSubsetPattern", "Phase-MNAR", "MNAR", "MNARTwoPhaseSubsetPattern",
            config={"two_phase_cheap_fraction": 0.4},
            table_params=(("Cheap feature fraction", "$f_{cheap}$", "0.4"),),
        ),
        Pattern(
            "censor_mnar", "MNARCensoringPattern", "Censor-MNAR", "MNAR", "MNARCensoringPattern",
            config={"censor_quantile": 0.25},
            table_params=(("Censoring quantile", "$q_{censor}$", "0.25"),),
        ),
    ]
}

# Missingness level used for every pattern in the main benchmark
DEFAULT_P = 0.4
# MCAR levels for the "RMSE vs. fraction missing" figure (paper Fig. 7)
MCAR_SWEEP_P = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def pattern_seed(dataset: str, pattern: Pattern, p: float, repeat: int, base_seed: int = 0) -> int:
    """Deterministic seed for one (dataset, pattern, p, repeat) mask."""
    key = f"{base_seed}|{dataset}|{pattern.code}|{p}|{repeat}".encode()
    return zlib.crc32(key)


def induce_missingness(X: np.ndarray, pattern: Pattern, p: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Standardize X, then induce missing values (NaN) in it with `pattern`.

    As in the paper, each column of the complete data is standardized before the mask
    is drawn; methods may re-standardize the observed entries themselves.
    Returns (X_missing, X_true).
    """
    import torch
    from tabimpute.model.encoders import normalize_data

    np.random.seed(seed)
    torch.manual_seed(seed)
    X_true = normalize_data(torch.from_numpy(np.asarray(X, dtype=np.float32)), clip=False)
    X_missing = pattern.make(p)._induce_missingness(X_true.clone())
    return X_missing.numpy(), X_true.numpy()
