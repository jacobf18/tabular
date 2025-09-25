"""
The module offers a flexible framework for creating diverse, realistic tabular datasets
with controlled properties, which can be used for training and evaluating in-context
learning models. Key features include:

- Controlled feature relationships and causal structures via multiple generation methods
- Customizable feature distributions with mixed continuous and categorical variables
- Flexible train/test splits optimized for in-context learning evaluation
- Batch generation capabilities with hierarchical parameter sharing
- Memory-efficient handling of variable-length datasets

The main class is PriorDataset, which provides an iterable interface for generating
an infinite stream of synthetic datasets with diverse characteristics.
"""

from __future__ import annotations

import os
import sys
import math
import warnings
from typing import Dict, Tuple, Union, Optional, Any
from abc import abstractmethod

import numpy as np
import joblib

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import IterableDataset

from .training_set_generation import MissingnessPrior

from .prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP
from .training_set_generation import ACTIVATION_FUNCTIONS

from .base_prior import Prior
from .scm_prior import SCMPrior

from .utils import DisablePrinting


warnings.filterwarnings(
    "ignore",
    message=".*The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)


class DummyPrior(Prior):
    """This class creates purely random data. This is useful for testing and debugging
    without the computational overhead of SCM-based generation.

    Parameters
    ----------
    batch_size : int, default=256
        Number of datasets to generate

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    device : str, default="cpu"
        Computation device
    """

    def __init__(
        self,
        batch_size: int = 256,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        device: str = "cpu",
    ):
        super().__init__(
            batch_size=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            min_train_size=min_train_size,
            max_train_size=max_train_size,
        )
        self.device = device

    @torch.no_grad()
    def get_batch(
        self, batch_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generates a batch of random datasets for testing purposes.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override, if None, uses self.batch_size

        Returns
        -------
        X : Tensor
            Features tensor of shape (batch_size, seq_len, max_features).
            Contains random Gaussian values for all features.

        y : Tensor
            Labels tensor of shape (batch_size, seq_len).
            Contains randomly assigned class labels.

        d : Tensor
            Number of features per dataset of shape (batch_size,).
            Always set to max_features for DummyPrior.

        seq_lens : Tensor
            Sequence length for each dataset of shape (batch_size,).
            All datasets share the same sequence length.

        train_sizes : Tensor
            Position for train/test split for each dataset of shape (batch_size,).
            All datasets share the same split position.
        """

        batch_size = batch_size or self.batch_size
        seq_len = self.sample_seq_len(
            self.min_seq_len, self.max_seq_len, log=self.log_seq_len
        )
        train_size = self.sample_train_size(
            self.min_train_size, self.max_train_size, seq_len
        )

        X = torch.randn(batch_size, seq_len, self.max_features, device=self.device)

        num_classes = np.random.randint(2, self.max_classes + 1)
        y = torch.randint(0, num_classes, (batch_size, seq_len), device=self.device)

        d = torch.full((batch_size,), self.max_features, device=self.device)
        seq_lens = torch.full((batch_size,), seq_len, device=self.device)
        train_sizes = torch.full((batch_size,), train_size, device=self.device)

        return X, y, d, seq_lens, train_sizes


def create_train_test_sets(
    X: Tensor, X_full: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create train and test sets from a matrix with missing values.

    Args:
        X (Tensor): Matrix with missing values.
        X_full (Tensor | None, optional): Full matrix. If provided, the function will use it to create the test target values.
        Defaults to None.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]: Train and test sets.
    """
    # Get missing indices in X
    missing_indices = torch.where(torch.isnan(X))

    non_missing_indices = torch.where(~torch.isnan(X))

    train_X = torch.zeros((len(non_missing_indices[0]), X.shape[0] + X.shape[1] - 2))
    train_y = torch.zeros(len(non_missing_indices[0]))
    test_X = torch.zeros((len(missing_indices[0]), X.shape[0] + X.shape[1] - 2))
    test_y = torch.zeros(len(missing_indices[0]))

    for k, (i, j) in enumerate(zip(non_missing_indices[0], non_missing_indices[1])):
        # Get row without j-th column
        row = torch.cat((X[i, :j], X[i, j + 1 :]))
        # Get column without i-th row
        col = torch.cat((X[:i, j], X[i + 1 :, j]))

        # Create train set
        train_X[k, :] = torch.cat((row, col))
        train_y[k] = X[i, j]

    for k, (i, j) in enumerate(zip(missing_indices[0], missing_indices[1])):
        # Get row without j-th column
        row = torch.cat((X[i, :j], X[i, j + 1 :]))
        # Get column without i-th row
        col = torch.cat((X[:i, j], X[i + 1 :, j]))

        # Create train set
        test_X[k, :] = torch.cat((row, col))
        if X_full is not None:
            test_y[k] = X_full[i, j]
        else:
            test_y[k] = torch.nan

    return train_X, train_y, test_X, test_y


class MissingDataPrior(DummyPrior):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def induce_missingness(self, X: Tensor) -> Tensor:
        """Induce missingness on a matrix.

        Args:
            X (Tensor): Matrix to induce missingness on.

        Returns:
        """
        pass

    # @torch.no_grad()
    # def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor]:
    #     """
    #     Generates a single valid dataset based on the provided parameters.

    #     Parameters
    #     ----------
    #     params : dict
    #         Hyperparameters for generating this specific dataset, including seq_len,
    #         train_size, num_features, num_classes, prior_type, device, etc.

    #     Returns
    #     -------
    #     tuple
    #         (X, y, d) where:
    #         - X: Features tensor of shape (seq_len, max_features)
    #         - y: Labels tensor of shape (seq_len,)
    #         - d: Number of active features after filtering (scalar Tensor)
    #     """
    #     X, _, d = super().generate_dataset(params)
    #     # X_missing = self.induce_mcar(X, self.mcar_prob_missing)
    #     X_missing = self.induce_missingness(X)

    #     train_X, train_y, test_X, test_y = create_train_test_sets(X_missing, X_full=X)

    #     return torch.cat((train_X, test_X)), torch.cat((train_y, test_y)), d

    @torch.no_grad()
    def get_batch(
        self, batch_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generates a batch of random datasets for testing purposes.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override, if None, uses self.batch_size

        Returns
        -------
        X : Tensor
            Features tensor of shape (batch_size, seq_len, max_features).
            Contains random Gaussian values for all features.

        y : Tensor
            Labels tensor of shape (batch_size, seq_len).
            Contains randomly assigned class labels.

        d : Tensor
            Number of features per dataset of shape (batch_size,).
            Always set to max_features for DummyPrior.

        seq_lens : Tensor
            Sequence length for each dataset of shape (batch_size,).
            All datasets share the same sequence length.

        train_sizes : Tensor
            Position for train/test split for each dataset of shape (batch_size,).
            All datasets share the same split position.
        """
        if batch_size is None:
            batch_size = self.batch_size

        X_batches, _, d, _, _ = super().get_batch(batch_size)
        train_sizes = torch.zeros(batch_size)
        seq_lens = torch.zeros(batch_size)

        X_list = []
        y_list = []

        for i in range(X_batches.shape[0]):
            X = X_batches[i, :]
            X_missing = self.induce_missingness(
                X
            )  # needs to always produce the same number of missing values

            train_X, train_y, test_X, test_y = create_train_test_sets(
                X_missing, X_full=X
            )

            X_full_missing = torch.cat((train_X, test_X))
            y_full = torch.cat((train_y, test_y))

            X_list.append(X_full_missing)
            y_list.append(y_full)

            train_sizes[i] = train_X.shape[0]
            seq_lens[i] = X_full_missing.shape[0]

        X_out = torch.stack(X_list, dim=0)
        y_out = torch.stack(y_list, dim=0)

        return X_out, y_out, d, seq_lens, train_sizes


class MCARPrior(MissingDataPrior):
    """
    Generates synthetic datasets using Structural Causal Models (SCM) and add MCAR missingness on top.

    The data generation process follows a hierarchical structure:
    1. Generate a list of parameters for each dataset, respecting group/subgroup sharing.
    2. Process the parameter list to generate datasets, applying necessary transformations and checks.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics

    batch_size_per_subgp : int, default=None
        Number of datasets per subgroup, with more similar causal structures
        If None, defaults to batch_size_per_gp

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    seq_len_per_gp : bool = False
        If True, sample sequence length per group, allowing variable-sized datasets

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets

    prior_type : str, default="mlp_scm"
        Type of prior: 'mlp_scm' (default), 'tree_scm', or 'mix_scm'
        'mix_scm' randomly selects between 'mlp_scm' and 'tree_scm' based on probabilities.

    fixed_hp : dict, default=DEFAULT_FIXED_HP
        Fixed structural configuration parameters

    sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Parameters sampled during generation

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors).

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda')
    """

    def __init__(
        self,
        num_missing: int = 10,
        mcar_prob_missing: float = 0.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_missing = num_missing
        self.mcar_prob_missing = mcar_prob_missing

    def induce_mcar_set_number_of_missing(self, matrix: Tensor, n_missing: int):
        """Induce MCAR missingness on a matrix and set the number of missing values to a fixed number.

        Args:
            matrix (Tensor): Matrix to induce MCAR missingness on.
            n_missing (int): Number of missing values to induce.

        Returns:
            Tensor: Matrix with MCAR missingness induced.
        """
        mat = matrix.clone()
        n_missing_indices = torch.randperm(mat.numel())[:n_missing]
        mat.view(-1)[n_missing_indices] = torch.nan
        return mat

    def induce_mcar(self, matrix: Tensor, p: float = 0.5):
        """Induce MCAR missingness on a matrix.

        Args:
            matrix (Tensor): Matrix to induce MCAR missingness on.
            p (float, optional): Probability of missingness. Defaults to 0.5.

        Returns:
            Tensor: Matrix with MCAR missingness induced.
        """
        mat = matrix.clone()
        mask = torch.rand(*mat.shape) < p
        mat[mask] = torch.nan
        return mat

    def induce_missingness(self, X: Tensor) -> Tensor:
        """Induce missingness on a matrix.

        Args:
            X (Tensor): Matrix to induce missingness on.

        Returns:
            Tensor: Matrix with MCAR missingness induced.
        """
        return self.induce_mcar_set_number_of_missing(X, self.num_missing)


class PriorDataset(IterableDataset):
    """
    Main dataset class that provides an infinite iterator over synthetic tabular datasets.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics

    batch_size_per_subgp : int, default=None
        Number of datasets per subgroup, with more similar causal structures
        If None, defaults to batch_size_per_gp

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution

    seq_len_per_gp : bool = False
        If True, sample sequence length per group, allowing variable-sized datasets

    min_train_size : int|float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int|float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets

    prior_type : str, default="mlp_scm"
        Type of prior: 'mlp_scm' (default), 'tree_scm', 'mix_scm', or 'dummy'

        1. SCM-based: Structural causal models with complex feature relationships
         - 'mlp_scm': MLP-based causal models
         - 'tree_scm': Tree-based causal models
         - 'mix_scm': Probabilistic mix of the above models

        2. Dummy: Randomly generated datasets for debugging

    scm_fixed_hp : dict, default=DEFAULT_FIXED_HP
        Fixed parameters for SCM-based priors

    scm_sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Parameters sampled during generation

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors)

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda')
    """

    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        min_train_size: Union[int, float] = 0.1,
        max_train_size: Union[int, float] = 0.9,
        replay_small: bool = False,
        prior_type: str = "mlp_scm",
        scm_fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
        scm_sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        device: str = "cpu",
        num_missing: int = 10,
        missingness_type: str = "mcar",
        missingness_generator_type: str = "linear_factor",
    ):
        super().__init__()
        if prior_type == "dummy":
            self.prior = DummyPrior(
                batch_size=batch_size,
                min_features=min_features,
                max_features=max_features,
                max_classes=max_classes,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                device=device,
            )
        elif prior_type in ["mlp_scm", "tree_scm", "mix_scm"]:
            self.prior = SCMPrior(
                batch_size=batch_size,
                batch_size_per_gp=batch_size_per_gp,
                batch_size_per_subgp=batch_size_per_subgp,
                min_features=min_features,
                max_features=max_features,
                max_classes=max_classes,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                seq_len_per_gp=seq_len_per_gp,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                replay_small=replay_small,
                prior_type=prior_type,
                fixed_hp=scm_fixed_hp,
                sampled_hp=scm_sampled_hp,
                n_jobs=n_jobs,
                num_threads_per_generate=num_threads_per_generate,
                device=device,
            )
        elif prior_type == "mcar":
            self.prior = MCARPrior(
                num_missing=num_missing,
                batch_size=batch_size,
                min_features=min_features,
                max_features=max_features,
                max_classes=max_classes,
                min_seq_len=min_seq_len,
                max_seq_len=max_seq_len,
                log_seq_len=log_seq_len,
                min_train_size=min_train_size,
                max_train_size=max_train_size,
                device=device,
            )
        elif prior_type == "missing":
            config = {
                "num_rows_low": min_seq_len,
                "num_rows_high": max_seq_len,
                "num_cols_low": min_features,
                "num_cols_high": max_features,
                "p_missing": 0.4,
                # SCM configs
                "num_nodes_low": 60,
                "num_nodes_high": 80,
                "graph_generation_method": ["MLP-Dropout", "Scale-Free"],
                "root_node_noise_dist": ["Normal", "Uniform"],
                "scm_activation_functions": list(ACTIVATION_FUNCTIONS.keys()),
                "xgb_n_estimators_exp_scale": 0.5,
                "xgb_max_depth_exp_scale": 0.5,
                "apply_feature_warping_prob": 0.1,
                "apply_quantization_prob": 0.1,
                # MNAR configs
                "threshold_quantile": 0.25,
                "n_core_items": 5,
                "n_genres": 3,
                "n_policies": 4,
                # Latent Factor configs
                "latent_rank_low": 1,
                "latent_rank_high": 11,
                "latent_spike_p": 0.3,
                "latent_slab_sigma": 2.0,
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
                "soft_polarization_alpha": 2.5,
                "soft_polarization_epsilon": 0.05,
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
                "censor_quantile": 0.1,
                "two_phase_cheap_fraction": 0.4,
                "two_phase_beta": 2.5,
                "skip_logic_p_noise": 0.9,
                "cold_start_fraction": 0.3,
                "cold_start_gamma": 0.15,
            }
            self.prior = MissingnessPrior(
                generator_type=missingness_generator_type,
                missingness_type=missingness_type,
                config=config,
                batch_size=batch_size,
                verbose=True,
            )
        else:
            raise ValueError(
                f"Unknown prior type '{prior_type}'. Available options: 'mlp_scm', 'tree_scm', 'mix_scm', or 'dummy'."
            )

        self.batch_size = batch_size
        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.min_features = min_features
        self.max_features = max_features
        self.max_classes = max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len
        self.seq_len_per_gp = seq_len_per_gp
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.device = device
        self.prior_type = prior_type

    def get_batch(
        self, batch_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generate a new batch of datasets.

        Parameters
        ----------
        batch_size : int, optional
            If provided, overrides the default batch size for this call

        Returns
        -------
        X : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is (batch_size, seq_len, max_features).
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random Gaussian values of (batch_size, seq_len, max_features).

        X : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is (batch_size, seq_len).
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random class labels of (batch_size, seq_len).

        d : Tensor
            Number of active features per dataset of shape (batch_size,).

        seq_lens : Tensor
            Sequence length for each dataset of shape (batch_size,).

        train_sizes : Tensor
            Position for train/test split for each dataset of shape (batch_size,).
        """
        return self.prior.get_batch(batch_size)

    def __iter__(self) -> "PriorDataset":
        """
        Returns an iterator that yields batches indefinitely.

        Returns
        -------
        self
            Returns self as an iterator
        """
        return self

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Returns the next batch from the iterator. Since this is an infinite
        iterator, it never raises StopIteration and instead continuously generates
        new synthetic data batches.
        """
        with DisablePrinting():
            return self.get_batch()

    def __repr__(self) -> str:
        """
        Returns a string representation of the dataset.

        Provides a detailed view of the dataset configuration for debugging
        and logging purposes.

        Returns
        -------
        str
            A formatted string with dataset parameters
        """
        return (
            f"PriorDataset(\n"
            f"  prior_type: {self.prior_type}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  batch_size_per_gp: {self.batch_size_per_gp}\n"
            f"  features: {self.min_features} - {self.max_features}\n"
            f"  max classes: {self.max_classes}\n"
            f"  seq_len: {self.min_seq_len or 'None'} - {self.max_seq_len}\n"
            f"  sequence length varies across groups: {self.seq_len_per_gp}\n"
            f"  train_size: {self.min_train_size} - {self.max_train_size}\n"
            f"  device: {self.device}\n"
            f")"
        )
