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
from scipy.stats import loguniform
import joblib

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nested import nested_tensor
from torch.utils.data import IterableDataset

from .mlp_scm import MLPSCM
from .tree_scm import TreeSCM
from .training_set_generation import MissingnessPrior

from .hp_sampling import HpSamplerList
from .reg2cls import Reg2Cls
from .prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP


warnings.filterwarnings(
    "ignore",
    message=".*The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)


class Prior:
    """
    Abstract base class for dataset prior generators.

    Defines the interface and common functionality for different types of
    synthetic dataset generators.

    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch

    min_features : int, default=2
        Minimum number of features per dataset

    max_features : int, default=100
        Maximum number of features per dataset

    max_classes : int, default=10
        Maximum number of target classes

    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len

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

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets
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
        replay_small: bool = False,
    ):
        self.batch_size = batch_size

        assert min_features <= max_features, "Invalid feature range"
        self.min_features = min_features
        self.max_features = max_features

        self.max_classes = max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len

        self.validate_train_size_range(min_train_size, max_train_size)
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.replay_small = replay_small

    @staticmethod
    def validate_train_size_range(
        min_train_size: Union[int, float], max_train_size: Union[int, float]
    ) -> None:
        """
        Checks if the training size range is valid.

        Parameters
        ----------
        min_train_size : int|float
            Minimum training size (position or ratio)

        max_train_size : int|float
            Maximum training size (position or ratio)

        Raises
        ------
        AssertionError
            If training size range is invalid
        ValueError
            If training size types are mismatched or invalid
        """
        # Check for numeric types only
        if not isinstance(min_train_size, (int, float)) or not isinstance(
            max_train_size, (int, float)
        ):
            raise TypeError("Training sizes must be int or float")

        # Check for valid ranges based on type
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            assert (
                0 < min_train_size < max_train_size
            ), "0 < min_train_size < max_train_size"
        elif isinstance(min_train_size, float) and isinstance(max_train_size, float):
            assert (
                0 < min_train_size < max_train_size < 1
            ), "0 < min_train_size < max_train_size < 1"
        else:
            raise ValueError(
                "Both training sizes must be of the same type (int or float)"
            )

    @staticmethod
    def sample_seq_len(
        min_seq_len: Optional[int],
        max_seq_len: int,
        log: bool = False,
        replay_small: bool = False,
    ) -> int:
        """
        Selects a random sequence length within the specified range.

        This method provides flexible sampling strategies for dataset sizes, including
        occasional re-sampling of smaller sequence lengths for better training diversity.

        Parameters
        ----------
        min_seq_len : int, optional
            Minimum sequence length. If None, returns max_seq_len directly.

        max_seq_len : int
            Maximum sequence length

        log : bool, default=False
            If True, sample from a log-uniform distribution to better
            cover the range of possible sizes

        replay_small : bool, default=False
            If True, occasionally sample smaller sequence lengths with
            specific distributions to ensure model robustness on smaller datasets

        Returns
        -------
        int
            The sampled sequence length
        """
        if min_seq_len is None:
            return max_seq_len

        if log:
            seq_len = int(loguniform.rvs(min_seq_len, max_seq_len))
        else:
            seq_len = np.random.randint(min_seq_len, max_seq_len)

        if replay_small:
            p = np.random.random()
            if p < 0.05:
                return np.random.randint(200, 1000)
            elif p < 0.3:
                return int(loguniform.rvs(1000, 10000))
            else:
                return seq_len
        else:
            return seq_len

    @staticmethod
    def sample_train_size(
        min_train_size: Union[int, float],
        max_train_size: Union[int, float],
        seq_len: int,
    ) -> int:
        """
        Selects a random training size within the specified range.

        This method handles both absolute position and fractional ratio approaches
        for determining the training/test split point.

        Parameters
        ----------
        min_train_size : int|float
            Minimum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        max_train_size : int|float
            Maximum training size. If int, used as absolute position.
            If float between 0 and 1, used as ratio of sequence length.

        seq_len : int
            Total sequence length

        Returns
        -------
        int
            The sampled training size position

        Raises
        ------
        ValueError
            If training size range has incompatible types
        """
        if isinstance(min_train_size, int) and isinstance(max_train_size, int):
            train_size = np.random.randint(min_train_size, max_train_size)
        elif isinstance(min_train_size, float) and isinstance(min_train_size, float):
            train_size = np.random.uniform(min_train_size, max_train_size)
            train_size = int(seq_len * train_size)
        else:
            raise ValueError("Invalid training size range.")
        return train_size

    @staticmethod
    def adjust_max_features(seq_len: int, max_features: int) -> int:
        """
        Adjusts the maximum number of features based on the sequence length.

        This method implements an adaptive feature limit that scales inversely
        with sequence length. Longer sequences are restricted to fewer features
        to prevent memory issues and excessive computation times while still
        maintaining dataset diversity and learning difficulty.

        Parameters
        ----------
        seq_len : int
            Sequence length (number of samples)

        max_features : int
            Original maximum number of features

        Returns
        -------
        int
            Adjusted maximum number of features, ensuring computational feasibility
        """
        if seq_len <= 10240:
            return min(100, max_features)
        elif 10240 < seq_len <= 20000:
            return min(80, max_features)
        elif 20000 < seq_len <= 30000:
            return min(60, max_features)
        elif 30000 < seq_len <= 40000:
            return min(40, max_features)
        elif 40000 < seq_len <= 50000:
            return min(30, max_features)
        elif 50000 < seq_len <= 60000:
            return min(20, max_features)
        elif 60000 < seq_len <= 65000:
            return min(15, max_features)
        else:
            return 10

    @staticmethod
    def delete_unique_features(X: Tensor, d: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Removes features that have only one unique value across all samples.

        Single-value features provide no useful information for learning since they
        have zero variance. This method identifies and removes such constant features
        to improve model training efficiency and stability. The removed features are
        replaced with zero padding to maintain tensor dimensions.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape (B, T, H) where:
            - B is batch size
            - T is sequence length
            - H is feature dimensionality

        d : Tensor
            Number of features per dataset of shape (B,), indicating how many
            features are actually used in each dataset (rest is padding)

        Returns
        -------
        tuple
            (X_new, d_new) where:
            - X_new is the filtered tensor with non-informative features removed
            - d_new is the updated feature count per dataset
        """

        def filter_unique_features(xi: Tensor, di: int) -> Tuple[Tensor, Tensor]:
            """Filters features with only one unique value from a single dataset."""
            num_features = xi.shape[-1]
            # Only consider actual features (up to di, ignoring padding)
            xi = xi[:, :di]
            # Identify features with more than one unique value (informative features)
            unique_mask = [len(torch.unique(xi[:, j])) > 1 for j in range(di)]
            di_new = sum(unique_mask)
            # Create new tensor with only informative features, padding the rest
            xi_new = F.pad(
                xi[:, unique_mask],
                pad=(0, num_features - di_new),
                mode="constant",
                value=0,
            )
            return xi_new, torch.tensor(di_new, device=xi.device)

        # Process each dataset in the batch independently
        filtered_results = [filter_unique_features(xi, di) for xi, di in zip(X, d)]
        X_new, d_new = [torch.stack(res) for res in zip(*filtered_results)]

        return X_new, d_new

    @staticmethod
    def sanity_check(
        X: Tensor,
        y: Tensor,
        train_size: int,
        n_attempts: int = 10,
        min_classes: int = 2,
    ) -> bool:
        """
        Verifies that both train and test sets contain all classes.

        For in-context learning to work properly, we need both the train and test
        sets to contain examples from all classes. This method checks this condition
        and attempts to fix invalid splits by randomly permuting the data.

        Parameters
        ----------
        X : Tensor
            Input features tensor of shape (B, T, H)

        y : Tensor
            Target labels tensor of shape (B, T)

        train_size : int
            Position to split the data into train and test sets

        n_attempts : int, default=10
            Number of random permutations to try for fixing invalid splits

        min_classes : int, default=2
            Minimum number of classes required in both train and test sets

        Returns
        -------
        bool
            True if all datasets have valid splits, False otherwise
        """

        def is_valid_split(yi: Tensor) -> bool:
            """Check if a single dataset has a valid train/test split."""
            # Guard against invalid train_size
            if train_size <= 0 or train_size >= yi.shape[0]:
                return False

            # A valid split requires both train and test sets to have the same classes
            # and at least min_classes different classes must be present
            unique_tr = torch.unique(yi[:train_size])
            unique_te = torch.unique(yi[train_size:])
            return (
                set(unique_tr.tolist()) == set(unique_te.tolist())
                and len(unique_tr) >= min_classes
            )

        # Check each dataset in the batch
        for i, (xi, yi) in enumerate(zip(X, y)):
            if is_valid_split(yi):
                continue

            # If the dataset has an invalid split, try to fix it with random permutations
            succeeded = False
            for _ in range(n_attempts):
                # Generate a random permutation of the samples
                perm = torch.randperm(yi.shape[0])
                yi_perm = yi[perm]
                xi_perm = xi[perm]
                # Check if the permutation results in a valid split
                if is_valid_split(yi_perm):
                    X[i], y[i] = xi_perm, yi_perm
                    succeeded = True
                    break

            if not succeeded:  # No valid split was found after all attempts
                return False

        return True


class SCMPrior(Prior):
    """
    Generates synthetic datasets using Structural Causal Models (SCM).

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
        fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
        sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
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
            replay_small=replay_small,
        )

        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.seq_len_per_gp = seq_len_per_gp
        self.prior_type = prior_type
        self.fixed_hp = fixed_hp
        self.sampled_hp = sampled_hp
        self.n_jobs = n_jobs
        self.num_threads_per_generate = num_threads_per_generate
        self.device = device

    def hp_sampling(self) -> Dict[str, Any]:
        """
        Sample hyperparameters for dataset generation.

        Returns
        -------
        dict
            Dictionary with sampled hyperparameters merged with fixed ones
        """
        hp_sampler = HpSamplerList(self.sampled_hp, device=self.device)
        return hp_sampler.sample()

    @torch.no_grad()
    def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generates a single valid dataset based on the provided parameters.

        Parameters
        ----------
        params : dict
            Hyperparameters for generating this specific dataset, including seq_len,
            train_size, num_features, num_classes, prior_type, device, etc.

        Returns
        -------
        tuple
            (X, y, d) where:
            - X: Features tensor of shape (seq_len, max_features)
            - y: Labels tensor of shape (seq_len,)
            - d: Number of active features after filtering (scalar Tensor)
        """

        if params["prior_type"] == "mlp_scm":
            prior_cls = MLPSCM
        elif params["prior_type"] == "tree_scm":
            prior_cls = TreeSCM
        else:
            raise ValueError(f"Unknown prior type {params['prior_type']}")

        while True:
            X, y = prior_cls(**params)()
            X, y = Reg2Cls(params)(X, y)

            # Add batch dim for single dataset to be compatible with delete_unique_features and sanity_check
            X, y = X.unsqueeze(0), y.unsqueeze(0)
            d = torch.tensor(
                [params["num_features"]], device=self.device, dtype=torch.long
            )

            # Only keep valid datasets with sufficient features and balanced classes
            X, d = self.delete_unique_features(X, d)
            if (d > 0).all() and self.sanity_check(X, y, params["train_size"]):
                return X.squeeze(0), y.squeeze(0), d.squeeze(0)

    @torch.no_grad()
    def get_batch(
        self, batch_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Generates a batch of datasets by first creating a parameter list and then processing it.

        Parameters
        ----------
        batch_size : int, optional
            Batch size override. If None, uses self.batch_size

        Returns
        -------
        X : Tensor or NestedTensor
            Features tensor. If seq_len_per_gp=False, shape is (batch_size, seq_len, max_features).
            If seq_len_per_gp=True, returns a NestedTensor.

        y : Tensor or NestedTensor
            Labels tensor. If seq_len_per_gp=False, shape is (batch_size, seq_len).
            If seq_len_per_gp=True, returns a NestedTensor.

        d : Tensor
            Number of active features per dataset after filtering, shape (batch_size,)

        seq_lens : Tensor
            Sequence length for each dataset, shape (batch_size,)

        train_sizes : Tensor
            Position for train/test split for each dataset, shape (batch_size,)
        """
        batch_size = batch_size or self.batch_size

        # Calculate number of groups and subgroups
        size_per_gp = min(self.batch_size_per_gp, batch_size)
        num_gps = math.ceil(batch_size / size_per_gp)

        size_per_subgp = min(self.batch_size_per_subgp, size_per_gp)

        # Generate parameters list for all datasets, preserving group and subgroup structure
        param_list = []
        global_seq_len = None
        global_train_size = None

        # Determine global seq_len/train_size if not per-group
        if not self.seq_len_per_gp:
            global_seq_len = self.sample_seq_len(
                self.min_seq_len,
                self.max_seq_len,
                log=self.log_seq_len,
                replay_small=self.replay_small,
            )
            global_train_size = self.sample_train_size(
                self.min_train_size, self.max_train_size, global_seq_len
            )

        # Generate parameters for each group
        for gp_idx in range(num_gps):
            # Determine actual size for this group (may be smaller for the last group)
            actual_gp_size = min(size_per_gp, batch_size - gp_idx * size_per_gp)
            if actual_gp_size <= 0:
                break

            group_sampled_hp = self.hp_sampling()
            # If per-group, sample seq_len and train_size for this group. Otherwise, use global ones
            if self.seq_len_per_gp:
                gp_seq_len = self.sample_seq_len(
                    self.min_seq_len,
                    self.max_seq_len,
                    log=self.log_seq_len,
                    replay_small=self.replay_small,
                )
                gp_train_size = self.sample_train_size(
                    self.min_train_size, self.max_train_size, gp_seq_len
                )
                # Adjust max features based on seq_len for this group
                gp_max_features = self.adjust_max_features(
                    gp_seq_len, self.max_features
                )
            else:
                gp_seq_len = global_seq_len
                gp_train_size = global_train_size
                gp_max_features = self.max_features

            # Calculate number of subgroups for this group
            num_subgps_in_gp = math.ceil(actual_gp_size / size_per_subgp)

            # Generate parameters for each subgroup
            for subgp_idx in range(num_subgps_in_gp):
                # Determine actual size for this subgroup
                actual_subgp_size = min(
                    size_per_subgp, actual_gp_size - subgp_idx * size_per_subgp
                )
                if actual_subgp_size <= 0:
                    break

                # Subgroups share prior type, number of features, and sampled HPs
                subgp_prior_type = self.get_prior()
                subgp_num_features = round(
                    np.random.uniform(self.min_features, gp_max_features)
                )
                subgp_sampled_hp = {
                    k: v() if callable(v) else v for k, v in group_sampled_hp.items()
                }

                # Generate parameters for each dataset in this subgroup
                for ds_idx in range(actual_subgp_size):
                    # Each dataset has its own number of classes
                    if np.random.random() > 0.5:
                        ds_num_classes = np.random.randint(2, self.max_classes + 1)
                    else:
                        ds_num_classes = 2

                    # Create parameters dictionary for this dataset
                    params = {
                        **self.fixed_hp,  # Fixed HPs
                        "seq_len": gp_seq_len,
                        "train_size": gp_train_size,
                        # If per-gp setting, use adjusted max features for this group because we use nested tensors
                        # If not per-gp setting, use global max features to fix size for concatenation
                        "max_features": (
                            gp_max_features
                            if self.seq_len_per_gp
                            else self.max_features
                        ),
                        **subgp_sampled_hp,  # sampled HPs for this group
                        "prior_type": subgp_prior_type,
                        "num_features": subgp_num_features,
                        "num_classes": ds_num_classes,
                        "device": self.device,
                    }
                    param_list.append(params)

        # Use joblib to generate datasets in parallel.
        # Note: the 'loky' backend does not support nested parallelism during DDP, whereas the 'threading' backend does.
        # However, 'threading' does not respect `inner_max_num_threads`.
        # Therefore, we stick with the 'loky' backend for parallelism, but this requires generating
        # the prior datasets separately from the training process and loading them from disk,
        # rather than generating them on-the-fly.
        if self.n_jobs > 1 and self.device == "cpu":
            with joblib.parallel_config(
                n_jobs=self.n_jobs,
                backend="loky",
                inner_max_num_threads=self.num_threads_per_generate,
            ):
                results = joblib.Parallel()(
                    joblib.delayed(self.generate_dataset)(params)
                    for params in param_list
                )
        else:
            results = [self.generate_dataset(params) for params in param_list]

        X_list, y_list, d_list = zip(*results)

        # Combine Results
        if self.seq_len_per_gp:
            # Use nested tensors for variable sequence lengths
            X = nested_tensor([x.to(self.device) for x in X_list], device=self.device)
            y = nested_tensor([y.to(self.device) for y in y_list], device=self.device)
        else:
            # Stack into regular tensors for fixed sequence length
            X = torch.stack(X_list).to(self.device)  # (B, T, H)
            y = torch.stack(y_list).to(self.device)  # (B, T)

        # Metadata (always regular tensors)
        d = torch.stack(d_list).to(
            self.device
        )  # Actual number of features after filtering out constant ones
        seq_lens = torch.tensor(
            [params["seq_len"] for params in param_list],
            device=self.device,
            dtype=torch.long,
        )
        train_sizes = torch.tensor(
            [params["train_size"] for params in param_list],
            device=self.device,
            dtype=torch.long,
        )

        return X, y, d, seq_lens, train_sizes

    def get_prior(self) -> str:
        """
        Determine which prior type to use for generation.

        For 'mix_scm' prior type, randomly selects between available priors
        based on configured probabilities.

        Returns
        -------
        str
            The selected prior type name
        """
        if self.prior_type == "mix_scm":
            return np.random.choice(
                ["mlp_scm", "tree_scm"], p=self.fixed_hp.get("mix_probas", [0.7, 0.3])
            )
        else:
            return self.prior_type


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
                'num_rows_low': 10, 'num_rows_high': 10, 'num_cols_low': 5, 'num_cols_high': 5.5,
                'p_missing': 0.4,
                # SCM configs
                'num_nodes_low': 60, 'num_nodes_high': 80, 'graph_generation_method': ['MLP-Dropout', 'Scale-Free'],
                'root_node_noise_dist': ['Normal', 'Uniform'], 'scm_activation_functions': list(ACTIVATION_FUNCTIONS.keys()),
                'xgb_n_estimators_exp_scale': 0.5, 'xgb_max_depth_exp_scale': 0.5,
                'apply_feature_warping_prob': 0.1, 'apply_quantization_prob': 0.1,
                # MNAR configs
                'threshold_quantile': 0.25, 'n_core_items': 5, 'n_genres': 3, 'n_policies': 4,
                # Latent Factor configs
                'latent_rank_low': 1, 'latent_rank_high': 11, 'latent_spike_p': 0.3, 'latent_slab_sigma': 2.0,
                # Non-linear Factor configs
                'spline_knot_k': [3, 5, 7], 'gp_length_scale_low': 0.3, 'gp_length_scale_high': 2.0,
                'fourier_dim_low': 100, 'fourier_dim_high': 501,
                # Robust-PCA configs
                'rpca_beta_a': 2, 'rpca_beta_b': 30,
                # Soft Polarization configs
                'soft_polarization_alpha': 2.5, 'soft_polarization_epsilon': 0.05,
                # User Cascade configs
                'cascade_n_genres': 5, 'cascade_delta': 1.5,
                # Cluster Level configs
                'cluster_level_n_row_clusters': 8, 'cluster_level_n_col_clusters': 8, 'cluster_level_tau_r_std': 1.0,
                # Spatial Block configs
                'spatial_block_n_blocks': 5, 'spatial_block_p_geom': 0.2,
                # Last few ones
                'censor_quantile': 0.1, 'two_phase_cheap_fraction': 0.4, 'two_phase_beta': 2.5,
                'skip_logic_p_noise': 0.9, 'cold_start_fraction': 0.3, 'cold_start_gamma': 0.15,
            }
            self.prior = MissingnessPrior(
                generator_type="latent_factor",
                missingness_type="mcar_fixed",
                config=config,
                num_missing = 10,
                batch_size=batch_size,
                verbose=True
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


class DisablePrinting:
    """Context manager to temporarily suppress printed output."""

    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.original_stdout
