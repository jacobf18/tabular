from .base_prior import Prior
import torch
from typing import Optional, Union, Tuple, Dict, Any
import math
import numpy as np
import joblib
from torch import Tensor
from .hp_sampling import HpSamplerList
from .mlp_scm import MLPSCM
from .tree_scm import TreeSCM
from .prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP
from .reg2cls import Reg2Cls
from torch.nested import nested_tensor


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

        X, y = prior_cls(**params)()
        # X, y = Reg2Cls(params)(X, y)

        # Add batch dim for single dataset to be compatible with delete_unique_features and sanity_check
        X, y = X.unsqueeze(0), y.unsqueeze(0)
        d = torch.tensor([params["num_features"]], device=self.device, dtype=torch.long)

        # Only keep valid datasets with sufficient features and balanced classes
        X, d = self.delete_unique_features(X, d)
        # if (d > 0).all() and self.sanity_check(X, y, params["train_size"]):
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
                # subgp_num_features = round(
                #     np.random.uniform(self.min_features, gp_max_features)
                # )
                subgp_num_features = gp_max_features
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
