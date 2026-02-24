from __future__ import annotations

from typing import Optional, Literal
from torch import nn, Tensor
import torch

from .config import ModelConfig
from .transformer import PerFeatureTransformer
from .encoders import torch_nanmean

import einops
import numpy as np
import importlib.resources as resources


class MCPFN(nn.Module):
    """A Tabular Foundation Model for Matrix Completion.

    MCPFN is a transformer-based architecture for matrix completion on tabular data.

    Parameters
    ----------

    """

    def __init__(
        self,
        embed_dim: int = 192,
        max_num_features: int = 50,
        features_per_group=2,
        nhead: int = 2,
        remove_duplicate_features: bool = True,
        num_buckets: Literal[1000, 5000] = 5000,
    ):
        super().__init__()

        self.config = ModelConfig(
            emsize=embed_dim,
            features_per_group=features_per_group,
            max_num_classes=10,  # won't be used
            nhead=nhead,
            remove_duplicate_features=remove_duplicate_features,
            num_buckets=num_buckets,
            max_num_features=max_num_features,
        )

        with resources.files("tabimpute.data").joinpath("encoder.pth").open("rb") as f:
            self.encoder = torch.load(f, weights_only=False)

        self.model = PerFeatureTransformer(
            config=self.config,
            encoder=self.encoder,
            n_out=5000,
            feature_positional_embedding="subspace",
        )

    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        d: Optional[Tensor] = None,
    ) -> Tensor:
        """Column-wise embedding -> row-wise interaction -> dataset-wise in-context learning.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - H is the number of features (columns)
            The first train_size positions contain training samples, and the remaining positions contain test samples.

        y_train : Tensor
            Training labels of shape (B, train_size) where:
             - B is the number of tables
             - train_size is the number of training samples provided for in-context learning

        d : Optional[Tensor], default=None
            The number of features per dataset. Used only in training mode.

        Returns
        -------
        Tensor
            For training mode:
              Raw logits of shape (B, T-train_size, max_classes), which will be further handled by the training code.

            For inference mode:
              Raw logits or probabilities for test samples of shape (B, T-train_size, num_classes).
        """
        X = einops.rearrange(X, "b t h -> t b h")
        y_train = einops.rearrange(y_train, "b t -> t b")

        out = self.model(
            X, y_train, single_eval_pos=X.shape[0], only_return_standard_out=False
        )

        return einops.rearrange(out, "t b h -> b t h")


class TabPFNModel(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.model = None
        self.device = device

    def forward(self, X: Tensor, y: Tensor, train_sizes: Tensor) -> Tensor:
        if self.model is None:
            from tabpfn import TabPFNRegressor

            reg = TabPFNRegressor(device=self.device)
            X_npy = X[0, :, :].cpu().numpy()
            y_npy = y[0, :].cpu().numpy()
            # fill nan values with mean of the y values
            y_npy[np.isnan(y_npy)] = np.nanmean(y_npy)
            reg.fit(X_npy, y_npy)
            self.model = reg.model_

        # Get the mask of test values
        mask = torch.zeros_like(y, dtype=torch.bool, device=self.device)
        for i in range(len(train_sizes)):
            mask[i, : train_sizes[i]] = True
        # Set values after train_size to nan for each row
        y[~mask] = torch.nan

        # Set the nan y values to mean of the y values
        batch_means = torch_nanmean(y, axis=1)
        y[~mask] = torch.repeat_interleave(batch_means, X.shape[1] - train_sizes, dim=0)

        X = einops.rearrange(X, "b t h -> t b h")
        y = einops.rearrange(y, "b t -> t b")
        out = self.model(
            X, y, single_eval_pos=X.shape[0], only_return_standard_out=False
        )["train_embeddings"]

        out = self.model.decoder_dict["standard"](out)

        return einops.rearrange(out, "t b h -> b t h")
