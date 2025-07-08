from __future__ import annotations

from typing import Optional
from torch import nn, Tensor
import torch

from mcpfn.model.config import ModelConfig
from mcpfn.model.transformer import PerFeatureTransformer

import einops


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
        features_per_group: int = 2,
        nhead: int = 2,
        remove_duplicate_features: bool = True,
        num_buckets: int = 5000,
        encoder_path: str = 'encoder.pth',
    ):
        super().__init__()
        
        self.config = ModelConfig(
            emsize=embed_dim,
            features_per_group=features_per_group,
            max_num_classes=10, # won't be used
            nhead=nhead,
            remove_duplicate_features=remove_duplicate_features,
            num_buckets=num_buckets,
            max_num_features=max_num_features
        )
        
        self.encoder = torch.load(encoder_path, weights_only=False)
        
        self.model = PerFeatureTransformer(config = self.config, encoder=self.encoder, n_out = 5000)

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

        # This code is written from TabICL, which uses a different order of dimensions
        # than the one used in TabPFN. So, we need to rearrange the dimensions to match
        # the order used in TabPFN and then rearrange the dimensions back to the order
        # used in TabICL.
        train_size = y_train.shape[1]
        
        X = einops.rearrange(X, 'b t h -> t b h')
        y_train = einops.rearrange(y_train, 'b t -> t b')
        
        out = self.model(X, y_train, single_eval_pos = train_size)
        
        return einops.rearrange(out, 't b h -> b t h')
