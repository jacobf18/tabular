from __future__ import annotations

from typing import Any, Optional, Literal
from torch import nn, Tensor
import torch

from .config import ModelConfig
from .transformer import PerFeatureTransformer
from . import encoders as local_encoders
from .encoders import torch_nanmean

import einops
import numpy as np
import importlib.resources as resources


def _register_legacy_tabpfn_encoder_aliases() -> None:
    """Expose local encoder symbols under the legacy TabPFN module path.

    The bundled ``encoder.pth`` was serialized against an older
    ``tabpfn.model.encoders`` module layout. Newer TabPFN versions re-export a
    different set of symbols there, so we register the local implementations
    before unpickling to keep checkpoint loading working.
    """

    import tabpfn.model.encoders as legacy_tabpfn_encoders

    for name, value in vars(local_encoders).items():
        if name.startswith("_") or hasattr(legacy_tabpfn_encoders, name):
            continue
        setattr(legacy_tabpfn_encoders, name, value)


def _patch_tabpfn_package_nan_handling_for_input_dtype() -> None:
    """Unpickled `encoder.pth` uses tabpfn's `NanHandlingEncoderStep`, not local encoders.

    `feature_means_` is float32 while activations can be bfloat16; align dtypes on assign.
    """
    try:
        from tabpfn.architectures.encoders.steps.nan_handling_encoder_step import (
            NanHandlingEncoderStep as TpfnNan,
        )
    except ImportError:
        return
    if getattr(TpfnNan, "_mcpfn_input_dtype_patch_applied", False):
        return

    def _transform(  # noqa: PLR0913 — mirrors upstream signature
        self: Any,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        x = state[self.in_keys[0]]
        nans_indicator: torch.Tensor | None = None
        if self.keep_nans:
            nans_indicator = (
                torch.isnan(x) * TpfnNan.nan_indicator
                + torch.logical_and(torch.isinf(x), torch.sign(x) == 1)
                * TpfnNan.inf_indicator
                + torch.logical_and(torch.isinf(x), torch.sign(x) == -1)
                * TpfnNan.neg_inf_indicator
            ).to(x.dtype)

        nan_mask = torch.logical_or(torch.isnan(x), torch.isinf(x))
        x = x.clone()
        repl = self.feature_means_.unsqueeze(0).expand_as(x).to(dtype=x.dtype)
        x[nan_mask] = repl[nan_mask]

        outputs: dict[str, torch.Tensor] = {self.out_keys[0]: x}
        if nans_indicator is not None:
            outputs[self.out_keys[1]] = nans_indicator
        return outputs

    TpfnNan._transform = _transform
    TpfnNan._mcpfn_input_dtype_patch_applied = True


_patch_tabpfn_package_nan_handling_for_input_dtype()


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

        _register_legacy_tabpfn_encoder_aliases()
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
