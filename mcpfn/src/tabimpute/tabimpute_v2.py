from __future__ import annotations

import importlib.resources as resources
import numpy as np
import torch
from typing import Any

from tabimpute.interface import ImputePFN
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model import TabImputeModel


class TabImputeV2(ImputePFN):
    """TabImpute V2 using the non-entry-wise-features model."""

    def __init__(
        self,
        device: str = "cpu",
        nhead: int = 2,
        preprocessors: list[Any] = None,
        checkpoint_path: str = None,
        max_num_rows: int = None,
        max_num_chunks: int = None,
        verbose: bool = False,
    ):
        if checkpoint_path is None:
            raise ValueError(
                "`checkpoint_path` is required for TabImputeV2 "
                "(the model used when entry_wise_features=False)."
            )

        self.device = device

        num_attention_heads = 32
        embedding_size = 32 * num_attention_heads
        mlp_hidden_size = 1024
        num_cls = 12
        num_layers = 12
        self.model = TabImputeModel(
            embedding_size=embedding_size,
            num_attention_heads=num_attention_heads,
            mlp_hidden_size=mlp_hidden_size,
            num_layers=num_layers,
            num_cls=num_cls,
            num_outputs=5000,
        ).to(self.device).to(torch.bfloat16)
        self.model.eval()

        with resources.files("tabimpute.data").joinpath("borders.pt").open("rb") as f:
            self.borders = torch.load(f, map_location=self.device)

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(checkpoint["model"], strict=False)

        self.preprocessors = preprocessors
        self.max_num_rows = max_num_rows
        if max_num_chunks is None:
            max_num_chunks = float("inf")
        self.max_num_chunks = max_num_chunks
        borders = self.borders.to(self.device)
        self.bar_distribution = FullSupportBarDistribution(borders=borders)
        self.verbose = verbose

    def get_imputation(
        self, X_normalized: np.ndarray, num_repeats: int = 1
    ) -> np.ndarray:
        if self.max_num_rows is not None:
            raise ValueError("`max_num_rows` is not supported in TabImputeV2.")
        return self._get_imputation_single(X_normalized, num_repeats=num_repeats)

    def _get_imputation_single(
        self, X_normalized: np.ndarray, num_repeats: int = 1
    ) -> np.ndarray:
        if num_repeats > 1:
            raise ValueError("`num_repeats > 1` is not supported in TabImputeV2.")

        X_normalized_tensor = torch.from_numpy(X_normalized).to(self.device)
        with torch.no_grad():
            X_normalized_tensor = X_normalized_tensor.unsqueeze(0).to(torch.bfloat16)
            preds = self.model(X_normalized_tensor)
            medians = self.bar_distribution.median(logits=preds).squeeze(0)

        X_imputed = medians.cpu().detach().numpy()
        missing_mask = np.isnan(X_normalized)
        X_normalized[missing_mask] = X_imputed[missing_mask]
        return X_normalized, X_imputed
