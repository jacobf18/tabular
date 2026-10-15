from __future__ import annotations

import importlib.resources as resources
from typing import Any, Callable

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torch.nn.modules.transformer import LayerNorm, Linear, MultiheadAttention

from tabimpute.interface import ImputePFN
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.positional import SinusoidalColumnEmbedding, SinusoidalRowEmbedding


class _LegacyTabImputeModel(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        num_outputs: int,
        num_cls: int,
    ):
        super().__init__()
        self.feature_encoder = _LegacyFeatureEncoder(embedding_size, num_cls)
        self.transformer_blocks = nn.ModuleList(
            [
                _LegacyTransformerEncoderLayer(
                    embedding_size, num_attention_heads, mlp_hidden_size
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = _LegacyDecoder(embedding_size, mlp_hidden_size, num_outputs)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.feature_encoder(src)
        for block in self.transformer_blocks:
            src = block(src)

        num_cls_rows = self.feature_encoder.last_num_cls_rows
        num_cls_cols = self.feature_encoder.last_num_cls_cols
        return self.decoder(src[:, num_cls_rows:, num_cls_cols:, :])


class _LegacyFeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int, num_cls: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_cls = num_cls
        self.last_num_cls_rows = None
        self.last_num_cls_cols = None
        self.observed_linear_layer = nn.Sequential(
            nn.Linear(1, embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, embedding_size),
        )
        self.row_embedding = SinusoidalRowEmbedding(embedding_size, damping_factor=0.1)
        self.column_embedding = SinusoidalColumnEmbedding(
            embedding_size, damping_factor=0.1
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embedding_size))
        self.row_cls_embedding = nn.Embedding(num_cls, embedding_size)
        self.col_cls_embedding = nn.Embedding(num_cls, embedding_size)

    def _compute_num_cls(self, _: int) -> int:
        return self.num_cls

    def _get_cls_tokens(
        self,
        num_cls: int,
        embedding: nn.Embedding,
        device: torch.device,
        dtype: torch.dtype,
    ):
        indices = torch.arange(num_cls, device=device, dtype=torch.long)
        return embedding(indices).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nan_mask = torch.isnan(x)
        x = torch.where(nan_mask, torch.zeros_like(x), x)
        embedded_observed = self.observed_linear_layer(x.unsqueeze(-1))
        nan_mask_expanded = nan_mask.unsqueeze(-1).expand_as(embedded_observed)

        batch_size, rows, cols = x.shape
        mask_expanded = self.mask_token.expand(
            batch_size, rows, cols, self.embedding_size
        )
        embedded = torch.where(nan_mask_expanded, mask_expanded, embedded_observed)
        embedded = self.column_embedding(embedded) + self.row_embedding(embedded)

        num_cls_rows = self._compute_num_cls(rows)
        num_cls_cols = self._compute_num_cls(cols)
        self.last_num_cls_rows = num_cls_rows
        self.last_num_cls_cols = num_cls_cols

        device = x.device
        dtype = embedded.dtype
        row_cls_tokens = self._get_cls_tokens(
            num_cls_cols, self.row_cls_embedding, device, dtype
        )
        row_cls = row_cls_tokens.unsqueeze(0).unsqueeze(0).expand(
            batch_size, rows, num_cls_cols, self.embedding_size
        )

        col_cls_tokens = self._get_cls_tokens(
            num_cls_rows, self.col_cls_embedding, device, dtype
        )
        col_cls = col_cls_tokens.unsqueeze(0).unsqueeze(2).expand(
            batch_size, num_cls_rows, cols, self.embedding_size
        )

        row_corner = row_cls_tokens.unsqueeze(0).expand(
            num_cls_rows, num_cls_cols, self.embedding_size
        )
        col_corner = col_cls_tokens.unsqueeze(1).expand(
            num_cls_rows, num_cls_cols, self.embedding_size
        )
        corner = (row_corner + col_corner).unsqueeze(0).expand(
            batch_size, num_cls_rows, num_cls_cols, self.embedding_size
        )

        top_block = torch.cat([corner, col_cls], dim=2)
        bottom_block = torch.cat([row_cls, embedded], dim=2)
        return torch.cat([top_block, bottom_block], dim=1)


class _LegacyTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(
            embedding_size,
            nhead,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.self_attention_between_features = MultiheadAttention(
            embedding_size,
            nhead,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)
        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.gelu = nn.GELU()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape

        src_norm = self.norm1(src)
        src_flat = src_norm.reshape(batch_size * rows_size, col_size, embedding_size)
        src_att = self.self_attention_between_features(
            src_flat, src_flat, src_flat, need_weights=False
        )[0]
        src = src + src_att.reshape(batch_size, rows_size, col_size, embedding_size)

        src_norm = self.norm2(src)
        src_t = src_norm.transpose(1, 2)
        src_flat = src_t.reshape(batch_size * col_size, rows_size, embedding_size)
        src_att = self.self_attention_between_datapoints(
            src_flat, src_flat, src_flat, need_weights=False
        )[0]
        src_att = src_att.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src + src_att.transpose(2, 1)

        src_norm = self.norm3(src)
        return src + self.linear2(self.gelu(self.linear1(src_norm)))


class _LegacyDecoder(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(embedding_size, mlp_hidden_size),
                nn.GELU(),
                nn.Linear(mlp_hidden_size, mlp_hidden_size),
                nn.GELU(),
                nn.Linear(mlp_hidden_size, num_outputs),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def get_v2_model_from_huggingface() -> str:
    return hf_hub_download(repo_id="Tabimpute/TabImpute", filename="tabimputev2.ckpt")


class TabImputeV2(ImputePFN):
    """Compatibility wrapper for the original TabImpute V2 checkpoint.

    The benchmark-selection paper imports ``tabimpute.tabimpute_v2.TabImputeV2``.
    The current package interface moved on to newer entry-wise checkpoints, while
    the V2 artifact is a direct matrix-to-matrix model. This class restores the
    historical API against the current model implementations.
    """

    def __init__(
        self,
        device: str = "cpu",
        nhead: int = 2,
        preprocessors: list[Any] | None = None,
        postprocessor: Callable[..., torch.Tensor] | None = None,
        postprocessor_kwargs: dict[str, Any] | None = None,
        checkpoint_path: str | None = None,
        max_num_rows: int | None = None,
        max_num_chunks: int | None = None,
        verbose: bool = False,
    ):
        del nhead
        if checkpoint_path is None:
            checkpoint_path = get_v2_model_from_huggingface()

        self.device = device
        self.model_family = "tabimpute_v2"
        self.direct_dtype = torch.bfloat16
        self.precision_mode = "full_bfloat16"
        self.direct_prior_index = None
        model_cfg = {
            "embedding_size": 1024,
            "num_attention_heads": 32,
            "mlp_hidden_size": 1024,
            "num_layers": 12,
            "num_outputs": 5000,
            "num_cls": 12,
        }

        self.model = _LegacyTabImputeModel(**model_cfg).to(
            self.device, dtype=torch.bfloat16
        )
        self.model.eval()

        with resources.files("tabimpute.data").joinpath("borders.pt").open("rb") as f:
            self.borders = torch.load(f, map_location=self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"], strict=True)

        self.preprocessors = preprocessors
        self.postprocessor = postprocessor
        self.postprocessor_kwargs = postprocessor_kwargs or {}
        self.max_num_rows = max_num_rows
        if max_num_chunks is None:
            max_num_chunks = float("inf")
        self.max_num_chunks = max_num_chunks
        self.max_num_cols = None
        self.bar_distribution = FullSupportBarDistribution(borders=self.borders.to(self.device))
        self.verbose = verbose
        self.entry_wise_features = False
        self.prediction_stat = "median"

    def _postprocess_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.postprocessor is None:
            return logits
        return self.postprocessor(
            logits=logits,
            borders=self.borders,
            **self.postprocessor_kwargs,
        )

    def _point_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        logits = self._postprocess_logits(logits.float())
        return self.bar_distribution.median(logits=logits)

    def _run_model_direct(
        self, X_batch: torch.Tensor, X_last: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        with torch.no_grad():
            preds = self.model(X_batch)
            medians_chunks = self._point_predictions(preds)
            medians_last = None
            if X_last is not None:
                preds_last = self.model(X_last)
                medians_last = self._point_predictions(preds_last)
        return medians_chunks, medians_last
