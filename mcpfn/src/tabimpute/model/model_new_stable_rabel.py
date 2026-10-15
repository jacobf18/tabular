from __future__ import annotations

import torch
from torch import nn

from tabimpute.model.model_new_stable import FeatureEncoderStable
from tabimpute.model.model_new_stable_memory_efficient import (
    TabImputeModelMemoryEfficient,
)
from tabimpute.model.rabel_encoder import RaBELValueEncoder


class FeatureEncoderPureRaBEL(FeatureEncoderStable):
    """TabImpute cell encoder that replaces the scalar MLP with RaBEL."""

    def __init__(
        self,
        embedding_size: int,
        num_cls: int,
        use_absolute_positional_embeddings: bool = False,
        positional_damping_factor: float = 0.1,
        embedding_dropout: float = 0.0,
        rabel_num_kernels: int = 64,
        rabel_sigma: float = 1.0,
        rabel_exponent_digits: int = 1,
        rabel_exponent_embedding_size: int = 32,
    ):
        super().__init__(
            embedding_size=embedding_size,
            num_cls=num_cls,
            use_absolute_positional_embeddings=use_absolute_positional_embeddings,
            positional_damping_factor=positional_damping_factor,
            embedding_dropout=embedding_dropout,
        )
        del self.observed_linear_layer
        self.rabel_encoder = RaBELValueEncoder(
            embedding_size=embedding_size,
            num_kernels=rabel_num_kernels,
            sigma=rabel_sigma,
            exponent_digits=rabel_exponent_digits,
            exponent_embedding_size=rabel_exponent_embedding_size,
        )

    def embed_cells(self, x: torch.Tensor) -> torch.Tensor:
        nan_mask = torch.isnan(x)
        x_clean = torch.where(nan_mask, torch.zeros_like(x), x)
        embedded_observed = self.rabel_encoder(x_clean.unsqueeze(-1))
        embedded = torch.where(
            nan_mask.unsqueeze(-1).expand_as(embedded_observed),
            self.mask_token.expand_as(embedded_observed),
            embedded_observed,
        )
        return embedded + self.missingness_embedding(nan_mask.to(torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embed_cells(x)
        batch_size, rows, cols, _ = embedded.shape

        if self.row_embedding is not None and self.column_embedding is not None:
            embedded = self.row_embedding(embedded)
            embedded = self.column_embedding(embedded)

        k_rows = self._compute_num_cls(rows)
        k_cols = self._compute_num_cls(cols)
        self.last_num_cls_rows = k_rows
        self.last_num_cls_cols = k_cols
        device = x.device
        dtype = embedded.dtype

        row_cls_tokens = self._get_cls_tokens(
            k_cols,
            self.row_cls_embedding,
            device,
            dtype,
        )
        row_cls = row_cls_tokens.unsqueeze(0).unsqueeze(0).expand(
            batch_size,
            rows,
            k_cols,
            self.embedding_size,
        )
        col_cls_tokens = self._get_cls_tokens(
            k_rows,
            self.col_cls_embedding,
            device,
            dtype,
        )
        col_cls = col_cls_tokens.unsqueeze(0).unsqueeze(2).expand(
            batch_size,
            k_rows,
            cols,
            self.embedding_size,
        )
        row_corner = row_cls_tokens.unsqueeze(0).expand(
            k_rows,
            k_cols,
            self.embedding_size,
        )
        col_corner = col_cls_tokens.unsqueeze(1).expand(
            k_rows,
            k_cols,
            self.embedding_size,
        )
        corner = (row_corner + col_corner).unsqueeze(0).expand(
            batch_size,
            k_rows,
            k_cols,
            self.embedding_size,
        )
        top_block = torch.cat([corner, col_cls], dim=2)
        bottom_block = torch.cat([row_cls, embedded], dim=2)
        return self.embedding_dropout(torch.cat([top_block, bottom_block], dim=1))


class TabImputeModelPureRaBEL(TabImputeModelMemoryEfficient):
    """Memory-efficient direct TabImpute with pure RaBEL scalar tokenization."""

    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        num_outputs: int,
        num_cls: int,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        rope_fraction: float = 1.0,
        use_absolute_positional_embeddings: bool = False,
        positional_damping_factor: float = 0.1,
        attention_dropout: float = 0.05,
        ffn_dropout: float = 0.10,
        drop_path_rate: float = 0.05,
        residual_scale_init: float = 0.10,
        rms_norm_eps: float = 1e-6,
        embedding_dropout: float = 0.0,
        attention_batch_chunk_size: int | None = None,
        use_activation_checkpointing: bool = True,
        checkpoint_use_reentrant: bool = False,
        rabel_num_kernels: int = 64,
        rabel_sigma: float = 1.0,
        rabel_exponent_digits: int = 1,
        rabel_exponent_embedding_size: int = 32,
    ):
        super().__init__(
            embedding_size=embedding_size,
            num_attention_heads=num_attention_heads,
            mlp_hidden_size=mlp_hidden_size,
            num_layers=num_layers,
            num_outputs=num_outputs,
            num_cls=num_cls,
            use_rope=use_rope,
            rope_base=rope_base,
            rope_fraction=rope_fraction,
            use_absolute_positional_embeddings=use_absolute_positional_embeddings,
            positional_damping_factor=positional_damping_factor,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            drop_path_rate=drop_path_rate,
            residual_scale_init=residual_scale_init,
            rms_norm_eps=rms_norm_eps,
            embedding_dropout=embedding_dropout,
            attention_batch_chunk_size=attention_batch_chunk_size,
            use_activation_checkpointing=use_activation_checkpointing,
            checkpoint_use_reentrant=checkpoint_use_reentrant,
        )
        self.feature_encoder = FeatureEncoderPureRaBEL(
            embedding_size=embedding_size,
            num_cls=num_cls,
            use_absolute_positional_embeddings=use_absolute_positional_embeddings,
            positional_damping_factor=positional_damping_factor,
            embedding_dropout=embedding_dropout,
            rabel_num_kernels=rabel_num_kernels,
            rabel_sigma=rabel_sigma,
            rabel_exponent_digits=rabel_exponent_digits,
            rabel_exponent_embedding_size=rabel_exponent_embedding_size,
        )


class TabImputeModel(TabImputeModelPureRaBEL):
    """Compatibility alias."""


__all__ = [
    "FeatureEncoderPureRaBEL",
    "TabImputeModel",
    "TabImputeModelPureRaBEL",
]
