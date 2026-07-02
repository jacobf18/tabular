from __future__ import annotations

import torch
from torch import nn

from tabimpute.model.model_compressed_cross import (
    ColumnCompressionModel,
    CompressedCrossTabImputeModel,
    RowCompressionModel,
)
from tabimpute.model.model_new_stable import (
    FeatureEncoderStable,
    TabImputeModelStable,
)
from tabimpute.model.rabel_encoder import RaBELValueEncoder


class FeatureEncoderRaBEL(FeatureEncoderStable):
    """Stable TabImpute feature encoder with a residual RaBEL value branch."""

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
        rabel_initial_scale: float = 0.05,
    ):
        super().__init__(
            embedding_size=embedding_size,
            num_cls=num_cls,
            use_absolute_positional_embeddings=use_absolute_positional_embeddings,
            positional_damping_factor=positional_damping_factor,
            embedding_dropout=embedding_dropout,
        )
        self.rabel_encoder = RaBELValueEncoder(
            embedding_size,
            num_kernels=rabel_num_kernels,
            sigma=rabel_sigma,
            exponent_digits=rabel_exponent_digits,
            exponent_embedding_size=rabel_exponent_embedding_size,
        )
        self.rabel_scale = nn.Parameter(torch.tensor(float(rabel_initial_scale)))

    def embed_cells(self, x: torch.Tensor) -> torch.Tensor:
        nan_mask = torch.isnan(x)
        x_clean = torch.where(nan_mask, torch.zeros_like(x), x)
        scalar_values = x_clean.unsqueeze(-1)
        legacy = self.observed_linear_layer(scalar_values)
        rabel = self.rabel_encoder(scalar_values)
        embedded_observed = legacy + self.rabel_scale.to(legacy.dtype) * rabel
        mask_expanded = self.mask_token.expand_as(embedded_observed)
        embedded = torch.where(
            nan_mask.unsqueeze(-1).expand_as(embedded_observed),
            mask_expanded,
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


class TabImputeModelRaBEL(TabImputeModelStable):
    """Stable direct TabImpute predictor with residual RaBEL tokenization."""

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
        rabel_num_kernels: int = 64,
        rabel_sigma: float = 1.0,
        rabel_exponent_digits: int = 1,
        rabel_exponent_embedding_size: int = 32,
        rabel_initial_scale: float = 0.05,
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
        )
        self.feature_encoder = FeatureEncoderRaBEL(
            embedding_size=embedding_size,
            num_cls=num_cls,
            use_absolute_positional_embeddings=use_absolute_positional_embeddings,
            positional_damping_factor=positional_damping_factor,
            embedding_dropout=embedding_dropout,
            rabel_num_kernels=rabel_num_kernels,
            rabel_sigma=rabel_sigma,
            rabel_exponent_digits=rabel_exponent_digits,
            rabel_exponent_embedding_size=rabel_exponent_embedding_size,
            rabel_initial_scale=rabel_initial_scale,
        )


class RaBELGridEmbedder(nn.Module):
    """Compressor scalar embedder with the same residual RaBEL branch."""

    def __init__(
        self,
        embedding_size: int,
        *,
        rabel_num_kernels: int = 64,
        rabel_sigma: float = 1.0,
        rabel_exponent_digits: int = 1,
        rabel_exponent_embedding_size: int = 32,
        rabel_initial_scale: float = 0.05,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.observed_linear_layer = nn.Sequential(
            nn.Linear(1, embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, embedding_size),
        )
        self.rabel_encoder = RaBELValueEncoder(
            embedding_size,
            num_kernels=rabel_num_kernels,
            sigma=rabel_sigma,
            exponent_digits=rabel_exponent_digits,
            exponent_embedding_size=rabel_exponent_embedding_size,
        )
        self.rabel_scale = nn.Parameter(torch.tensor(float(rabel_initial_scale)))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embedding_size))
        self.missingness_embedding = nn.Embedding(2, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nan_mask = torch.isnan(x)
        x_clean = torch.where(nan_mask, torch.zeros_like(x), x)
        scalar_values = x_clean.unsqueeze(-1)
        legacy = self.observed_linear_layer(scalar_values)
        rabel = self.rabel_encoder(scalar_values)
        observed = legacy + self.rabel_scale.to(legacy.dtype) * rabel
        embedded = torch.where(
            nan_mask.unsqueeze(-1).expand_as(observed),
            self.mask_token.expand_as(observed),
            observed,
        )
        return embedded + self.missingness_embedding(nan_mask.to(torch.long))


def _rabel_embedder_kwargs(config: dict) -> dict:
    return {
        "rabel_num_kernels": config.get("rabel_num_kernels", 64),
        "rabel_sigma": config.get("rabel_sigma", 1.0),
        "rabel_exponent_digits": config.get("rabel_exponent_digits", 1),
        "rabel_exponent_embedding_size": config.get(
            "rabel_exponent_embedding_size",
            32,
        ),
        "rabel_initial_scale": config.get("rabel_initial_scale", 0.05),
    }


class RowCompressionModelRaBEL(RowCompressionModel):
    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        k_row: int,
        num_cls: int = 12,
        attention_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        residual_scale_init: float = 1.0,
        rms_norm_eps: float = 1e-6,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        rope_fraction: float = 1.0,
        use_activation_checkpointing: bool = True,
        rabel_config: dict | None = None,
    ):
        super().__init__(
            embedding_size=embedding_size,
            num_attention_heads=num_attention_heads,
            mlp_hidden_size=mlp_hidden_size,
            num_layers=num_layers,
            k_row=k_row,
            num_cls=num_cls,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            drop_path_rate=drop_path_rate,
            residual_scale_init=residual_scale_init,
            rms_norm_eps=rms_norm_eps,
            use_rope=use_rope,
            rope_base=rope_base,
            rope_fraction=rope_fraction,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.scalar_embedder = RaBELGridEmbedder(
            embedding_size,
            **_rabel_embedder_kwargs(rabel_config or {}),
        )


class ColumnCompressionModelRaBEL(ColumnCompressionModel):
    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        k_col: int,
        num_cls: int = 12,
        attention_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        residual_scale_init: float = 1.0,
        rms_norm_eps: float = 1e-6,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        rope_fraction: float = 1.0,
        use_activation_checkpointing: bool = True,
        rabel_config: dict | None = None,
    ):
        super().__init__(
            embedding_size=embedding_size,
            num_attention_heads=num_attention_heads,
            mlp_hidden_size=mlp_hidden_size,
            num_layers=num_layers,
            k_col=k_col,
            num_cls=num_cls,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            drop_path_rate=drop_path_rate,
            residual_scale_init=residual_scale_init,
            rms_norm_eps=rms_norm_eps,
            use_rope=use_rope,
            rope_base=rope_base,
            rope_fraction=rope_fraction,
            use_activation_checkpointing=use_activation_checkpointing,
        )
        self.scalar_embedder = RaBELGridEmbedder(
            embedding_size,
            **_rabel_embedder_kwargs(rabel_config or {}),
        )


class CompressedCrossTabImputeModelRaBEL(CompressedCrossTabImputeModel):
    """Cross-compressed model whose predictor and compressors use RaBEL."""

    def _embed_target_real_tokens(self, target_chunk: torch.Tensor) -> torch.Tensor:
        return self.predictor.feature_encoder.embed_cells(target_chunk)

    @torch.no_grad()
    def initialize_compressor_rabel_from_predictor(self) -> None:
        """Copy only the new RaBEL branch, preserving trained compressor weights."""
        predictor_encoder = self.predictor.feature_encoder
        for compressor in (self.row_compressor, self.column_compressor):
            embedder = compressor.scalar_embedder
            embedder.rabel_encoder.load_state_dict(
                predictor_encoder.rabel_encoder.state_dict()
            )
            embedder.rabel_scale.copy_(predictor_encoder.rabel_scale)

    @torch.no_grad()
    def initialize_compressor_embedders_from_predictor(self) -> None:
        predictor_encoder = self.predictor.feature_encoder
        for compressor in (self.row_compressor, self.column_compressor):
            embedder = compressor.scalar_embedder
            embedder.observed_linear_layer.load_state_dict(
                predictor_encoder.observed_linear_layer.state_dict()
            )
            embedder.rabel_encoder.load_state_dict(
                predictor_encoder.rabel_encoder.state_dict()
            )
            embedder.rabel_scale.copy_(predictor_encoder.rabel_scale)
            embedder.mask_token.copy_(predictor_encoder.mask_token)
            embedder.missingness_embedding.load_state_dict(
                predictor_encoder.missingness_embedding.state_dict()
            )
            if compressor.num_cls == predictor_encoder.num_cls:
                compressor.row_cls_embedding.load_state_dict(
                    predictor_encoder.row_cls_embedding.state_dict()
                )
                compressor.col_cls_embedding.load_state_dict(
                    predictor_encoder.col_cls_embedding.state_dict()
                )


__all__ = [
    "ColumnCompressionModelRaBEL",
    "CompressedCrossTabImputeModelRaBEL",
    "FeatureEncoderRaBEL",
    "RaBELGridEmbedder",
    "RaBELValueEncoder",
    "RowCompressionModelRaBEL",
    "TabImputeModelRaBEL",
]
