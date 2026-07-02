from __future__ import annotations

import torch
from torch import nn

from tabimpute.model.model_compressed_cross import (
    ColumnCompressionModel,
    CompressedCrossTabImputeModel,
    RowCompressionModel,
)
from tabimpute.model.model_new_stable_rabel import TabImputeModelPureRaBEL
from tabimpute.model.rabel_encoder import RaBELValueEncoder


class PureRaBELGridEmbedder(nn.Module):
    """Pure RaBEL scalar embedder for compressor input grids."""

    def __init__(
        self,
        embedding_size: int,
        *,
        rabel_num_kernels: int = 64,
        rabel_sigma: float = 1.0,
        rabel_exponent_digits: int = 1,
        rabel_exponent_embedding_size: int = 32,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.rabel_encoder = RaBELValueEncoder(
            embedding_size=embedding_size,
            num_kernels=rabel_num_kernels,
            sigma=rabel_sigma,
            exponent_digits=rabel_exponent_digits,
            exponent_embedding_size=rabel_exponent_embedding_size,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embedding_size))
        self.missingness_embedding = nn.Embedding(2, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nan_mask = torch.isnan(x)
        x_clean = torch.where(nan_mask, torch.zeros_like(x), x)
        observed = self.rabel_encoder(x_clean.unsqueeze(-1))
        embedded = torch.where(
            nan_mask.unsqueeze(-1).expand_as(observed),
            self.mask_token.expand_as(observed),
            observed,
        )
        return embedded + self.missingness_embedding(nan_mask.to(torch.long))


def _pure_rabel_embedder_kwargs(config: dict) -> dict:
    return {
        "rabel_num_kernels": config.get("rabel_num_kernels", 64),
        "rabel_sigma": config.get("rabel_sigma", 1.0),
        "rabel_exponent_digits": config.get("rabel_exponent_digits", 1),
        "rabel_exponent_embedding_size": config.get(
            "rabel_exponent_embedding_size",
            32,
        ),
    }


class RowCompressionModelPureRaBEL(RowCompressionModel):
    """Row compressor with pure RaBEL scalar tokenization."""

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
        self.scalar_embedder = PureRaBELGridEmbedder(
            embedding_size,
            **_pure_rabel_embedder_kwargs(rabel_config or {}),
        )


class ColumnCompressionModelPureRaBEL(ColumnCompressionModel):
    """Column compressor with pure RaBEL scalar tokenization."""

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
        self.scalar_embedder = PureRaBELGridEmbedder(
            embedding_size,
            **_pure_rabel_embedder_kwargs(rabel_config or {}),
        )


class CompressedCrossTabImputeModelPureRaBEL(CompressedCrossTabImputeModel):
    """Cross-compressed TabImpute with pure RaBEL in all three models."""

    def _embed_target_real_tokens(self, target_chunk: torch.Tensor) -> torch.Tensor:
        return self.predictor.feature_encoder.embed_cells(target_chunk)

    @torch.no_grad()
    def initialize_compressor_embedders_from_predictor(self) -> None:
        predictor_encoder = self.predictor.feature_encoder
        for compressor in (self.row_compressor, self.column_compressor):
            embedder = compressor.scalar_embedder
            embedder.rabel_encoder.load_state_dict(
                predictor_encoder.rabel_encoder.state_dict()
            )
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
    "ColumnCompressionModelPureRaBEL",
    "CompressedCrossTabImputeModelPureRaBEL",
    "PureRaBELGridEmbedder",
    "RowCompressionModelPureRaBEL",
    "TabImputeModelPureRaBEL",
]
