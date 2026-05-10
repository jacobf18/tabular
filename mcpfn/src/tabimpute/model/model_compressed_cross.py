from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from tabimpute.model.model_new_stable import (
    TransformerEncoderLayerStable,
    _build_drop_path_schedule,
)


@dataclass(frozen=True)
class ChunkSpan:
    start: int
    end: int


def make_chunk_spans(size: int, chunk_size: int) -> list[ChunkSpan]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_size >= size:
        return [ChunkSpan(0, size)]
    spans: list[ChunkSpan] = []
    start = 0
    while start + chunk_size < size:
        spans.append(ChunkSpan(start, start + chunk_size))
        start += chunk_size
    if start < size:
        final_span = ChunkSpan(max(0, size - chunk_size), size)
        if not spans or spans[-1] != final_span:
            spans.append(final_span)
    return spans


class ScalarGridEmbedder(nn.Module):
    """Embeds scalar matrix entries into TabImpute-width tokens without CLS."""

    def __init__(self, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.observed_linear_layer = nn.Sequential(
            nn.Linear(1, embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, embedding_size),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embedding_size))
        self.missingness_embedding = nn.Embedding(2, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nan_mask = torch.isnan(x)
        x_clean = torch.where(nan_mask, torch.zeros_like(x), x)
        embedded_observed = self.observed_linear_layer(x_clean.unsqueeze(-1))
        mask_expanded = self.mask_token.expand_as(embedded_observed)
        embedded = torch.where(
            nan_mask.unsqueeze(-1).expand_as(embedded_observed),
            mask_expanded,
            embedded_observed,
        )
        return embedded + self.missingness_embedding(nan_mask.to(torch.long))


class AxialEmbeddingTransformer(nn.Module):
    """Small TabImpute-style axial transformer over already-embedded grids."""

    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        attention_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        residual_scale_init: float = 1.0,
        rms_norm_eps: float = 1e-6,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        rope_fraction: float = 1.0,
        use_activation_checkpointing: bool = True,
        checkpoint_use_reentrant: bool = False,
    ):
        super().__init__()
        self.use_activation_checkpointing = use_activation_checkpointing
        self.checkpoint_use_reentrant = checkpoint_use_reentrant
        drop_path_rates = _build_drop_path_schedule(num_layers, drop_path_rate)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayerStable(
                    embedding_size=embedding_size,
                    nhead=num_attention_heads,
                    mlp_hidden_size=mlp_hidden_size,
                    rms_norm_eps=rms_norm_eps,
                    attention_dropout=attention_dropout,
                    ffn_dropout=ffn_dropout,
                    drop_path=drop_path_rates[layer_idx],
                    residual_scale_init=residual_scale_init,
                    use_rope=use_rope,
                    rope_base=rope_base,
                    rope_fraction=rope_fraction,
                )
                for layer_idx in range(num_layers)
            ]
        )

    def _run_block(
        self,
        block: TransformerEncoderLayerStable,
        x: torch.Tensor,
        num_cls_rows: int = 0,
        num_cls_cols: int = 0,
    ) -> torch.Tensor:
        if (
            self.use_activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(
                lambda hidden: block(
                    hidden,
                    num_cls_rows=num_cls_rows,
                    num_cls_cols=num_cls_cols,
                ),
                x,
                use_reentrant=self.checkpoint_use_reentrant,
            )
        return block(x, num_cls_rows=num_cls_rows, num_cls_cols=num_cls_cols)

    def forward(
        self,
        x: torch.Tensor,
        num_cls_rows: int = 0,
        num_cls_cols: int = 0,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = self._run_block(
                block,
                x,
                num_cls_rows=num_cls_rows,
                num_cls_cols=num_cls_cols,
            )
        return x


class RowCompressionModel(nn.Module):
    """Compresses a column-aligned source chunk into K latent rows."""

    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        k_row: int,
        attention_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        residual_scale_init: float = 1.0,
        rms_norm_eps: float = 1e-6,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        rope_fraction: float = 1.0,
        use_activation_checkpointing: bool = True,
    ):
        super().__init__()
        self.k_row = k_row
        self.scalar_embedder = ScalarGridEmbedder(embedding_size)
        self.query_rows = nn.Parameter(torch.zeros(1, k_row, 1, embedding_size))
        self.transformer = AxialEmbeddingTransformer(
            embedding_size=embedding_size,
            num_attention_heads=num_attention_heads,
            mlp_hidden_size=mlp_hidden_size,
            num_layers=num_layers,
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

    def forward(self, target_chunk: torch.Tensor, source_chunk: torch.Tensor) -> torch.Tensor:
        if target_chunk.ndim != 3 or source_chunk.ndim != 3:
            raise ValueError("target_chunk and source_chunk must be B x rows x cols")
        if target_chunk.shape[0] != source_chunk.shape[0]:
            raise ValueError("target_chunk and source_chunk batch sizes must match")
        if target_chunk.shape[2] != source_chunk.shape[2]:
            raise ValueError("row compression requires matching column counts")

        real_tokens = self.scalar_embedder(torch.cat([target_chunk, source_chunk], dim=1))
        batch_size, _, cols, _ = real_tokens.shape
        query = self.query_rows.expand(batch_size, self.k_row, cols, -1)
        tokens = torch.cat([real_tokens, query], dim=1)
        tokens = self.transformer(tokens)
        return tokens[:, -self.k_row :, :, :]


class ColumnCompressionModel(nn.Module):
    """Compresses a row-aligned source chunk into K latent columns."""

    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        k_col: int,
        attention_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        residual_scale_init: float = 1.0,
        rms_norm_eps: float = 1e-6,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        rope_fraction: float = 1.0,
        use_activation_checkpointing: bool = True,
    ):
        super().__init__()
        self.k_col = k_col
        self.scalar_embedder = ScalarGridEmbedder(embedding_size)
        self.query_cols = nn.Parameter(torch.zeros(1, 1, k_col, embedding_size))
        self.transformer = AxialEmbeddingTransformer(
            embedding_size=embedding_size,
            num_attention_heads=num_attention_heads,
            mlp_hidden_size=mlp_hidden_size,
            num_layers=num_layers,
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

    def forward(self, target_chunk: torch.Tensor, source_chunk: torch.Tensor) -> torch.Tensor:
        if target_chunk.ndim != 3 or source_chunk.ndim != 3:
            raise ValueError("target_chunk and source_chunk must be B x rows x cols")
        if target_chunk.shape[0] != source_chunk.shape[0]:
            raise ValueError("target_chunk and source_chunk batch sizes must match")
        if target_chunk.shape[1] != source_chunk.shape[1]:
            raise ValueError("column compression requires matching row counts")

        real_tokens = self.scalar_embedder(torch.cat([target_chunk, source_chunk], dim=2))
        batch_size, rows, _, _ = real_tokens.shape
        query = self.query_cols.expand(batch_size, rows, self.k_col, -1)
        tokens = torch.cat([real_tokens, query], dim=2)
        tokens = self.transformer(tokens)
        return tokens[:, :, -self.k_col :, :]


class CompressedCrossTabImputeModel(nn.Module):
    """Cross-only chunk compression wrapper around direct TabImpute."""

    def __init__(
        self,
        predictor: nn.Module,
        row_compressor: RowCompressionModel,
        column_compressor: ColumnCompressionModel,
        k_row: int,
        k_col: int,
    ):
        super().__init__()
        self.predictor = predictor
        self.row_compressor = row_compressor
        self.column_compressor = column_compressor
        self.k_row = k_row
        self.k_col = k_col
        self.embedding_size = predictor.feature_encoder.embedding_size
        self.off_axis_token = nn.Parameter(torch.zeros(1, 1, 1, self.embedding_size))

    def _embed_target_real_tokens(self, target_chunk: torch.Tensor) -> torch.Tensor:
        encoder = self.predictor.feature_encoder
        nan_mask = torch.isnan(target_chunk)
        x_clean = torch.where(nan_mask, torch.zeros_like(target_chunk), target_chunk)
        embedded_observed = encoder.observed_linear_layer(x_clean.unsqueeze(-1))
        mask_expanded = encoder.mask_token.expand_as(embedded_observed)
        embedded = torch.where(
            nan_mask.unsqueeze(-1).expand_as(embedded_observed),
            mask_expanded,
            embedded_observed,
        )
        return embedded + encoder.missingness_embedding(nan_mask.to(torch.long))

    def _prepend_predictor_cls(self, real_tokens: torch.Tensor) -> torch.Tensor:
        encoder = self.predictor.feature_encoder
        batch_size, rows, cols, embedding_size = real_tokens.shape
        k_rows = encoder._compute_num_cls(rows)
        k_cols = encoder._compute_num_cls(cols)
        encoder.last_num_cls_rows = k_rows
        encoder.last_num_cls_cols = k_cols
        device = real_tokens.device
        dtype = real_tokens.dtype

        row_cls_tokens = encoder._get_cls_tokens(
            k_cols,
            encoder.row_cls_embedding,
            device,
            dtype,
        )
        row_cls = row_cls_tokens.unsqueeze(0).unsqueeze(0).expand(
            batch_size,
            rows,
            k_cols,
            embedding_size,
        )

        col_cls_tokens = encoder._get_cls_tokens(
            k_rows,
            encoder.col_cls_embedding,
            device,
            dtype,
        )
        col_cls = col_cls_tokens.unsqueeze(0).unsqueeze(2).expand(
            batch_size,
            k_rows,
            cols,
            embedding_size,
        )

        row_corner = row_cls_tokens.unsqueeze(0).expand(
            k_rows,
            k_cols,
            embedding_size,
        )
        col_corner = col_cls_tokens.unsqueeze(1).expand(
            k_rows,
            k_cols,
            embedding_size,
        )
        corner = (row_corner + col_corner).unsqueeze(0).expand(
            batch_size,
            k_rows,
            k_cols,
            embedding_size,
        )

        top_block = torch.cat([corner, col_cls], dim=2)
        bottom_block = torch.cat([row_cls, real_tokens], dim=2)
        return torch.cat([top_block, bottom_block], dim=1)

    def _run_predictor_from_embeddings(
        self,
        real_tokens: torch.Tensor,
        decode_rows: int,
        decode_cols: int,
    ) -> torch.Tensor:
        encoder = self.predictor.feature_encoder
        if encoder.row_embedding is not None and encoder.column_embedding is not None:
            real_tokens = encoder.row_embedding(real_tokens)
            real_tokens = encoder.column_embedding(real_tokens)
        real_tokens = encoder.embedding_dropout(real_tokens)
        src = self._prepend_predictor_cls(real_tokens)
        num_cls_rows = encoder.last_num_cls_rows
        num_cls_cols = encoder.last_num_cls_cols

        for block in self.predictor.transformer_blocks:
            if hasattr(self.predictor, "_run_block"):
                src = self.predictor._run_block(
                    block,
                    src,
                    num_cls_rows=num_cls_rows,
                    num_cls_cols=num_cls_cols,
                )
            else:
                src = block(src, num_cls_rows=num_cls_rows, num_cls_cols=num_cls_cols)

        feats = src[
            :,
            num_cls_rows : num_cls_rows + decode_rows,
            num_cls_cols : num_cls_cols + decode_cols,
            :,
        ]
        return self.predictor.decoder(feats)

    def _split_chunks(
        self,
        x: torch.Tensor,
        chunk_rows: int,
        chunk_cols: int,
    ) -> tuple[list[ChunkSpan], list[ChunkSpan]]:
        if x.ndim != 3:
            raise ValueError("Expected batched matrix tensor B x rows x cols")
        return make_chunk_spans(x.shape[1], chunk_rows), make_chunk_spans(
            x.shape[2],
            chunk_cols,
        )

    def _target_logits(
        self,
        x_missing: torch.Tensor,
        row_spans: list[ChunkSpan],
        col_spans: list[ChunkSpan],
        row_idx: int,
        col_idx: int,
    ) -> torch.Tensor:
        row_span = row_spans[row_idx]
        col_span = col_spans[col_idx]
        target = x_missing[
            :,
            row_span.start : row_span.end,
            col_span.start : col_span.end,
        ]
        target_rows = target.shape[1]
        target_cols = target.shape[2]
        target_tokens = self._embed_target_real_tokens(target)

        latent_rows = []
        for source_row_idx, source_row_span in enumerate(row_spans):
            if source_row_idx == row_idx:
                continue
            source = x_missing[
                :,
                source_row_span.start : source_row_span.end,
                col_span.start : col_span.end,
            ]
            latent_rows.append(self.row_compressor(target, source))
        if latent_rows:
            row_context = torch.cat(latent_rows, dim=1)
        else:
            row_context = target_tokens.new_zeros(
                target_tokens.shape[0],
                0,
                target_cols,
                self.embedding_size,
            )

        latent_cols = []
        for source_col_idx, source_col_span in enumerate(col_spans):
            if source_col_idx == col_idx:
                continue
            source = x_missing[
                :,
                row_span.start : row_span.end,
                source_col_span.start : source_col_span.end,
            ]
            latent_cols.append(self.column_compressor(target, source))
        if latent_cols:
            col_context = torch.cat(latent_cols, dim=2)
        else:
            col_context = target_tokens.new_zeros(
                target_tokens.shape[0],
                target_rows,
                0,
                self.embedding_size,
            )

        context_rows = row_context.shape[1]
        context_cols = col_context.shape[2]
        if context_cols:
            top = torch.cat([target_tokens, col_context], dim=2)
        else:
            top = target_tokens

        if context_rows:
            if context_cols:
                off_axis = self.off_axis_token.expand(
                    target_tokens.shape[0],
                    context_rows,
                    context_cols,
                    self.embedding_size,
                )
                bottom = torch.cat([row_context, off_axis], dim=2)
            else:
                bottom = row_context
            real_tokens = torch.cat([top, bottom], dim=1)
        else:
            real_tokens = top

        return self._run_predictor_from_embeddings(
            real_tokens,
            decode_rows=target_rows,
            decode_cols=target_cols,
        )

    def forward_target_chunk(
        self,
        x_missing: torch.Tensor,
        chunk_rows: int,
        chunk_cols: int,
        target_row_idx: int,
        target_col_idx: int,
    ) -> torch.Tensor:
        row_spans, col_spans = self._split_chunks(x_missing, chunk_rows, chunk_cols)
        return self._target_logits(
            x_missing,
            row_spans,
            col_spans,
            target_row_idx,
            target_col_idx,
        )

    def forward_full_matrix(
        self,
        x_missing: torch.Tensor,
        x_target: torch.Tensor,
        chunk_rows: int,
        chunk_cols: int,
        criterion: nn.Module,
    ) -> torch.Tensor:
        row_spans, col_spans = self._split_chunks(x_missing, chunk_rows, chunk_cols)
        losses = []
        for row_idx, row_span in enumerate(row_spans):
            for col_idx, col_span in enumerate(col_spans):
                logits = self._target_logits(
                    x_missing,
                    row_spans,
                    col_spans,
                    row_idx,
                    col_idx,
                )
                target = x_target[
                    :,
                    row_span.start : row_span.end,
                    col_span.start : col_span.end,
                ]
                missing_mask = torch.isnan(
                    x_missing[
                        :,
                        row_span.start : row_span.end,
                        col_span.start : col_span.end,
                    ]
                )
                if missing_mask.any():
                    chunk_losses = criterion(logits[missing_mask], target[missing_mask])
                    losses.append(chunk_losses.mean())
        if not losses:
            return x_target.new_zeros(())
        return torch.stack(losses).mean()

    @torch.no_grad()
    def impute_normalized(
        self,
        x_missing: torch.Tensor,
        chunk_rows: int,
        chunk_cols: int,
        bar_distribution: nn.Module,
    ) -> torch.Tensor:
        row_spans, col_spans = self._split_chunks(x_missing, chunk_rows, chunk_cols)
        prediction_sum = torch.zeros_like(x_missing, dtype=torch.float32)
        prediction_counts = torch.zeros_like(x_missing, dtype=torch.float32)
        for row_idx, row_span in enumerate(row_spans):
            for col_idx, col_span in enumerate(col_spans):
                logits = self._target_logits(
                    x_missing,
                    row_spans,
                    col_spans,
                    row_idx,
                    col_idx,
                )
                medians = bar_distribution.median(logits).to(dtype=torch.float32)
                row_slice = slice(row_span.start, row_span.end)
                col_slice = slice(col_span.start, col_span.end)
                missing_mask = torch.isnan(x_missing[:, row_slice, col_slice])
                sum_chunk = prediction_sum[:, row_slice, col_slice]
                count_chunk = prediction_counts[:, row_slice, col_slice]
                sum_chunk[missing_mask] += medians[missing_mask]
                count_chunk[missing_mask] += 1.0
        output = x_missing.clone()
        has_predictions = prediction_counts > 0
        output[has_predictions] = (
            prediction_sum[has_predictions] / prediction_counts[has_predictions]
        ).to(dtype=output.dtype)
        return output


def build_compressor_config_from_predictor(
    predictor_config: dict,
    *,
    k_row: int,
    k_col: int,
    num_layers: int | None = None,
    mlp_hidden_size: int | None = None,
) -> dict:
    return {
        "embedding_size": predictor_config["embedding_size"],
        "num_attention_heads": predictor_config["num_attention_heads"],
        "mlp_hidden_size": mlp_hidden_size
        if mlp_hidden_size is not None
        else predictor_config["mlp_hidden_size"],
        "num_layers": num_layers
        if num_layers is not None
        else predictor_config["num_layers"],
        "attention_dropout": predictor_config.get("attention_dropout", 0.0),
        "ffn_dropout": predictor_config.get("ffn_dropout", 0.0),
        "drop_path_rate": predictor_config.get("drop_path_rate", 0.0),
        "residual_scale_init": predictor_config.get("residual_scale_init", 1.0),
        "rms_norm_eps": predictor_config.get("rms_norm_eps", 1e-6),
        "use_rope": predictor_config.get("use_rope", True),
        "rope_base": predictor_config.get("rope_base", 10000.0),
        "rope_fraction": predictor_config.get("rope_fraction", 1.0),
        "k_row": k_row,
        "k_col": k_col,
        "use_activation_checkpointing": predictor_config.get(
            "use_activation_checkpointing",
            True,
        ),
    }


__all__ = [
    "ChunkSpan",
    "ColumnCompressionModel",
    "CompressedCrossTabImputeModel",
    "RowCompressionModel",
    "build_compressor_config_from_predictor",
    "make_chunk_spans",
]
