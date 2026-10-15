from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from tabimpute.model.model_new_stable import DecoderStable, DropPath, RMSNorm
from tabimpute.model.rabel_encoder import RaBELValueEncoder


class FeedForward(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(embedding_size, hidden_size)
        self.value = nn.Linear(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate(x)) * self.value(x)
        return self.out(self.dropout(x))


class BipartiteMessagePassingLayer(nn.Module):
    """One observed-entry message-passing block over row and column latents."""

    def __init__(
        self,
        embedding_size: int,
        mlp_hidden_size: int,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        residual_scale_init: float = 0.1,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.value_to_row = nn.Linear(embedding_size, embedding_size, bias=False)
        self.col_to_row = nn.Linear(embedding_size, embedding_size, bias=False)
        self.row_message_norm = RMSNorm(embedding_size, eps=rms_norm_eps)
        self.row_message_out = nn.Linear(embedding_size, embedding_size)

        self.value_to_col = nn.Linear(embedding_size, embedding_size, bias=False)
        self.row_to_col = nn.Linear(embedding_size, embedding_size, bias=False)
        self.col_message_norm = RMSNorm(embedding_size, eps=rms_norm_eps)
        self.col_message_out = nn.Linear(embedding_size, embedding_size)

        self.row_state_norm = RMSNorm(embedding_size, eps=rms_norm_eps)
        self.col_state_norm = RMSNorm(embedding_size, eps=rms_norm_eps)
        self.row_agg_norm = RMSNorm(embedding_size, eps=rms_norm_eps)
        self.col_agg_norm = RMSNorm(embedding_size, eps=rms_norm_eps)
        self.row_update = FeedForward(embedding_size, mlp_hidden_size, dropout)
        self.col_update = FeedForward(embedding_size, mlp_hidden_size, dropout)

        self.row_drop_path = DropPath(drop_path)
        self.col_drop_path = DropPath(drop_path)
        self.row_residual_scale = nn.Parameter(
            torch.full((1,), float(residual_scale_init))
        )
        self.col_residual_scale = nn.Parameter(
            torch.full((1,), float(residual_scale_init))
        )

    def forward(
        self,
        row_state: torch.Tensor,
        col_state: torch.Tensor,
        value_emb: torch.Tensor,
        observed: torch.Tensor,
        row_counts: torch.Tensor,
        col_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed = observed.unsqueeze(-1)

        row_message = self.value_to_row(value_emb) + self.col_to_row(
            self.col_state_norm(col_state)
        ).unsqueeze(1)
        row_message = self.row_message_out(
            F.silu(self.row_message_norm(row_message))
        )
        row_agg = (row_message * observed).sum(dim=2) / row_counts.unsqueeze(-1)

        col_message = self.value_to_col(value_emb) + self.row_to_col(
            self.row_state_norm(row_state)
        ).unsqueeze(2)
        col_message = self.col_message_out(
            F.silu(self.col_message_norm(col_message))
        )
        col_agg = (col_message * observed).sum(dim=1) / col_counts.unsqueeze(-1)

        row_update = self.row_update(
            self.row_state_norm(row_state) + self.row_agg_norm(row_agg)
        )
        col_update = self.col_update(
            self.col_state_norm(col_state) + self.col_agg_norm(col_agg)
        )
        row_state = row_state + self.row_drop_path(
            self.row_residual_scale * row_update
        )
        col_state = col_state + self.col_drop_path(
            self.col_residual_scale * col_update
        )
        return row_state, col_state


class ValueWeightedBipartiteMessagePassingLayer(BipartiteMessagePassingLayer):
    """Observed-value-weighted row/column cross messages."""

    def __init__(
        self,
        embedding_size: int,
        mlp_hidden_size: int,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        residual_scale_init: float = 0.1,
        rms_norm_eps: float = 1e-6,
        value_weighted_scale_init: float = 0.01,
    ) -> None:
        super().__init__(
            embedding_size=embedding_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,
            drop_path=drop_path,
            residual_scale_init=residual_scale_init,
            rms_norm_eps=rms_norm_eps,
        )
        self.value_col_to_row = nn.Linear(embedding_size, embedding_size, bias=False)
        self.value_row_to_col = nn.Linear(embedding_size, embedding_size, bias=False)
        self.row_value_agg_norm = RMSNorm(embedding_size, eps=rms_norm_eps)
        self.col_value_agg_norm = RMSNorm(embedding_size, eps=rms_norm_eps)
        self.row_value_scale = nn.Parameter(
            torch.full((1,), float(value_weighted_scale_init))
        )
        self.col_value_scale = nn.Parameter(
            torch.full((1,), float(value_weighted_scale_init))
        )

    def forward(
        self,
        row_state: torch.Tensor,
        col_state: torch.Tensor,
        value_emb: torch.Tensor,
        observed: torch.Tensor,
        row_counts: torch.Tensor,
        col_counts: torch.Tensor,
        clean_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed_expanded = observed.unsqueeze(-1)

        row_message = self.value_to_row(value_emb) + self.col_to_row(
            self.col_state_norm(col_state)
        ).unsqueeze(1)
        row_message = self.row_message_out(
            F.silu(self.row_message_norm(row_message))
        )
        row_agg = (row_message * observed_expanded).sum(dim=2) / row_counts.unsqueeze(-1)

        col_message = self.value_to_col(value_emb) + self.row_to_col(
            self.row_state_norm(row_state)
        ).unsqueeze(2)
        col_message = self.col_message_out(
            F.silu(self.col_message_norm(col_message))
        )
        col_agg = (col_message * observed_expanded).sum(dim=1) / col_counts.unsqueeze(-1)

        weighted_values = clean_values * observed
        row_weighted = torch.bmm(
            weighted_values,
            self.value_col_to_row(self.col_state_norm(col_state)),
        ) / row_counts.unsqueeze(-1)
        col_weighted = torch.bmm(
            weighted_values.transpose(1, 2),
            self.value_row_to_col(self.row_state_norm(row_state)),
        ) / col_counts.unsqueeze(-1)

        row_update = self.row_update(
            self.row_state_norm(row_state)
            + self.row_agg_norm(row_agg)
            + self.row_value_scale * self.row_value_agg_norm(row_weighted)
        )
        col_update = self.col_update(
            self.col_state_norm(col_state)
            + self.col_agg_norm(col_agg)
            + self.col_value_scale * self.col_value_agg_norm(col_weighted)
        )
        row_state = row_state + self.row_drop_path(
            self.row_residual_scale * row_update
        )
        col_state = col_state + self.col_drop_path(
            self.col_residual_scale * col_update
        )
        return row_state, col_state


class RowColumnLatentDecoder(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        mlp_hidden_size: int,
        num_outputs: int,
        rms_norm_eps: float = 1e-6,
        interaction_scale_init: float = 0.1,
        feature_size: int | None = None,
        extra_decoder_hidden_size: int | None = None,
        extra_decoder_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_size = int(embedding_size)
        self.feature_size = int(feature_size or embedding_size)
        self.row_proj = nn.Linear(embedding_size, self.feature_size, bias=False)
        self.col_proj = nn.Linear(embedding_size, self.feature_size, bias=False)
        self.interaction_proj = nn.Linear(
            embedding_size,
            self.feature_size,
            bias=False,
        )
        self.norm = RMSNorm(self.feature_size, eps=rms_norm_eps)
        self.interaction_scale = nn.Parameter(
            torch.full((1,), float(interaction_scale_init))
        )
        self.decoder = DecoderStable(self.feature_size, mlp_hidden_size, num_outputs)
        self.extra_decoder = None
        if extra_decoder_hidden_size is not None and int(extra_decoder_hidden_size) > 0:
            self.extra_decoder = DecoderStable(
                self.feature_size,
                int(extra_decoder_hidden_size),
                num_outputs,
            )
            self.extra_decoder_scale = nn.Parameter(
                torch.tensor(float(extra_decoder_scale_init), dtype=torch.float32)
            )

    def features_for_pairs(
        self,
        row_state: torch.Tensor,
        col_state: torch.Tensor,
    ) -> torch.Tensor:
        features = self.row_proj(row_state) + self.col_proj(col_state)
        features = features + self.interaction_scale * self.interaction_proj(
            row_state * col_state
        )
        return self.norm(features)

    def forward_pairs(
        self,
        row_state: torch.Tensor,
        col_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode_features(self.features_for_pairs(row_state, col_state))

    def forward_grid(
        self,
        row_state: torch.Tensor,
        col_state: torch.Tensor,
    ) -> torch.Tensor:
        row = row_state.unsqueeze(2)
        col = col_state.unsqueeze(1)
        features = self.row_proj(row) + self.col_proj(col)
        features = features + self.interaction_scale * self.interaction_proj(
            row * col
        )
        return self.decode_features(self.norm(features))

    def grid_features(
        self,
        row_state: torch.Tensor,
        col_state: torch.Tensor,
    ) -> torch.Tensor:
        row = row_state.unsqueeze(2)
        col = col_state.unsqueeze(1)
        features = self.row_proj(row) + self.col_proj(col)
        features = features + self.interaction_scale * self.interaction_proj(
            row * col
        )
        return self.norm(features)

    def decode_features(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(features)
        if self.extra_decoder is not None:
            logits = logits + self.extra_decoder_scale.to(logits.dtype) * (
                self.extra_decoder(features).to(logits.dtype)
            )
        return logits


class TabImputeModelRowColumnLatent(nn.Module):
    """Matrix-completion model with explicit row/column latent states.

    This is intentionally separate from the TabImpute axial-attention family.
    It treats observed entries as edges in a bipartite row-column graph and
    decodes a cell from its final row and column states.
    """

    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int | None = None,
        mlp_hidden_size: int = 1024,
        num_layers: int = 8,
        num_outputs: int = 5000,
        num_cls: int | None = None,
        use_rope: bool | None = None,
        rope_base: float | None = None,
        rope_fraction: float | None = None,
        use_absolute_positional_embeddings: bool = False,
        positional_damping_factor: float | None = None,
        attention_dropout: float | None = None,
        ffn_dropout: float = 0.05,
        drop_path_rate: float = 0.0,
        residual_scale_init: float = 0.1,
        rms_norm_eps: float = 1e-6,
        embedding_dropout: float = 0.0,
        attention_batch_chunk_size: int | None = None,
        use_activation_checkpointing: bool = True,
        checkpoint_use_reentrant: bool = False,
        rabel_num_kernels: int = 64,
        rabel_sigma: float = 1.0,
        rabel_exponent_digits: int = 1,
        rabel_exponent_embedding_size: int = 32,
        count_hidden_size: int | None = None,
        decoder_interaction_scale_init: float = 0.1,
        decoder_feature_size: int | None = None,
        decoder_hidden_size: int | None = None,
        extra_decoder_hidden_size: int | None = None,
        extra_decoder_scale_init: float = 0.0,
        use_axis_identity_features: bool = False,
        axis_identity_hidden_size: int | None = None,
        axis_identity_scale_init: float = 0.05,
        positional_encoding_kind: str | None = None,
        positional_num_frequencies: int = 32,
        positional_max_size: int = 1024,
        use_value_weighted_messages: bool = False,
        value_weighted_message_scale_init: float = 0.01,
        **_: object,
    ) -> None:
        super().__init__()
        del (
            num_attention_heads,
            num_cls,
            use_rope,
            rope_base,
            rope_fraction,
            attention_dropout,
            attention_batch_chunk_size,
        )
        self.embedding_size = embedding_size
        self.num_outputs = num_outputs
        self.decoder_feature_size = int(decoder_feature_size or embedding_size)
        self.decoder_hidden_size = int(decoder_hidden_size or mlp_hidden_size)
        self.use_activation_checkpointing = use_activation_checkpointing
        self.checkpoint_use_reentrant = checkpoint_use_reentrant
        self.use_value_weighted_messages = bool(use_value_weighted_messages)

        self.value_encoder = RaBELValueEncoder(
            embedding_size=embedding_size,
            num_kernels=rabel_num_kernels,
            sigma=rabel_sigma,
            exponent_digits=rabel_exponent_digits,
            exponent_embedding_size=rabel_exponent_embedding_size,
        )
        self.value_norm = RMSNorm(embedding_size, eps=rms_norm_eps)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.row_init = nn.Parameter(torch.zeros(1, 1, embedding_size))
        self.col_init = nn.Parameter(torch.zeros(1, 1, embedding_size))
        self.use_axis_identity_features = bool(use_axis_identity_features)
        self.use_absolute_positional_embeddings = bool(use_absolute_positional_embeddings)
        self.positional_damping_factor = float(
            0.1 if positional_damping_factor is None else positional_damping_factor
        )
        if positional_encoding_kind is None:
            positional_encoding_kind = (
                "sinusoidal_index"
                if self.use_absolute_positional_embeddings
                else "none"
            )
        self.positional_encoding_kind = str(positional_encoding_kind).lower()
        if self.positional_encoding_kind in {"", "false", "off"}:
            self.positional_encoding_kind = "none"
        self.positional_num_frequencies = int(positional_num_frequencies)
        self.positional_max_size = int(positional_max_size)
        count_hidden_size = int(count_hidden_size or max(64, embedding_size // 4))
        self.row_count_encoder = nn.Sequential(
            nn.Linear(2, count_hidden_size),
            nn.GELU(),
            nn.Linear(count_hidden_size, embedding_size),
        )
        self.col_count_encoder = nn.Sequential(
            nn.Linear(2, count_hidden_size),
            nn.GELU(),
            nn.Linear(count_hidden_size, embedding_size),
        )
        identity_hidden_size = int(axis_identity_hidden_size or count_hidden_size)
        if self.use_axis_identity_features:
            self.row_identity_encoder = nn.Sequential(
                nn.Linear(4, identity_hidden_size),
                nn.GELU(),
                nn.Linear(identity_hidden_size, embedding_size),
            )
            self.col_identity_encoder = nn.Sequential(
                nn.Linear(4, identity_hidden_size),
                nn.GELU(),
                nn.Linear(identity_hidden_size, embedding_size),
            )
            self.row_identity_scale = nn.Parameter(
                torch.full((1,), float(axis_identity_scale_init))
            )
            self.col_identity_scale = nn.Parameter(
                torch.full((1,), float(axis_identity_scale_init))
            )
        else:
            self.row_identity_encoder = None
            self.col_identity_encoder = None
            self.row_identity_scale = None
            self.col_identity_scale = None

        if self.positional_encoding_kind in {"fourier_mlp", "poly_fourier_mlp"}:
            feature_dim = 1 + 2 * self.positional_num_frequencies
            if self.positional_encoding_kind == "poly_fourier_mlp":
                feature_dim += 2
            self.row_positional_encoder = nn.Sequential(
                nn.Linear(feature_dim, identity_hidden_size),
                nn.GELU(),
                nn.Linear(identity_hidden_size, embedding_size),
            )
            self.col_positional_encoder = nn.Sequential(
                nn.Linear(feature_dim, identity_hidden_size),
                nn.GELU(),
                nn.Linear(identity_hidden_size, embedding_size),
            )
        else:
            self.row_positional_encoder = None
            self.col_positional_encoder = None

        if self.positional_encoding_kind == "learned_table":
            if self.positional_max_size <= 1:
                raise ValueError("positional_max_size must be > 1 for learned_table")
            self.row_positional_table = nn.Parameter(
                torch.empty(1, self.positional_max_size, embedding_size)
            )
            self.col_positional_table = nn.Parameter(
                torch.empty(1, self.positional_max_size, embedding_size)
            )
            nn.init.normal_(self.row_positional_table, mean=0.0, std=0.02)
            nn.init.normal_(self.col_positional_table, mean=0.0, std=0.02)
        else:
            self.row_positional_table = None
            self.col_positional_table = None
        self.row_input_proj = nn.Linear(embedding_size, embedding_size)
        self.col_input_proj = nn.Linear(embedding_size, embedding_size)
        self.row_init_norm = RMSNorm(embedding_size, eps=rms_norm_eps)
        self.col_init_norm = RMSNorm(embedding_size, eps=rms_norm_eps)

        if num_layers <= 1:
            drop_path_rates = [drop_path_rate] * max(0, num_layers)
        else:
            drop_path_rates = torch.linspace(
                0.0, drop_path_rate, steps=num_layers
            ).tolist()
        message_layer_class = (
            ValueWeightedBipartiteMessagePassingLayer
            if self.use_value_weighted_messages
            else BipartiteMessagePassingLayer
        )
        self.message_layers = nn.ModuleList(
            [
                message_layer_class(
                    embedding_size=embedding_size,
                    mlp_hidden_size=mlp_hidden_size,
                    dropout=ffn_dropout,
                    drop_path=drop_path_rates[layer_idx],
                    residual_scale_init=residual_scale_init,
                    rms_norm_eps=rms_norm_eps,
                    **(
                        {
                            "value_weighted_scale_init": (
                                value_weighted_message_scale_init
                            )
                        }
                        if self.use_value_weighted_messages
                        else {}
                    ),
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.output_decoder = RowColumnLatentDecoder(
            embedding_size=embedding_size,
            mlp_hidden_size=self.decoder_hidden_size,
            num_outputs=num_outputs,
            rms_norm_eps=rms_norm_eps,
            interaction_scale_init=decoder_interaction_scale_init,
            feature_size=self.decoder_feature_size,
            extra_decoder_hidden_size=extra_decoder_hidden_size,
            extra_decoder_scale_init=extra_decoder_scale_init,
        )

    def set_attention_capture(self, enabled: bool, store_weights: bool = True) -> None:
        del enabled, store_weights

    def get_attention_diagnostics(self) -> list[dict[str, object]]:
        return []

    def _encode_values(
        self,
        src: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nan_mask = torch.isnan(src)
        observed = (~nan_mask).to(dtype=src.dtype)
        clean = torch.where(nan_mask, torch.zeros_like(src), src)
        value_emb = self.value_encoder(clean.unsqueeze(-1))
        value_emb = self.value_norm(value_emb)
        value_emb = self.embedding_dropout(value_emb)

        row_counts = observed.sum(dim=2).clamp_min(1.0)
        col_counts = observed.sum(dim=1).clamp_min(1.0)
        row_summary = (value_emb * observed.unsqueeze(-1)).sum(dim=2) / row_counts.unsqueeze(-1)
        col_summary = (value_emb * observed.unsqueeze(-1)).sum(dim=1) / col_counts.unsqueeze(-1)
        return value_emb, observed, row_counts, col_counts, row_summary, col_summary

    def _count_features(
        self,
        counts: torch.Tensor,
        axis_size: int,
    ) -> torch.Tensor:
        denom = max(float(axis_size), 1.0)
        count_frac = counts / denom
        log_count = torch.log1p(counts) / torch.log1p(
            torch.tensor(denom, device=counts.device, dtype=counts.dtype)
        )
        return torch.stack([count_frac, log_count], dim=-1)

    def _axis_identity_features(
        self,
        *,
        batch_size: int,
        axis_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if axis_size <= 1:
            pos = torch.zeros((axis_size,), device=device, dtype=dtype)
        else:
            pos = torch.linspace(-1.0, 1.0, steps=axis_size, device=device, dtype=dtype)
        abs_pos = pos.abs()
        sin_pos = torch.sin(torch.pi * pos)
        cos_pos = torch.cos(torch.pi * pos)
        features = torch.stack([pos, abs_pos, sin_pos, cos_pos], dim=-1)
        return features.unsqueeze(0).expand(batch_size, axis_size, 4)

    def _sinusoidal_axis_embedding(
        self,
        *,
        batch_size: int,
        axis_size: int,
        device: torch.device,
        dtype: torch.dtype,
        normalized: bool = False,
    ) -> torch.Tensor:
        if normalized:
            position = self._axis_positions(
                axis_size=axis_size,
                device=device,
                dtype=torch.float32,
            ).unsqueeze(1)
            num_bands = (self.embedding_size + 1) // 2
            frequencies = torch.exp(
                torch.linspace(
                    torch.log(torch.tensor(torch.pi, device=device)),
                    torch.log(torch.tensor(64.0 * torch.pi, device=device)),
                    steps=num_bands,
                    device=device,
                    dtype=torch.float32,
                )
            )
            pe = torch.cat(
                [
                    torch.sin(position * frequencies),
                    torch.cos(position * frequencies),
                ],
                dim=1,
            )[:, : self.embedding_size]
        else:
            position = torch.arange(axis_size, device=device, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.embedding_size, 2, device=device, dtype=torch.float32)
                * (-torch.log(torch.tensor(10000.0, device=device)) / self.embedding_size)
            )
            pe = torch.zeros(axis_size, self.embedding_size, device=device, dtype=torch.float32)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.to(dtype=dtype).unsqueeze(0).expand(batch_size, axis_size, self.embedding_size)

    def _axis_positions(
        self,
        *,
        axis_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if axis_size <= 1:
            return torch.zeros((axis_size,), device=device, dtype=dtype)
        return torch.linspace(-1.0, 1.0, steps=axis_size, device=device, dtype=dtype)

    def _fourier_position_features(
        self,
        *,
        batch_size: int,
        axis_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        pos = self._axis_positions(axis_size=axis_size, device=device, dtype=torch.float32)
        pos_col = pos.unsqueeze(1)
        frequencies = torch.exp(
            torch.linspace(
                torch.log(torch.tensor(torch.pi, device=device)),
                torch.log(torch.tensor(64.0 * torch.pi, device=device)),
                steps=self.positional_num_frequencies,
                device=device,
                dtype=torch.float32,
            )
        )
        features = [pos_col, torch.sin(pos_col * frequencies), torch.cos(pos_col * frequencies)]
        if self.positional_encoding_kind == "poly_fourier_mlp":
            features.extend([pos_col.square(), pos_col.abs()])
        out = torch.cat(features, dim=1).to(dtype=dtype)
        return out.unsqueeze(0).expand(batch_size, axis_size, out.shape[-1])

    def _learned_table_position_embedding(
        self,
        table: torch.Tensor,
        *,
        batch_size: int,
        axis_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if axis_size <= table.shape[1]:
            emb = table[:, :axis_size]
        else:
            emb = F.interpolate(
                table.transpose(1, 2),
                size=axis_size,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
        return emb.to(dtype=dtype).expand(batch_size, axis_size, self.embedding_size)

    def _positional_axis_embedding(
        self,
        *,
        axis: str,
        batch_size: int,
        axis_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        kind = self.positional_encoding_kind
        if kind == "none":
            return None
        if kind == "sinusoidal_index":
            return self._sinusoidal_axis_embedding(
                batch_size=batch_size,
                axis_size=axis_size,
                device=device,
                dtype=dtype,
                normalized=False,
            )
        if kind == "sinusoidal_normalized":
            return self._sinusoidal_axis_embedding(
                batch_size=batch_size,
                axis_size=axis_size,
                device=device,
                dtype=dtype,
                normalized=True,
            )
        if kind in {"fourier_mlp", "poly_fourier_mlp"}:
            features = self._fourier_position_features(
                batch_size=batch_size,
                axis_size=axis_size,
                device=device,
                dtype=dtype,
            )
            encoder = (
                self.row_positional_encoder
                if axis == "row"
                else self.col_positional_encoder
            )
            if encoder is None:
                raise RuntimeError(f"Missing positional encoder for kind={kind}")
            return encoder(features)
        if kind == "learned_table":
            table = (
                self.row_positional_table
                if axis == "row"
                else self.col_positional_table
            )
            if table is None:
                raise RuntimeError("Missing learned positional table")
            return self._learned_table_position_embedding(
                table,
                batch_size=batch_size,
                axis_size=axis_size,
                dtype=dtype,
            )
        raise ValueError(f"Unsupported positional_encoding_kind={kind!r}")

    def _initial_states(
        self,
        row_summary: torch.Tensor,
        col_summary: torch.Tensor,
        row_counts: torch.Tensor,
        col_counts: torch.Tensor,
        rows: int,
        cols: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = row_summary.shape[0]
        row_state = self.row_init.expand(batch_size, rows, self.embedding_size)
        col_state = self.col_init.expand(batch_size, cols, self.embedding_size)
        row_state = row_state + self.row_input_proj(row_summary)
        col_state = col_state + self.col_input_proj(col_summary)
        row_state = row_state + self.row_count_encoder(
            self._count_features(row_counts, cols)
        )
        col_state = col_state + self.col_count_encoder(
            self._count_features(col_counts, rows)
        )
        if self.use_axis_identity_features:
            if (
                self.row_identity_encoder is None
                or self.col_identity_encoder is None
                or self.row_identity_scale is None
                or self.col_identity_scale is None
            ):
                raise RuntimeError("Axis identity modules are not initialized")
            row_state = row_state + self.row_identity_scale * self.row_identity_encoder(
                self._axis_identity_features(
                    batch_size=batch_size,
                    axis_size=rows,
                    device=row_state.device,
                    dtype=row_state.dtype,
                )
            )
            col_state = col_state + self.col_identity_scale * self.col_identity_encoder(
                self._axis_identity_features(
                    batch_size=batch_size,
                    axis_size=cols,
                    device=col_state.device,
                    dtype=col_state.dtype,
                )
            )
        row_pos = self._positional_axis_embedding(
            axis="row",
            batch_size=batch_size,
            axis_size=rows,
            device=row_state.device,
            dtype=row_state.dtype,
        )
        if row_pos is not None:
            row_state = row_state + self.positional_damping_factor * row_pos
        col_pos = self._positional_axis_embedding(
            axis="col",
            batch_size=batch_size,
            axis_size=cols,
            device=col_state.device,
            dtype=col_state.dtype,
        )
        if col_pos is not None:
            col_state = col_state + self.positional_damping_factor * col_pos
        return self.row_init_norm(row_state), self.col_init_norm(col_state)

    def _run_layer(
        self,
        layer: BipartiteMessagePassingLayer,
        row_state: torch.Tensor,
        col_state: torch.Tensor,
        value_emb: torch.Tensor,
        observed: torch.Tensor,
        row_counts: torch.Tensor,
        col_counts: torch.Tensor,
        clean_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(layer, ValueWeightedBipartiteMessagePassingLayer):
            if clean_values is None:
                raise RuntimeError("clean_values are required for value-weighted messages")
            if (
                self.use_activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                return checkpoint(
                    lambda r, c, v, o, rc, cc, x: layer(r, c, v, o, rc, cc, x),
                    row_state,
                    col_state,
                    value_emb,
                    observed,
                    row_counts,
                    col_counts,
                    clean_values,
                    use_reentrant=self.checkpoint_use_reentrant,
                )
            return layer(
                row_state,
                col_state,
                value_emb,
                observed,
                row_counts,
                col_counts,
                clean_values,
            )
        if (
            self.use_activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(
                lambda r, c, v, o, rc, cc: layer(r, c, v, o, rc, cc),
                row_state,
                col_state,
                value_emb,
                observed,
                row_counts,
                col_counts,
                use_reentrant=self.checkpoint_use_reentrant,
            )
        return layer(row_state, col_state, value_emb, observed, row_counts, col_counts)

    def encode_latents(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if src.ndim != 3:
            raise ValueError(f"Expected src shape B x rows x cols, got {tuple(src.shape)}")
        batch_size, rows, cols = src.shape
        del batch_size
        (
            value_emb,
            observed,
            row_counts,
            col_counts,
            row_summary,
            col_summary,
        ) = self._encode_values(src)
        clean_values = torch.where(torch.isnan(src), torch.zeros_like(src), src)
        row_state, col_state = self._initial_states(
            row_summary,
            col_summary,
            row_counts,
            col_counts,
            rows,
            cols,
        )
        for layer in self.message_layers:
            row_state, col_state = self._run_layer(
                layer,
                row_state,
                col_state,
                value_emb,
                observed,
                row_counts,
                col_counts,
                clean_values,
            )
        return row_state, col_state

    def forward(
        self,
        src: torch.Tensor,
        return_embeddings: bool = False,
        decode_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        row_state, col_state = self.encode_latents(src)
        if return_embeddings:
            return self.output_decoder.grid_features(row_state, col_state)

        if decode_mask is not None:
            if decode_mask.shape != src.shape:
                raise ValueError(
                    f"decode_mask shape {tuple(decode_mask.shape)} must match "
                    f"input shape {tuple(src.shape)}"
                )
            batch_idx, row_idx, col_idx = decode_mask.nonzero(as_tuple=True)
            if batch_idx.numel() == 0:
                return row_state.new_empty((0, self.num_outputs))
            rows = row_state[batch_idx, row_idx]
            cols = col_state[batch_idx, col_idx]
            return self.output_decoder.forward_pairs(rows, cols)

        return self.output_decoder.forward_grid(row_state, col_state)


class TabImputeModel(TabImputeModelRowColumnLatent):
    """Compatibility alias."""


__all__ = [
    "BipartiteMessagePassingLayer",
    "ValueWeightedBipartiteMessagePassingLayer",
    "RowColumnLatentDecoder",
    "TabImputeModel",
    "TabImputeModelRowColumnLatent",
]
