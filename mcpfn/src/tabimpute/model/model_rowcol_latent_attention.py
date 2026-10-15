from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from tabimpute.model.model_new_stable import DropPath, RMSNorm
from tabimpute.model.model_new_stable import _summarize_attention
from tabimpute.model.model_rowcol_latent import (
    FeedForward,
    TabImputeModelRowColumnLatent,
)


class LatentSelfAttentionLayer(nn.Module):
    """Permutation-equivariant self-attention over row or column latent states."""

    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        attention_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        drop_path: float = 0.0,
        residual_scale_init: float = 0.1,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if embedding_size % num_attention_heads != 0:
            raise ValueError(
                "embedding_size must be divisible by num_attention_heads, got "
                f"{embedding_size} and {num_attention_heads}."
            )
        self.num_attention_heads = num_attention_heads
        self.head_dim = embedding_size // num_attention_heads
        self.attn_norm = RMSNorm(embedding_size, eps=rms_norm_eps)
        self.qkv = nn.Linear(embedding_size, 3 * embedding_size, bias=False)
        self.attn_out = nn.Linear(embedding_size, embedding_size)
        self.ffn_norm = RMSNorm(embedding_size, eps=rms_norm_eps)
        self.ffn = FeedForward(embedding_size, mlp_hidden_size, ffn_dropout)
        self.drop_path = DropPath(drop_path)
        self.attention_dropout = float(attention_dropout)
        self.attn_residual_scale = nn.Parameter(
            torch.full((1,), float(residual_scale_init))
        )
        self.ffn_residual_scale = nn.Parameter(
            torch.full((1,), float(residual_scale_init))
        )
        self.capture_attention_weights = False
        self.store_attention_weights = True
        self.last_attention_weights = None
        self.last_attention_stats = None

    def set_attention_capture(self, enabled: bool, store_weights: bool = True) -> None:
        self.capture_attention_weights = bool(enabled)
        self.store_attention_weights = bool(store_weights)
        if not enabled:
            self.last_attention_weights = None
            self.last_attention_stats = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, embedding_size = x.shape
        qkv = self.qkv(self.attn_norm(x))
        qkv = qkv.view(
            batch_size,
            num_tokens,
            3,
            self.num_attention_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        if self.capture_attention_weights:
            logits = torch.matmul(query, key.transpose(-2, -1)) * (
                self.head_dim ** -0.5
            )
            probs = torch.softmax(logits, dim=-1)
            self.last_attention_stats = _summarize_attention(probs, logits)
            self.last_attention_weights = (
                probs.detach().cpu() if self.store_attention_weights else None
            )
            probs = F.dropout(
                probs,
                p=self.attention_dropout,
                training=self.training,
            )
            attn = torch.matmul(probs, value)
        else:
            self.last_attention_weights = None
            self.last_attention_stats = None
            attn = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.attention_dropout if self.training else 0.0,
            )
        attn = attn.transpose(1, 2).reshape(batch_size, num_tokens, embedding_size)
        x = x + self.drop_path(self.attn_residual_scale * self.attn_out(attn))
        x = x + self.drop_path(self.ffn_residual_scale * self.ffn(self.ffn_norm(x)))
        return x

    def get_attention_weights(self):
        return self.last_attention_weights

    def get_attention_stats(self):
        return self.last_attention_stats


class TabImputeModelRowColumnLatentAttention(TabImputeModelRowColumnLatent):
    """Row/column latent model with row-wise and column-wise latent attention.

    This is a middle ground between the fully factorized row/column message model
    and the older dense cell-token axial transformer. It still only stores row
    and column states, but each message-passing layer is followed by set
    self-attention over rows and over columns.
    """

    def __init__(
        self,
        *args,
        num_attention_heads: int = 8,
        latent_attention_dropout: float | None = None,
        latent_memory_residual_start_layer: int | None = None,
        latent_memory_residual_scale_init: float = 0.0,
        latent_memory_residual_normalize: bool = True,
        **kwargs,
    ) -> None:
        attention_dropout = float(kwargs.get("attention_dropout", 0.0) or 0.0)
        if latent_attention_dropout is None:
            latent_attention_dropout = attention_dropout
        super().__init__(*args, num_attention_heads=num_attention_heads, **kwargs)

        embedding_size = self.embedding_size
        mlp_hidden_size = int(kwargs.get("mlp_hidden_size", 1024))
        num_layers = len(self.message_layers)
        ffn_dropout = float(kwargs.get("ffn_dropout", 0.05))
        drop_path_rate = float(kwargs.get("drop_path_rate", 0.0))
        residual_scale_init = float(kwargs.get("residual_scale_init", 0.1))
        rms_norm_eps = float(kwargs.get("rms_norm_eps", 1e-6))
        if num_layers <= 1:
            drop_path_rates = [drop_path_rate] * max(0, num_layers)
        else:
            drop_path_rates = torch.linspace(
                0.0, drop_path_rate, steps=num_layers
            ).tolist()

        self.row_attention_layers = nn.ModuleList(
            [
                LatentSelfAttentionLayer(
                    embedding_size=embedding_size,
                    num_attention_heads=num_attention_heads,
                    mlp_hidden_size=mlp_hidden_size,
                    attention_dropout=float(latent_attention_dropout),
                    ffn_dropout=ffn_dropout,
                    drop_path=drop_path_rates[layer_idx],
                    residual_scale_init=residual_scale_init,
                    rms_norm_eps=rms_norm_eps,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.col_attention_layers = nn.ModuleList(
            [
                LatentSelfAttentionLayer(
                    embedding_size=embedding_size,
                    num_attention_heads=num_attention_heads,
                    mlp_hidden_size=mlp_hidden_size,
                    attention_dropout=float(latent_attention_dropout),
                    ffn_dropout=ffn_dropout,
                    drop_path=drop_path_rates[layer_idx],
                    residual_scale_init=residual_scale_init,
                    rms_norm_eps=rms_norm_eps,
                )
                for layer_idx in range(num_layers)
            ]
        )
        if latent_memory_residual_start_layer is None:
            self.use_latent_memory_residual = False
            self.latent_memory_residual_start_layer = num_layers + 1
            self.row_memory_residual_scale = None
            self.col_memory_residual_scale = None
            self.row_memory_residual_norm = None
            self.col_memory_residual_norm = None
        else:
            start_layer = int(latent_memory_residual_start_layer)
            if start_layer < 0 or start_layer >= num_layers:
                raise ValueError(
                    "latent_memory_residual_start_layer must be in "
                    f"[0, {num_layers - 1}], got {start_layer}."
                )
            self.use_latent_memory_residual = True
            self.latent_memory_residual_start_layer = start_layer
            self.row_memory_residual_scale = nn.Parameter(
                torch.full((num_layers,), float(latent_memory_residual_scale_init))
            )
            self.col_memory_residual_scale = nn.Parameter(
                torch.full((num_layers,), float(latent_memory_residual_scale_init))
            )
            if latent_memory_residual_normalize:
                self.row_memory_residual_norm = RMSNorm(
                    embedding_size,
                    eps=rms_norm_eps,
                )
                self.col_memory_residual_norm = RMSNorm(
                    embedding_size,
                    eps=rms_norm_eps,
                )
            else:
                self.row_memory_residual_norm = nn.Identity()
                self.col_memory_residual_norm = nn.Identity()

    def set_attention_capture(self, enabled: bool, store_weights: bool = True) -> None:
        for layer in self.row_attention_layers:
            layer.set_attention_capture(enabled, store_weights=store_weights)
        for layer in self.col_attention_layers:
            layer.set_attention_capture(enabled, store_weights=store_weights)

    def get_attention_diagnostics(self) -> list[dict[str, object]]:
        diagnostics = []
        for layer_idx, (row_layer, col_layer) in enumerate(
            zip(self.row_attention_layers, self.col_attention_layers)
        ):
            diagnostics.append(
                {
                    "layer": layer_idx,
                    "rows": {
                        "weights": row_layer.get_attention_weights(),
                        "stats": row_layer.get_attention_stats(),
                    },
                    "cols": {
                        "weights": col_layer.get_attention_weights(),
                        "stats": col_layer.get_attention_stats(),
                    },
                }
            )
        return diagnostics

    def _run_latent_attention(
        self,
        layer: LatentSelfAttentionLayer,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.use_activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(
                lambda x: layer(x),
                state,
                use_reentrant=self.checkpoint_use_reentrant,
            )
        return layer(state)

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
        row_memory = row_state
        col_memory = col_state
        for layer_idx, (message_layer, row_attn, col_attn) in enumerate(zip(
            self.message_layers,
            self.row_attention_layers,
            self.col_attention_layers,
        )):
            row_state, col_state = self._run_layer(
                message_layer,
                row_state,
                col_state,
                value_emb,
                observed,
                row_counts,
                col_counts,
                clean_values,
            )
            row_state = self._run_latent_attention(row_attn, row_state)
            col_state = self._run_latent_attention(col_attn, col_state)
            if (
                self.use_latent_memory_residual
                and layer_idx >= self.latent_memory_residual_start_layer
            ):
                row_scale = self.row_memory_residual_scale[layer_idx].to(
                    dtype=row_state.dtype
                )
                col_scale = self.col_memory_residual_scale[layer_idx].to(
                    dtype=col_state.dtype
                )
                row_state = row_state + row_scale * self.row_memory_residual_norm(
                    row_memory
                )
                col_state = col_state + col_scale * self.col_memory_residual_norm(
                    col_memory
                )
        return row_state, col_state


class TabImputeModel(TabImputeModelRowColumnLatentAttention):
    """Compatibility alias."""


__all__ = [
    "LatentSelfAttentionLayer",
    "TabImputeModel",
    "TabImputeModelRowColumnLatentAttention",
]
