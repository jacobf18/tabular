from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from tabimpute.model.model_new_stable import (
    AxialRoPESelfAttention,
    DecoderStable,
    DropPath,
    RMSNorm,
    _apply_rope_to_subset,
    _build_drop_path_schedule,
    _summarize_attention,
)
from tabimpute.model.model_new_stable_rabel import FeatureEncoderPureRaBEL


class ScaledAxialRoPESelfAttention(AxialRoPESelfAttention):
    """Axial RoPE attention with a learnable softmax logit scale.

    The base model uses the standard ``1 / sqrt(head_dim)`` attention scale.
    This variant multiplies queries by a learned scalar before attention, which
    is equivalent to multiplying the attention logits by that scalar while
    preserving the fast SDPA path.
    """

    def __init__(
        self,
        *args,
        logit_scale_init: float = 1.0,
        min_logit_scale: float = 0.125,
        max_logit_scale: float = 32.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if logit_scale_init <= 0:
            raise ValueError("logit_scale_init must be positive.")
        if min_logit_scale <= 0 or max_logit_scale < min_logit_scale:
            raise ValueError("Invalid logit scale clamp range.")
        self.log_attention_scale = nn.Parameter(
            torch.tensor(float(math.log(logit_scale_init)))
        )
        self.min_logit_scale = float(min_logit_scale)
        self.max_logit_scale = float(max_logit_scale)

    def _attention_multiplier(self, dtype: torch.dtype) -> torch.Tensor:
        scale = self.log_attention_scale.exp().clamp(
            self.min_logit_scale,
            self.max_logit_scale,
        )
        return scale.to(dtype=dtype)

    def forward(self, x: torch.Tensor, rope_start_index: int = 0) -> torch.Tensor:
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        if self.rope is not None and rope_start_index < q.size(2):
            rope_seq_len = q.size(2) - rope_start_index
            cos, sin = self.rope.get_cos_sin(rope_seq_len, q.device, q.dtype)
            q = _apply_rope_to_subset(q, cos, sin, rope_start_index, self.rotary_dim)
            k = _apply_rope_to_subset(k, cos, sin, rope_start_index, self.rotary_dim)

        attention_multiplier = self._attention_multiplier(q.dtype)
        q = q * attention_multiplier

        if self.capture_attention_weights:
            logits = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            probs = torch.softmax(logits, dim=-1)
            self.last_attention_stats = {
                **_summarize_attention(probs, logits),
                "attention_logit_scale": float(attention_multiplier.detach().cpu()),
            }
            self.last_attention_weights = (
                probs.detach().cpu() if self.store_attention_weights else None
            )
            probs = F.dropout(probs, p=self.dropout, training=self.training)
            attn_out = torch.matmul(probs, v)
        else:
            self.last_attention_weights = None
            self.last_attention_stats = None
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(
            x.size(0), x.size(1), self.embedding_size
        )
        return self.out_proj(attn_out)


class TransformerEncoderLayerScaledAttention(nn.Module):
    """Stable axial transformer block with scaled feature/datapoint attention."""

    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        drop_path: float = 0.0,
        residual_scale_init: float = 0.1,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        rope_fraction: float = 1.0,
        attention_batch_chunk_size: int | None = None,
        feature_logit_scale_init: float = 1.0,
        datapoint_logit_scale_init: float = 4.0,
        min_logit_scale: float = 0.125,
        max_logit_scale: float = 32.0,
    ):
        super().__init__()
        self.self_attention_between_features = ScaledAxialRoPESelfAttention(
            embedding_size=embedding_size,
            nhead=nhead,
            dropout=attention_dropout,
            use_rope=use_rope,
            rope_base=rope_base,
            rope_fraction=rope_fraction,
            logit_scale_init=feature_logit_scale_init,
            min_logit_scale=min_logit_scale,
            max_logit_scale=max_logit_scale,
        )
        self.self_attention_between_datapoints = ScaledAxialRoPESelfAttention(
            embedding_size=embedding_size,
            nhead=nhead,
            dropout=attention_dropout,
            use_rope=use_rope,
            rope_base=rope_base,
            rope_fraction=rope_fraction,
            logit_scale_init=datapoint_logit_scale_init,
            min_logit_scale=min_logit_scale,
            max_logit_scale=max_logit_scale,
        )

        self.norm1 = RMSNorm(embedding_size, eps=rms_norm_eps)
        self.norm2 = RMSNorm(embedding_size, eps=rms_norm_eps)
        self.norm3 = RMSNorm(embedding_size, eps=rms_norm_eps)
        self.ff_gate = nn.Linear(embedding_size, mlp_hidden_size)
        self.ff_value = nn.Linear(embedding_size, mlp_hidden_size)
        self.ff_out = nn.Linear(mlp_hidden_size, embedding_size)
        self.ff_dropout = nn.Dropout(ffn_dropout)
        self.feature_drop_path = DropPath(drop_path)
        self.datapoint_drop_path = DropPath(drop_path)
        self.ffn_drop_path = DropPath(drop_path)
        self.feature_residual_scale = nn.Parameter(torch.full((1,), residual_scale_init))
        self.datapoint_residual_scale = nn.Parameter(torch.full((1,), residual_scale_init))
        self.ffn_residual_scale = nn.Parameter(torch.full((1,), residual_scale_init))
        self.attention_batch_chunk_size = attention_batch_chunk_size
        self._last_feature_attention_weights = None
        self._last_datapoint_attention_weights = None
        self._last_feature_attention_stats = None
        self._last_datapoint_attention_stats = None

    def set_attention_capture(self, enabled: bool, store_weights: bool = True) -> None:
        self.self_attention_between_features.set_attention_capture(
            enabled,
            store_weights=store_weights,
        )
        self.self_attention_between_datapoints.set_attention_capture(
            enabled,
            store_weights=store_weights,
        )
        if not enabled:
            self._last_feature_attention_weights = None
            self._last_datapoint_attention_weights = None
            self._last_feature_attention_stats = None
            self._last_datapoint_attention_stats = None

    def _run_axis_attention(
        self,
        attention: ScaledAxialRoPESelfAttention,
        src_flat: torch.Tensor,
        rope_start_index: int,
    ) -> torch.Tensor:
        chunk_size = self.attention_batch_chunk_size
        if (
            chunk_size is None
            or chunk_size <= 0
            or src_flat.size(0) <= chunk_size
            or attention.capture_attention_weights
        ):
            return attention(src_flat, rope_start_index=rope_start_index)
        chunks = [
            attention(chunk, rope_start_index=rope_start_index)
            for chunk in src_flat.split(chunk_size, dim=0)
        ]
        return torch.cat(chunks, dim=0)

    def forward(
        self,
        src: torch.Tensor,
        num_cls_rows: int,
        num_cls_cols: int,
    ) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape

        src_norm = self.norm1(src)
        src_flat = src_norm.reshape(batch_size * rows_size, col_size, embedding_size)
        src_att = self._run_axis_attention(
            self.self_attention_between_features,
            src_flat,
            rope_start_index=num_cls_cols,
        )
        self._last_feature_attention_weights = (
            self.self_attention_between_features.last_attention_weights
        )
        self._last_feature_attention_stats = (
            self.self_attention_between_features.last_attention_stats
        )
        src_att = src_att.reshape(batch_size, rows_size, col_size, embedding_size)
        src = src + self.feature_drop_path(self.feature_residual_scale * src_att)

        src_norm = self.norm2(src)
        src_t = src_norm.transpose(1, 2)
        src_flat = src_t.reshape(batch_size * col_size, rows_size, embedding_size)
        src_att = self._run_axis_attention(
            self.self_attention_between_datapoints,
            src_flat,
            rope_start_index=num_cls_rows,
        )
        self._last_datapoint_attention_weights = (
            self.self_attention_between_datapoints.last_attention_weights
        )
        self._last_datapoint_attention_stats = (
            self.self_attention_between_datapoints.last_attention_stats
        )
        src_att = src_att.reshape(batch_size, col_size, rows_size, embedding_size)
        src_att = src_att.transpose(2, 1)
        src = src + self.datapoint_drop_path(self.datapoint_residual_scale * src_att)

        src_norm = self.norm3(src)
        src_mlp = F.silu(self.ff_gate(src_norm)) * self.ff_value(src_norm)
        src_mlp = self.ff_dropout(src_mlp)
        src_mlp = self.ff_out(src_mlp)
        return src + self.ffn_drop_path(self.ffn_residual_scale * src_mlp)

    def get_attention_weights(self, attention_type: str = "features"):
        if attention_type == "features":
            return self._last_feature_attention_weights
        if attention_type == "datapoints":
            return self._last_datapoint_attention_weights
        raise ValueError(
            f"attention_type must be 'features' or 'datapoints', got {attention_type}"
        )

    def get_attention_stats(self, attention_type: str = "features"):
        if attention_type == "features":
            return self._last_feature_attention_stats
        if attention_type == "datapoints":
            return self._last_datapoint_attention_stats
        raise ValueError(
            f"attention_type must be 'features' or 'datapoints', got {attention_type}"
        )


class TabImputeModelPureRaBELScaledAttention(nn.Module):
    """Pure RaBEL TabImpute with learnable feature/datapoint attention scale."""

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
        feature_logit_scale_init: float = 1.0,
        datapoint_logit_scale_init: float = 4.0,
        min_logit_scale: float = 0.125,
        max_logit_scale: float = 32.0,
    ):
        super().__init__()
        self.num_cls = num_cls
        self.use_activation_checkpointing = use_activation_checkpointing
        self.checkpoint_use_reentrant = checkpoint_use_reentrant
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

        drop_path_rates = _build_drop_path_schedule(num_layers, drop_path_rate)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderLayerScaledAttention(
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
                    attention_batch_chunk_size=attention_batch_chunk_size,
                    feature_logit_scale_init=feature_logit_scale_init,
                    datapoint_logit_scale_init=datapoint_logit_scale_init,
                    min_logit_scale=min_logit_scale,
                    max_logit_scale=max_logit_scale,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.decoder = (
            DecoderStable(embedding_size, mlp_hidden_size, num_outputs)
            if num_outputs is not None and num_outputs > 0
            else None
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        missing = [
            key
            for key in result.missing_keys
            if not key.endswith(".log_attention_scale")
        ]
        if strict and (missing or result.unexpected_keys):
            raise RuntimeError(
                "Error(s) in loading state_dict for "
                f"{self.__class__.__name__}: missing={missing}, "
                f"unexpected={result.unexpected_keys}"
            )
        return result

    def set_attention_capture(self, enabled: bool, store_weights: bool = True) -> None:
        for block in self.transformer_blocks:
            block.set_attention_capture(enabled, store_weights=store_weights)

    def get_attention_diagnostics(self) -> list[dict[str, object]]:
        diagnostics = []
        for layer_idx, block in enumerate(self.transformer_blocks):
            diagnostics.append(
                {
                    "layer": layer_idx,
                    "features": {
                        "weights": block.get_attention_weights("features"),
                        "stats": block.get_attention_stats("features"),
                    },
                    "datapoints": {
                        "weights": block.get_attention_weights("datapoints"),
                        "stats": block.get_attention_stats("datapoints"),
                    },
                }
            )
        return diagnostics

    def _run_block(
        self,
        block: TransformerEncoderLayerScaledAttention,
        src: torch.Tensor,
        num_cls_rows: int,
        num_cls_cols: int,
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
                src,
                use_reentrant=self.checkpoint_use_reentrant,
            )
        return block(src, num_cls_rows=num_cls_rows, num_cls_cols=num_cls_cols)

    def forward(
        self,
        src: torch.Tensor,
        return_embeddings: bool = False,
        decode_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src = self.feature_encoder(src)
        num_cls_rows = self.feature_encoder.last_num_cls_rows
        num_cls_cols = self.feature_encoder.last_num_cls_cols

        for block in self.transformer_blocks:
            src = self._run_block(
                block,
                src,
                num_cls_rows=num_cls_rows,
                num_cls_cols=num_cls_cols,
            )

        feats = src[:, num_cls_rows:, num_cls_cols:, :]
        if return_embeddings or self.decoder is None:
            return feats
        if decode_mask is not None:
            if decode_mask.shape != feats.shape[:-1]:
                raise ValueError(
                    f"decode_mask shape {tuple(decode_mask.shape)} must match "
                    f"feature grid shape {tuple(feats.shape[:-1])}"
                )
            feats = feats[decode_mask]
        return self.decoder(feats)


class TabImputeModel(TabImputeModelPureRaBELScaledAttention):
    """Compatibility alias."""


__all__ = [
    "ScaledAxialRoPESelfAttention",
    "TabImputeModel",
    "TabImputeModelPureRaBELScaledAttention",
    "TransformerEncoderLayerScaledAttention",
]
