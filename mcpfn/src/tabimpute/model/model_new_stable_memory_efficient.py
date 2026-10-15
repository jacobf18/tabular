from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from tabimpute.model.model_new_stable import (
    DecoderStable,
    FeatureEncoderStable,
    TransformerEncoderLayerStable,
    _build_drop_path_schedule,
)


class TabImputeModelMemoryEfficient(torch.nn.Module):
    """Memory-efficient sibling of the stable direct TabImpute model.

    This keeps the same module names and tensor shapes as
    ``model_new_stable.TabImputeModel`` so checkpoints are state-dict compatible.
    The main difference is optional activation checkpointing across transformer
    blocks during training.
    """

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
    ):
        super().__init__()
        self.num_cls = num_cls
        self.use_activation_checkpointing = use_activation_checkpointing
        self.checkpoint_use_reentrant = checkpoint_use_reentrant

        self.feature_encoder = FeatureEncoderStable(
            embedding_size=embedding_size,
            num_cls=num_cls,
            use_absolute_positional_embeddings=use_absolute_positional_embeddings,
            positional_damping_factor=positional_damping_factor,
            embedding_dropout=embedding_dropout,
        )

        drop_path_rates = _build_drop_path_schedule(num_layers, drop_path_rate)
        self.transformer_blocks = torch.nn.ModuleList(
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
                    attention_batch_chunk_size=attention_batch_chunk_size,
                )
                for layer_idx in range(num_layers)
            ]
        )
        if num_outputs is not None and num_outputs > 0:
            self.decoder = DecoderStable(embedding_size, mlp_hidden_size, num_outputs)
        else:
            self.decoder = None

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
        block: TransformerEncoderLayerStable,
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


class TabImputeModel(TabImputeModelMemoryEfficient):
    """Drop-in class name for training code imports."""


__all__ = ["TabImputeModel", "TabImputeModelMemoryEfficient"]
