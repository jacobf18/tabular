from __future__ import annotations

import torch
from torch import nn

from tabimpute.model.model_new_stable import RMSNorm
from tabimpute.model.model_new_stable_rabel import TabImputeModelPureRaBEL


class RowColumnContextAdapter(nn.Module):
    """Inject row/column summary tokens into per-cell decoder features."""

    def __init__(
        self,
        embedding_size: int,
        *,
        context_scale_init: float = 0.05,
        include_interaction: bool = True,
    ):
        super().__init__()
        self.include_interaction = include_interaction
        self.row_norm = RMSNorm(embedding_size)
        self.col_norm = RMSNorm(embedding_size)
        self.cell_norm = RMSNorm(embedding_size)
        self.row_proj = nn.Linear(embedding_size, embedding_size)
        self.col_proj = nn.Linear(embedding_size, embedding_size)
        self.interaction_proj = (
            nn.Linear(embedding_size, embedding_size)
            if include_interaction
            else None
        )
        self.context_scale = nn.Parameter(torch.tensor(float(context_scale_init)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.row_proj.weight)
        nn.init.zeros_(self.row_proj.bias)
        nn.init.xavier_uniform_(self.col_proj.weight)
        nn.init.zeros_(self.col_proj.bias)
        if self.interaction_proj is not None:
            nn.init.xavier_uniform_(self.interaction_proj.weight)
            nn.init.zeros_(self.interaction_proj.bias)

    def forward(
        self,
        cell_features: torch.Tensor,
        row_summary: torch.Tensor,
        col_summary: torch.Tensor,
    ) -> torch.Tensor:
        row_context = self.row_proj(self.row_norm(row_summary))
        col_context = self.col_proj(self.col_norm(col_summary))
        context = row_context + col_context
        if self.interaction_proj is not None:
            interaction = self.row_norm(row_summary) * self.col_norm(col_summary)
            context = context + self.interaction_proj(interaction)
        return cell_features + self.context_scale.to(cell_features.dtype) * self.cell_norm(context)


class TabImputeModelPureRaBELContextDecoder(TabImputeModelPureRaBEL):
    """Pure RaBEL TabImpute with explicit row/column context at decode time."""

    def __init__(
        self,
        *args,
        context_scale_init: float = 0.05,
        context_include_interaction: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.context_adapter = RowColumnContextAdapter(
            self.transformer_blocks[0].norm1.weight.numel(),
            context_scale_init=context_scale_init,
            include_interaction=context_include_interaction,
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        missing = [
            key
            for key in result.missing_keys
            if not key.startswith("context_adapter.")
        ]
        if strict and (missing or result.unexpected_keys):
            raise RuntimeError(
                "Error(s) in loading state_dict for "
                f"{self.__class__.__name__}: missing={missing}, "
                f"unexpected={result.unexpected_keys}"
            )
        return result

    def _decoder_context_features(
        self,
        src: torch.Tensor,
        num_cls_rows: int,
        num_cls_cols: int,
        decode_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        cell_features = src[:, num_cls_rows:, num_cls_cols:, :]
        row_summary = src[:, num_cls_rows:, :num_cls_cols, :].mean(dim=2)
        col_summary = src[:, :num_cls_rows, num_cls_cols:, :].mean(dim=1)

        if decode_mask is None:
            row_context = row_summary.unsqueeze(2).expand_as(cell_features)
            col_context = col_summary.unsqueeze(1).expand_as(cell_features)
            return self.context_adapter(cell_features, row_context, col_context)

        if decode_mask.shape != cell_features.shape[:-1]:
            raise ValueError(
                f"decode_mask shape {tuple(decode_mask.shape)} must match "
                f"feature grid shape {tuple(cell_features.shape[:-1])}"
            )
        batch_idx, row_idx, col_idx = decode_mask.nonzero(as_tuple=True)
        selected_cells = cell_features[batch_idx, row_idx, col_idx]
        selected_rows = row_summary[batch_idx, row_idx]
        selected_cols = col_summary[batch_idx, col_idx]
        return self.context_adapter(selected_cells, selected_rows, selected_cols)

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

        if return_embeddings or self.decoder is None:
            return src[:, num_cls_rows:, num_cls_cols:, :]
        feats = self._decoder_context_features(
            src,
            num_cls_rows=num_cls_rows,
            num_cls_cols=num_cls_cols,
            decode_mask=decode_mask,
        )
        return self.decoder(feats)


class TabImputeModel(TabImputeModelPureRaBELContextDecoder):
    """Compatibility alias."""


__all__ = [
    "RowColumnContextAdapter",
    "TabImputeModel",
    "TabImputeModelPureRaBELContextDecoder",
]
