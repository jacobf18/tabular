from __future__ import annotations

import importlib.resources
import math

import torch
import torch.nn.functional as F
from torch import nn

from tabimpute.model.model_rowcol_latent_attention import (
    TabImputeModelRowColumnLatentAttention,
)


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class TabImputeModelRowColumnALSHybrid(TabImputeModelRowColumnLatentAttention):
    """Row/column latent encoder with explicit unrolled low-rank ALS refinement."""

    def __init__(
        self,
        *args,
        als_rank: int = 10,
        als_steps: int = 4,
        als_ridge: float = 1e-3,
        als_update_damping: float = 0.85,
        als_ridge_scale_by_obs: bool = False,
        als_solve_method: str = "cholesky",
        als_jitter: float = 1e-6,
        als_factor_norm_clip: float | None = None,
        als_init_scale: float = 0.05,
        logit_sigma_init: float = 0.025,
        logit_sigma_min: float = 0.003,
        use_neural_residual: bool = False,
        neural_residual_scale_init: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if als_rank <= 0:
            raise ValueError(f"als_rank must be positive, got {als_rank}.")
        if als_steps < 0:
            raise ValueError(f"als_steps must be non-negative, got {als_steps}.")
        if als_ridge <= 0.0:
            raise ValueError(f"als_ridge must be positive, got {als_ridge}.")
        if not 0.0 < als_update_damping <= 1.0:
            raise ValueError(
                "als_update_damping must be in (0, 1], got "
                f"{als_update_damping}."
            )
        if als_solve_method not in {"cholesky", "solve"}:
            raise ValueError(
                "als_solve_method must be 'cholesky' or 'solve', got "
                f"{als_solve_method!r}."
            )
        if als_jitter < 0.0:
            raise ValueError(f"als_jitter must be non-negative, got {als_jitter}.")
        if als_factor_norm_clip is not None and als_factor_norm_clip <= 0.0:
            raise ValueError(
                "als_factor_norm_clip must be positive when set, got "
                f"{als_factor_norm_clip}."
            )
        if logit_sigma_init <= logit_sigma_min:
            raise ValueError("logit_sigma_init must be greater than logit_sigma_min.")

        self.als_rank = int(als_rank)
        self.als_steps = int(als_steps)
        self.als_ridge = float(als_ridge)
        self.als_update_damping = float(als_update_damping)
        self.als_ridge_scale_by_obs = bool(als_ridge_scale_by_obs)
        self.als_solve_method = str(als_solve_method)
        self.als_jitter = float(als_jitter)
        self.als_factor_norm_clip = (
            None if als_factor_norm_clip is None else float(als_factor_norm_clip)
        )
        self.logit_sigma_min = float(logit_sigma_min)
        self.use_neural_residual = bool(use_neural_residual)

        self.row_factor_head = nn.Linear(self.embedding_size, self.als_rank)
        self.col_factor_head = nn.Linear(self.embedding_size, self.als_rank)
        self.row_bias_head = nn.Linear(self.embedding_size, 1)
        self.col_bias_head = nn.Linear(self.embedding_size, 1)
        self.prediction_scale = nn.Parameter(torch.ones(()))
        self.prediction_bias = nn.Parameter(torch.zeros(()))
        self.bias_scale = nn.Parameter(torch.zeros(()))
        self.logit_sigma_unconstrained = nn.Parameter(
            torch.tensor(
                _inverse_softplus(logit_sigma_init - logit_sigma_min),
                dtype=torch.float32,
            )
        )
        if self.use_neural_residual:
            self.neural_residual_head = nn.Linear(self.embedding_size, 1)
            self.neural_residual_scale = nn.Parameter(
                torch.full((1,), float(neural_residual_scale_init))
            )
        else:
            self.neural_residual_head = None
            self.neural_residual_scale = None

        nn.init.normal_(self.row_factor_head.weight, mean=0.0, std=als_init_scale)
        nn.init.normal_(self.col_factor_head.weight, mean=0.0, std=als_init_scale)
        nn.init.zeros_(self.row_factor_head.bias)
        nn.init.zeros_(self.col_factor_head.bias)
        nn.init.zeros_(self.row_bias_head.weight)
        nn.init.zeros_(self.row_bias_head.bias)
        nn.init.zeros_(self.col_bias_head.weight)
        nn.init.zeros_(self.col_bias_head.bias)

        borders_path = importlib.resources.files("tabimpute") / "data" / "borders.pt"
        with importlib.resources.as_file(borders_path) as path:
            borders = torch.load(path, map_location="cpu")
        if borders.numel() - 1 != self.num_outputs:
            raise ValueError(
                f"num_outputs={self.num_outputs} does not match borders "
                f"({borders.numel() - 1})."
            )
        bucket_means = borders[:-1] + 0.5 * (borders[1:] - borders[:-1])
        self.register_buffer("bucket_means", bucket_means.float(), persistent=False)

    def _initial_factors(
        self,
        row_state: torch.Tensor,
        col_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row_factors = self.row_factor_head(row_state).float()
        col_factors = self.col_factor_head(col_state).float()
        row_bias = self.row_bias_head(row_state).squeeze(-1).float()
        col_bias = self.col_bias_head(col_state).squeeze(-1).float()
        return row_factors, col_factors, row_bias, col_bias

    def _solve_regularized_system(
        self,
        system: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        system = 0.5 * (system + system.transpose(-1, -2))
        if self.als_jitter > 0.0:
            eye = torch.eye(
                self.als_rank,
                device=system.device,
                dtype=system.dtype,
            )
            system = system + self.als_jitter * eye.view(
                *((1,) * (system.ndim - 2)),
                self.als_rank,
                self.als_rank,
            )
        if self.als_solve_method == "solve":
            return torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)

        chol, info = torch.linalg.cholesky_ex(system)
        if torch.any(info > 0):
            return torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)
        return torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)

    def _apply_factor_norm_clip(self, factors: torch.Tensor) -> torch.Tensor:
        if self.als_factor_norm_clip is None:
            return factors
        norms = torch.linalg.vector_norm(factors, dim=-1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(self.als_factor_norm_clip / norms, max=1.0)
        return factors * scale

    def _als_refine(
        self,
        src: torch.Tensor,
        row_factors: torch.Tensor,
        col_factors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clean = torch.where(torch.isnan(src), torch.zeros_like(src), src).float()
        observed = (~torch.isnan(src)).float()
        eye = torch.eye(
            self.als_rank,
            device=src.device,
            dtype=torch.float32,
        )
        row_factors = row_factors.float()
        col_factors = col_factors.float()
        for _ in range(self.als_steps):
            row_system = torch.einsum(
                "brc,bck,bcl->brkl",
                observed,
                col_factors,
                col_factors,
            )
            row_rhs = torch.einsum("brc,brc,bck->brk", observed, clean, col_factors)
            row_obs = observed.sum(dim=2).clamp_min(1.0)
            row_ridge = self.als_ridge * (
                row_obs if self.als_ridge_scale_by_obs else torch.ones_like(row_obs)
            )
            row_system = row_system + (
                row_ridge.view(*row_ridge.shape, 1, 1)
                * eye.view(1, 1, self.als_rank, self.als_rank)
            )
            row_solved = self._solve_regularized_system(row_system, row_rhs)
            row_factors = (
                (1.0 - self.als_update_damping) * row_factors
                + self.als_update_damping * row_solved
            )
            row_factors = self._apply_factor_norm_clip(row_factors)

            col_system = torch.einsum(
                "brc,brk,brl->bckl",
                observed,
                row_factors,
                row_factors,
            )
            col_rhs = torch.einsum("brc,brc,brk->bck", observed, clean, row_factors)
            col_obs = observed.sum(dim=1).clamp_min(1.0)
            col_ridge = self.als_ridge * (
                col_obs if self.als_ridge_scale_by_obs else torch.ones_like(col_obs)
            )
            col_system = col_system + (
                col_ridge.view(*col_ridge.shape, 1, 1)
                * eye.view(1, 1, self.als_rank, self.als_rank)
            )
            col_solved = self._solve_regularized_system(col_system, col_rhs)
            col_factors = (
                (1.0 - self.als_update_damping) * col_factors
                + self.als_update_damping * col_solved
            )
            col_factors = self._apply_factor_norm_clip(col_factors)
        return row_factors, col_factors

    def _point_grid(
        self,
        src: torch.Tensor,
        row_state: torch.Tensor,
        col_state: torch.Tensor,
    ) -> torch.Tensor:
        row_factors, col_factors, row_bias, col_bias = self._initial_factors(
            row_state,
            col_state,
        )
        row_factors, col_factors = self._als_refine(src, row_factors, col_factors)
        point = torch.einsum("brk,bck->brc", row_factors, col_factors)
        point = self.prediction_scale.float() * point + self.prediction_bias.float()
        point = point + self.bias_scale.float() * (
            row_bias.unsqueeze(2) + col_bias.unsqueeze(1)
        )
        if self.use_neural_residual:
            if self.neural_residual_head is None or self.neural_residual_scale is None:
                raise RuntimeError("Neural residual modules are not initialized.")
            features = self.output_decoder.grid_features(row_state, col_state)
            residual = self.neural_residual_head(features).squeeze(-1).float()
            point = point + self.neural_residual_scale.float() * residual
        return point

    def _point_pairs(
        self,
        point_grid: torch.Tensor,
        decode_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_idx, row_idx, col_idx = decode_mask.nonzero(as_tuple=True)
        if batch_idx.numel() == 0:
            return point_grid.new_empty((0,))
        return point_grid[batch_idx, row_idx, col_idx]

    def _logits_from_point(self, point: torch.Tensor) -> torch.Tensor:
        centers = self.bucket_means.to(device=point.device, dtype=torch.float32)
        sigma = F.softplus(self.logit_sigma_unconstrained).float()
        sigma = sigma + self.logit_sigma_min
        logits = -0.5 * ((centers.view(*((1,) * point.ndim), -1) - point.unsqueeze(-1)) / sigma).square()
        logits = logits - logits.amax(dim=-1, keepdim=True)
        return logits.to(dtype=point.dtype)

    def forward(
        self,
        src: torch.Tensor,
        return_embeddings: bool = False,
        decode_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        row_state, col_state = self.encode_latents(src)
        if return_embeddings:
            return self.output_decoder.grid_features(row_state, col_state)

        point_grid = self._point_grid(src, row_state, col_state)
        if decode_mask is not None:
            if decode_mask.shape != src.shape:
                raise ValueError(
                    f"decode_mask shape {tuple(decode_mask.shape)} must match "
                    f"input shape {tuple(src.shape)}"
                )
            return self._logits_from_point(self._point_pairs(point_grid, decode_mask))
        return self._logits_from_point(point_grid)


class TabImputeModel(TabImputeModelRowColumnALSHybrid):
    """Compatibility alias."""


__all__ = [
    "TabImputeModel",
    "TabImputeModelRowColumnALSHybrid",
]
