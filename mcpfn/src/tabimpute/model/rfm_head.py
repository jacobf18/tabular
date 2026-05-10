from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RFMHead(nn.Module):
    """Kernel-ridge regression head for per-cell matrix imputation.

    For each batch element b, given per-cell embeddings Phi (R, C, D), an
    observation mask (R, C) where 1 = observed and 0 = missing, and the dense
    target tensor (R, C):
      - sample S "anchor" cells from the observed set;
      - solve K_ss @ alpha = y_s with K_ss = (Phi_s M Phi_s^T) + lam * I;
      - predict y_hat for every missing cell via K_ts @ alpha.
    Loss is the MSE on missing cells only.

    M is parameterized as M = L @ L^T with L of shape (D, r) when ``agop_rank``
    is set; otherwise M = I (plain dot-product kernel). Initialization of L
    uses N(0, 1/r) entries so E[L @ L^T] = I (warm-starts near dot-product).
    """

    def __init__(
        self,
        lambda_init: float = 1e-2,
        lambda_floor: float = 1e-4,
        agop_rank: int | None = None,
        embed_dim: int | None = None,
    ):
        super().__init__()
        # softplus(log_lambda) + floor == lambda_init at init time. The naive
        # inverse softplus log(expm1(x)) overflows fp32 for x > ~80, so use the
        # large-x approximation log(expm1(x)) ≈ x for big values.
        x = float(lambda_init)
        if x > 20.0:
            inv_sp = x
        else:
            inv_sp = float(torch.log(torch.expm1(torch.tensor(x))).item())
        self.log_lambda = nn.Parameter(torch.tensor(inv_sp))
        self.lambda_floor = float(lambda_floor)
        if agop_rank is not None:
            assert embed_dim is not None and agop_rank > 0
            std = 1.0 / math.sqrt(agop_rank)
            self.L = nn.Parameter(torch.randn(embed_dim, agop_rank) * std)
        else:
            self.register_parameter("L", None)

    @property
    def lam(self) -> torch.Tensor:
        return F.softplus(self.log_lambda) + self.lambda_floor

    def _solve_one(
        self,
        phi: torch.Tensor,         # (R*C, D) — fp32
        target: torch.Tensor,      # (R*C,)   — fp32
        observed: torch.Tensor,    # (R*C,)   — bool
        missing: torch.Tensor,     # (R*C,)   — bool
        num_anchors: int,
        generator: torch.Generator | None,
    ):
        obs_idx = torch.nonzero(observed, as_tuple=False).squeeze(-1)
        mis_idx = torch.nonzero(missing, as_tuple=False).squeeze(-1)

        if obs_idx.numel() == 0 or mis_idx.numel() == 0:
            empty = phi.new_zeros(0)
            return empty, empty

        S = min(num_anchors, int(obs_idx.numel()))
        if int(obs_idx.numel()) > S:
            perm = torch.randperm(
                int(obs_idx.numel()),
                device=phi.device,
                generator=generator,
            )[:S]
            anchor_idx = obs_idx[perm]
        else:
            anchor_idx = obs_idx

        Phi_s = phi.index_select(0, anchor_idx)             # (S, D)
        y_s = target.index_select(0, anchor_idx)             # (S,)
        Phi_t = phi.index_select(0, mis_idx)                 # (T, D)
        y_t = target.index_select(0, mis_idx)                # (T,)

        eye = torch.eye(int(Phi_s.shape[0]), device=phi.device, dtype=phi.dtype)
        if self.L is not None:
            L = self.L.to(phi.dtype)
            Phi_sL = Phi_s @ L                               # (S, r)
            Phi_tL = Phi_t @ L                               # (T, r)
            K_ss = Phi_sL @ Phi_sL.transpose(-1, -2) + self.lam.to(phi.dtype) * eye
            K_ts = Phi_tL @ Phi_sL.transpose(-1, -2)         # (T, S)
        else:
            K_ss = Phi_s @ Phi_s.transpose(-1, -2) + self.lam.to(phi.dtype) * eye
            K_ts = Phi_t @ Phi_s.transpose(-1, -2)           # (T, S)

        alpha = torch.linalg.solve(K_ss, y_s.unsqueeze(-1)).squeeze(-1)  # (S,)
        y_hat_t = K_ts @ alpha                                            # (T,)
        return y_hat_t, y_t

    def forward(
        self,
        feats: torch.Tensor,        # (B, R, C, D)
        target: torch.Tensor,       # (B, R, C)
        observed_mask: torch.Tensor,  # (B, R, C) bool — True iff cell is observed
        num_anchors: int = 1024,
        generator: torch.Generator | None = None,
    ):
        B, R, C, D = feats.shape
        assert target.shape == (B, R, C)
        assert observed_mask.shape == (B, R, C)

        feats32 = feats.float().reshape(B, R * C, D)
        target32 = target.float().reshape(B, R * C)
        obs_flat = observed_mask.reshape(B, R * C)
        mis_flat = ~obs_flat

        sq_errs = []
        abs_errs = []
        total_missing = 0
        for b in range(B):
            y_hat_t, y_t = self._solve_one(
                feats32[b], target32[b], obs_flat[b], mis_flat[b],
                num_anchors=num_anchors, generator=generator,
            )
            n = int(y_hat_t.numel())
            if n == 0:
                continue
            sq_errs.append((y_hat_t - y_t).pow(2).sum())
            abs_errs.append((y_hat_t - y_t).abs().sum().detach())
            total_missing += n

        if total_missing == 0:
            zero = feats.new_zeros((), dtype=torch.float32)
            return zero, zero.detach()

        loss_missing = torch.stack(sq_errs).sum() / total_missing
        mae_missing = torch.stack(abs_errs).sum() / total_missing
        return loss_missing.to(feats.dtype), mae_missing.to(feats.dtype)
