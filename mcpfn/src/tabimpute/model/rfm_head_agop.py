from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RFMHeadAGOP(nn.Module):
    """Laplace-kernel ridge head with closed-form AGOP iteration.

    Drop-in replacement for ``RFMHead`` that follows the actual RFM recipe:
    a Laplace kernel on per-cell embeddings with a Mahalanobis ``M`` that is
    *recomputed* per patch by AGOP fixed-point iteration rather than learned
    by gradient descent.

    Per patch (one batch element):
      1. Sample S anchor cells from the observed set.
      2. For ``iters`` rounds:
           a. Build pairwise distances under current ``√M``.
           b. Set bandwidth γ = median(dist) * exp(log_gamma_scale).
           c. K_ss = exp(-D_ss / γ); solve (K_ss + λI) α = y_S.
           d. (Unless last iter) compute ∇f at anchors closed-form,
              SVD-truncate to a low-rank factor ``√M = Σ Vᵀ``, normalize.
      3. Predict missing cells with K_ts α under the final (M, γ, α).

    ``M`` is stored implicitly as the factor ``sqrtM_factor`` of shape (k, D),
    where k ≤ min(|S|, D) is data-driven via SVD truncation. ``M = Fᵀ F``
    where F is ``sqrtM_factor``; ``√M v`` for any vector v is just F v
    (since M is symmetric PSD).

    Learnable parameters: ``log_lambda`` (ridge) and ``log_gamma_scale``
    (multiplicative correction on the median-distance bandwidth). No ``L``,
    no ``agop_rank``, no ``embed_dim``. State dict is two scalars.
    """

    def __init__(
        self,
        lambda_init: float = 1e-2,
        lambda_floor: float = 1e-4,
        gamma_scale_init: float = 1.0,
        iters: int = 3,
        svd_rel_threshold: float = 1e-3,
        gamma_floor: float = 1e-4,
        detach_M: bool = True,
    ):
        super().__init__()
        # softplus(log_lambda) + floor == lambda_init at init time. Match the
        # numerically-safe inverse-softplus from RFMHead.
        x = float(lambda_init)
        if x > 20.0:
            inv_sp = x
        else:
            inv_sp = float(torch.log(torch.expm1(torch.tensor(x))).item())
        self.log_lambda = nn.Parameter(torch.tensor(inv_sp))
        self.log_gamma_scale = nn.Parameter(
            torch.tensor(math.log(float(gamma_scale_init)))
        )
        self.lambda_floor = float(lambda_floor)
        self.iters = int(iters)
        self.svd_rel_threshold = float(svd_rel_threshold)
        self.gamma_floor = float(gamma_floor)
        self.detach_M = bool(detach_M)

    @property
    def lam(self) -> torch.Tensor:
        return F.softplus(self.log_lambda) + self.lambda_floor

    @staticmethod
    def _pairwise_dist_sq(
        A: torch.Tensor,                 # (n_a, D)
        B: torch.Tensor,                 # (n_b, D)
        sqrtM_factor: torch.Tensor | None,  # None or (k, D)
    ) -> torch.Tensor:
        if sqrtM_factor is None:
            A_p, B_p = A, B
        else:
            A_p = A @ sqrtM_factor.transpose(-1, -2)   # (n_a, k)
            B_p = B @ sqrtM_factor.transpose(-1, -2)   # (n_b, k)
        A_norm = (A_p * A_p).sum(dim=-1)               # (n_a,)
        B_norm = (B_p * B_p).sum(dim=-1)               # (n_b,)
        cross = A_p @ B_p.transpose(-1, -2)            # (n_a, n_b)
        return (A_norm.unsqueeze(-1) + B_norm.unsqueeze(0) - 2 * cross).clamp_min(0)

    def _anchor_gradients(
        self,
        Phi_s: torch.Tensor,             # (S, D)
        alpha: torch.Tensor,             # (S,)
        K_ss_no_ridge: torch.Tensor,     # (S, S)
        D_ss: torch.Tensor,              # (S, S) — Euclidean dist under M
        gamma: torch.Tensor,             # scalar
        sqrtM_factor: torch.Tensor | None,
    ) -> torch.Tensor:
        """Closed-form ∇f(z) at each anchor z = Phi_s[j] for Laplace kernel.

        For K(x, z) = exp(-||√M(z-x)|| / γ):
          ∇_z K(x, z) = K(x, z) * (-1/γ) * M (z - x) / ||√M(z - x)||.
        Then ∇f(z) = Σ_i α_i ∇_z K(x_i, z).
        Vectorizes the inner sum and excludes the i=j term (distance 0).
        """
        eps = 1e-8
        # W[j, i] = α_i * K(x_i, Phi_s[j]) / ||√M(Phi_s[j] - x_i)||
        W = K_ss_no_ridge * alpha.unsqueeze(0) / (D_ss + eps)   # (S, S)
        # Zero the diagonal: gradient term at z = x_i is undefined (norm=0)
        # but its limit is 0 in the smooth-kernel sense.
        idx = torch.arange(W.shape[0], device=W.device)
        W = W.clone()
        W[idx, idx] = 0.0

        W_sum = W.sum(dim=1)                                    # (S,)
        # residual[j, :] = W_sum[j] * Phi_s[j] - Σ_i W[j, i] * Phi_s[i]
        residual = W_sum.unsqueeze(-1) * Phi_s - W @ Phi_s      # (S, D)

        if sqrtM_factor is None:
            M_residual = residual
        else:
            # M v = sqrtM_factor.T @ sqrtM_factor @ v  (M is symmetric PSD)
            M_residual = (residual @ sqrtM_factor.transpose(-1, -2)) @ sqrtM_factor
        return (-1.0 / gamma) * M_residual                      # (S, D)

    @staticmethod
    def _robust_solve(K_ss: torch.Tensor, y_s: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
        """Solve (K_ss) alpha = y_s with jittered retry if K_ss is singular.

        In training on synthetic data, occasional batches produce anchor
        embeddings that are nearly identical (Laplace kernel rows then become
        nearly linearly dependent). The default ridge isn't always enough to
        keep K_ss strictly invertible. Add increasing jitter until the solve
        succeeds, or fall back to lstsq.
        """
        rhs = y_s.unsqueeze(-1)
        try:
            return torch.linalg.solve(K_ss, rhs).squeeze(-1)
        except torch._C._LinAlgError:
            pass
        # First retry with small jitter.
        scale = K_ss.diagonal().abs().max().clamp_min(1.0)
        for jitter_mult in (1e-5, 1e-3, 1e-1):
            try:
                K_jit = K_ss + (jitter_mult * scale) * eye
                return torch.linalg.solve(K_jit, rhs).squeeze(-1)
            except torch._C._LinAlgError:
                continue
        # Last resort: lstsq (always returns a min-norm solution).
        return torch.linalg.lstsq(K_ss + (0.5 * scale) * eye, rhs).solution.squeeze(-1)

    def _update_sqrtM(self, G: torch.Tensor) -> torch.Tensor | None:
        """SVD-truncate G/√S to produce a low-rank √M factor, normalized to ||√M||_op = 1."""
        S = G.shape[0]
        G_scaled = G / math.sqrt(S)
        U, sv, Vt = torch.linalg.svd(G_scaled, full_matrices=False)
        sv_max = sv.max()
        if sv_max.item() < 1e-12:
            # Degenerate AGOP (all gradients ~0); revert to M = I.
            return None
        keep = sv > self.svd_rel_threshold * sv_max
        if int(keep.sum().item()) == 0:
            return None
        sv_kept = sv[keep]
        Vt_kept = Vt[keep, :]
        # √M_normalized = (σ_i / σ_max) v_i v_iᵀ, stored as factor:
        #   sqrtM_factor = (σ_kept / σ_max)[:, None] * Vt_kept,  shape (k, D)
        sqrtM_factor = (sv_kept / sv_max).unsqueeze(-1) * Vt_kept
        return sqrtM_factor

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

        n_obs = int(obs_idx.numel())
        S = min(num_anchors, n_obs)
        if n_obs > S:
            perm = torch.randperm(n_obs, device=phi.device, generator=generator)[:S]
            anchor_idx = obs_idx[perm]
        else:
            anchor_idx = obs_idx

        Phi_s = phi.index_select(0, anchor_idx)             # (S, D)
        y_s = target.index_select(0, anchor_idx)             # (S,)
        Phi_t = phi.index_select(0, mis_idx)                 # (T, D)
        y_t = target.index_select(0, mis_idx)                # (T,)

        eye = torch.eye(int(Phi_s.shape[0]), device=phi.device, dtype=phi.dtype)
        gamma_scale = torch.exp(self.log_gamma_scale.to(phi.dtype))

        sqrtM_factor: torch.Tensor | None = None
        alpha: torch.Tensor | None = None
        gamma: torch.Tensor | None = None

        for t in range(self.iters):
            D_ss_sq = self._pairwise_dist_sq(Phi_s, Phi_s, sqrtM_factor)
            D_ss = D_ss_sq.sqrt()                                       # (S, S)

            # γ = exp(log_gamma_scale) * median of pairwise distances
            # Use only off-diagonal entries (diagonal is 0).
            off_diag = D_ss[~torch.eye(S, dtype=torch.bool, device=phi.device)]
            gamma_base = off_diag.median().clamp_min(self.gamma_floor)
            gamma = gamma_scale * gamma_base

            K_ss_no_ridge = torch.exp(-D_ss / gamma)                    # (S, S)
            K_ss = K_ss_no_ridge + self.lam.to(phi.dtype) * eye
            alpha = self._robust_solve(K_ss, y_s, eye)

            if t < self.iters - 1:
                G = self._anchor_gradients(
                    Phi_s, alpha, K_ss_no_ridge, D_ss, gamma, sqrtM_factor,
                )
                new_factor = self._update_sqrtM(G)
                if new_factor is not None and self.detach_M and self.training:
                    new_factor = new_factor.detach()
                sqrtM_factor = new_factor

        # Final predict.
        D_ts_sq = self._pairwise_dist_sq(Phi_t, Phi_s, sqrtM_factor)
        K_ts = torch.exp(-D_ts_sq.sqrt() / gamma)                       # (T, S)
        y_hat_t = K_ts @ alpha                                          # (T,)
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

    def predict(
        self,
        feats: torch.Tensor,            # (B, R, C, D)
        target: torch.Tensor,           # (B, R, C) — values at observed cells (anything at missing)
        observed_mask: torch.Tensor,    # (B, R, C) bool — True iff cell is observed
        num_anchors: int = 1024,
        generator: torch.Generator | None = None,
        grad_enabled: bool = False,
    ) -> torch.Tensor:
        """Return per-cell predictions of the same shape as ``target``.

        Observed cells carry NaN in the output (caller is expected to substitute
        the original values there). Missing cells carry the kernel-ridge
        predictions in standardized space.

        If ``grad_enabled`` is True, gradients flow through the head's two
        scalars (log_lambda, log_gamma_scale). Used by per-matrix tuning.
        """
        ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
        with ctx:
            B, R, C, D = feats.shape
            feats32 = feats.float().reshape(B, R * C, D)
            target32 = target.float().reshape(B, R * C)
            obs_flat = observed_mask.reshape(B, R * C)
            mis_flat = ~obs_flat

            out = torch.full_like(target32, float("nan"))

            for b in range(B):
                mis_idx = torch.nonzero(mis_flat[b], as_tuple=False).squeeze(-1)
                if mis_idx.numel() == 0 or obs_flat[b].sum() == 0:
                    continue
                y_hat_t, _ = self._solve_one(
                    feats32[b], target32[b], obs_flat[b], mis_flat[b],
                    num_anchors=num_anchors, generator=generator,
                )
                out[b].index_copy_(0, mis_idx, y_hat_t.float())

            return out.reshape(B, R, C).to(feats.dtype)
