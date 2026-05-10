"""Paper method: User-Item Gaussian Kernel variant (Algorithm 4).

Reference: "Nearest Neighbors for Matrix Estimation Interpreted as Blind
Regression for Latent Variable Model" (Lee, Li, Shah, Song; arXiv:1705.04867),
section on real-data experiments.

Math summary (with beta = 2):
  - For each pair of users (u, v) let Q_row(u, v) be the set of columns both
    have observed. Sample-variance row dissimilarity is
        s_row(u, v) = Var[ Z[u, Q] - Z[v, Q] ] (unbiased, divided by |Q|-1).
  - For each pair of items (i, j) let Q_col(i, j) be the set of users that
    rated both. Sample-variance column dissimilarity
        s_col(i, j) = Var[ Z[Q, i] - Z[Q, j] ].
  - Combined dissimilarity for a (u, v, i, j) quadruple
        D(u, v, i, j) = max( s_row(u, v), s_col(i, j) ).
  - Gaussian kernel weight
        w = exp( -lambda * D ).
    (With beta = 2 in the paper's notation the kernel reduces to straight
    exp(-lambda * s); the "beta" knob selects the exponent applied to the
    dissimilarity.)
  - First-order estimate
        z_hat(u, i) = sum_{(v,j) valid} w * ( Z[v, i] + Z[u, j] - Z[v, j] )
                        / sum_{(v,j) valid} w
    where "valid" means all four entries (u, j), (v, i), (v, j), (u, i-anchor)
    are observed.

Speed:
  - The dissimilarity matrices are computed in closed form with three dense
    matmuls over the (up to 2000 x 2000) observed mask + value matrices, so
    the O(m^2 + n^2) precomputation is fast.
  - For each withheld entry we restrict to the intersection of {v where
    Z[v,i] observed} and {j where Z[u,j] observed}, then trim to the top-K
    closest candidates by their marginal dissimilarity. K is set by
    ``k_neighbors`` (default 100). This is a documented approximation: with
    K >= typical row/column density it reproduces the exact algorithm.

Fallback hierarchy if no valid quadruples exist:
  1. row mean + global-mean correction
  2. column mean
  3. global mean
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from .base import BaseMatrixCompleter, MethodResult


class PaperGaussianKernel(BaseMatrixCompleter):
    name = "paper_gaussian_kernel"

    def __init__(
        self,
        beta: float = 2.0,
        lmbda: float = 1.0,
        k_neighbors: int = 100,
        min_overlap: int = 2,
    ):
        super().__init__(beta=beta, lmbda=lmbda, k_neighbors=k_neighbors, min_overlap=min_overlap)
        self.beta = float(beta)
        self.lmbda = float(lmbda)
        self.k_neighbors = int(k_neighbors)
        self.min_overlap = int(min_overlap)

    # ------------------------------------------------------------------ math
    @staticmethod
    def _pairwise_sample_var_dissim(
        Z: np.ndarray, M: np.ndarray, min_overlap: int
    ) -> np.ndarray:
        """Pairwise row-wise sample-variance dissimilarity.

        Z is (n, d), already zero where unobserved. M is (n, d) boolean mask.

        Returns an (n, n) float32 matrix where entry (u, v) is the sample
        variance of (Z[u, Q] - Z[v, Q]) for their observed-column overlap Q.
        Entries with overlap < ``min_overlap`` are set to +inf (treated as
        "no valid similarity").
        """
        Zf = Z.astype(np.float32, copy=False)
        Mf = M.astype(np.float32, copy=False)
        Z2 = Zf * Zf

        # All pairwise sums over the intersection of observed columns.
        # (Z @ M.T)[u,v]  = sum_j m_vj * z_uj              (Z is zero off-mask)
        # (M @ Z.T)[u,v]  = (Z @ M.T)[v,u]
        ZMt = Zf @ Mf.T
        Z2Mt = Z2 @ Mf.T
        ZZt = Zf @ Zf.T

        d_sum = ZMt - ZMt.T
        d_sq_sum = Z2Mt + Z2Mt.T - 2.0 * ZZt
        overlap = (Mf @ Mf.T).astype(np.float32)

        # Sample variance = (d_sq_sum - d_sum^2 / overlap) / (overlap - 1)
        # Guard divisions.
        safe_n = np.where(overlap >= max(min_overlap, 2), overlap, np.nan).astype(np.float32)
        var = (d_sq_sum - (d_sum * d_sum) / safe_n) / (safe_n - 1.0)

        # Numerical cleanup: tiny negatives from cancellation -> clamp to 0.
        var = np.where(var < 0.0, 0.0, var)
        # Diagonal is Var of zero diff == 0 when overlap > 0, but keep as +inf
        # to avoid self-matching in the top-K step.
        np.fill_diagonal(var, np.inf)
        # Missing overlap -> +inf
        var = np.where(np.isnan(var), np.inf, var)
        return var.astype(np.float32)

    # ------------------------------------------------------------- predict
    def fit_predict(
        self,
        train_matrix: sp.csr_matrix,
        test_rows: np.ndarray,
        test_cols: np.ndarray,
        *,
        rating_scale: Optional[tuple[float, float]] = None,
        dataset: Optional[str] = None,
        withhold_fraction: Optional[float] = None,
    ) -> MethodResult:
        n_rows, n_cols = train_matrix.shape
        Z = np.asarray(train_matrix.toarray(), dtype=np.float32)
        M = (Z != 0.0)

        # Precompute dissimilarities.
        s_row = self._pairwise_sample_var_dissim(Z, M, self.min_overlap)          # (m, m)
        s_col = self._pairwise_sample_var_dissim(Z.T, M.T, self.min_overlap)      # (n, n)

        row_means = np.zeros(n_rows, dtype=np.float32)
        col_means = np.zeros(n_cols, dtype=np.float32)
        row_counts = M.sum(axis=1)
        col_counts = M.sum(axis=0)
        np.divide(Z.sum(axis=1), np.maximum(row_counts, 1), out=row_means, where=row_counts > 0)
        np.divide(Z.sum(axis=0), np.maximum(col_counts, 1), out=col_means, where=col_counts > 0)
        global_mean = self._global_mean(train_matrix)

        preds = np.empty(test_rows.size, dtype=np.float32)
        fallback = 0
        lam = self.lmbda

        for idx in range(test_rows.size):
            u = int(test_rows[idx])
            i = int(test_cols[idx])

            # candidates: v where Z[v, i] observed, v != u
            v_mask = M[:, i].copy()
            v_mask[u] = False
            v_candidates = np.nonzero(v_mask)[0]

            # candidates: j where Z[u, j] observed, j != i
            j_mask = M[u, :].copy()
            j_mask[i] = False
            j_candidates = np.nonzero(j_mask)[0]

            if v_candidates.size == 0 or j_candidates.size == 0:
                # Fallback: row mean + column offset.
                preds[idx] = self._fallback(u, i, row_means, col_means, row_counts, col_counts, global_mean)
                fallback += 1
                continue

            # Top-K by marginal dissimilarity.
            if v_candidates.size > self.k_neighbors:
                dv = s_row[u, v_candidates]
                top_v = np.argpartition(dv, self.k_neighbors - 1)[: self.k_neighbors]
                v_candidates = v_candidates[top_v]
            if j_candidates.size > self.k_neighbors:
                dj = s_col[i, j_candidates]
                top_j = np.argpartition(dj, self.k_neighbors - 1)[: self.k_neighbors]
                j_candidates = j_candidates[top_j]

            # Restrict to (v, j) pairs where Z[v, j] is also observed.
            Z_vj = Z[np.ix_(v_candidates, j_candidates)]          # (nv, nj)
            M_vj = M[np.ix_(v_candidates, j_candidates)]          # (nv, nj) boolean
            s_row_uv = s_row[u, v_candidates]                      # (nv,)
            s_col_ij = s_col[i, j_candidates]                      # (nj,)
            D = np.maximum(s_row_uv[:, None], s_col_ij[None, :])   # (nv, nj)

            # Guard against inf / nan dissimilarities (overlap < min_overlap).
            valid = M_vj & np.isfinite(D)
            if not valid.any():
                preds[idx] = self._fallback(u, i, row_means, col_means, row_counts, col_counts, global_mean)
                fallback += 1
                continue

            w = np.exp(-lam * D, where=valid, out=np.zeros_like(D))
            w = w * valid

            contrib = Z[v_candidates, i][:, None] + Z[u, j_candidates][None, :] - Z_vj
            num = float(np.sum(w * contrib))
            den = float(np.sum(w))
            if den <= 1e-12 or not np.isfinite(den):
                preds[idx] = self._fallback(u, i, row_means, col_means, row_counts, col_counts, global_mean)
                fallback += 1
                continue
            preds[idx] = num / den

        preds = self._clip_scale(preds, rating_scale)
        return MethodResult(
            predictions=preds,
            fallback_count=fallback,
            extra={"beta": self.beta, "lambda": self.lmbda, "k_neighbors": self.k_neighbors},
        )

    # ----------------------------------------------------------- fallback
    @staticmethod
    def _fallback(
        u: int,
        i: int,
        row_means: np.ndarray,
        col_means: np.ndarray,
        row_counts: np.ndarray,
        col_counts: np.ndarray,
        global_mean: float,
    ) -> float:
        if row_counts[u] > 0 and col_counts[i] > 0:
            return 0.5 * (float(row_means[u]) + float(col_means[i]))
        if row_counts[u] > 0:
            return float(row_means[u])
        if col_counts[i] > 0:
            return float(col_means[i])
        return float(global_mean)


def build(
    beta: float = 2.0,
    lmbda: float = 1.0,
    k_neighbors: int = 100,
    min_overlap: int = 2,
    **_: object,
) -> PaperGaussianKernel:
    return PaperGaussianKernel(
        beta=beta, lmbda=lmbda, k_neighbors=k_neighbors, min_overlap=min_overlap
    )
