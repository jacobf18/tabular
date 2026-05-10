"""User-user collaborative filtering with mean-adjusted cosine similarity.

Prediction formula:
    r_hat(u, i) = mu_u + (sum_{v in N_k(u; i)} sim(u, v) * (r(v, i) - mu_v))
                       / (sum_{v in N_k(u; i)} |sim(u, v)|)
with N_k(u; i) = top-k users that have rated item i, ranked by similarity.

Similarity uses shrunk cosine over the mean-centered rating vectors restricted
to column overlap.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from .base import BaseMatrixCompleter, MethodResult


class UserUserCF(BaseMatrixCompleter):
    name = "user_user_cf"

    def __init__(self, k_neighbors: int = 50, shrinkage: float = 25.0):
        super().__init__(k_neighbors=k_neighbors, shrinkage=shrinkage)
        self.k_neighbors = int(k_neighbors)
        self.shrinkage = float(shrinkage)

    def _compute_similarity(self, Z: np.ndarray, M: np.ndarray, row_means: np.ndarray) -> np.ndarray:
        # Mean-center observed entries only
        Zc = (Z - row_means[:, None]) * M
        norms = np.sqrt(np.sum(Zc * Zc, axis=1))  # (m,)
        denom = np.outer(norms, norms) + 1e-12
        sim = (Zc @ Zc.T) / denom
        # Shrinkage: multiply by overlap / (overlap + shrinkage)
        overlap = (M.astype(np.float32) @ M.astype(np.float32).T)
        sim *= overlap / (overlap + self.shrinkage)
        np.fill_diagonal(sim, 0.0)
        return sim.astype(np.float32)

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
        M = (Z != 0.0).astype(np.float32)
        row_counts = M.sum(axis=1)
        col_counts = M.sum(axis=0)
        row_means = np.zeros(n_rows, dtype=np.float32)
        col_means = np.zeros(n_cols, dtype=np.float32)
        np.divide(Z.sum(axis=1), np.maximum(row_counts, 1), out=row_means, where=row_counts > 0)
        np.divide(Z.sum(axis=0), np.maximum(col_counts, 1), out=col_means, where=col_counts > 0)
        global_mean = self._global_mean(train_matrix)

        sim = self._compute_similarity(Z, M.astype(bool), row_means)

        preds = np.empty(test_rows.size, dtype=np.float32)
        fallback = 0
        for idx in range(test_rows.size):
            u = int(test_rows[idx])
            i = int(test_cols[idx])
            v_mask = M[:, i].astype(bool).copy()
            v_mask[u] = False
            cand = np.nonzero(v_mask)[0]
            if cand.size == 0:
                preds[idx] = self._simple_fallback(u, i, row_means, col_means, row_counts, col_counts, global_mean)
                fallback += 1
                continue
            sims = sim[u, cand]
            # top-k by absolute similarity
            if cand.size > self.k_neighbors:
                top = np.argpartition(-np.abs(sims), self.k_neighbors - 1)[: self.k_neighbors]
                cand = cand[top]
                sims = sims[top]
            r_vi = Z[cand, i]
            mu_v = row_means[cand]
            num = float(np.sum(sims * (r_vi - mu_v)))
            den = float(np.sum(np.abs(sims)))
            if den < 1e-12:
                preds[idx] = float(row_means[u]) if row_counts[u] > 0 else float(global_mean)
                fallback += 1
                continue
            preds[idx] = float(row_means[u]) + num / den

        preds = self._clip_scale(preds, rating_scale)
        return MethodResult(
            predictions=preds,
            fallback_count=fallback,
            extra={"k_neighbors": self.k_neighbors, "shrinkage": self.shrinkage},
        )

    @staticmethod
    def _simple_fallback(u, i, row_means, col_means, row_counts, col_counts, global_mean):
        if row_counts[u] > 0:
            return float(row_means[u])
        if col_counts[i] > 0:
            return float(col_means[i])
        return float(global_mean)


def build(k_neighbors: int = 50, shrinkage: float = 25.0, **_: object) -> UserUserCF:
    return UserUserCF(k_neighbors=k_neighbors, shrinkage=shrinkage)
