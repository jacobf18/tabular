"""SoftImpute baseline.

First tries to use the ``fancyimpute.SoftImpute`` package if installed.
Falls back to a compact local implementation (Mazumder, Hastie, Tibshirani
2010) based on soft-thresholded truncated SVD.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from .base import BaseMatrixCompleter, MethodResult


def _try_import_fancyimpute():
    try:
        from fancyimpute import SoftImpute  # type: ignore
        return SoftImpute
    except Exception:
        return None


class _SoftImputeImpl:
    """Local SoftImpute implementation using truncated SVD soft-thresholding.

    The update rule at step t is
        X_{t+1} = SoftSVT_lambda( P_Omega(X) + P_Omega^c(X_t) )
    where SoftSVT_lambda zeroes singular values below ``lambda`` and shrinks
    the rest by ``lambda``.
    """

    def __init__(self, lmbda: float, max_rank: int, max_iters: int, tol: float):
        self.lmbda = float(lmbda)
        self.max_rank = int(max_rank)
        self.max_iters = int(max_iters)
        self.tol = float(tol)

    def fit(self, X_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Initialize with observed entries and zero-filled holes (already
        # mean-centered upstream so zero is the right imputation prior).
        X = np.where(mask, X_obs, 0.0).astype(np.float32)

        use_torch = False
        try:
            import torch  # type: ignore
            use_torch = torch.cuda.is_available()
        except Exception:
            torch = None
            use_torch = False

        if use_torch:
            return self._fit_torch(X_obs, mask, X, torch)
        return self._fit_numpy(X_obs, mask, X)

    def _fit_torch(self, X_obs: np.ndarray, mask: np.ndarray, X: np.ndarray, torch) -> np.ndarray:
        device = "cuda"
        X_t = torch.as_tensor(X, device=device, dtype=torch.float32)
        mask_t = torch.as_tensor(mask, device=device)
        X_obs_t = torch.as_tensor(X_obs, device=device, dtype=torch.float32)
        k = min(self.max_rank, min(X.shape) - 1)
        prev_frob = None
        for _ in range(self.max_iters):
            # Randomized low-rank SVD on GPU — much faster than full SVD.
            try:
                U, S, V = torch.svd_lowrank(X_t, q=min(k + 10, min(X_t.shape)))
            except RuntimeError:
                break
            U = U[:, :k]
            S = S[:k]
            V = V[:, :k]
            S_thresh = torch.clamp(S - self.lmbda, min=0.0)
            nz = int((S_thresh > 0).sum().item())
            if nz == 0:
                X_new = torch.zeros_like(X_t)
            else:
                X_new = (U[:, :nz] * S_thresh[:nz]) @ V[:, :nz].T
            X_new = torch.where(mask_t, X_obs_t, X_new)
            frob = float(torch.linalg.norm(X_new - X_t).item())
            X_t = X_new
            if prev_frob is not None and abs(prev_frob - frob) / (prev_frob + 1e-9) < self.tol:
                break
            prev_frob = frob
        return X_t.detach().cpu().numpy().astype(np.float32)

    def _fit_numpy(self, X_obs: np.ndarray, mask: np.ndarray, X: np.ndarray) -> np.ndarray:
        try:
            from sklearn.utils.extmath import randomized_svd  # type: ignore

            def _svd(M, k):
                return randomized_svd(M, n_components=k, random_state=0, n_iter=2)
        except Exception:
            def _svd(M, k):
                U, S, Vt = np.linalg.svd(M, full_matrices=False)
                return U[:, :k], S[:k], Vt[:k, :]

        prev_frob = None
        for _ in range(self.max_iters):
            k = min(self.max_rank, min(X.shape) - 1)
            try:
                U, S, Vt = _svd(X, k)
            except Exception:
                break
            S_thresh = np.maximum(S - self.lmbda, 0.0)
            rank = int((S_thresh > 0).sum())
            if rank == 0:
                X_new = np.zeros_like(X)
            else:
                X_new = (U[:, :rank] * S_thresh[:rank]) @ Vt[:rank, :]
            X_new[mask] = X_obs[mask]
            frob = float(np.linalg.norm(X_new - X))
            X = X_new
            if prev_frob is not None and abs(prev_frob - frob) / (prev_frob + 1e-9) < self.tol:
                break
            prev_frob = frob
        return X


class SoftImputeWrapper(BaseMatrixCompleter):
    name = "softimpute"

    def __init__(
        self,
        lmbda: float = 2.0,
        max_rank: int = 30,
        max_iters: int = 100,
        tol: float = 1e-4,
        prefer_fancyimpute: bool = False,
    ):
        super().__init__(lmbda=lmbda, max_rank=max_rank, max_iters=max_iters, tol=tol)
        self.lmbda = float(lmbda)
        self.max_rank = int(max_rank)
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.prefer_fancyimpute = prefer_fancyimpute

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
        Z = np.asarray(train_matrix.toarray(), dtype=np.float32)
        M = (Z != 0.0)
        # Work in mean-centered space so zeros don't bias SoftImpute.
        global_mean = self._global_mean(train_matrix)

        SoftImputeExt = _try_import_fancyimpute() if self.prefer_fancyimpute else None
        if SoftImputeExt is not None:
            # fancyimpute expects NaN-coded missing values and works on the raw
            # scale; we still center so ``lambda`` has a consistent meaning.
            X_in = np.where(M, Z - global_mean, np.nan).astype(np.float32)
            X_fit = SoftImputeExt(
                max_rank=self.max_rank,
                max_iters=self.max_iters,
                shrinkage_value=self.lmbda,
                convergence_threshold=self.tol,
                verbose=False,
            ).fit_transform(X_in)
            X_filled = np.asarray(X_fit) + global_mean
            backend = "fancyimpute"
        else:
            X_zero = np.where(M, Z - global_mean, 0.0).astype(np.float32)
            impl = _SoftImputeImpl(self.lmbda, self.max_rank, self.max_iters, self.tol)
            X_fit = impl.fit(X_zero, M)
            X_filled = X_fit + global_mean
            backend = "local"

        preds = X_filled[test_rows, test_cols].astype(np.float32)
        # Replace any residual NaN with global mean.
        fallback = int(np.isnan(preds).sum())
        preds = np.where(np.isnan(preds), global_mean, preds)
        preds = self._clip_scale(preds, rating_scale)
        return MethodResult(
            predictions=preds,
            fallback_count=fallback,
            extra={
                "lambda": self.lmbda,
                "max_rank": self.max_rank,
                "max_iters": self.max_iters,
                "backend": backend,
            },
        )


def build(
    lmbda: float = 2.0,
    max_rank: int = 30,
    max_iters: int = 100,
    tol: float = 1e-4,
    prefer_fancyimpute: bool = True,
    **_: object,
) -> SoftImputeWrapper:
    return SoftImputeWrapper(
        lmbda=lmbda,
        max_rank=max_rank,
        max_iters=max_iters,
        tol=tol,
        prefer_fancyimpute=prefer_fancyimpute,
    )
