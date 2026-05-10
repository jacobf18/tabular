from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.utils.extmath import randomized_svd

from .interface import ImputePFN


def standardize_observed(
    X: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize columns using observed entries only."""
    X = np.asarray(X, dtype=np.float64)
    if mask is None:
        mask = ~np.isnan(X)

    if X.shape != mask.shape:
        raise ValueError("X and mask must have the same shape")

    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)

    means = np.where(np.isnan(means), 0.0, means)
    stds = np.where(np.isnan(stds) | (stds < 1e-8), 1.0, stds)

    X_scaled = X.copy()
    X_scaled[mask] = (X[mask] - means[np.where(mask)[1]]) / stds[np.where(mask)[1]]
    return X_scaled, means, stds


def unstandardize(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Undo column-wise standardization."""
    return np.asarray(X, dtype=np.float64) * stds + means


def _run_imputer_on_standardized(
    X_scaled: np.ndarray,
    tabimpute_model: ImputePFN,
) -> np.ndarray:
    """Run ImputePFN without applying its built-in normalization again."""
    num_cols = X_scaled.shape[1]
    _, X_full = tabimpute_model.impute(
        X_scaled.copy(),
        return_full=True,
        means=np.zeros(num_cols, dtype=np.float64),
        stds=np.ones(num_cols, dtype=np.float64),
    )
    return np.asarray(X_full, dtype=np.float64)


def svd_deflation_impute(
    X_obs: np.ndarray,
    tabimpute_model: ImputePFN,
    k: int = 10,
    n_iter: int = 4,
    verbose: bool = False,
    tol: float = 1e-3,
) -> np.ndarray:
    """
    Iterative SVD deflation wrapper around TabImpute for high-rank completion.

    Parameters
    ----------
    X_obs:
        Partially observed matrix with ``np.nan`` for missing entries.
    tabimpute_model:
        An instantiated ``ImputePFN`` model from ``interface.py``.
    k:
        Number of singular components to peel off per iteration.
    n_iter:
        Maximum number of deflation iterations.
    verbose:
        If ``True``, print per-iteration residual norms.
    tol:
        Early stopping threshold on the observed-entry residual Frobenius norm.
    """
    X_obs = np.asarray(X_obs, dtype=np.float64)
    if X_obs.ndim != 2:
        raise ValueError("X_obs must be a 2-dimensional array")
    if k <= 0:
        raise ValueError("k must be positive")
    if n_iter < 0:
        raise ValueError("n_iter must be non-negative")
    if tol < 0:
        raise ValueError("tol must be non-negative")
    if not isinstance(tabimpute_model, ImputePFN):
        raise TypeError("tabimpute_model must be an instance of ImputePFN")

    mask = ~np.isnan(X_obs)
    if not np.any(mask):
        raise ValueError("X_obs must contain at least one observed entry")
    if np.all(mask):
        return X_obs.copy()

    n_components = min(k, X_obs.shape[0], X_obs.shape[1])
    X_accumulated = np.zeros_like(X_obs, dtype=np.float64)
    X_residual = X_obs.copy()

    for iteration_idx in range(n_iter):
        X_scaled, means, stds = standardize_observed(X_residual, mask=mask)
        Y_scaled = _run_imputer_on_standardized(X_scaled, tabimpute_model)
        Y_full = unstandardize(Y_scaled, means, stds)

        U, S, Vt = randomized_svd(Y_full, n_components=n_components, random_state=0)
        Y_k = (U * S) @ Vt
        X_accumulated += Y_k

        X_residual = X_obs.copy()
        X_residual[mask] = X_obs[mask] - X_accumulated[mask]

        residual_norm = float(np.sqrt(np.sum(np.square(X_residual[mask]))))
        if verbose:
            print(f"Iter {iteration_idx + 1}: residual norm = {residual_norm:.4f}")
        if residual_norm < tol:
            break

    X_scaled, means, stds = standardize_observed(X_residual, mask=mask)
    Y_final_scaled = _run_imputer_on_standardized(X_scaled, tabimpute_model)
    Y_final = unstandardize(Y_final_scaled, means, stds)

    X_completed = X_obs.copy()
    X_hat = X_accumulated + Y_final
    X_completed[~mask] = X_hat[~mask]
    return X_completed


@dataclass
class TabImputeSVDDeflation:
    """Convenience wrapper that owns an ``ImputePFN`` and applies SVD deflation."""

    tabimpute_model: ImputePFN
    k: int = 10
    n_iter: int = 4
    tol: float = 1e-3
    verbose: bool = False

    def impute(self, X_obs: np.ndarray) -> np.ndarray:
        return svd_deflation_impute(
            X_obs=X_obs,
            tabimpute_model=self.tabimpute_model,
            k=self.k,
            n_iter=self.n_iter,
            verbose=self.verbose,
            tol=self.tol,
        )

    @classmethod
    def from_imputepfn(
        cls,
        k: int = 10,
        n_iter: int = 4,
        tol: float = 1e-3,
        verbose: bool = False,
        **imputepfn_kwargs,
    ) -> "TabImputeSVDDeflation":
        """Instantiate the wrapper with a new ``ImputePFN`` directly."""
        return cls(
            tabimpute_model=ImputePFN(**imputepfn_kwargs),
            k=k,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
        )
