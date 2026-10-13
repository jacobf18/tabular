"""Imputation error metrics. All are computed on the masked (imputed) entries only."""
from __future__ import annotations

import numpy as np
from scipy import stats


def rmse(X_true: np.ndarray, X_imputed: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((X_true[mask] - X_imputed[mask]) ** 2)))


def mae(X_true: np.ndarray, X_imputed: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(np.abs(X_true[mask] - X_imputed[mask])))


def normalized_rmse_columnwise(X_true: np.ndarray, X_imputed: np.ndarray, mask: np.ndarray) -> float:
    """
    Average column-wise normalized RMSE (the RMSE reported in the paper).

    For each column j with missing entries: NRMSE_j = RMSE_j / sigma_j, where both
    are computed over the missing entries of column j and sigma_j is the standard
    deviation of the true values. Columns whose true values are constant are scored
    0 if they were imputed exactly and skipped otherwise.
    """
    scores = []
    for j in range(X_true.shape[1]):
        col_mask = mask[:, j]
        if not col_mask.any():
            continue
        true_vals = X_true[col_mask, j]
        col_rmse = np.sqrt(np.mean((true_vals - X_imputed[col_mask, j]) ** 2))
        sigma = np.std(true_vals)
        if sigma < 1e-3:
            if col_rmse < 1e-3:
                scores.append(0.0)
            continue
        scores.append(col_rmse / sigma)
    return float(np.mean(scores)) if scores else 0.0


def _masked_columns(X_true, X_imputed, mask):
    for j in range(X_true.shape[1]):
        col_mask = mask[:, j]
        if col_mask.sum() < 2:
            continue
        true_vals, imp_vals = X_true[col_mask, j], X_imputed[col_mask, j]
        if np.isnan(true_vals).any() or np.isnan(imp_vals).any():
            continue
        yield true_vals, imp_vals


def r_squared_columnwise(X_true: np.ndarray, X_imputed: np.ndarray, mask: np.ndarray) -> float:
    """Mean over columns of the squared correlation between true and imputed values."""
    scores = []
    for true_vals, imp_vals in _masked_columns(X_true, X_imputed, mask):
        if np.isclose(np.std(imp_vals), 0, atol=1e-3) or np.isclose(np.std(true_vals), 0, atol=1e-3):
            scores.append(0.0)
            continue
        r2 = np.corrcoef(imp_vals, true_vals)[0, 1] ** 2
        scores.append(0.0 if np.isnan(r2) else r2)
    return float(np.mean(scores)) if scores else 0.0


def wasserstein_columnwise(X_true: np.ndarray, X_imputed: np.ndarray, mask: np.ndarray) -> float:
    """Mean over columns of the 1-Wasserstein distance between true and imputed values."""
    scores = []
    for true_vals, imp_vals in _masked_columns(X_true, X_imputed, mask):
        w1 = stats.wasserstein_distance(true_vals, imp_vals)
        scores.append(0.0 if np.isnan(w1) else w1)
    return float(np.mean(scores)) if scores else 0.0


METRICS = {
    "nrmse": normalized_rmse_columnwise,
    "r2": r_squared_columnwise,
    "wasserstein": wasserstein_columnwise,
    "rmse": rmse,
    "mae": mae,
}
