"""Reproducible train/test splits over the observed entries of a sparse matrix."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.sparse as sp


@dataclass
class ObservedSplit:
    """Result of withholding a fraction of observed entries.

    ``train_matrix`` retains the un-held-out observed entries (all others zero).
    ``test_rows``/``test_cols``/``test_values`` describe the held-out entries.
    """

    train_matrix: sp.csr_matrix
    train_rows: np.ndarray
    train_cols: np.ndarray
    train_values: np.ndarray
    test_rows: np.ndarray
    test_cols: np.ndarray
    test_values: np.ndarray
    seed: int
    fraction: float


def withhold_observed(
    matrix: sp.csr_matrix,
    fraction: float,
    seed: int,
) -> ObservedSplit:
    """Uniformly withhold ``fraction`` of non-zero entries of ``matrix``."""
    coo = matrix.tocoo()
    rows = coo.row
    cols = coo.col
    vals = coo.data.astype(np.float32)
    n_obs = rows.size
    n_test = int(round(fraction * n_obs))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_obs)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    train_matrix = sp.coo_matrix(
        (vals[train_idx], (rows[train_idx], cols[train_idx])), shape=matrix.shape, dtype=np.float32
    ).tocsr()

    return ObservedSplit(
        train_matrix=train_matrix,
        train_rows=rows[train_idx],
        train_cols=cols[train_idx],
        train_values=vals[train_idx],
        test_rows=rows[test_idx],
        test_cols=cols[test_idx],
        test_values=vals[test_idx],
        seed=seed,
        fraction=fraction,
    )


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    if predictions.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((predictions.astype(np.float64) - targets.astype(np.float64)) ** 2)))


def dense_from_train(split: ObservedSplit) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dense_train, observed_mask) suitable for dense-matrix methods."""
    m = split.train_matrix
    dense = np.asarray(m.toarray(), dtype=np.float32)
    mask = np.asarray(dense != 0.0, dtype=bool)
    return dense, mask
