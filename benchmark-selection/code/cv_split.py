"""Balanced k-fold cross-validation with training subsampling.

Usage:
    from cv_split import cv_folds, cv_train_val

    folds = cv_folds(M, n_folds=10, seed=42)
    for fold_idx in range(n_folds):
        for p in holdout_fracs:
            train_idx, val_idx = cv_train_val(folds, fold_idx, p, M, seed=42)
"""

import numpy as np


def cv_folds(M: int, n_folds: int = 10, seed: int = 42) -> list[np.ndarray]:
    """Partition M items into n_folds balanced groups via random permutation.

    Returns a list of n_folds arrays of indices.  Sizes differ by at most 1.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    return list(np.array_split(perm, n_folds))


def cv_train_val(
    folds: list[np.ndarray],
    fold_idx: int,
    holdout_frac: float,
    M: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) for one fold and holdout fraction.

    Parameters
    ----------
    folds : list of arrays from cv_folds().
    fold_idx : which fold to hold out as the validation set.
    holdout_frac : fraction of *total* data that is NOT used for training.
        The validation set is always one fold (~1/n_folds of M).
        The training set is subsampled from the remaining folds so that
        |train| = round((1 - holdout_frac) * M), capped at the pool size.
        For holdout_frac = 1/n_folds this uses the entire pool.
    M : total number of items (for computing target train size).
    seed : base seed; combined with fold_idx and holdout_frac for
        reproducible subsampling independent of iteration order.
    """
    n_folds = len(folds)
    val_idx = folds[fold_idx]
    pool_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

    n_train = min(round((1 - holdout_frac) * M), len(pool_idx))

    if n_train >= len(pool_idx):
        train_idx = pool_idx
    else:
        # Deterministic sub-seed so result is independent of iteration order
        sub_seed = (seed, fold_idx, int(holdout_frac * 10000))
        rng = np.random.default_rng(sub_seed)
        train_idx = rng.permutation(pool_idx)[:n_train]

    return train_idx, val_idx
