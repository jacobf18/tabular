"""Context-only baseline: xRFM trained on (X_row[r] ⊕ X_col[c]) → Y[r, c].

No transformer backbone. No residual correction. Just: take the per-row and
per-column context vectors, concatenate them to form an 8-dim input per cell,
fit xRFM on observed cells, predict missing cells.

The point of this baseline (per the discussion in xrfm/context.txt): does
context-only prediction beat MatFormer on regimes where context is supposed
to carry signal? If yes, the existing boost pipeline is failing to extract
something usable. If no, context is either redundant (visible regime) or
unrecoverable from observed cells (hidden regime).
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np


_RFM_PARAMS = {
    "model": {
        "kernel": "l2_high_dim",
        "exponent": 1.0,
        "bandwidth": 10.0,
        "diag": True,
        "bandwidth_mode": "constant",
    },
    "fit": {
        "reg": 1e-3,
        "iters": 3,
        "early_stop_rfm": True,
        "verbose": False,
    },
}


def predict_context_only(
    Y: np.ndarray,
    mask_missing: np.ndarray,
    X_row: np.ndarray,
    X_col: np.ndarray,
    *,
    max_train_cells: int = 5000,
    val_fraction: float = 0.2,
    device: str = "cuda",
    seed: int = 0,
) -> np.ndarray:
    """Fit xRFM on observed (r, c) cells with features [X_row[r], X_col[c]],
    predict missing cells. Returns a full (N, T) matrix where observed cells
    are passed through from Y and missing cells are model predictions.

    Args:
      Y:            (N, T) noisy data matrix (Y = M + noise from generate.py).
      mask_missing: (N, T) bool, True = missing cell.
      X_row:        (N, d_r) row context.
      X_col:        (T, d_c) col context.
      max_train_cells: cap on number of observed cells used for training
                       (prevents O(n^3) solve from blowing up at 200x200).
      val_fraction: held-out fraction of training cells for xRFM early-stop.
      device:       'cuda' or 'cpu'.
      seed:         RNG seed for subsample + train/val split.
    """
    from xrfm import xRFM

    N, T = Y.shape
    assert mask_missing.shape == (N, T)
    assert X_row.shape[0] == N
    assert X_col.shape[0] == T

    # All observed (r, c) coordinates.
    obs_rows, obs_cols = np.where(~mask_missing)
    n_obs = obs_rows.size
    if n_obs < 8:
        # Degenerate: fall back to mean.
        out = Y.astype(np.float32).copy()
        out[mask_missing] = float(np.nanmean(Y[~mask_missing]))
        return out

    rng = np.random.default_rng(seed)

    # Subsample observed cells if there are too many.
    if n_obs > max_train_cells:
        sub = rng.choice(n_obs, size=max_train_cells, replace=False)
        obs_rows_train = obs_rows[sub]
        obs_cols_train = obs_cols[sub]
    else:
        obs_rows_train = obs_rows
        obs_cols_train = obs_cols
    n_train_pool = obs_rows_train.size

    # Build per-cell features: concat row context and col context.
    feat_train = np.concatenate(
        [X_row[obs_rows_train], X_col[obs_cols_train]], axis=1
    ).astype(np.float32)
    y_train = Y[obs_rows_train, obs_cols_train].astype(np.float32)

    # Train/val split for xRFM's internal early stop.
    n_val = max(1, int(round(n_train_pool * val_fraction)))
    n_val = min(n_val, n_train_pool - 4)
    perm = rng.permutation(n_train_pool)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    X_tr = feat_train[tr_idx]
    y_tr = y_train[tr_idx]
    X_va = feat_train[val_idx]
    y_va = y_train[val_idx]

    # Missing cells -> features.
    miss_rows, miss_cols = np.where(mask_missing)
    if miss_rows.size == 0:
        return Y.astype(np.float32).copy()
    feat_miss = np.concatenate(
        [X_row[miss_rows], X_col[miss_cols]], axis=1
    ).astype(np.float32)

    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        model = xRFM(
            rfm_params=_RFM_PARAMS,
            max_leaf_size=60_000,
            refill_size=1500,
            device=device,
            tuning_metric="mse",
            random_state=seed,
        )
        model.fit(X_tr, y_tr, X_va, y_va)
        pred = model.predict(feat_miss)

    pred = np.asarray(pred, dtype=np.float32).reshape(-1)

    out = Y.astype(np.float32).copy()
    out[miss_rows, miss_cols] = pred
    return out
