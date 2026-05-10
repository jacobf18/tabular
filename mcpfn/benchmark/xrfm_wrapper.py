"""
xRFM wrapper for tabimpute benchmark.

Column-wise regression imputer built on the upstream `xrfm` package
(https://github.com/dmbeaglehole/xrfm). For each column with missingness,
fits an xRFM on the observed rows using the other columns (mean-filled +
binary missingness indicators) as features, then predicts the missing
entries.

This is a standalone baseline: it does not load or depend on any MCPFN
checkpoint and does not interact with the existing benchmark runners.
Wire it in by importing `XRFMImputer` or calling `impute_xrfm`.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from typing import Optional

import numpy as np
import torch

from xrfm import xRFM


DEFAULT_RFM_PARAMS = {
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


def _build_features(X_filled: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Concatenate mean-filled values and a binary missingness mask along columns.

    Resulting feature dim is 2*d. Caller is responsible for dropping the
    target column's value and indicator before fitting/predicting that
    column (see `_drop_target_columns`).
    """
    return np.concatenate([X_filled, mask.astype(np.float32)], axis=1)


def _drop_target_columns(features: np.ndarray, target_col: int, d: int) -> np.ndarray:
    """Drop the target column's value and mask features to prevent leakage.

    Features are laid out as [values (d) | mask (d)]; remove indices
    `target_col` and `target_col + d`.
    """
    keep = np.ones(2 * d, dtype=bool)
    keep[target_col] = False
    keep[target_col + d] = False
    return features[:, keep]


def _mean_fill(X: np.ndarray) -> np.ndarray:
    """Mean-fill NaNs column-wise. All-NaN columns get 0."""
    out = X.copy().astype(np.float32)
    col_means = np.nanmean(out, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    inds = np.where(np.isnan(out))
    out[inds] = np.take(col_means, inds[1])
    return out


class XRFMImputer:
    """Column-wise xRFM imputer for continuous tabular data with NaNs.

    Parameters
    ----------
    device : str or None
        Torch device. Defaults to 'cuda' if available else 'cpu'.
    val_fraction : float
        Fraction of observed rows held out per column for xRFM's leaf
        validation / iteration selection. Must be in (0, 0.5].
    mice_iters : int
        Number of MICE-style outer passes. 0 means single pass.
    leaf_size : int
        xRFM `max_leaf_size` (paper's C). 60k matches the paper.
    refill_size : int
        xRFM `refill_size` (paper's N for split-direction subsample).
    random_state : int
        Seed for the held-out validation split.
    rfm_params : dict or None
        Overrides for the per-leaf RFM hyperparameters. If None, uses
        `DEFAULT_RFM_PARAMS` (Laplace-like kernel, diag AGOP, T=3, ridge=1e-3).
    xrfm_kwargs : dict or None
        Extra kwargs forwarded to the `xRFM` constructor.
    verbose : bool
        If False, suppresses xRFM's tqdm + stdout chatter.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        val_fraction: float = 0.2,
        mice_iters: int = 0,
        leaf_size: int = 60_000,
        refill_size: int = 1500,
        random_state: int = 0,
        rfm_params: Optional[dict] = None,
        xrfm_kwargs: Optional[dict] = None,
        verbose: bool = False,
    ):
        if not (0.0 < val_fraction <= 0.5):
            raise ValueError(f"val_fraction must be in (0, 0.5], got {val_fraction}")
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.val_fraction = float(val_fraction)
        self.mice_iters = int(mice_iters)
        self.leaf_size = int(leaf_size)
        self.refill_size = int(refill_size)
        self.random_state = int(random_state)
        self.rfm_params = rfm_params if rfm_params is not None else DEFAULT_RFM_PARAMS
        self.xrfm_kwargs = xrfm_kwargs or {}
        self.verbose = bool(verbose)

    def _fit_predict_column(
        self,
        features_obs: np.ndarray,
        y_obs: np.ndarray,
        features_query: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Fit xRFM on observed rows, predict on query rows."""
        n_obs = features_obs.shape[0]
        # Need at least a couple of training and a couple of validation points.
        if n_obs < 4:
            return np.full(features_query.shape[0], float(y_obs.mean()), dtype=np.float32)

        n_val = max(1, int(round(n_obs * self.val_fraction)))
        n_val = min(n_val, n_obs - 1)
        idx = rng.permutation(n_obs)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        X_tr = features_obs[tr_idx].astype(np.float32, copy=False)
        y_tr = y_obs[tr_idx].astype(np.float32, copy=False)
        X_va = features_obs[val_idx].astype(np.float32, copy=False)
        y_va = y_obs[val_idx].astype(np.float32, copy=False)

        if self.verbose:
            stdout_ctx = contextlib.nullcontext()
            stderr_ctx = contextlib.nullcontext()
        else:
            stdout_ctx = contextlib.redirect_stdout(io.StringIO())
            stderr_ctx = contextlib.redirect_stderr(io.StringIO())
        with stdout_ctx, stderr_ctx:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = xRFM(
                    rfm_params=self.rfm_params,
                    max_leaf_size=self.leaf_size,
                    refill_size=self.refill_size,
                    device=self.device,
                    tuning_metric="mse",
                    random_state=self.random_state,
                    **self.xrfm_kwargs,
                )
                model.fit(X_tr, y_tr, X_va, y_va)
                pred = model.predict(features_query.astype(np.float32, copy=False))

        pred = np.asarray(pred).reshape(-1)
        if pred.shape[0] != features_query.shape[0]:
            raise RuntimeError(
                f"xRFM returned {pred.shape[0]} predictions for "
                f"{features_query.shape[0]} query rows"
            )
        return pred.astype(np.float32, copy=False)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        mask = np.isnan(X).astype(np.float32)
        if not mask.any():
            return X.copy()

        rng = np.random.default_rng(self.random_state)

        X_imputed = _mean_fill(X)
        cols_with_nan = [c for c in range(d) if mask[:, c].any()]

        n_passes = max(1, self.mice_iters if self.mice_iters > 0 else 1)
        for pass_idx in range(n_passes):
            features_full = _build_features(X_imputed, mask)
            new_imputed = X_imputed.copy()
            for c in cols_with_nan:
                obs_rows = mask[:, c] == 0.0
                miss_rows = ~obs_rows
                if not miss_rows.any():
                    continue
                features = _drop_target_columns(features_full, target_col=c, d=d)
                features_obs = features[obs_rows]
                features_query = features[miss_rows]
                y_obs = X[obs_rows, c]

                pred = self._fit_predict_column(
                    features_obs, y_obs, features_query, rng
                )
                new_imputed[miss_rows, c] = pred

            X_imputed = new_imputed

        return X_imputed


def impute_xrfm(X_missing: np.ndarray, **kwargs) -> np.ndarray:
    """Convenience function. Fits and returns imputed array."""
    return XRFMImputer(**kwargs).fit_transform(X_missing)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 200, 6
    X_true = rng.standard_normal((n, d)).astype(np.float32)
    # Inject signal: col 0 = f(col 1, col 2)
    X_true[:, 0] = X_true[:, 1] ** 2 + 0.5 * X_true[:, 2] + 0.1 * rng.standard_normal(n)

    miss = rng.random((n, d)) < 0.2
    X_missing = X_true.copy()
    X_missing[miss] = np.nan

    print(f"Input shape={X_missing.shape}, "
          f"NaN frac={float(np.isnan(X_missing).mean()):.3f}")

    imputer = XRFMImputer(verbose=False, val_fraction=0.2)
    X_hat = imputer.fit_transform(X_missing)

    assert X_hat.shape == X_missing.shape, X_hat.shape
    assert not np.isnan(X_hat).any(), "imputed array still has NaNs"

    rmse_xrfm = float(np.sqrt(((X_hat[miss] - X_true[miss]) ** 2).mean()))
    mean_baseline = _mean_fill(X_missing)
    rmse_mean = float(np.sqrt(((mean_baseline[miss] - X_true[miss]) ** 2).mean()))
    print(f"RMSE on masked entries:  xRFM={rmse_xrfm:.4f}   mean-fill={rmse_mean:.4f}")
    print("Smoke test OK")
