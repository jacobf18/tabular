"""
Boosted Matrix Completion
=========================
L2Boost in the space of low-rank matrices.

At each round t:
    1. Fit a base completer to the residual matrix R^(t) (NaN on unobserved entries)
    2. Accumulate:  M_hat += eta * completer.fit_transform(R^(t))
    3. Update residual on observed entries: R^(t+1)[Omega] = R^(t)[Omega] - eta * L^(t)[Omega]

The base completer can be anything with the interface:
    fit_transform(X: np.ndarray) -> np.ndarray
where X has np.nan on unobserved entries and returns a fully dense matrix.

Included base completers
------------------------
    SoftImputeCompleter   — classic nuclear-norm / SVT solver
    TabImputeCompleter    — adapter wrapping ImputePFN (default) or any sklearn-style imputer
    NNMFCompleter         — rank-k NMF on observed entries (non-negative only)
"""

from __future__ import annotations

import warnings
from typing import Optional, Protocol

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Protocol / interface
# ---------------------------------------------------------------------------

class MatrixCompleter(Protocol):
    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Parameters
        ----------
        X : (m, n) array with np.nan on unobserved entries.

        Returns
        -------
        X_filled : (m, n) dense array — every entry populated.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete base completers
# ---------------------------------------------------------------------------

class SoftImputeCompleter:
    """
    Iterative SVD / SoftImpute.

    Iterates:
        1. Fill NaNs with current estimate.
        2. SVD + soft-threshold singular values by `lambda_`.
        3. Repeat until convergence.

    Reference: Mazumder, Hastie & Tibshirani (2010).
    """

    def __init__(
        self,
        rank: int = 5,
        lambda_: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.rank = rank
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        X = np.array(X, dtype=float)
        omega = ~np.isnan(X)

        # Initialise: column means on observed entries
        col_means = np.where(omega.any(axis=0), np.nanmean(X, axis=0), 0.0)
        Z = np.where(omega, X, col_means[np.newaxis, :])

        for _ in range(self.max_iter):
            Z_old = Z.copy()

            # Truncated SVD
            k = min(self.rank, min(X.shape) - 1)
            try:
                U, s, Vt = np.linalg.svd(Z, full_matrices=False)
            except np.linalg.LinAlgError:
                break
            U, s, Vt = U[:, :k], s[:k], Vt[:k, :]

            # Soft-threshold singular values
            s_thresh = np.maximum(s - self.lambda_, 0.0)
            Z_new = (U * s_thresh) @ Vt

            # Keep observed entries fixed
            Z = np.where(omega, X, Z_new)

            # Convergence check on unobserved entries
            delta = np.linalg.norm(Z - Z_old) / (np.linalg.norm(Z_old) + 1e-10)
            if delta < self.tol:
                break

        return Z


class NNMFCompleter:
    """
    Non-negative MF on observed entries via multiplicative updates.
    Only appropriate for non-negative matrices.

    Parameters
    ----------
    rank : int
        Number of latent factors.
    max_iter : int
        Multiplicative update iterations.
    """

    def __init__(self, rank: int = 5, max_iter: int = 200, random_state: Optional[int] = None):
        self.rank = rank
        self.max_iter = max_iter
        self.rng = np.random.default_rng(random_state)

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        X = np.array(X, dtype=float)
        omega = ~np.isnan(X)

        if np.any(X[omega] < 0):
            warnings.warn("NNMFCompleter received negative values; clamping to 0.")
            X = np.where(omega, np.maximum(X, 0), np.nan)

        m, n = X.shape
        k = min(self.rank, min(m, n))

        W = self.rng.uniform(0.01, 1.0, (m, k))
        H = self.rng.uniform(0.01, 1.0, (k, n))

        # Masked reconstruction: only update on Omega
        V = np.where(omega, X, W @ H)  # fill NaNs with current estimate

        eps = 1e-10
        for _ in range(self.max_iter):
            WH = W @ H + eps
            # Multiplicative updates on observed entries only
            V_obs = np.where(omega, X, WH)  # use current estimate for missing
            H = H * (W.T @ V_obs) / (W.T @ WH + eps)
            WH = W @ H + eps
            W = W * (V_obs @ H.T) / (WH @ H.T + eps)

        return W @ H


# ---------------------------------------------------------------------------
# Boosted Matrix Completion
# ---------------------------------------------------------------------------


class BoostedMatrixCompletion:
    """
    L2Boost for matrix completion.

    Before the first boosting round, a single SoftImputeCompleter pass is run
    on the masked matrix (with the same validation masking as later rounds).
    Training and reported initial MSE use ``learning_rate`` times that
    completion as the starting ``M_hat``; subsequent rounds fit residuals.

    Parameters
    ----------
    base_completer : MatrixCompleter
        Any object with a fit_transform(X) method.
    n_estimators : int
        Number of boosting rounds T.
    learning_rate : float
        Shrinkage parameter eta in (0, 1].
    early_stopping_rounds : int or None
        Stop if validation loss does not improve for this many consecutive
        rounds. Requires val_mask to be passed to fit().
    verbose : bool
        Print round-level loss.

    Attributes
    ----------
    estimators_ : list of (m, n) arrays
        Dense outputs L^(t) from each boosting round (before shrinkage).
        Does not include the one-shot SoftImpute initialization.
    initial_completion_ : (m, n) array
        Dense matrix from a single SoftImputeCompleter pass before boosting.
    train_losses_ : list of float
        MSE on observed train entries per round.
    val_losses_ : list of float
        MSE on validation mask entries per round (if provided).
    """

    def __init__(
        self,
        base_completer: MatrixCompleter,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        early_stopping_rounds: Optional[int] = None,
        verbose: bool = False,
    ):
        self.base_completer = base_completer
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: NDArray[np.float64],
        val_mask: Optional[NDArray[np.bool_]] = None,
    ) -> "BoostedMatrixCompletion":
        """
        Fit the boosted model.

        Parameters
        ----------
        X : (m, n) array
            Matrix with np.nan on unobserved entries. NaN positions define
            the observed mask Omega (entries we supervise the residual on).
        val_mask : (m, n) bool array or None
            True where entries should be held out for early stopping.
            These entries are masked to NaN before each round's completer call
            (treated as unobserved) so there is no leakage.
        """
        X = np.array(X, dtype=float)
        m, n = X.shape

        train_mask = ~np.isnan(X)                     # Omega
        if val_mask is not None:
            val_mask = np.asarray(val_mask, dtype=bool)
            # Remove val entries from training supervision
            train_mask = train_mask & ~val_mask

        self.estimators_: list[NDArray] = []
        self.train_losses_: list[float] = []
        self.val_losses_: list[float] = []

        # One-shot SoftImpute warm start (same masking as boosting rounds)
        R = X.copy()
        R_input = R.copy()
        if val_mask is not None:
            R_input[val_mask] = np.nan
        init_completer = (
            self.base_completer
            if isinstance(self.base_completer, SoftImputeCompleter)
            else SoftImputeCompleter()
        )
        self.initial_completion_ = init_completer.fit_transform(R_input)

        # Running prediction (dense); same shrinkage as subsequent weak learners
        # M_hat = self.learning_rate * self.initial_completion_.copy()
        M_hat = np.zeros((m, n)) 
        R[train_mask] = X[train_mask] - M_hat[train_mask]

        best_val_loss = np.inf
        no_improve = 0

        initial_train_mse = float(
            np.mean((X[train_mask] - M_hat[train_mask]) ** 2)
        )
        if self.verbose:
            val_str = ""
            if val_mask is not None:
                initial_val_mse = float(
                    np.mean((X[val_mask] - M_hat[val_mask]) ** 2)
                )
                val_str = f"  val_mse={initial_val_mse:.6f}"
            print(f"[Initial] train_mse={initial_train_mse:.6f}{val_str}")

        for t in range(self.n_estimators):
            # Mask val entries from the completer (no leakage)
            R_input = R.copy()
            if val_mask is not None:
                R_input[val_mask] = np.nan

            # Fit weak learner to current residual
            L = self.base_completer.fit_transform(R_input)  # (m, n) dense

            # Accumulate
            M_hat = M_hat + self.learning_rate * L
            self.estimators_.append(L)

            # Update residual on TRAIN observed entries only
            R[train_mask] = X[train_mask] - M_hat[train_mask]

            # Losses
            train_loss = float(np.mean((X[train_mask] - M_hat[train_mask]) ** 2))
            self.train_losses_.append(train_loss)

            if val_mask is not None:
                val_loss = float(np.mean((X[val_mask] - M_hat[val_mask]) ** 2))
                self.val_losses_.append(val_loss)
            else:
                val_loss = None

            if self.verbose:
                val_str = f"  val_mse={val_loss:.6f}" if val_loss is not None else ""
                print(f"[Round {t+1:>3d}]  train_mse={train_loss:.6f}{val_str}")

            # Early stopping
            if val_loss is not None and self.early_stopping_rounds is not None:
                if val_loss < best_val_loss - 1e-8:
                    best_val_loss = val_loss
                    no_improve = 0
                    self._best_n_estimators = t + 1
                else:
                    no_improve += 1
                if no_improve >= self.early_stopping_rounds:
                    if self.verbose:
                        print(f"Early stopping at round {t+1}.")
                    # Trim to best checkpoint
                    self.estimators_ = self.estimators_[:self._best_n_estimators]
                    break

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, shape: Optional[tuple[int, int]] = None) -> NDArray[np.float64]:
        """
        Return the accumulated low-rank estimate.

        Parameters
        ----------
        shape : ignored — present for API symmetry.

        Returns
        -------
        M_hat : (m, n) dense array.
        """
        if not self.estimators_:
            return self.learning_rate * self.initial_completion_.copy()
        m, n = self.initial_completion_.shape
        M_hat = np.zeros((m, n))
        for L in self.estimators_:
            M_hat += self.learning_rate * L
        return M_hat

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def effective_rank(self, threshold: float = 0.99) -> int:
        """
        Effective rank of M_hat: smallest k such that top-k singular values
        capture `threshold` fraction of total spectral energy.
        """
        M_hat = self.predict()
        s = np.linalg.svd(M_hat, compute_uv=False)
        cumvar = np.cumsum(s**2) / (np.sum(s**2) + 1e-12)
        return int(np.searchsorted(cumvar, threshold) + 1)


# ---------------------------------------------------------------------------
# TabImpute PFN adapter
# ---------------------------------------------------------------------------


class TabImputeCompleter:
    """
    Adapter for ImputePFN (default) or any imputer with ``fit_transform`` or
    ``impute`` and NaN support.

    Parameters
    ----------
    imputer
        If None, a default :class:`tabimpute.interface.ImputePFN` is created.
        Otherwise pass a fitted-style imputer: ``fit_transform(X)`` (sklearn)
        or ``impute(X)`` (``ImputePFN``).
    """

    def __init__(self, imputer=None):
        from tabimpute.interface import ImputePFN

        self.imputer = ImputePFN(device='cuda', entry_wise_features=True)

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        imputer = self.imputer
        if hasattr(imputer, "fit_transform"):
            return np.asarray(imputer.fit_transform(X), dtype=np.float64)
        return np.asarray(imputer.impute(X), dtype=np.float64)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Ground truth: rank-3 matrix
    m, n, r = 80, 60, 15
    U = rng.standard_normal((m, r))
    V = rng.standard_normal((n, r))
    M_true = U @ V.T

    # Observe 40% of entries, hold out 10% for validation
    obs_frac, val_frac = 0.40, 0.10
    all_idx = np.array([(i, j) for i in range(m) for j in range(n)])
    perm = rng.permutation(len(all_idx))

    n_obs = int(obs_frac * len(all_idx))
    n_val = int(val_frac * len(all_idx))

    obs_idx = all_idx[perm[:n_obs]]
    val_idx = all_idx[perm[n_obs: n_obs + n_val]]

    X = np.full((m, n), np.nan)
    for i, j in obs_idx:
        X[i, j] = M_true[i, j] + rng.normal(0, 0.1)

    val_mask = np.zeros((m, n), dtype=bool)
    for i, j in val_idx:
        val_mask[i, j] = True
        X[i, j] = M_true[i, j]   # revealed but held out of training loss

    # Run boosted TabImpute (ImputePFN via TabImputeCompleter)
    model = BoostedMatrixCompletion(
        base_completer=TabImputeCompleter(),
        n_estimators=60,
        learning_rate=0.1,
        early_stopping_rounds=8,
        verbose=True,
    )
    # M_initial = SoftImputeCompleter(rank=3, lambda_=0.5).fit_transform(X.copy())
    M_initial = TabImputeCompleter().fit_transform(X.copy())
    model.fit(X, val_mask=val_mask)

    M_hat = model.predict()
    
    baseline_recovery_mse = float(np.mean((M_true[~val_mask] - M_initial[~val_mask]) ** 2))
    test_mse = float(np.mean((M_true[~val_mask] - M_hat[~val_mask]) ** 2))
    print(f"\nBaseline recovery MSE: {baseline_recovery_mse:.5f}")
    print(f"Recovery MSE: {test_mse:.5f}")
    print(f"Effective rank of M_hat:    {model.effective_rank()}")