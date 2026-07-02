"""Synthetic contextual matrix-completion generator.

Mimics Section 5.1 of "Robust Matrix Estimation with Side Information".

Each draw produces:
  - X_row: (N, 4) row context  ~ Unif & Gaussian (per paper)
  - X_col: (T, 4) column context  ~ Unif & Gaussian (per paper)
  - M    : (N, T) clean matrix = α1 M1 + α2 M2 + α3 M3 + α4 M4 (Frobenius-normalized)
  - Y    : M + N(0, σ²) noise
  - mask : (N, T) bool, True = MISSING (treated × post quadrant)

Components:
  M1 — both contexts drive (rank K1=17, polynomial factors of x and z)
  M2 — row context only   (rank K2=3, polynomial factors of x, random col loadings)
  M3 — column context only (rank K3=3, random row loadings, polynomial factors of z)
  M4 — neither             (rank K4=3, random factors both sides — pure low-rank)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _poly4_features(X: np.ndarray) -> np.ndarray:
    """X: (N, 4) -> Phi: (N, 17) with [1, x_d, x_d^2, x_d^3, x_d^4 for d=0..3]."""
    N, D = X.shape
    assert D == 4, "expected 4 row/col features per paper"
    cols = [np.ones((N, 1), dtype=np.float64)]
    for d in range(D):
        xd = X[:, d:d + 1]
        for p in range(1, 5):
            cols.append(xd ** p)
    return np.concatenate(cols, axis=1)  # (N, 17)


def _poly_factor(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """Random polynomial factors of X: returns (N, K) where each column is
    a degree-4 polynomial in X (4-dim) with N(0,1) coefficients."""
    Phi = _poly4_features(X)          # (N, 17)
    B = rng.standard_normal((Phi.shape[1], K))  # (17, K)
    return Phi @ B                    # (N, K)


def _random_factor(N: int, K: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((N, K))


def _frob_normalize(M: np.ndarray, target_frob: float) -> np.ndarray:
    nrm = float(np.linalg.norm(M, "fro"))
    if nrm < 1e-12:
        return M
    return M * (target_frob / nrm)


def _mar_mask(Y: np.ndarray, p_missing: float, rng: np.random.Generator) -> np.ndarray:
    """UCI-style MAR mask. Pick a random subset of "predictor" columns; the
    remaining "target" columns get per-row missingness probability = sigmoid
    of a random linear combo of predictor values (calibrated so the marginal
    missingness rate over target columns is ≈ p_missing).

    Predictor columns are never missing under this pattern. Mirrors
    src/tabimpute/prior/training_set_generation.py:MARPattern.
    """
    N, T = Y.shape
    num_predictors = int(rng.integers(1, max(2, T // 2)))
    predictor_idx = rng.choice(T, num_predictors, replace=False)
    target_idx = np.setdiff1d(np.arange(T), predictor_idx)
    predictors = np.where(np.isnan(Y[:, predictor_idx]), 0.0,
                          Y[:, predictor_idx])
    log_odds = predictors @ rng.standard_normal(num_predictors)
    # Bias so the marginal P(miss) on target cells is ~ p_missing.
    threshold = np.quantile(log_odds, 1.0 - p_missing)
    probs = 1.0 / (1.0 + np.exp(-(log_odds - threshold)))   # shape (N,)
    mask = np.zeros((N, T), dtype=bool)
    for c in target_idx:
        mask[:, c] = rng.random(N) < probs
    return mask


def _mnar_mask(Y: np.ndarray, p_missing: float, rng: np.random.Generator) -> np.ndarray:
    """UCI-style MNAR mask. Pick a random subset of target columns; in each,
    missingness probability is sigmoid of the cell's own value × a random
    {-2,-1,1,2} coefficient, calibrated to marginal rate p_missing.

    Mirrors src/tabimpute/prior/training_set_generation.py:MNARPattern.
    """
    N, T = Y.shape
    num_targets = int(rng.integers(1, T + 1))
    target_idx = rng.choice(T, num_targets, replace=False)
    mask = np.zeros((N, T), dtype=bool)
    coef_choices = np.array([-2, -1, 1, 2])
    for c in target_idx:
        col = Y[:, c]
        coef = float(rng.choice(coef_choices))
        log_odds = col * coef
        finite = np.isfinite(log_odds)
        if not finite.any():
            continue
        threshold = np.quantile(log_odds[finite], 1.0 - p_missing)
        probs = 1.0 / (1.0 + np.exp(-(log_odds - threshold)))
        probs = np.where(finite, probs, 0.0)
        mask[:, c] = rng.random(N) < probs
    return mask


@dataclass
class Draw:
    X_row: np.ndarray   # (N, 4)
    X_col: np.ndarray   # (T, 4)
    M: np.ndarray       # (N, T) clean
    Y: np.ndarray       # (N, T) noisy
    mask: np.ndarray    # (N, T) bool, True = missing
    alphas: tuple
    N0: int
    T0: int
    noise_std: float
    seed: int


def make_draw(
    N: int = 200,
    T: int = 200,
    alphas: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    K: Sequence[int] = (17, 3, 3, 3),
    N0: int | None = None,
    T0: int | None = None,
    noise_std: float = 0.5,
    seed: int = 0,
    missingness: str = "block",
    missing_prob: float = 0.4,
) -> Draw:
    rng = np.random.default_rng(seed)
    K1, K2, K3, K4 = K

    # Row & column contexts (per paper distributions).
    X = np.stack([
        rng.uniform(-1.0, 1.0, N),
        rng.uniform(-0.5, 0.5, N),
        rng.normal(0.0, 0.2, N),
        rng.normal(0.0, 0.3, N),
    ], axis=1)
    Z = np.stack([
        rng.uniform(-1.0, 1.0, T),
        rng.uniform(-0.5, 0.5, T),
        rng.normal(0.0, 0.2, T),
        rng.normal(0.0, 0.3, T),
    ], axis=1)

    # Components.
    G1 = _poly_factor(X, K1, rng)        # (N, K1)
    H1 = _poly_factor(Z, K1, rng)        # (T, K1)
    M1 = G1 @ H1.T

    G2 = _poly_factor(X, K2, rng)
    H2 = _random_factor(T, K2, rng)
    M2 = G2 @ H2.T

    G3 = _random_factor(N, K3, rng)
    H3 = _poly_factor(Z, K3, rng)
    M3 = G3 @ H3.T

    G4 = _random_factor(N, K4, rng)
    H4 = _random_factor(T, K4, rng)
    M4 = G4 @ H4.T

    target = 2.0 * np.sqrt(N * T)
    M1 = _frob_normalize(M1, target)
    M2 = _frob_normalize(M2, target)
    M3 = _frob_normalize(M3, target)
    M4 = _frob_normalize(M4, target)

    a1, a2, a3, a4 = alphas
    M = a1 * M1 + a2 * M2 + a3 * M3 + a4 * M4
    Y = M + rng.normal(0.0, noise_std, (N, T))

    # Missingness mask. Multiple modes:
    #   "block": MNAR bottom-right rectangle (mimics treated x post quadrant)
    #   "mcar" : each cell independently missing with prob missing_prob
    #   "mar"  : target cols missing as fn of predictor-col values (UCI MARPattern)
    #   "mnar" : each target col missing as fn of its own value (UCI MNARPattern)
    if N0 is None:
        N0 = N // 2
    if T0 is None:
        T0 = T // 2
    if missingness == "block":
        mask = np.zeros((N, T), dtype=bool)
        mask[N0:, T0:] = True
    elif missingness == "mcar":
        mask = rng.random((N, T)) < float(missing_prob)
    elif missingness == "mar":
        mask = _mar_mask(Y, float(missing_prob), rng)
    elif missingness == "mnar":
        mask = _mnar_mask(Y, float(missing_prob), rng)
    else:
        raise ValueError(f"Unknown missingness={missingness!r}; "
                         "expected one of 'block', 'mcar', 'mar', 'mnar'.")

    return Draw(
        X_row=X.astype(np.float32),
        X_col=Z.astype(np.float32),
        M=M.astype(np.float32),
        Y=Y.astype(np.float32),
        mask=mask,
        alphas=tuple(alphas),
        N0=int(N0),
        T0=int(T0),
        noise_std=float(noise_std),
        seed=int(seed),
    )


def observed_matrix(draw: Draw) -> np.ndarray:
    """Return Y with NaN at the missing block (the observed/input matrix)."""
    Yobs = draw.Y.astype(np.float32).copy()
    Yobs[draw.mask] = np.nan
    return Yobs


def make_hidden_context_draw(
    N: int = 200,
    T: int = 200,
    treatment_strength: float = 1.0,
    K_base: int = 5,
    K_treat: int = 5,
    N0: int | None = None,
    T0: int | None = None,
    noise_std: float = 0.5,
    seed: int = 0,
) -> Draw:
    """Variant where context-driven signal appears ONLY in the missing block.

    Setup:
      - M_base : rank-K_base random low-rank matrix, supported everywhere.
                 NO context dependence. Defines the predictable baseline.
      - M_treat: polynomial-in-(x_r, z_c) signal, MASKED to the bottom-right
                 (treated × post) block only. Supported only where mask=True.

    Observed cells: M_base + noise (no x_r/z_c information).
    Missing cells : M_base + treatment_strength * M_treat + noise.

    Implication for residual-correction:
      - At observed cells: residual = noise. xRFM trains on noise → predicts 0.
      - At missing cells : true signal = M_treat[r, c], which depends on x_r/z_c
        but is NOT present anywhere in observed rows. The ONLY way for any
        residual learner to recover this signal is via explicit context features.
    """
    rng = np.random.default_rng(seed)
    if N0 is None:
        N0 = N // 2
    if T0 is None:
        T0 = T // 2

    X = np.stack([
        rng.uniform(-1.0, 1.0, N),
        rng.uniform(-0.5, 0.5, N),
        rng.normal(0.0, 0.2, N),
        rng.normal(0.0, 0.3, N),
    ], axis=1)
    Z = np.stack([
        rng.uniform(-1.0, 1.0, T),
        rng.uniform(-0.5, 0.5, T),
        rng.normal(0.0, 0.2, T),
        rng.normal(0.0, 0.3, T),
    ], axis=1)

    G_base = _random_factor(N, K_base, rng)
    H_base = _random_factor(T, K_base, rng)
    M_base = G_base @ H_base.T
    M_base = _frob_normalize(M_base, 2.0 * np.sqrt(N * T))

    G_treat = _poly_factor(X, K_treat, rng)
    H_treat = _poly_factor(Z, K_treat, rng)
    M_treat_full = G_treat @ H_treat.T
    M_treat = np.zeros_like(M_treat_full)
    M_treat[N0:, T0:] = M_treat_full[N0:, T0:]
    # Normalize on the active block so treatment_strength is interpretable
    # as a multiplier on signal in the missing block.
    n_active = max(1, (N - N0) * (T - T0))
    M_treat = _frob_normalize(M_treat, 2.0 * np.sqrt(n_active))

    M = M_base + treatment_strength * M_treat
    Y = M + rng.normal(0.0, noise_std, (N, T))

    mask = np.zeros((N, T), dtype=bool)
    mask[N0:, T0:] = True

    return Draw(
        X_row=X.astype(np.float32),
        X_col=Z.astype(np.float32),
        M=M.astype(np.float32),
        Y=Y.astype(np.float32),
        mask=mask,
        alphas=(float(treatment_strength),),
        N0=int(N0),
        T0=int(T0),
        noise_std=float(noise_std),
        seed=int(seed),
    )


TREATMENT_SETTINGS = {
    "zero":   0.0,
    "small":  0.5,
    "medium": 1.0,
    "large":  2.0,
}


# Six illustrative alpha settings.
ALPHA_SETTINGS = {
    "pure_interaction":  (1.00, 0.00, 0.00, 0.00),
    "row_only":          (0.00, 1.00, 0.00, 0.00),
    "col_only":          (0.00, 0.00, 1.00, 0.00),
    "low_rank_only":     (0.00, 0.00, 0.00, 1.00),
    "balanced":          (0.25, 0.25, 0.25, 0.25),
    "context_heavy":     (0.60, 0.20, 0.10, 0.10),
}


if __name__ == "__main__":
    d = make_draw(N=64, T=64, alphas=ALPHA_SETTINGS["balanced"], seed=0)
    print("X_row", d.X_row.shape, "X_col", d.X_col.shape)
    print("M", d.M.shape, "Y", d.Y.shape, "mask", d.mask.shape,
          "missing frac=", float(d.mask.mean()))
    print("noise_std=", d.noise_std)
    print("M frob=", float(np.linalg.norm(d.M, 'fro')),
          " sqrt(NT)*2=", 2 * np.sqrt(d.M.size))
