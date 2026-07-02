"""Agarwal-Choi-Yuan four-component estimator (arxiv:2603.24833).

Estimator for the model
    Y = M + noise,
    M = M1 + M2 + M3 + M4
where
    M1 = G1(X) Q1(Z)^T   (interaction; both contexts)
    M2 = G2(X) V1^T      (row-context driven, latent col factor)
    M3 = W1 Q2(Z)^T      (row latent factor, col-context driven)
    M4 = W2 V2^T         (pure latent low-rank)

Algorithm:
  1. Build sieve projectors  P_X = Phi (Phi^T Phi)^-1 Phi^T   (degree-4 poly basis)
                              P_Z = Psi (Psi^T Psi)^-1 Psi^T
  2. M_hat_1 = P_X Y P_Z                                          (closed form)
  3. M_hat_2 = SoftSVD( P_X Y (I - P_Z) ;  nu_2 )
     M_hat_3 = SoftSVD( (I - P_X) Y P_Z ;  nu_3 )
     M_hat_4 = SoftSVD( (I - P_X) Y (I - P_Z) ;  nu_4 )
  4. M_hat = sum.

Penalties (from the paper, scaled by sigma):
  nu_2 = c * sigma * sqrt(T)
  nu_3 = c * sigma * sqrt(N)
  nu_4 = c * sigma * sqrt(N + T)
with c a tunable multiplier (default 1.0).

Missingness handling. Two modes:
  - 'mean_fill' (default): replace missing entries with column means (over
    observed) before running the algorithm. Simple, works for any pattern,
    biased but standard for MC baselines.
  - 'ipw_mcar': for MCAR with known per-cell prob p_obs (= 1 - p_miss),
    use Y_tilde = Y_obs / p_obs at observed cells, 0 elsewhere. This is
    the inverse-probability-weighted version sketched in the paper for
    MCAR data.
"""
from __future__ import annotations

import numpy as np

from generate import _poly4_features


def _poly_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial features up to `degree` per coordinate (no cross-terms),
    plus a constant. For 4-dim X: J = 1 + 4*degree features.

    degree=1 → J=5  (paper default in arxiv:2603.24833 experiments)
    degree=2 → J=9
    degree=3 → J=13
    degree=4 → J=17 (matches the generator)
    degree=5 → J=21 (oversmoothed, slightly more variance)
    """
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    N, D = X.shape
    cols = [np.ones((N, 1), dtype=np.float64)]
    for d in range(D):
        xd = X[:, d:d + 1]
        for p in range(1, degree + 1):
            cols.append(xd ** p)
    return np.concatenate(cols, axis=1)


def _projector(B: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Hat matrix B (B^T B + ridge I)^-1 B^T  for a basis matrix B."""
    BtB = B.T @ B
    BtB += ridge * np.eye(BtB.shape[0])
    return B @ np.linalg.solve(BtB, B.T)


def _soft_thresholded_svd(A: np.ndarray, threshold: float) -> np.ndarray:
    """Singular-value soft-thresholding: shrink each sigma by `threshold`,
    clip at 0. Equivalent to argmin ||A - X||_F^2 + 2*threshold * ||X||_*."""
    if threshold <= 0:
        return A.copy()
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_shrunk = np.maximum(s - threshold, 0.0)
    if not np.any(s_shrunk > 0):
        return np.zeros_like(A)
    return (U * s_shrunk) @ Vt


def _mean_fill(Y_with_nan: np.ndarray) -> np.ndarray:
    out = Y_with_nan.copy().astype(np.float64)
    col_mean = np.nanmean(out, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    nan_idx = np.where(np.isnan(out))
    out[nan_idx] = np.take(col_mean, nan_idx[1])
    return out


def _acy_one_pass(
    Y_in: np.ndarray, P_X: np.ndarray, P_Z: np.ndarray,
    nu_2: float, nu_3: float, nu_4: float,
) -> np.ndarray:
    """One-pass ACY decomposition assuming Y_in is fully observed."""
    N, T = Y_in.shape
    I_N_minus_P_X = np.eye(N) - P_X
    I_T_minus_P_Z = np.eye(T) - P_Z
    M1 = P_X @ Y_in @ P_Z
    M2 = _soft_thresholded_svd(P_X @ Y_in @ I_T_minus_P_Z, nu_2)
    M3 = _soft_thresholded_svd(I_N_minus_P_X @ Y_in @ P_Z, nu_3)
    M4 = _soft_thresholded_svd(I_N_minus_P_X @ Y_in @ I_T_minus_P_Z, nu_4)
    return M1 + M2 + M3 + M4


def estimate_acy(
    Y_obs: np.ndarray,
    mask_missing: np.ndarray,
    X_row: np.ndarray,
    X_col: np.ndarray,
    noise_std: float = 0.5,
    penalty_mult: float = 1.0,
    missingness_mode: str = "mean_fill",
    p_obs: float | None = None,
    em_iters: int = 0,
    em_tol: float = 1e-4,
    poly_degree: int = 4,
) -> np.ndarray:
    """Return the ACY estimate Mhat (N, T) for the partially-observed matrix.

    Args:
      Y_obs:        (N, T) values; NaN at missing cells (or any sentinel —
                    we use mask_missing to identify missing).
      mask_missing: (N, T) bool, True = missing.
      X_row:        (N, d_r) row context.
      X_col:        (T, d_c) col context.
      noise_std:    sigma in nu_k = c * sigma * sqrt(...). Default 0.5
                    matches our synthetic generator.
      penalty_mult: c multiplier on the noise-std-scaled penalties.
      missingness_mode: 'mean_fill' or 'ipw_mcar'.
      p_obs:        per-cell observation probability; required for ipw_mcar.
    """
    N, T = Y_obs.shape
    Y = np.asarray(Y_obs, dtype=np.float64)
    Y_with_nan = np.where(mask_missing, np.nan, Y)

    if missingness_mode == "mean_fill":
        Y_in = _mean_fill(Y_with_nan)
    elif missingness_mode == "ipw_mcar":
        if p_obs is None or not (0.0 < p_obs <= 1.0):
            raise ValueError(f"ipw_mcar needs valid p_obs (got {p_obs!r})")
        Y_in = np.where(mask_missing, 0.0, Y) / p_obs
    else:
        raise ValueError(f"Unknown missingness_mode={missingness_mode!r}")

    # Sieve bases. degree=4 → J=17 matches the generator (oracle basis).
    # degree=1 → J=5 matches the paper's own experimental default.
    Phi = _poly_features(np.asarray(X_row, dtype=np.float64), poly_degree)
    Psi = _poly_features(np.asarray(X_col, dtype=np.float64), poly_degree)

    P_X = _projector(Phi)   # (N, N)
    P_Z = _projector(Psi)   # (T, T)

    nu_2 = penalty_mult * noise_std * np.sqrt(T)
    nu_3 = penalty_mult * noise_std * np.sqrt(N)
    nu_4 = penalty_mult * noise_std * np.sqrt(N + T)

    Mhat = _acy_one_pass(Y_in, P_X, P_Z, nu_2, nu_3, nu_4)

    # SoftImpute-style EM: re-impute missing cells with current estimate, refit.
    if em_iters > 0 and mask_missing.any():
        Y_em = Y_in.copy()
        for _ in range(em_iters):
            prev = Mhat.copy()
            Y_em = np.where(mask_missing, Mhat, Y_in)
            Mhat = _acy_one_pass(Y_em, P_X, P_Z, nu_2, nu_3, nu_4)
            if np.linalg.norm(Mhat - prev, "fro") / max(np.linalg.norm(prev, "fro"), 1e-12) < em_tol:
                break

    return Mhat.astype(np.float32)


if __name__ == "__main__":
    # Smoke test: generate one matrix and check the estimator runs end-to-end.
    from generate import make_draw, observed_matrix
    d = make_draw(N=200, T=200, alphas=(1.0, 0, 0, 0), seed=0,
                  missingness="mcar", missing_prob=0.4)
    Mhat = estimate_acy(d.Y, d.mask, d.X_row, d.X_col,
                        noise_std=d.noise_std, penalty_mult=1.0,
                        missingness_mode="ipw_mcar", p_obs=0.6)
    rmse = float(np.sqrt(np.mean((Mhat[d.mask] - d.M[d.mask])**2)))
    print(f"pure_interaction MCAR  RMSE on missing block = {rmse:.4f}")
