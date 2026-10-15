"""EM algorithm for Gaussian covariance estimation under incomplete observations.

Implements the algorithm described in Section 4.3 of the paper.  Given a
score matrix B ∈ R^{M×N} with missing entries indicated by an observation
mask O ∈ {0,1}^{M×N}, we estimate (μ, Σ) via EM under the model
B_{i·} ~ N(μ, Σ), treating missing entries as latent variables (MAR).

References
----------
Dempster, Laird & Rubin (1977).  Maximum likelihood from incomplete data
    via the EM algorithm.  JRSS-B 39(1).
Little & Rubin (2019).  Statistical Analysis with Missing Data, 3rd ed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def _mean_from_observed(
    B: NDArray[np.floating],
    O: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Per-benchmark mean from observed entries (Eq. mean-missing).

    μ̂_j = (Σ_i O_{ij} B_{ij}) / (Σ_i O_{ij}).
    """
    col_sums = O.sum(axis=0)  # (N,)
    col_sums = np.maximum(col_sums, 1.0)  # avoid division by zero
    return (B * O).sum(axis=0) / col_sums


def _pairwise_complete_cov(
    B: NDArray[np.floating],
    O: NDArray[np.floating],
    mu: NDArray[np.floating],
) -> NDArray[np.floating]:
    r"""Pairwise-complete covariance estimate (Eq. pairwise-cov).

    Σ̂^{pw}_{jk} = (B̄^T B̄)_{jk} / (O^T O - 1)_{jk}  where B̄ = O⊙(B - μ).
    """
    B_centered = O * (B - mu[None, :])  # (M, N)
    gram = B_centered.T @ B_centered  # (N, N)
    pair_counts = O.T @ O  # (N, N)
    # need at least 2 observations per pair for an unbiased estimate
    denom = np.maximum(pair_counts - 1.0, 1.0)
    return gram / denom


def _project_psd(
    S: NDArray[np.floating],
    eps: float = 1e-6,
) -> NDArray[np.floating]:
    """Project a symmetric matrix onto the PSD cone (clamp eigenvalues)."""
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, eps)
    return (eigvecs * eigvals[None, :]) @ eigvecs.T


# ---------------------------------------------------------------------------
# E-step
# ---------------------------------------------------------------------------

def _e_step(
    B: NDArray[np.floating],
    O: NDArray[np.floating],
    mu: NDArray[np.floating],
    Sigma: NDArray[np.floating],
) -> tuple[NDArray[np.floating], list[NDArray[np.floating]]]:
    r"""E-step: impute missing entries and compute conditional covariances.

    For each model i with observed set B_i and missing set B̄_i:

        E[B_{i,B̄_i} | B_{i,B_i}]  (Eq. em-emean)
        C_i = Cov(B_{i,B̄_i} | B_{i,B_i})  (Eq. em-ecov)

    Returns
    -------
    B_tilde : (M, N) completed score matrix.
    C_list  : length-M list of (N, N) sparse correction matrices.
              C_list[i][j,k] is non-zero only when both j,k ∈ B̄_i.
    """
    M, N = B.shape
    B_tilde = B.copy()
    C_list: list[NDArray[np.floating]] = []

    for i in range(M):
        obs = np.where(O[i] == 1)[0]
        mis = np.where(O[i] == 0)[0]

        C_i = np.zeros((N, N))

        if len(mis) == 0:
            # fully observed — nothing to impute
            C_list.append(C_i)
            continue

        if len(obs) == 0:
            # fully missing — impute with marginal mean, C_i = full Sigma
            B_tilde[i, :] = mu
            C_i[:, :] = Sigma
            C_list.append(C_i)
            continue

        # Partition Sigma into observed/missing blocks
        Sigma_oo = Sigma[np.ix_(obs, obs)]
        Sigma_mo = Sigma[np.ix_(mis, obs)]
        Sigma_mm = Sigma[np.ix_(mis, mis)]

        # Solve via Cholesky for numerical stability; add a small ridge
        # if Sigma_oo is near-singular (e.g. M < N or very few observations).
        try:
            L_oo, low = cho_factor(Sigma_oo)
        except np.linalg.LinAlgError:
            Sigma_oo = Sigma_oo + 1e-6 * np.eye(len(obs))
            L_oo, low = cho_factor(Sigma_oo)

        # Sigma_mo @ Sigma_oo^{-1}
        # Equivalently solve Sigma_oo @ X^T = Sigma_mo^T  →  X = (solve)^T
        SooInv_SmoT = cho_solve((L_oo, low), Sigma_mo.T)  # (|obs|, |mis|)

        # Conditional mean  (Eq. em-emean)
        residual = B[i, obs] - mu[obs]
        cond_mean = mu[mis] + Sigma_mo @ cho_solve((L_oo, low), residual)
        B_tilde[i, mis] = cond_mean

        # Conditional covariance  (Eq. em-ecov)
        cond_cov = Sigma_mm - Sigma_mo @ SooInv_SmoT  # (|mis|, |mis|)

        # Place into full N×N matrix at (mis, mis) positions
        C_i[np.ix_(mis, mis)] = cond_cov
        C_list.append(C_i)

    return B_tilde, C_list


# ---------------------------------------------------------------------------
# M-step
# ---------------------------------------------------------------------------

def _m_step(
    B_tilde: NDArray[np.floating],
    C_list: list[NDArray[np.floating]],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    r"""M-step: re-estimate μ, Σ from completed data + correction.

    μ^{(t+1)} = (1/M) B̃^T 1_M                        (Eq. em-mstep-mu)
    Σ^{(t+1)} = (1/M) B̄̃^T B̄̃  +  (1/M) Σ_i C_i     (Eq. em-mstep-sigma)
    """
    M, N = B_tilde.shape
    mu_new = B_tilde.mean(axis=0)  # (N,)
    B_centered = B_tilde - mu_new[None, :]  # (M, N)
    Sigma_new = (B_centered.T @ B_centered) / M

    # Correction term: average of conditional covariance matrices
    C_avg = np.zeros((N, N))
    for C_i in C_list:
        C_avg += C_i
    C_avg /= M

    Sigma_new += C_avg
    return mu_new, Sigma_new


# ---------------------------------------------------------------------------
# Observed-data log-likelihood (for convergence monitoring)
# ---------------------------------------------------------------------------

def _observed_loglik(
    B: NDArray[np.floating],
    O: NDArray[np.floating],
    mu: NDArray[np.floating],
    Sigma: NDArray[np.floating],
) -> float:
    r"""Observed-data log-likelihood Σ_i log p(B_{i, B_i} | μ, Σ).

    Each model i contributes the log-density of a |B_i|-variate Gaussian.
    When a submatrix is near-singular, a small ridge is added for stability.
    """
    M, N = B.shape
    ll = 0.0
    for i in range(M):
        obs = np.where(O[i] == 1)[0]
        if len(obs) == 0:
            continue
        d = len(obs)
        Sigma_oo = Sigma[np.ix_(obs, obs)]
        residual = B[i, obs] - mu[obs]
        try:
            L_oo, low = cho_factor(Sigma_oo)
        except np.linalg.LinAlgError:
            Sigma_oo = Sigma_oo + 1e-6 * np.eye(d)
            try:
                L_oo, low = cho_factor(Sigma_oo)
            except np.linalg.LinAlgError:
                return -np.inf
        log_det = 2.0 * np.sum(np.log(np.diag(L_oo)))
        maha = residual @ cho_solve((L_oo, low), residual)
        ll += -0.5 * (d * np.log(2 * np.pi) + log_det + maha)
    return float(ll)


# ---------------------------------------------------------------------------
# Main EM routine
# ---------------------------------------------------------------------------

def em_cov(
    B: NDArray[np.floating],
    O: NDArray[np.floating] | None = None,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
    eps_psd: float = 1e-6,
    shrinkage: float | str = 0.0,
    verbose: bool = False,
) -> dict:
    r"""EM estimation of (μ, Σ) from an incomplete score matrix.

    Parameters
    ----------
    B : (M, N) score matrix.  Unobserved entries may be any value (e.g. 0
        or NaN); they are identified by the mask *O*.
    O : (M, N) binary observation mask, 1 = observed.  If *None*, entries
        that are NaN in *B* are treated as missing; the rest as observed.
    max_iter : maximum EM iterations.
    tol : relative change in log-likelihood for convergence.
    eps_psd : floor for eigenvalues when projecting to PSD cone.
    shrinkage : Ledoit-Wolf shrinkage intensity α ∈ [0, 1].  At each M-step,
        Σ ← (1-α)Σ + α·(tr(Σ)/N)·I.  Use ``"auto"`` to set α = (N-M)/N
        when M < N (proportional to rank deficiency), 0 otherwise.
    verbose : print iteration-level diagnostics.

    Returns
    -------
    dict with keys:
        mu      : (N,) estimated mean vector.
        Sigma   : (N, N) estimated covariance matrix.
        B_imp   : (M, N) completed score matrix from the final E-step.
        n_iter  : number of EM iterations run.
        loglik  : list of observed-data log-likelihood values.
        converged : whether the algorithm converged within *max_iter*.
    """
    B = np.asarray(B, dtype=np.float64)
    M, N = B.shape

    # Build observation mask from NaNs if not provided
    if O is None:
        O = (~np.isnan(B)).astype(np.float64)
        B = np.nan_to_num(B, nan=0.0)
    else:
        O = np.asarray(O, dtype=np.float64)

    # --- Initialisation (§4.3, "Initialization" paragraph) ---
    mu = _mean_from_observed(B, O)
    Sigma_pw = _pairwise_complete_cov(B, O, mu)
    Sigma = _project_psd(Sigma_pw, eps=eps_psd)

    # Compute shrinkage intensity
    auto_shrink = isinstance(shrinkage, str) and shrinkage == "auto"
    if auto_shrink:
        alpha = max(0.0, (N - M) / N) if M < N else 0.0
    else:
        alpha = float(shrinkage)

    # Apply shrinkage to initial estimate to ensure PD for E-step
    if alpha > 0:
        target = np.trace(Sigma) / N * np.eye(N)
        Sigma = (1 - alpha) * Sigma + alpha * target

    ll_prev = _observed_loglik(B, O, mu, Sigma)
    ll_history = [ll_prev]

    if verbose:
        frac_missing = 1.0 - O.sum() / O.size
        print(f"EM init  | missing {frac_missing:.1%} | "
              f"loglik {ll_prev:.4f}"
              + (f" | shrinkage α={alpha:.3f}" if alpha > 0 else ""))

    converged = False
    for it in range(1, max_iter + 1):
        # E-step
        B_tilde, C_list = _e_step(B, O, mu, Sigma)

        # M-step
        mu_new, Sigma_new = _m_step(B_tilde, C_list)

        # Ensure PSD (clamp small/negative eigenvalues)
        Sigma_new = _project_psd(Sigma_new, eps=eps_psd)

        # Monitor convergence via parameter change (robust to rank deficiency)
        param_change = (np.linalg.norm(Sigma_new - Sigma, 'fro')
                        / max(np.linalg.norm(Sigma, 'fro'), 1e-10))

        mu, Sigma = mu_new, Sigma_new

        # Monitor log-likelihood (informational; may not be monotone
        # under rank deficiency)
        ll = _observed_loglik(B, O, mu, Sigma)
        ll_history.append(ll)

        if verbose:
            delta = ll - ll_prev
            print(f"EM iter {it:3d} | loglik {ll:.4f} | "
                  f"Δ {delta:+.2e} | ΔΣ {param_change:.2e}")

        if param_change < tol and it > 1:
            converged = True
            if verbose:
                print(f"EM converged after {it} iterations "
                      f"(ΔΣ/‖Σ‖ = {param_change:.2e}).")
            break

        ll_prev = ll

    # Apply post-hoc shrinkage if requested (ensures PD for downstream use)
    if alpha > 0:
        target = np.trace(Sigma) / N * np.eye(N)
        Sigma = (1 - alpha) * Sigma + alpha * target
        if verbose:
            print(f"  Post-hoc shrinkage α={alpha:.3f} applied.")

    # Final E-step to get imputed matrix consistent with final parameters
    B_imp, _ = _e_step(B, O, mu, Sigma)

    return {
        "mu": mu,
        "Sigma": Sigma,
        "B_imp": B_imp,
        "n_iter": it if max_iter > 0 else 0,
        "loglik": ll_history,
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    # Generate ground-truth
    N = 6  # benchmarks
    M = 50  # models
    mu_true = np.random.randn(N)
    A = np.random.randn(N, N)
    Sigma_true = A @ A.T + 0.1 * np.eye(N)

    B_full = np.random.multivariate_normal(mu_true, Sigma_true, size=M)

    # Mask ~30% of entries at random
    rng = np.random.default_rng(0)
    O = rng.random((M, N)) > 0.3
    O = O.astype(np.float64)
    # Ensure every row has at least one observation
    for i in range(M):
        if O[i].sum() == 0:
            O[i, rng.integers(N)] = 1.0
    # Ensure every column has at least two observations
    for j in range(N):
        if O[:, j].sum() < 2:
            choices = np.where(O[:, j] == 0)[0]
            O[choices[:2], j] = 1.0

    B_obs = B_full * O  # zero out missing entries

    result = em_cov(B_obs, O, verbose=True, max_iter=100)

    mu_err = np.linalg.norm(result["mu"] - mu_true)
    Sigma_err = (np.linalg.norm(result["Sigma"] - Sigma_true, "fro")
                 / np.linalg.norm(Sigma_true, "fro"))

    print(f"\n‖μ̂ − μ‖₂          = {mu_err:.4f}")
    print(f"‖Σ̂ − Σ‖_F / ‖Σ‖_F = {Sigma_err:.4f}")
    print(f"Converged: {result['converged']} in {result['n_iter']} iters")
