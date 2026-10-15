"""Greedy benchmark selection via entropy and mutual information.

Implements:
  - Algorithm 1 (greedy entropy): select the benchmark with largest
    conditional variance at each step (= pivoted Cholesky).
  - Algorithm 2 (greedy MI): select the benchmark maximizing
    log σ²_{v|S} + log P_{vv}, where P_{vv} is the complement precision
    diagonal.
"""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray


def random_select(
    Sigma: NDArray[np.floating],
    k: int,
    seed: Union[int, tuple] = 0,
    benchmark_names: Optional[NDArray] = None,
) -> dict:
    """Select k benchmarks in uniformly random order.

    Computes the same metrics as greedy_entropy (residual variance,
    log-determinants) but picks benchmarks uniformly at random without
    replacement, serving as a baseline for comparison.

    Parameters
    ----------
    Sigma : (N, N) covariance (or correlation) matrix, must be PSD.
    k : number of benchmarks to select.
    seed : random seed (int or tuple for deterministic sub-seeding).
    benchmark_names : optional (N,) array of names for reporting.

    Returns
    -------
    dict with keys:
        selected : list of k selected indices (in random order).
        names    : list of k names (if benchmark_names provided).
        log_dets : (k,) cumulative log-det values after each selection.
        residual_variance : (k,) residual variance fraction after each step.
    """
    N = Sigma.shape[0]
    assert 1 <= k <= N

    rng = np.random.default_rng(seed)
    total_var = np.trace(Sigma)

    perm = rng.permutation(N)
    selected = perm[:k].tolist()

    log_dets = []
    residual_variance = []
    RIDGE = 1e-8

    sel_set = set()
    for t in range(1, k + 1):
        sel_set.add(selected[t - 1])
        sel = selected[:t]
        R_AA = Sigma[np.ix_(sel, sel)]

        sign, logdet = np.linalg.slogdet(R_AA)
        log_dets.append(logdet if sign > 0 else -np.inf)

        unsel = np.array([j for j in range(N) if j not in sel_set])
        if len(unsel) > 0:
            R_UA = Sigma[np.ix_(unsel, sel)]
            R_AA_reg = R_AA + RIDGE * np.eye(t)
            W = np.linalg.solve(R_AA_reg, R_UA.T).T
            cond_var = np.diag(Sigma)[unsel] - np.sum(W * R_UA, axis=1)
            resid = np.sum(np.maximum(cond_var, 0.0))
        else:
            resid = 0.0
        residual_variance.append(resid / total_var)

    result = {
        "selected": selected,
        "log_dets": np.array(log_dets),
        "residual_variance": np.array(residual_variance),
    }
    if benchmark_names is not None:
        result["names"] = [str(benchmark_names[j]) for j in selected]

    return result


def greedy_entropy(
    Sigma: NDArray[np.floating],
    k: int,
    benchmark_names: Optional[NDArray] = None,
) -> dict:
    """Select k benchmarks by greedy entropy maximization.

    Parameters
    ----------
    Sigma : (N, N) covariance (or correlation) matrix, must be PSD.
    k : number of benchmarks to select.
    benchmark_names : optional (N,) array of names for reporting.

    Returns
    -------
    dict with keys:
        selected : list of k selected indices (in selection order).
        names    : list of k names (if benchmark_names provided).
        d        : (N,) final residual diagonal (conditional variances).
        log_dets : (k,) cumulative log-det values after each selection.
        residual_variance : (k,) residual variance fraction after each step.
    """
    N = Sigma.shape[0]
    assert Sigma.shape == (N, N)
    assert 1 <= k <= N

    total_var = np.trace(Sigma)

    # Residual diagonal = conditional variances
    d = np.diag(Sigma).copy().astype(np.float64)
    # Cholesky rows: L[j] stores the partial row for benchmark j
    L = np.zeros((N, k), dtype=np.float64)

    selected = []
    log_dets = []
    residual_variance = []
    cum_logdet = 0.0

    for t in range(k):
        # Select pivot with largest conditional variance
        # (mask already-selected to -inf)
        d_masked = d.copy()
        for j in selected:
            d_masked[j] = -np.inf
        j_star = int(np.argmax(d_masked))
        selected.append(j_star)

        cum_logdet += np.log(max(d[j_star], 1e-300))
        log_dets.append(cum_logdet)

        # Cholesky update for all remaining benchmarks
        sqrt_d = np.sqrt(max(d[j_star], 1e-300))
        for j in range(N):
            if j in selected and j != j_star:
                continue
            L[j, t] = (Sigma[j, j_star] - L[j, :t] @ L[j_star, :t]) / sqrt_d
            if j != j_star:
                d[j] -= L[j, t] ** 2

        # Residual variance = sum of conditional variances of unselected
        resid = sum(d[j] for j in range(N) if j not in selected)
        residual_variance.append(resid / total_var)

    result = {
        "selected": selected,
        "d": d,
        "log_dets": np.array(log_dets),
        "residual_variance": np.array(residual_variance),
    }
    if benchmark_names is not None:
        result["names"] = [str(benchmark_names[j]) for j in selected]

    return result


# ── Complement precision diagonal ────────────────────────────────

def _complement_precision_diag(Sigma: NDArray, comp: NDArray) -> NDArray:
    """Compute diagonal of (Sigma[comp, comp])^{-1}.

    Uses Cholesky with eigendecomposition fallback for near-singular blocks.
    Returns array of length len(comp).
    """
    from scipy.linalg import solve_triangular

    m = len(comp)
    Sigma_sub = Sigma[np.ix_(comp, comp)]

    try:
        L = np.linalg.cholesky(Sigma_sub)
        # diag(Σ^{-1}) = column norms² of L^{-1}
        L_inv = solve_triangular(L, np.eye(m), lower=True)
        return np.sum(L_inv ** 2, axis=0)
    except np.linalg.LinAlgError:
        # Eigendecomposition fallback: Σ^{-1} = V Λ^{-1} V^T
        eigvals, eigvecs = np.linalg.eigh(Sigma_sub)
        eigvals = np.maximum(eigvals, 1e-10)
        return np.sum(eigvecs ** 2 / eigvals[None, :], axis=1)


# ── Greedy MI maximization ───────────────────────────────────────

def greedy_mi(
    Sigma: NDArray[np.floating],
    k: int,
    benchmark_names: Optional[NDArray] = None,
) -> dict:
    """Select k benchmarks by greedy mutual information maximization.

    At each step, selects the benchmark v maximizing the MI gain
        Δ(v | S) = ½[log σ²_{v|S} + log P_{vv}]
    where σ²_{v|S} is the forward conditional variance (Cholesky) and
    P_{vv} = [Σ_{V\\S, V\\S}^{-1}]_{vv} is the complement precision diagonal.

    The complement precision is recomputed from scratch at each step via
    Cholesky factorization, avoiding the numerically unstable rank-1
    downdate of the explicit inverse.

    Parameters
    ----------
    Sigma : (N, N) covariance (or correlation) matrix, must be PSD.
    k : number of benchmarks to select.
    benchmark_names : optional (N,) array of names for reporting.

    Returns
    -------
    dict with keys:
        selected : list of k selected indices (in selection order).
        names    : list of k names (if benchmark_names provided).
        d        : (N,) final residual diagonal (conditional variances).
        log_dets : (k,) cumulative log-det of selected block.
        residual_variance : (k,) residual variance fraction after each step.
        mi_values : (k,) mutual information I(X_A; X_{V\\A}) after each step.
    """
    N = Sigma.shape[0]
    assert Sigma.shape == (N, N)
    assert 1 <= k <= N

    total_var = np.trace(Sigma)

    # log det(Σ_V) — constant term in MI formula
    sign_full, logdet_full = np.linalg.slogdet(Sigma)
    if sign_full <= 0:
        eigvals_full = np.linalg.eigvalsh(Sigma)
        logdet_full = np.sum(np.log(np.maximum(eigvals_full, 1e-300)))

    # Forward Cholesky state (identical to greedy_entropy)
    d = np.diag(Sigma).copy().astype(np.float64)
    L = np.zeros((N, k), dtype=np.float64)

    selected = []
    selected_set = set()
    log_dets = []
    residual_variance = []
    cum_logdet = 0.0

    for t in range(k):
        # Complement indices (sorted for reproducibility)
        comp = np.array(sorted(set(range(N)) - selected_set))

        # Complement precision diagonal via fresh factorization
        P_diag_comp = _complement_precision_diag(Sigma, comp)

        # Map back to full index space
        P_full = np.zeros(N)
        for i, idx in enumerate(comp):
            P_full[idx] = P_diag_comp[i]

        # Selection: argmax_{v ∈ complement} [log d_v + log P_{vv}]
        best_j = int(comp[0])
        best_score = -np.inf
        for j in comp:
            dj = d[j]
            pj = P_full[j]
            if dj > 1e-300 and pj > 1e-300:
                score = np.log(dj) + np.log(pj)
            else:
                score = -np.inf
            if score > best_score:
                best_score = score
                best_j = int(j)

        j_star = best_j
        selected.append(j_star)
        selected_set.add(j_star)

        # Forward Cholesky update
        cum_logdet += np.log(max(d[j_star], 1e-300))
        log_dets.append(cum_logdet)

        sqrt_d = np.sqrt(max(d[j_star], 1e-300))
        for j in range(N):
            if j in selected_set and j != j_star:
                continue
            L[j, t] = (Sigma[j, j_star] - L[j, :t] @ L[j_star, :t]) / sqrt_d
            if j != j_star:
                d[j] -= L[j, t] ** 2

        # Residual variance
        resid = sum(d[j] for j in range(N) if j not in selected_set)
        residual_variance.append(resid / total_var)

    # Compute MI = ½[log det(Σ_A) + log det(Σ_{V\A}) - log det(Σ_V)]
    mi_values = []
    for kk in range(1, k + 1):
        logdet_A = log_dets[kk - 1]
        comp_kk = np.array(sorted(set(range(N)) - set(selected[:kk])))
        if len(comp_kk) == 0:
            mi_values.append(0.0)
            continue
        Sigma_comp = Sigma[np.ix_(comp_kk, comp_kk)]
        sign_c, logdet_c = np.linalg.slogdet(Sigma_comp)
        if sign_c <= 0:
            eig_c = np.linalg.eigvalsh(Sigma_comp)
            logdet_c = np.sum(np.log(np.maximum(eig_c, 1e-300)))
        mi_values.append(0.5 * (logdet_A + logdet_c - logdet_full))

    result = {
        "selected": selected,
        "d": d,
        "log_dets": np.array(log_dets),
        "residual_variance": np.array(residual_variance),
        "mi_values": np.array(mi_values),
    }
    if benchmark_names is not None:
        result["names"] = [str(benchmark_names[j]) for j in selected]

    return result
