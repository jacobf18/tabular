"""Compare greedy entropy vs mutual information selection via 10-fold CV.

For each dataset, uses the 10% holdout (most training data) setting.
Runs both selection methods on each fold, evaluates imputation R² and
residual variance, then plots the comparison.
"""

import pathlib
import sys
import time
import numpy as np
from collections import defaultdict

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir, figures_dir, logs_dir

DATA = data_dir()
OUT = figures_dir()
LOGS = logs_dir()
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

from greedy_select import greedy_entropy, greedy_mi, random_select
from em_cov import em_cov
from cv_split import cv_folds, cv_train_val

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_MAX = 15
N_FOLDS = 10
HOLDOUT_FRAC = 0.1
SEED = 42

DATASETS = ["mmlu", "mteb", "merged"]

DISPLAY_NAME = {
    "mmlu": "MMLU",
    "mteb": "MTEB",
    "merged": "Merged",
}


# ── Covariance estimation (same as eval_greedy_all.py) ───────────

def estimate_corr_pairwise(B_train, O_train):
    col_count = np.maximum(O_train.sum(axis=0), 1.0)
    col_mean = (B_train * O_train).sum(axis=0) / col_count
    B_c = O_train * (B_train - col_mean[None, :])
    col_var = (B_c ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    col_std = np.sqrt(np.maximum(col_var, 1e-12))

    B_std = O_train * (B_train - col_mean[None, :]) / col_std[None, :]
    gram = B_std.T @ B_std
    pair_counts = O_train.T @ O_train
    denom = np.maximum(pair_counts - 1.0, 1.0)
    R = gram / denom

    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-6)
    R = (eigvecs * eigvals[None, :]) @ eigvecs.T
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(R), 1e-12))
    R = R * np.outer(d_inv, d_inv)
    return R, col_mean, col_std


def estimate_corr_em(B_train, O_train):
    M_tr, N_tr = B_train.shape
    obs_frac = O_train.sum() / (M_tr * N_tr)
    col_count = np.maximum(O_train.sum(axis=0), 1.0)
    col_mean = (B_train * O_train).sum(axis=0) / col_count
    B_c = O_train * (B_train - col_mean[None, :])
    col_var = (B_c ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    col_std_floor = 0.01 * (np.abs(col_mean) + 1.0)
    col_std = np.sqrt(np.maximum(col_var, col_std_floor ** 2))
    B_std = O_train * (B_train - col_mean[None, :]) / col_std[None, :]

    regularized = M_tr < N_tr or obs_frac < 0.5
    use_shrink = "auto" if M_tr < N_tr else 0.0
    em_eps = 1e-3 if regularized else 1e-6
    em_iter = 1000 if regularized else 500
    result = em_cov(B_std, O_train, max_iter=em_iter, tol=5e-4,
                    shrinkage=use_shrink, eps_psd=em_eps, verbose=False)
    Sigma = result["Sigma"]
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    R = Sigma * np.outer(d_inv, d_inv)
    return R, col_mean, col_std


def estimate_corr(B_train, O_train):
    M_tr, N_tr = B_train.shape
    obs_frac = O_train.sum() / (M_tr * N_tr)
    if obs_frac < 0.9 or M_tr < N_tr:
        return estimate_corr_em(B_train, O_train)
    else:
        return estimate_corr_pairwise(B_train, O_train)


# ── Imputation ───────────────────────────────────────────────────

def impute_and_evaluate(B_test, O_test, selected, R, col_mean, col_std, N):
    A = set(selected)
    M_test = B_test.shape[0]
    B_test_std = O_test * (B_test - col_mean[None, :]) / col_std[None, :]
    B_test_std = np.clip(B_test_std, -10, 10) * O_test
    RIDGE = 0.01

    # Check if all models share the same observation pattern (fast path)
    obs_A = np.array([j for j in selected if O_test[0, j] == 1])
    eval_idx = np.array([j for j in range(N) if j not in A and O_test[0, j] == 1])
    all_same = (len(obs_A) > 0 and len(eval_idx) > 0 and
                np.all(O_test[:, list(selected)] == O_test[0, list(selected)]) and
                np.all(O_test[:, eval_idx] == O_test[0, eval_idx]))

    if all_same:
        R_oo = R[np.ix_(obs_A, obs_A)] + RIDGE * np.eye(len(obs_A))
        R_eo = R[np.ix_(eval_idx, obs_A)]
        W = np.linalg.solve(R_oo, R_eo.T).T
        X_obs = B_test_std[:, obs_A]
        X_true = B_test_std[:, eval_idx]
        X_pred = X_obs @ W.T
        ss_res = np.sum((X_true - X_pred) ** 2)
        ss_tot = np.sum(X_true ** 2)
        n_eval = M_test * len(eval_idx)
    else:
        ss_res = 0.0
        ss_tot = 0.0
        n_eval = 0
        for i in range(M_test):
            obs_A_i = np.array([j for j in selected if O_test[i, j] == 1])
            eval_i = np.array([j for j in range(N) if j not in A and O_test[i, j] == 1])
            if len(obs_A_i) == 0 or len(eval_i) == 0:
                continue
            R_oo = R[np.ix_(obs_A_i, obs_A_i)] + RIDGE * np.eye(len(obs_A_i))
            R_eo = R[np.ix_(eval_i, obs_A_i)]
            W = np.linalg.solve(R_oo, R_eo.T).T
            x_obs = B_test_std[i, obs_A_i]
            x_true = B_test_std[i, eval_i]
            x_pred = W @ x_obs
            ss_res += np.sum((x_true - x_pred) ** 2)
            ss_tot += np.sum(x_true ** 2)
            n_eval += len(eval_i)

    if n_eval == 0:
        return {"r2": np.nan, "rmse": np.nan}
    return {"r2": 1.0 - ss_res / max(ss_tot, 1e-12),
            "rmse": np.sqrt(ss_res / n_eval)}


# ── Run one dataset ──────────────────────────────────────────────

def run_dataset(ds_name):
    print(f"\n{'#'*70}")
    print(f"# Dataset: {ds_name.upper()}")
    print(f"{'#'*70}")

    d = np.load(DATA / f"{ds_name}.matrix.npz", allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = d["benchmark_names"]
    M, N = B.shape
    obs_frac = O.sum() / (M * N)
    print(f"{M} models × {N} tasks, {obs_frac*100:.1f}% observed", flush=True)

    folds = cv_folds(M, N_FOLDS, SEED)

    p = HOLDOUT_FRAC
    metrics = {"entropy": defaultdict(list), "mi": defaultdict(list),
               "random": defaultdict(list)}

    for fold_idx in range(N_FOLDS):
        t0 = time.time()
        train_idx, val_idx = cv_train_val(folds, fold_idx, p, M, SEED)

        B_train, O_train = B[train_idx], O[train_idx]
        B_test, O_test = B[val_idx], O[val_idx]
        M_train = len(train_idx)
        M_test = len(val_idx)

        print(f"  Fold {fold_idx}: {M_train} train / {M_test} test",
              end="", flush=True)

        try:
            R, col_mean, col_std = estimate_corr(B_train, O_train)
        except Exception as e:
            print(f" — estimation failed: {e}", flush=True)
            for method in ["entropy", "mi", "random"]:
                for metric in ["r2", "rmse", "resid_var", "mi_val"]:
                    metrics[method][metric].append([np.nan] * K_MAX)
            continue

        # Run all three methods
        res_ent = greedy_entropy(R, K_MAX, benchmark_names=benchmark_names)
        res_mi = greedy_mi(R, K_MAX, benchmark_names=benchmark_names)
        rand_seed = (SEED + 7, fold_idx, int(p * 10000))
        res_rand = random_select(R, K_MAX, seed=rand_seed,
                                 benchmark_names=benchmark_names)

        # Compute MI values for all methods
        sign_full, logdet_full = np.linalg.slogdet(R)
        if sign_full <= 0:
            eigv = np.linalg.eigvalsh(R)
            logdet_full = np.sum(np.log(np.maximum(eigv, 1e-300)))

        for method, result in [("entropy", res_ent), ("mi", res_mi),
                               ("random", res_rand)]:
            fold_r2, fold_rmse, fold_rv, fold_mi = [], [], [], []
            for k in range(1, K_MAX + 1):
                selected = result["selected"][:k]
                fold_rv.append(result["residual_variance"][k - 1])
                imp = impute_and_evaluate(B_test, O_test, selected, R,
                                          col_mean, col_std, N)
                fold_r2.append(imp["r2"])
                fold_rmse.append(imp["rmse"])

                # MI of this subset
                R_AA = R[np.ix_(selected, selected)]
                s_a, ld_a = np.linalg.slogdet(R_AA)
                ld_a = ld_a if s_a > 0 else -1e30
                comp = sorted(set(range(N)) - set(selected))
                R_CC = R[np.ix_(comp, comp)]
                s_c, ld_c = np.linalg.slogdet(R_CC)
                ld_c = ld_c if s_c > 0 else -1e30
                fold_mi.append(0.5 * (ld_a + ld_c - logdet_full))

            metrics[method]["r2"].append(fold_r2)
            metrics[method]["rmse"].append(fold_rmse)
            metrics[method]["resid_var"].append(fold_rv)
            metrics[method]["mi_val"].append(fold_mi)

        elapsed = time.time() - t0
        print(f"  Ent R²@5={metrics['entropy']['r2'][-1][4]:.3f}"
              f"  MI R²@5={metrics['mi']['r2'][-1][4]:.3f}"
              f"  Rand R²@5={metrics['random']['r2'][-1][4]:.3f}"
              f"  ({elapsed:.1f}s)", flush=True)

    # Convert to arrays
    for method in ["entropy", "mi", "random"]:
        for key in metrics[method]:
            metrics[method][key] = np.array(metrics[method][key])

    return metrics, benchmark_names


# ── Plotting ─────────────────────────────────────────────────────

def plot_comparison(ds_name, metrics):
    """3-panel comparison: R², residual variance, MI value."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ks = np.arange(1, K_MAX + 1)

    colors = {"entropy": "#4477AA", "mi": "#CC3311", "random": "#888888"}
    labels = {"entropy": "Entropy (Alg. 1)", "mi": "Mutual Info (Alg. 2)",
              "random": "Random"}
    styles = {"entropy": "o-", "mi": "o-", "random": "x--"}

    for method in ["entropy", "mi", "random"]:
        c = colors[method]
        lab = labels[method]
        ls = styles[method]
        lw = 1.5 if method != "random" else 1.0
        ms = 4 if method != "random" else 3
        valid = ~np.all(np.isnan(metrics[method]["r2"]), axis=1)
        if not valid.any():
            continue

        mean_r2 = np.nanmean(metrics[method]["r2"][valid], axis=0)
        std_r2 = np.nanstd(metrics[method]["r2"][valid], axis=0)
        mean_rv = np.nanmean(metrics[method]["resid_var"][valid], axis=0)

        axes[0].plot(ks, mean_r2, ls, color=c, label=lab,
                     markersize=ms, linewidth=lw)
        axes[0].fill_between(ks, mean_r2 - std_r2, mean_r2 + std_r2,
                             color=c, alpha=0.15)
        axes[1].semilogy(ks, mean_rv, ls, color=c, label=lab,
                         markersize=ms, linewidth=lw)

    # MI values for all three methods
    for method in ["entropy", "mi", "random"]:
        c = colors[method]
        lab = labels[method]
        ls = styles[method]
        lw = 1.5 if method != "random" else 1.0
        ms = 4 if method != "random" else 3
        if "mi_val" in metrics[method]:
            valid = ~np.all(np.isnan(metrics[method]["mi_val"]), axis=1)
            if valid.any():
                mean_mi = np.nanmean(metrics[method]["mi_val"][valid], axis=0)
                axes[2].plot(ks, mean_mi, ls, color=c, label=lab,
                             markersize=ms, linewidth=lw)

    axes[0].set_ylabel("Imputation $R^2$")
    axes[1].set_ylabel("Residual variance fraction")
    axes[2].set_ylabel("Mutual information $I(X_{\\mathcal{A}}; X_{\\bar{\\mathcal{A}}})$")

    for ax in axes:
        ax.set_xlabel("Selected benchmarks $k$")
        ax.set_xlim(0.5, K_MAX + 0.5)
        ax.set_xticks([1, 3, 5, 7, 10, 13, 15])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = OUT / f"entropy_vs_mi_{ds_name}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\nPlot saved to {out_path}", flush=True)
    plt.close()


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    total_t0 = time.time()

    for ds_name in DATASETS:
        ds_t0 = time.time()
        metrics, bench_names = run_dataset(ds_name)
        plot_comparison(ds_name, metrics)

        # Save results
        save_dict = {"benchmark_names": bench_names}
        for method in ["entropy", "mi", "random"]:
            for key, arr in metrics[method].items():
                save_dict[f"{method}_{key}"] = arr
        np.savez(LOGS / f"entropy_vs_mi_{ds_name}.npz", **save_dict)

        ds_elapsed = time.time() - ds_t0
        print(f"\n{ds_name.upper()} completed in {ds_elapsed:.0f}s", flush=True)

    total_elapsed = time.time() - total_t0
    print(f"\nAll datasets completed in {total_elapsed:.0f}s", flush=True)
