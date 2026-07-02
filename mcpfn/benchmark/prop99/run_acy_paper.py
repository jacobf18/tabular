"""Re-implement the ACY paper's Section 5.2 Prop99 experiment.

Setup (mirrors arxiv:2603.24833 Section 5.2):
  - Panel:        states × years cigsale, 1970-2000 (S=39, T=31).
  - Row context:  5 state-level covariates (built by build_acy_covariates.py).
  - Col context:  3 year-level covariates.
  - Missingness:  per iteration, pick 8 random states to be "treated"; mask
                  their cells in periods T_0+1 through T (post-cutoff).
                  Test block = 8 × (T − T_0).
  - T_0 sweep:    {10, 15, 20, 25}.
  - Iterations:   100 per T_0.

Methods:
  - acy_deg2     : ACY with degree-2 polynomial basis (the paper's choice).
  - acy_cv       : ACY with CV-selected polynomial degree over {1..4}.
  - matformer    : MatFormer alone (no context).
  - boost_a005   : MatFormer + xRFM residual, α=0.05.
  - hybrid       : ACY-CV → boost-on-residual.

Metrics (per paper):
  - amse_cell        : mean((Mhat - true)^2) over all missing cells.
  - amse_year_avg    : for each year, average prediction & truth across 8
                       treated states, MSE on those averages.
  - amse_overall_avg : average across the 8×(T-T_0) test block, squared diff
                       to truth's overall average.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '/root/tabular/mcpfn/src')
sys.path.insert(0, '/root/tabular/mcpfn/benchmark/synthetic_context')

from acy_estimator import estimate_acy, _poly_features, _projector, _mean_fill  # noqa: E402
from tabimpute.interface_rfm import BoostMatFormerImputer  # noqa: E402


ROUND2_T2_CONFIG = {'config': {
    'embedding_size': 768, 'num_attention_heads': 16, 'mlp_hidden_size': 1536,
    'num_layers': 8, 'num_outputs': 5000, 'num_cls': 12, 'use_rope': True,
    'rope_base': 10000.0, 'rope_fraction': 0.5,
    'use_absolute_positional_embeddings': False, 'positional_damping_factor': 0.1,
    'attention_dropout': 0.0, 'ffn_dropout': 0.05, 'drop_path_rate': 0.01,
    'residual_scale_init': 0.5, 'embedding_dropout': 0.0, 'rms_norm_eps': 1e-6,
}}
CHECKPOINT_PATH = '/mnt/checkpoints_sweep/workdir/dwip_v2-trial_002/checkpoint_17000.pth'

PROP99_DIR = Path("/root/tabular/mcpfn/benchmark/prop99")
DATA_DIR = PROP99_DIR / "acy_paper_data"

CV_DEGREE_GRID = [1, 2, 3, 4]


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[int]]:
    panel = pd.read_csv(DATA_DIR / "panel_cigsale.csv", index_col=0)
    state_covs = pd.read_csv(DATA_DIR / "covariates_states.csv", index_col=0)
    year_covs = pd.read_csv(DATA_DIR / "covariates_years.csv", index_col=0)

    # Align state order across panel and state_covs.
    state_covs = state_covs.reindex(panel.index)
    M = panel.to_numpy(dtype=np.float64)        # (S, T)
    Xrow = state_covs.to_numpy(dtype=np.float64)  # (S, d_row)
    Xcol = year_covs.to_numpy(dtype=np.float64)   # (T, d_col)
    states = panel.index.tolist()
    years = [int(c) for c in panel.columns]
    return M, Xrow, Xcol, states, years


def amse_metrics(pred: np.ndarray, truth: np.ndarray,
                 treated_rows: np.ndarray, post_cols: np.ndarray) -> dict[str, float]:
    """Per-iteration metrics on the test block (treated_rows × post_cols)."""
    rr = treated_rows[:, None]
    cc = post_cols[None, :]
    p = pred[rr, cc]     # (n_treat, n_post)
    t = truth[rr, cc]

    amse_cell = float(np.mean((p - t) ** 2))
    # year-by-year averages across treated states
    year_pred = p.mean(axis=0)
    year_true = t.mean(axis=0)
    amse_year = float(np.mean((year_pred - year_true) ** 2))
    # overall average across all treated × post cells
    amse_overall = float((p.mean() - t.mean()) ** 2)
    return {
        "amse_cell": amse_cell,
        "amse_year": amse_year,
        "amse_overall": amse_overall,
    }


def run_acy(M, Xrow, Xcol, mask, degree, em_iters, noise_std):
    return estimate_acy(
        M, mask, Xrow, Xcol,
        noise_std=noise_std, penalty_mult=1.0,
        missingness_mode="mean_fill", em_iters=em_iters,
        poly_degree=degree,
    ).astype(np.float64)


def cv_select_degree(M, Xrow, Xcol, mask, em_iters, noise_std,
                     seed, val_frac=0.1):
    obs_rows, obs_cols = np.where(~mask)
    n_obs = obs_rows.size
    n_val = max(10, int(round(val_frac * n_obs)))
    rng = np.random.default_rng(seed + 999)
    perm = rng.permutation(n_obs)
    val_r = obs_rows[perm[:n_val]]
    val_c = obs_cols[perm[:n_val]]
    mask_train = mask.copy()
    mask_train[val_r, val_c] = True

    best_mse, best_deg = float("inf"), CV_DEGREE_GRID[0]
    for deg in CV_DEGREE_GRID:
        cv_em = min(20, em_iters)
        Mhat = run_acy(M, Xrow, Xcol, mask_train, deg, cv_em, noise_std)
        mse = float(np.mean((Mhat[val_r, val_c] - M[val_r, val_c]) ** 2))
        if mse < best_mse:
            best_mse, best_deg = mse, deg
    return best_deg


def run_one_iteration(
    M, Xrow, Xcol, treated_rows, post_cols, imputer_boost, noise_std,
    em_iters=50, seed=0,
):
    S, T = M.shape
    mask = np.zeros((S, T), dtype=bool)
    rr = treated_rows[:, None]
    cc = post_cols[None, :]
    mask[rr, cc] = True
    Y_obs = np.where(mask, np.nan, M).astype(np.float32)

    preds = {}

    # --- ACY at the paper's degree=2 ---
    preds["acy_deg2"] = run_acy(M, Xrow, Xcol, mask, 2, em_iters, noise_std)

    # --- ACY-CV ---
    best_deg = cv_select_degree(M, Xrow, Xcol, mask, em_iters, noise_std, seed)
    preds["acy_cv"] = run_acy(M, Xrow, Xcol, mask, best_deg, em_iters, noise_std)

    # --- MatFormer alone (boost imputer with α=0 ≈ pure MatFormer) ---
    # Easier: call ImputePFN.impute directly on the imputer to skip the xRFM step.
    from tabimpute.interface import ImputePFN
    with torch.no_grad():
        X_mlp, _ = ImputePFN.impute(imputer_boost, Y_obs, return_full=True)
    preds["matformer"] = np.asarray(X_mlp, dtype=np.float64)

    # --- Boost α=0.05 ---
    imputer_boost.blend_alpha = 0.05
    with torch.no_grad():
        X_boost = imputer_boost.impute(Y_obs)
    preds["boost_a005"] = np.asarray(X_boost, dtype=np.float64)

    # --- Hybrid: ACY-CV → boost-on-residual ---
    M_hat_acy = preds["acy_cv"]
    Y_resid = np.where(mask, np.nan, M - M_hat_acy).astype(np.float32)
    imputer_boost.blend_alpha = 0.1   # standard hybrid α
    with torch.no_grad():
        M_hat_resid = imputer_boost.impute(Y_resid)
    preds["hybrid"] = M_hat_acy + np.asarray(M_hat_resid, dtype=np.float64)

    return preds, best_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_iters", type=int, default=100)
    ap.add_argument("--n_treated", type=int, default=8)
    ap.add_argument("--T0_list", nargs="+", type=int, default=[10, 15, 20, 25])
    ap.add_argument("--em_iters", type=int, default=50)
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--out_csv", type=Path,
                    default=DATA_DIR / "per_iteration_metrics.csv")
    args = ap.parse_args()

    M, Xrow, Xcol, states, years = load_data()
    S, T = M.shape
    print(f"Panel: S={S} states × T={T} years (years {years[0]}..{years[-1]})")
    print(f"Row context: {Xrow.shape}  Col context: {Xcol.shape}")

    # Estimate noise std from cigsale's natural scale. Cigsale values range
    # ~50-250 packs/capita; signal std ≈ 30. Use a rough fraction.
    noise_std = float(np.nanstd(M)) * 0.1  # 10% of total signal std as noise
    print(f"Using noise_std = {noise_std:.3f} (= 0.1 × std of cigsale)")

    print(f"Loading BoostMatFormerImputer ...", flush=True)
    imputer_boost = BoostMatFormerImputer(
        device='cuda', entry_wise_features=False,
        json_config=ROUND2_T2_CONFIG, checkpoint_path=CHECKPOINT_PATH,
        max_num_rows=S, max_num_cols=T, blend_alpha=0.05,
    )

    rng_master = np.random.default_rng(args.seed_base)
    rows = []
    t_start = time.time()

    for T0 in args.T0_list:
        if T0 >= T:
            raise SystemExit(f"T0={T0} >= T={T}; nothing to test.")
        post_cols = np.arange(T0, T)
        print(f"\n=== T0 = {T0}  (test cols {T0}..{T-1}, {len(post_cols)} periods)")
        for it in range(args.n_iters):
            seed = args.seed_base + 1000 * T0 + it
            iter_rng = np.random.default_rng(seed)
            treated_rows = iter_rng.choice(S, args.n_treated, replace=False)

            t0 = time.time()
            preds, deg = run_one_iteration(
                M, Xrow, Xcol, treated_rows, post_cols, imputer_boost,
                noise_std=noise_std, em_iters=args.em_iters, seed=seed,
            )
            t_iter = time.time() - t0

            for method, pred in preds.items():
                m = amse_metrics(pred, M, treated_rows, post_cols)
                rows.append({
                    "T0": T0, "iter": it, "method": method,
                    "amse_cell": m["amse_cell"],
                    "amse_year": m["amse_year"],
                    "amse_overall": m["amse_overall"],
                    "cv_deg": deg if method in {"acy_cv", "hybrid"} else None,
                    "n_treated": args.n_treated,
                    "test_cells": args.n_treated * len(post_cols),
                    "runtime_s": t_iter,
                })

            if it % 10 == 0 or it == args.n_iters - 1:
                # Print per-method running mean amse_cell
                df_tmp = pd.DataFrame(rows)
                summ = df_tmp[df_tmp.T0 == T0].groupby("method")["amse_cell"].mean()
                summ_str = "  ".join(f"{k}={v:7.2f}" for k, v in summ.items())
                print(f"  [it {it:3d}/{args.n_iters}]  iter_time={t_iter:.1f}s  "
                      f"means: {summ_str}  ({time.time()-t_start:.0f}s)",
                      flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {len(rows)} rows to {args.out_csv}")
    print()
    print("=== Mean AMSE by (T0, method) — paper's metric: amse_cell ===")
    print(df.groupby(["T0", "method"])["amse_cell"].mean().round(2)
            .unstack("method").to_string())


if __name__ == "__main__":
    main()
