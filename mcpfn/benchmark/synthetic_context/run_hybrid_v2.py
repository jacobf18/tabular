"""Variant B: ACY-style four-component decomposition with MatFormer-boost
applied to each of the three SVT pieces (M_2, M_3, M_4).

Per matrix:
  1. Mean-fill Y → Y_filled.
  2. CV-select polynomial degree d via observed-cell holdout (scoring full ACY).
  3. Build P_X, P_Z from degree-d polynomial basis on (X_row, X_col).
  4. M̂_1 = P_X Y_filled P_Z                                    [closed-form, no boost]
  5. For each k in {2, 3, 4}:
       proj_k = sign-pattern * Y_filled * sign-pattern
       Y_k    = where(mask_missing, NaN, proj_k)
       M̂_k    = BoostMatFormerImputer.impute(Y_k)
  6. Final M̂ = M̂_1 + M̂_2 + M̂_3 + M̂_4.

This replaces ACY's soft-thresholded SVD on M_2/M_3/M_4 with MatFormer's
pretrained low-rank prior + xRFM residual correction. M_1 retains the
closed-form sieve projection from the paper.
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

from generate import ALPHA_SETTINGS, make_draw  # noqa: E402
from acy_estimator import (  # noqa: E402
    _poly_features, _projector, _mean_fill, estimate_acy,
)
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

DEGREE_GRID = [1, 2, 3, 4, 5, 6]


def rmse(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))


def cv_select_degree(
    Y: np.ndarray, mask_missing: np.ndarray,
    X_row: np.ndarray, X_col: np.ndarray,
    val_frac: float, seed: int, noise_std: float,
) -> int:
    """Cheap CV: full ACY at em=20 over 6 degrees, score on val cells."""
    obs_rows, obs_cols = np.where(~mask_missing)
    n_obs = obs_rows.size
    n_val = max(10, int(round(val_frac * n_obs)))
    rng = np.random.default_rng(seed + 1000)
    perm = rng.permutation(n_obs)
    val_sel = perm[:n_val]
    val_r = obs_rows[val_sel]
    val_c = obs_cols[val_sel]

    mask_train = mask_missing.copy()
    mask_train[val_r, val_c] = True

    best_mse, best_deg = float("inf"), DEGREE_GRID[0]
    for deg in DEGREE_GRID:
        Mhat_cv = estimate_acy(
            Y, mask_train, X_row, X_col,
            noise_std=noise_std, penalty_mult=1.0,
            missingness_mode="mean_fill", em_iters=20,
            poly_degree=deg,
        )
        mse = float(np.mean((Mhat_cv[val_r, val_c] - Y[val_r, val_c]) ** 2))
        if mse < best_mse:
            best_mse, best_deg = mse, deg
    return best_deg


def hybrid_v2(
    draw, imputer: BoostMatFormerImputer, val_frac: float, seed: int,
    noise_std: float,
) -> tuple[np.ndarray, int]:
    Y = draw.Y.astype(np.float64)
    mask_missing = draw.mask
    X_row = draw.X_row.astype(np.float64)
    X_col = draw.X_col.astype(np.float64)
    N, T = Y.shape

    best_deg = cv_select_degree(Y, mask_missing, X_row, X_col,
                                val_frac, seed, noise_std)

    # Mean-fill once.
    Y_for_fit = np.where(mask_missing, np.nan, Y)
    Y_filled = _mean_fill(Y_for_fit)

    # Build projectors with chosen degree.
    Phi = _poly_features(X_row, best_deg)
    Psi = _poly_features(X_col, best_deg)
    P_X = _projector(Phi)
    P_Z = _projector(Psi)
    I_minus_P_X = np.eye(N) - P_X
    I_minus_P_Z = np.eye(T) - P_Z

    # M_1: closed-form projection (the paper's approach for this piece).
    M_hat_1 = P_X @ Y_filled @ P_Z

    # Helper: run boost on a projected piece (NaN at original missing).
    def boost_on_piece(piece: np.ndarray) -> np.ndarray:
        Y_piece = np.where(mask_missing, np.nan, piece).astype(np.float32)
        with torch.no_grad():
            out = imputer.impute(Y_piece)
        return np.asarray(out, dtype=np.float64)

    # M_2, M_3, M_4: replace soft-SVT with boost.
    M_hat_2 = boost_on_piece(P_X         @ Y_filled @ I_minus_P_Z)
    M_hat_3 = boost_on_piece(I_minus_P_X @ Y_filled @ P_Z)
    M_hat_4 = boost_on_piece(I_minus_P_X @ Y_filled @ I_minus_P_Z)

    M_hat = M_hat_1 + M_hat_2 + M_hat_3 + M_hat_4
    return M_hat, 1 + 4 * best_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--settings", nargs="+", default=list(ALPHA_SETTINGS.keys()))
    ap.add_argument("--missingness",
                    choices=["block", "mcar", "mar", "mnar"], default="block")
    ap.add_argument("--missing_prob", type=float, default=0.4)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--noise_std", type=float, default=0.5)
    ap.add_argument("--method_name", type=str, default="hybrid_v2")
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/root/tabular/mcpfn/benchmark/synthetic_context/results"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "per_draw_metrics.csv"

    print(f"Loading BoostMatFormerImputer (α={args.alpha}) ...", flush=True)
    imputer = BoostMatFormerImputer(
        device='cuda', entry_wise_features=False,
        json_config=ROUND2_T2_CONFIG, checkpoint_path=CHECKPOINT_PATH,
        max_num_rows=args.N, max_num_cols=args.T,
        blend_alpha=args.alpha,
    )

    rows = []
    t_start = time.time()
    j_choices: dict[int, int] = {}

    for setting in args.settings:
        alphas = ALPHA_SETTINGS[setting]
        for seed in range(args.n_seeds):
            draw = make_draw(N=args.N, T=args.T, alphas=alphas, seed=seed,
                             missingness=args.missingness,
                             missing_prob=args.missing_prob)
            M_true = draw.M
            mask_missing = draw.mask

            t0 = time.time()
            M_hat, J = hybrid_v2(draw, imputer, args.val_frac, seed, args.noise_std)
            t = time.time() - t0
            r_miss = rmse(M_hat[mask_missing], M_true[mask_missing])
            r_obs = rmse(M_hat[~mask_missing], M_true[~mask_missing])

            j_choices[J] = j_choices.get(J, 0) + 1
            rows.append({
                "setting": setting, "alphas": str(alphas), "seed": seed,
                "method": args.method_name,
                "rmse_missing": r_miss, "rmse_observed": r_obs,
                "runtime_seconds": t,
                "N": args.N, "T": args.T, "blend_alpha": args.alpha,
                "cv_J_chosen": J,
            })
            print(f"[{setting} s{seed}] hybrid_v2 J*={J}  rmse_miss={r_miss:.4f}  "
                  f"({time.time()-t_start:.0f}s)", flush=True)

    print(f"\nJ-choice distribution: {dict(sorted(j_choices.items()))}")

    new_df = pd.DataFrame(rows)
    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        old_df = old_df[old_df["method"] != args.method_name].copy()
        all_cols = list(set(old_df.columns) | set(new_df.columns))
        old_df = old_df.reindex(columns=all_cols)
        new_df = new_df.reindex(columns=all_cols)
        out_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out_df = new_df
    out_df.to_csv(csv_path, index=False)
    print(f"Wrote {len(rows)} {args.method_name} rows to {csv_path}")


if __name__ == "__main__":
    main()
