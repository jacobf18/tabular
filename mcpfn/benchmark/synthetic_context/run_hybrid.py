"""Hybrid estimator: ACY M_1 sieve projection + boost (MatFormer + xRFM-residual)
on the residual.

Variant A from the design discussion. Pipeline per matrix:
  1.  (optional CV) Pick polynomial degree d for sieve basis by holding out a
      fraction of observed cells, scoring M̂_1 = P_X Y_filled P_Z at val cells.
  2.  Compute M̂_1 = P_X Y_filled P_Z using all observed cells.
  3.  Y_resid = Y - M̂_1, with NaN preserved at originally missing cells.
  4.  M̂_resid = BoostMatFormerImputer.impute(Y_resid).
  5.  Final estimate M̂ = M̂_1 + M̂_resid.

Rationale: ACY's sieve projection cleanly extracts the smooth-in-(x_r, z_c)
component without invoking matrix-completion machinery. Whatever's left is
exactly what MatFormer (pretrained on low-rank prior) + xRFM-residual is
well-suited for. Two complementary mechanisms, stacked.
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

from generate import ALPHA_SETTINGS, make_draw, observed_matrix  # noqa: E402
from acy_estimator import _poly_features, _projector, _mean_fill, estimate_acy  # noqa: E402
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
    val_frac: float, seed: int, em_iters: int, noise_std: float,
) -> int:
    """Hold out val_frac of observed cells; pick polynomial degree that
    minimizes FULL ACY estimator's RMSE on those val cells. Runs the same
    sieve+SVT+EM pipeline as the final fit (just over each degree in
    DEGREE_GRID), and picks the degree with lowest val MSE."""
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
    cv_em_iters = min(20, em_iters)   # Cap CV EM iters — discriminating J
    # doesn't need full convergence, just enough to compare degrees.
    for deg in DEGREE_GRID:
        Mhat_cv = estimate_acy(
            Y, mask_train, X_row, X_col,
            noise_std=noise_std, penalty_mult=1.0,
            missingness_mode="mean_fill", em_iters=cv_em_iters,
            poly_degree=deg,
        )
        mse = float(np.mean((Mhat_cv[val_r, val_c] - Y[val_r, val_c]) ** 2))
        if mse < best_mse:
            best_mse, best_deg = mse, deg
    return best_deg, best_mse


def hybrid_one(
    draw, imputer: BoostMatFormerImputer, val_frac: float, seed: int,
    em_iters: int, noise_std: float,
) -> tuple[np.ndarray, int]:
    Y = draw.Y.astype(np.float64)
    mask_missing = draw.mask
    X_row = draw.X_row.astype(np.float64)
    X_col = draw.X_col.astype(np.float64)

    # 1. CV-select polynomial degree, scoring via full ACY (M_1+M_2+M_3+M_4+EM).
    best_deg, _ = cv_select_degree(
        Y, mask_missing, X_row, X_col, val_frac, seed, em_iters, noise_std,
    )

    # 2. Full ACY estimate using all observed cells with the chosen degree.
    M_hat_acy = estimate_acy(
        Y, mask_missing, X_row, X_col,
        noise_std=noise_std, penalty_mult=1.0,
        missingness_mode="mean_fill", em_iters=em_iters,
        poly_degree=best_deg,
    ).astype(np.float64)

    # 3. Residual matrix: NaN at originally missing cells, value-residual elsewhere.
    Y_resid = np.where(mask_missing, np.nan, Y - M_hat_acy).astype(np.float32)

    # 4. Boost on residual.
    with torch.no_grad():
        M_hat_resid = imputer.impute(Y_resid)
    M_hat_resid = np.asarray(M_hat_resid, dtype=np.float64)

    # 5. Combine.
    M_hat = M_hat_acy + M_hat_resid
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
    ap.add_argument("--em_iters", type=int, default=50)
    ap.add_argument("--noise_std", type=float, default=0.5)
    ap.add_argument("--method_name", type=str, default="hybrid")
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/root/tabular/mcpfn/benchmark/synthetic_context/results"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "per_draw_metrics.csv"

    print(f"Loading BoostMatFormerImputer (α={args.alpha}) ...", flush=True)
    imputer = BoostMatFormerImputer(
        device='cuda',
        entry_wise_features=False,
        json_config=ROUND2_T2_CONFIG,
        checkpoint_path=CHECKPOINT_PATH,
        max_num_rows=args.N, max_num_cols=args.T,
        blend_alpha=args.alpha,
    )

    rows = []
    t_start = time.time()
    j_choices: dict[int, int] = {}

    for setting in args.settings:
        if setting not in ALPHA_SETTINGS:
            raise SystemExit(f"unknown setting: {setting}")
        alphas = ALPHA_SETTINGS[setting]
        for seed in range(args.n_seeds):
            draw = make_draw(N=args.N, T=args.T, alphas=alphas, seed=seed,
                             missingness=args.missingness,
                             missing_prob=args.missing_prob)
            M_true = draw.M
            mask_missing = draw.mask

            t0 = time.time()
            M_hat, J = hybrid_one(draw, imputer, args.val_frac, seed,
                                  args.em_iters, args.noise_std)
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
            print(f"[{setting} s{seed}] hybrid J*={J}  rmse_miss={r_miss:.4f}  "
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
