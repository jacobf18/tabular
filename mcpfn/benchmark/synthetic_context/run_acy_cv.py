"""ACY with cross-validated sieve dimension J.

For each (setting, seed):
  1. Generate the matrix; identify observed and missing cells.
  2. Hold out a random fraction of observed cells as val.
  3. For each candidate J in J_GRID:
       - Treat val cells as missing (add to mask).
       - Fit ACY (mean-fill + EM SoftImpute, sieve basis of degree implied by J).
       - Score: MSE on val cells.
  4. Pick J* with lowest val MSE.
  5. Refit ACY with J* using ALL observed cells (no held-out val).
  6. Compute RMSE on actual missing cells.

This puts ACY on a level epistemic field with MatFormer / boost on the basis
question — no oracle knowledge of the polynomial degree.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/root/tabular/mcpfn/benchmark/synthetic_context')

from generate import ALPHA_SETTINGS, make_draw, observed_matrix  # noqa: E402
from acy_estimator import estimate_acy  # noqa: E402


# Sweep J = 1 + 4*degree for degrees {1, 2, 3, 4, 5} = J values {5, 9, 13, 17, 21}.
# Plus degree=0 (intercept only) and a slightly oversmoothed degree=6 (J=25).
DEGREE_GRID = [1, 2, 3, 4, 5, 6]


def rmse(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--settings", nargs="+", default=list(ALPHA_SETTINGS.keys()))
    ap.add_argument("--missingness",
                    choices=["block", "mcar", "mar", "mnar"], default="block")
    ap.add_argument("--missing_prob", type=float, default=0.4)
    ap.add_argument("--noise_std", type=float, default=0.5)
    ap.add_argument("--penalty_mult", type=float, default=1.0)
    ap.add_argument("--em_iters", type=int, default=50)
    ap.add_argument("--val_frac", type=float, default=0.1,
                    help="Fraction of observed cells to hold out for J selection.")
    ap.add_argument("--method_name", type=str, default="acy_cv")
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/root/tabular/mcpfn/benchmark/synthetic_context/results"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "per_draw_metrics.csv"

    print(f"ACY-CV: J_grid={[1+4*d for d in DEGREE_GRID]}, "
          f"val_frac={args.val_frac}, em_iters={args.em_iters}, "
          f"noise_std={args.noise_std}, missingness={args.missingness}",
          flush=True)

    rows = []
    t_start = time.time()
    j_choices: dict[int, int] = {j: 0 for j in [1 + 4*d for d in DEGREE_GRID]}

    for setting in args.settings:
        if setting not in ALPHA_SETTINGS:
            raise SystemExit(f"unknown setting: {setting}")
        alphas = ALPHA_SETTINGS[setting]
        for seed in range(args.n_seeds):
            draw = make_draw(N=args.N, T=args.T, alphas=alphas, seed=seed,
                             missingness=args.missingness,
                             missing_prob=args.missing_prob)
            mask_missing = draw.mask
            M_true = draw.M
            Y = draw.Y.copy()

            # Build val set: pick a fraction of observed cells.
            obs_rows, obs_cols = np.where(~mask_missing)
            n_obs = obs_rows.size
            n_val = max(10, int(round(args.val_frac * n_obs)))
            cv_rng = np.random.default_rng(seed + 1000)
            perm = cv_rng.permutation(n_obs)
            val_sel = perm[:n_val]
            val_r = obs_rows[val_sel]
            val_c = obs_cols[val_sel]

            # Train-time mask: original missing OR held-out val.
            mask_train = mask_missing.copy()
            mask_train[val_r, val_c] = True

            best_J = None
            best_mse = float("inf")
            for degree in DEGREE_GRID:
                Mhat_cv = estimate_acy(
                    Y, mask_train, draw.X_row, draw.X_col,
                    noise_std=args.noise_std,
                    penalty_mult=args.penalty_mult,
                    missingness_mode="mean_fill",
                    em_iters=args.em_iters,
                    poly_degree=degree,
                )
                mse = float(np.mean((Mhat_cv[val_r, val_c] - Y[val_r, val_c]) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_J = 1 + 4 * degree
                    best_degree = degree

            # Final refit using all observed cells with best degree.
            Mhat = estimate_acy(
                Y, mask_missing, draw.X_row, draw.X_col,
                noise_std=args.noise_std,
                penalty_mult=args.penalty_mult,
                missingness_mode="mean_fill",
                em_iters=args.em_iters,
                poly_degree=best_degree,
            )
            r_miss = rmse(Mhat[mask_missing], M_true[mask_missing])
            r_obs = rmse(Mhat[~mask_missing], M_true[~mask_missing])

            j_choices[best_J] += 1
            rows.append({
                "setting": setting, "alphas": str(alphas), "seed": seed,
                "method": args.method_name,
                "rmse_missing": r_miss, "rmse_observed": r_obs,
                "runtime_seconds": 0.0,
                "N": args.N, "T": args.T, "blend_alpha": np.nan,
                "cv_J_chosen": best_J,
            })
            print(f"[{setting} s{seed}] acy_cv J*={best_J} "
                  f"val_mse={best_mse:.4f}  rmse_miss={r_miss:.4f}  "
                  f"({time.time()-t_start:.0f}s)", flush=True)

    print(f"\nJ-choice distribution: {j_choices}")

    new_df = pd.DataFrame(rows)
    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        old_df = old_df[old_df["method"] != args.method_name].copy()
        for col in old_df.columns:
            if col not in new_df.columns:
                new_df[col] = np.nan
        for col in new_df.columns:
            if col not in old_df.columns:
                old_df[col] = np.nan
        # Use union of columns
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
