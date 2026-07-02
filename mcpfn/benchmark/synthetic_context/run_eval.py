"""Run methods on the synthetic-context benchmark.

For each (alpha-setting, seed):
  - Generate a draw (N x T with bottom-right block missing).
  - Run methods:
       matformer       : MatFormer alone (return_full=True predictions)
       boost_no_context    : MatFormer + xRFM residual, no context features
       boost_row_context   : MatFormer + xRFM residual, with row context (x_i) injected
       boost_row_col_context : both row and col context (col is constant per-column-fit,
                           so this mostly tests whether it hurts or no-ops)

Writes per_draw_metrics.csv with one row per (setting, seed, method).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/root/tabular/mcpfn/src')
sys.path.insert(0, '/root/tabular/mcpfn/benchmark/synthetic_context')

from generate import ALPHA_SETTINGS, make_draw, observed_matrix  # noqa: E402
from context_only import predict_context_only  # noqa: E402
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


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_matformer(imputer: BoostMatFormerImputer, X: np.ndarray) -> np.ndarray:
    """Return MatFormer's imputed matrix (observed cells preserved, missing
    cells filled with MLP-decoder predictions). Bypasses the xRFM correction."""
    import torch
    from tabimpute.interface import ImputePFN
    with torch.no_grad():
        X_imputed_mlp, _X_full = ImputePFN.impute(imputer, X, return_full=True)
    return np.asarray(X_imputed_mlp, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--settings", nargs="+", default=list(ALPHA_SETTINGS.keys()))
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="Boost blend coefficient.")
    ap.add_argument("--max_rows", type=int, default=200)
    ap.add_argument("--max_cols", type=int, default=200)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/root/tabular/mcpfn/benchmark/synthetic_context/results"))
    ap.add_argument("--limit_seeds", type=int, default=0,
                    help="If >0, only run this many seeds per setting.")
    ap.add_argument("--missingness",
                    choices=["block", "mcar", "mar", "mnar"], default="block")
    ap.add_argument("--missing_prob", type=float, default=0.4,
                    help="Per-cell missingness probability when --missingness=mcar.")
    ap.add_argument("--context_scale", type=float, default=1.0,
                    help="Upweight applied to (standardized) context features inside "
                         "the boost's xRFM. 1.0 = drown-out default. ~10 puts context "
                         "and data on equal kernel-distance footing for C=200, d_ctx=4.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "per_draw_metrics.csv"

    print(f"Loading BoostMatFormerImputer (α={args.alpha}) ...", flush=True)
    imputer = BoostMatFormerImputer(
        device='cuda',
        entry_wise_features=False,
        json_config=ROUND2_T2_CONFIG,
        checkpoint_path=CHECKPOINT_PATH,
        max_num_rows=args.max_rows,
        max_num_cols=args.max_cols,
        blend_alpha=args.alpha,
        context_scale=args.context_scale,
    )

    rows = []
    t_start = time.time()
    n_seeds = args.limit_seeds if args.limit_seeds > 0 else args.n_seeds

    for setting in args.settings:
        if setting not in ALPHA_SETTINGS:
            raise SystemExit(f"unknown setting: {setting}")
        alphas = ALPHA_SETTINGS[setting]

        for seed in range(n_seeds):
            draw = make_draw(
                N=args.N, T=args.T, alphas=alphas, seed=seed,
                missingness=args.missingness, missing_prob=args.missing_prob,
            )
            Xobs = observed_matrix(draw)
            M_true = draw.M
            mask_missing = draw.mask  # bool
            mask_observed = ~mask_missing

            # --- MatFormer alone ---
            t0 = time.time()
            X_mf = run_matformer(imputer, Xobs)
            t_mf = time.time() - t0
            rmse_mf_miss = rmse(X_mf[mask_missing], M_true[mask_missing])
            rmse_mf_obs = rmse(X_mf[mask_observed], M_true[mask_observed])

            # --- Boost no context ---
            t0 = time.time()
            X_b0 = imputer.impute(Xobs)
            t_b0 = time.time() - t0
            rmse_b0_miss = rmse(X_b0[mask_missing], M_true[mask_missing])
            rmse_b0_obs = rmse(X_b0[mask_observed], M_true[mask_observed])

            # --- Boost with row context ---
            t0 = time.time()
            X_b1 = imputer.impute(Xobs, row_context=draw.X_row)
            t_b1 = time.time() - t0
            rmse_b1_miss = rmse(X_b1[mask_missing], M_true[mask_missing])
            rmse_b1_obs = rmse(X_b1[mask_observed], M_true[mask_observed])

            # --- Boost with row + col context ---
            t0 = time.time()
            X_b2 = imputer.impute(Xobs, row_context=draw.X_row, col_context=draw.X_col)
            t_b2 = time.time() - t0
            rmse_b2_miss = rmse(X_b2[mask_missing], M_true[mask_missing])
            rmse_b2_obs = rmse(X_b2[mask_observed], M_true[mask_observed])

            # --- Context-only xRFM (no MatFormer, just (X_row[r], X_col[c]) -> Y[r,c]) ---
            t0 = time.time()
            X_co = predict_context_only(
                Y=draw.Y,
                mask_missing=mask_missing,
                X_row=draw.X_row,
                X_col=draw.X_col,
                seed=seed,
            )
            t_co = time.time() - t0
            rmse_co_miss = rmse(X_co[mask_missing], M_true[mask_missing])
            rmse_co_obs = rmse(X_co[mask_observed], M_true[mask_observed])

            for method, rmse_miss, rmse_obs, runtime in [
                ("matformer",          rmse_mf_miss, rmse_mf_obs, t_mf),
                ("boost_no_context",       rmse_b0_miss, rmse_b0_obs, t_b0),
                ("boost_row_context",      rmse_b1_miss, rmse_b1_obs, t_b1),
                ("boost_row_col_context",  rmse_b2_miss, rmse_b2_obs, t_b2),
                ("context_only",       rmse_co_miss, rmse_co_obs, t_co),
            ]:
                rows.append({
                    "setting": setting,
                    "alphas": str(alphas),
                    "seed": seed,
                    "method": method,
                    "rmse_missing": rmse_miss,
                    "rmse_observed": rmse_obs,
                    "runtime_seconds": runtime,
                    "N": args.N, "T": args.T,
                    "blend_alpha": args.alpha,
                })

            print(f"[{setting} s{seed}] mf={rmse_mf_miss:.4f}  b0={rmse_b0_miss:.4f}  "
                  f"b1={rmse_b1_miss:.4f}  b2={rmse_b2_miss:.4f}  "
                  f"co={rmse_co_miss:.4f}  "
                  f"({time.time()-t_start:.0f}s)", flush=True)

            # Flush after every draw so partial runs are recoverable.
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nWrote {csv_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
