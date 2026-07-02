"""Hidden-context variant: x_r-driven signal exists ONLY in the missing block.

This is the regime where AGOP cannot recover x_r from raw features because
no observed cell depends on x_r. Tests whether explicit context-feature
injection genuinely helps when it has to.

Same 4 methods as run_eval.py, but on `make_hidden_context_draw` draws.
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

from generate import TREATMENT_SETTINGS, make_hidden_context_draw, observed_matrix  # noqa: E402
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
    import torch
    from tabimpute.interface import ImputePFN
    with torch.no_grad():
        X_imputed_mlp, _ = ImputePFN.impute(imputer, X, return_full=True)
    return np.asarray(X_imputed_mlp, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--settings", nargs="+", default=list(TREATMENT_SETTINGS.keys()))
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Boost blend coefficient. Use 1.0 (no damping) for "
                         "clean isolation of the context-injection effect.")
    ap.add_argument("--max_rows", type=int, default=200)
    ap.add_argument("--max_cols", type=int, default=200)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/root/tabular/mcpfn/benchmark/synthetic_context/results_hidden"))
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
    )

    rows = []
    t_start = time.time()

    for setting in args.settings:
        if setting not in TREATMENT_SETTINGS:
            raise SystemExit(f"unknown setting: {setting}")
        ts = TREATMENT_SETTINGS[setting]

        for seed in range(args.n_seeds):
            draw = make_hidden_context_draw(
                N=args.N, T=args.T, treatment_strength=ts, seed=seed,
            )
            Xobs = observed_matrix(draw)
            M_true = draw.M
            mask_missing = draw.mask

            t0 = time.time()
            X_mf = run_matformer(imputer, Xobs)
            t_mf = time.time() - t0
            rmse_mf = rmse(X_mf[mask_missing], M_true[mask_missing])

            t0 = time.time()
            X_b0 = imputer.impute(Xobs)
            t_b0 = time.time() - t0
            rmse_b0 = rmse(X_b0[mask_missing], M_true[mask_missing])

            t0 = time.time()
            X_b1 = imputer.impute(Xobs, row_context=draw.X_row)
            t_b1 = time.time() - t0
            rmse_b1 = rmse(X_b1[mask_missing], M_true[mask_missing])

            t0 = time.time()
            X_b2 = imputer.impute(Xobs, row_context=draw.X_row, col_context=draw.X_col)
            t_b2 = time.time() - t0
            rmse_b2 = rmse(X_b2[mask_missing], M_true[mask_missing])

            t0 = time.time()
            X_co = predict_context_only(
                Y=draw.Y,
                mask_missing=mask_missing,
                X_row=draw.X_row,
                X_col=draw.X_col,
                seed=seed,
            )
            t_co = time.time() - t0
            rmse_co = rmse(X_co[mask_missing], M_true[mask_missing])

            for method, rm, rt in [
                ("matformer",             rmse_mf, t_mf),
                ("boost_no_context",      rmse_b0, t_b0),
                ("boost_row_context",     rmse_b1, t_b1),
                ("boost_row_col_context", rmse_b2, t_b2),
                ("context_only",          rmse_co, t_co),
            ]:
                rows.append({
                    "setting": setting, "treatment_strength": ts, "seed": seed,
                    "method": method, "rmse_missing": rm, "runtime_seconds": rt,
                    "N": args.N, "T": args.T, "blend_alpha": args.alpha,
                })

            print(f"[{setting} ts={ts} s{seed}] mf={rmse_mf:.4f}  "
                  f"b0={rmse_b0:.4f}  b1={rmse_b1:.4f}  b2={rmse_b2:.4f}  "
                  f"co={rmse_co:.4f}  "
                  f"({time.time()-t_start:.0f}s)", flush=True)

            pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nWrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
