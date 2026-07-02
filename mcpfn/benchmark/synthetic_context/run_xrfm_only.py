"""Add pure xRFM (no MatFormer) baseline rows to existing per_draw_metrics.csv.

Runs XRFMImputer (same column-wise xRFM as the UCI bar in xrfm_wrapper.py) on
the same (setting, seed) draws as run_eval.py uses, for either block or mcar
missingness, then appends 'xrfm' rows to the existing CSV.

Usage:
  python run_xrfm_only.py --missingness block
  python run_xrfm_only.py --missingness mcar --missing_prob 0.4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/root/tabular/mcpfn/benchmark')
sys.path.insert(0, '/root/tabular/mcpfn/benchmark/synthetic_context')

from generate import ALPHA_SETTINGS, make_draw, observed_matrix  # noqa: E402
from xrfm_wrapper import XRFMImputer  # noqa: E402


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
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/root/tabular/mcpfn/benchmark/synthetic_context/results"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "per_draw_metrics.csv"

    print(f"Loading XRFMImputer (column-wise xRFM, no MatFormer) ...", flush=True)
    imputer = XRFMImputer(device='cuda', val_fraction=0.2, random_state=0,
                          verbose=False)

    rows = []
    t_start = time.time()

    for setting in args.settings:
        if setting not in ALPHA_SETTINGS:
            raise SystemExit(f"unknown setting: {setting}")
        alphas = ALPHA_SETTINGS[setting]
        for seed in range(args.n_seeds):
            draw = make_draw(N=args.N, T=args.T, alphas=alphas, seed=seed,
                             missingness=args.missingness,
                             missing_prob=args.missing_prob)
            Xobs = observed_matrix(draw)
            M_true = draw.M
            mask_missing = draw.mask
            mask_observed = ~mask_missing

            t0 = time.time()
            X_xrfm = imputer.fit_transform(Xobs)
            t = time.time() - t0
            r_miss = rmse(X_xrfm[mask_missing], M_true[mask_missing])
            r_obs = rmse(X_xrfm[mask_observed], M_true[mask_observed])
            rows.append({
                "setting": setting, "alphas": str(alphas), "seed": seed,
                "method": "xrfm",
                "rmse_missing": r_miss, "rmse_observed": r_obs,
                "runtime_seconds": t,
                "N": args.N, "T": args.T, "blend_alpha": np.nan,
            })
            print(f"[{setting} s{seed}] xrfm rmse_miss={r_miss:.4f}  "
                  f"({time.time()-t_start:.0f}s)", flush=True)

    new_df = pd.DataFrame(rows)
    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        old_df = old_df[old_df["method"] != "xrfm"].copy()
        # Align columns
        for col in old_df.columns:
            if col not in new_df.columns:
                new_df[col] = np.nan
        new_df = new_df[old_df.columns]
        out_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out_df = new_df
    out_df.to_csv(csv_path, index=False)
    print(f"\nWrote {len(rows)} xrfm rows to {csv_path}")


if __name__ == "__main__":
    main()
