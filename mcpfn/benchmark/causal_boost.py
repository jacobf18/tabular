"""Apply MatFormer + xRFM-residual boost to causal benchmarks (CPS, Kansas, Prop99).

Per draw of each benchmark:
  1. Load observed.npy (matrix with NaN at treated cells) and the pre-computed
     MatFormer baseline (canonical dwip-v2 .npy per benchmark).
  2. Compute residuals at observed cells: r_obs = X_obs - matformer_obs (in
     standardized space).
  3. For each column with NaN, fit xRFM column-wise on residuals (same
     machinery as boost_matformer in interface_rfm.py) and predict residual
     at treated cells.
  4. Combine: y_final[treated] = matformer[treated] + α · r_predicted[treated]
     (α = 0.1, in standardized space) and de-standardize.
  5. Save boost_matformer_a0p1.npy in the same draw directory.

Mirrors the UCI boost result (xRFM residual on raw features, α=0.1).

Usage:
    python causal_boost.py --benchmark CPS
    python causal_boost.py --benchmark kansas
    python causal_boost.py --benchmark prop99
    python causal_boost.py --benchmark all
"""
from __future__ import annotations

import argparse
import contextlib
import io
import time
import warnings
from pathlib import Path

import numpy as np

# Default xRFM hyperparameters — match what was used for UCI boost.
_DEFAULT_RFM_PARAMS = {
    "model": {
        "kernel": "l2_high_dim",
        "exponent": 1.0,
        "bandwidth": 10.0,
        "diag": True,
        "bandwidth_mode": "constant",
    },
    "fit": {
        "reg": 1e-3,
        "iters": 3,
        "early_stop_rfm": True,
        "verbose": False,
    },
}

# (benchmark_name → canonical MatFormer .npy basename) mapping.
MATFORMER_BASE = {
    "CPS": "dwip-v2-2",
    "kansas": "dwip-v2-1",
    "prop99": "dwip-v2-1",
}

BENCHMARK_DIR = Path("/root/tabular/mcpfn/benchmark")


def _mean_fill(X: np.ndarray) -> np.ndarray:
    out = X.copy().astype(np.float32)
    col_means = np.nanmean(out, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    inds = np.where(np.isnan(out))
    out[inds] = np.take(col_means, inds[1])
    return out


def _build_features(X_filled, mask, target_col, d):
    feats = np.concatenate([X_filled, mask.astype(np.float32)], axis=1)
    keep = np.ones(2 * d, dtype=bool)
    keep[target_col] = False
    keep[target_col + d] = False
    return feats[:, keep]


def _fit_predict_residual(features_obs, residual_obs, features_query, rng,
                          val_fraction=0.2, random_state=0, device="cuda"):
    """Fit xRFM on (features, residuals) at observed cells, predict at query."""
    from xrfm import xRFM
    n_obs = features_obs.shape[0]
    n_q = features_query.shape[0]
    if n_obs < 4 or n_q == 0:
        return np.zeros(n_q, dtype=np.float32)
    n_val = max(1, int(round(n_obs * val_fraction)))
    n_val = min(n_val, n_obs - 1)
    idx = rng.permutation(n_obs)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    X_tr = features_obs[tr_idx].astype(np.float32, copy=False)
    y_tr = residual_obs[tr_idx].astype(np.float32, copy=False)
    X_va = features_obs[val_idx].astype(np.float32, copy=False)
    y_va = residual_obs[val_idx].astype(np.float32, copy=False)
    X_q = features_query.astype(np.float32, copy=False)

    stdout_ctx = contextlib.redirect_stdout(io.StringIO())
    stderr_ctx = contextlib.redirect_stderr(io.StringIO())
    with stdout_ctx, stderr_ctx, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = xRFM(
            rfm_params=_DEFAULT_RFM_PARAMS,
            max_leaf_size=60_000, refill_size=1500,
            device=device, tuning_metric="mse", random_state=random_state,
        )
        try:
            model.fit(X_tr, y_tr, X_va, y_va)
            pred = model.predict(X_q)
        except Exception:
            return np.zeros(n_q, dtype=np.float32)
    pred = np.asarray(pred).reshape(-1).astype(np.float32)
    if pred.shape[0] != n_q:
        return np.zeros(n_q, dtype=np.float32)
    return pred


def boost_one_draw(draw_dir: Path, matformer_basename: str, alpha: float,
                   out_basename: str, force: bool, device: str) -> str:
    """Compute boost prediction for one draw. Returns status string."""
    observed_p = draw_dir / "observed.npy"
    matformer_p = draw_dir / f"{matformer_basename}.npy"
    out_p = draw_dir / f"{out_basename}.npy"

    if not observed_p.is_file():
        return "skip: no observed.npy"
    if not matformer_p.is_file():
        return f"skip: no {matformer_basename}.npy"
    if out_p.exists() and not force:
        return "skip: exists"

    X = np.load(observed_p).astype(np.float64)                 # (R, C) with NaN at treated
    matformer = np.load(matformer_p).astype(np.float64)         # (R, C) full predictions
    if X.shape != matformer.shape:
        return f"shape mismatch ({X.shape} vs {matformer.shape})"
    R, C = X.shape

    # Standardize per column.
    col_mean = np.nanmean(X, axis=0)
    col_std = np.nanstd(X, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    col_std = np.where(np.isnan(col_std), 1.0, col_std)
    col_std = np.where(col_std > 1e-12, col_std, 1.0)
    X_std = (X - col_mean) / col_std
    y_mlp_std = (matformer - col_mean) / col_std

    # Standardized residuals at observed cells (NaN at treated).
    residual_std = X_std - y_mlp_std

    # Features for xRFM (mean-fill + mask of OTHER columns).
    rng = np.random.default_rng(0)
    mask = np.isnan(X_std).astype(np.float32)
    X_filled = _mean_fill(X_std)

    r_pred_full = np.zeros((R, C), dtype=np.float32)
    cols_with_nan = [c for c in range(C) if mask[:, c].any()]
    for c in cols_with_nan:
        obs_rows_c = mask[:, c] == 0.0
        miss_rows_c = ~obs_rows_c
        if not miss_rows_c.any() or obs_rows_c.sum() < 4:
            continue
        features = _build_features(X_filled, mask, c, C)
        r_pred = _fit_predict_residual(
            features[obs_rows_c],
            residual_std[obs_rows_c, c].astype(np.float32),
            features[miss_rows_c],
            rng, device=device,
        )
        r_pred_full[miss_rows_c, c] = r_pred

    # Combine in standardized space, de-standardize.
    X_std_final = X_std.copy()
    miss_mask = np.isnan(X_std)
    X_std_final[miss_mask] = (y_mlp_std + alpha * r_pred_full)[miss_mask]
    X_out = X_std_final * col_std + col_mean

    np.save(out_p, X_out.astype(np.float32))
    return "ok"


def find_draws(benchmark: str) -> list[Path]:
    base = BENCHMARK_DIR / benchmark / "causal_eval"
    draws = sorted([p for p in base.rglob("draw_*") if p.is_dir()])
    return draws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["CPS", "kansas", "prop99", "all"],
                    default="all")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out_basename", type=str, default="boost_matformer_a0p1")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N draws per benchmark (0 = no limit).")
    args = ap.parse_args()

    benchmarks = ["CPS", "kansas", "prop99"] if args.benchmark == "all" else [args.benchmark]

    total_ok = 0
    total_skip = 0
    total_err = 0
    for bm in benchmarks:
        draws = find_draws(bm)
        if args.limit > 0:
            draws = draws[: args.limit]
        matformer = MATFORMER_BASE[bm]
        print(f"\n[{bm}] {len(draws)} draws  MatFormer base: {matformer}.npy",
              flush=True)
        t0 = time.time()
        ok = 0; sk = 0; er = 0
        for i, draw in enumerate(draws, 1):
            status = boost_one_draw(
                draw, matformer, args.alpha, args.out_basename,
                args.force, args.device,
            )
            if status == "ok":
                ok += 1
            elif status.startswith("skip"):
                sk += 1
            else:
                er += 1
                print(f"  [{i}/{len(draws)}] {draw.name}: {status}", flush=True)
            if i % 50 == 0:
                dt = time.time() - t0
                rate = i / dt if dt > 0 else 0
                eta = (len(draws) - i) / rate if rate > 0 else 0
                print(f"  [{bm}] {i}/{len(draws)} ok={ok} skip={sk} err={er}  "
                      f"rate={rate:.1f}/s  eta={eta/60:.1f}min", flush=True)
        dt = time.time() - t0
        print(f"[{bm}] DONE in {dt/60:.1f} min  ok={ok} skip={sk} err={er}",
              flush=True)
        total_ok += ok; total_skip += sk; total_err += er

    print(f"\n[total] ok={total_ok}  skip={total_skip}  err={total_err}")


if __name__ == "__main__":
    main()
