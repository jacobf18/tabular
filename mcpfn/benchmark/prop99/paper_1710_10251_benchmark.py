#!/usr/bin/env python3
from __future__ import annotations

"""
Benchmark generator and evaluation harness for the masking schemes in
Athey et al. (2017/2022), "Matrix Completion Methods for Causal Panel Data Models"
(arXiv:1710.10251).

What this script reproduces faithfully from the paper:
- California smoking donor-pool benchmark:
  * use the control units only (drop California)
  * create pseudo-treatment under either
      1) simultaneous adoption: Nt random units treated from T0+1 onward
      2) staggered adoption: Nt random units treated from a random date after T0
  * evaluate by RMSE on the hidden entries
- Stock-market benchmark:
  * for each (N, T) with N*T fixed, take the first T periods and N random units
  * hide the second half of periods for a random half of units (25% missing overall)
  * evaluate by average RMSE on the hidden entries
- MC-NNM style tuning rule:
  * choose lambda by the paper's cross-validation recipe using random subsets Ok of O
  * report average RMSE and 95% normal-approximation intervals

What this script does NOT fully reproduce:
- the paper's exact stock universe (2453 stocks over 3082 trading days), unless you
  provide a matching returns matrix
- all baseline estimators from the paper (DID, VT-EN, HR-EN, SC-ADH)
  The script includes a practical MC-NNM implementation and two simple baselines
  (two-way fixed effects and column mean) so you can run the benchmark end-to-end.

Examples
--------
# 1) Build California-smoking pseudo-treatment draws
python paper_1710_10251_benchmark.py build-california \
  --output-dir ./ca_smoking_simul \
  --design simultaneous \
  --nt 8 \
  --t0-ratios 0.35 0.5 0.65 0.8 0.9 \
  --reps 100 \
  --seed 1

# 2) Evaluate MC-NNM on those draws
python paper_1710_10251_benchmark.py evaluate \
  --input-dir ./ca_smoking_simul \
  --method mcnnm \
  --output-csv ./ca_smoking_simul/results_mcnnm.csv

# 3) Build stock-style draws from a user-provided returns matrix CSV
#    (rows = units/stocks, columns = time periods)
python paper_1710_10251_benchmark.py build-stocks \
  --stock-csv ./returns_matrix.csv \
  --output-dir ./stock_benchmark \
  --pairs 490,10 245,20 140,35 98,50 70,70 50,98 35,140 20,245 10,490 \
  --reps 50 \
  --seed 1

# 4) If you do not have a stock matrix, generate a synthetic factor model matrix and
#    apply the paper's stock-style missingness pattern
python paper_1710_10251_benchmark.py build-stocks \
  --output-dir ./stock_synth \
  --synthetic-stock-units 3000 \
  --synthetic-stock-periods 3082 \
  --pairs 70,70 35,140 20,245 \
  --reps 20 \
  --seed 1
"""

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

SMOKING_RDATASET_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/Synth/smoking.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ca = sub.add_parser("build-california", help="Build California-smoking pseudo-treatment draws.")
    p_ca.add_argument("--output-dir", type=Path, required=True)
    p_ca.add_argument("--smoking-csv", type=Path, default=None,
                      help="Optional local path to the smoking dataset. If omitted, uses the Rdatasets mirror.")
    p_ca.add_argument("--outcome-col", type=str, default="cigsale")
    p_ca.add_argument("--design", choices=["simultaneous", "staggered"], required=True)
    p_ca.add_argument("--nt", type=int, required=True,
                      help="Number of treated units per draw. Paper examples: 8 for simultaneous, 35 for staggered.")
    p_ca.add_argument("--t0-ratios", type=float, nargs="+", required=True,
                      help="Pretreatment ratios T0/T. Example: 0.35 0.5 0.65 0.8 0.9")
    p_ca.add_argument("--reps", type=int, default=100)
    p_ca.add_argument("--seed", type=int, default=1)
    p_ca.add_argument("--force", action="store_true")

    p_st = sub.add_parser("build-stocks", help="Build stock-style block-missingness draws.")
    p_st.add_argument("--output-dir", type=Path, required=True)
    p_st.add_argument("--stock-csv", type=Path, default=None,
                      help="Optional CSV containing a returns matrix. Rows should be units, columns periods by default.")
    p_st.add_argument("--orient", choices=["rows_are_units", "cols_are_units"], default="rows_are_units")
    p_st.add_argument("--pairs", type=str, nargs="+", required=True,
                      help="Pairs N,T written as N,T. Example: 490,10 245,20 70,70 20,245 10,490")
    p_st.add_argument("--reps", type=int, default=50)
    p_st.add_argument("--seed", type=int, default=1)
    p_st.add_argument("--force", action="store_true")
    p_st.add_argument("--synthetic-stock-units", type=int, default=None,
                      help="If --stock-csv is omitted, generate a synthetic stock-like factor matrix with this many units.")
    p_st.add_argument("--synthetic-stock-periods", type=int, default=None,
                      help="If --stock-csv is omitted, generate a synthetic stock-like factor matrix with this many periods.")
    p_st.add_argument("--synthetic-rank", type=int, default=6)
    p_st.add_argument("--synthetic-noise", type=float, default=0.05)

    p_ev = sub.add_parser("evaluate", help="Evaluate imputers on a benchmark directory.")
    p_ev.add_argument("--input-dir", type=Path, required=True,
                      help="Directory produced by build-california or build-stocks.")
    p_ev.add_argument("--method", choices=["mcnnm", "two_way_fe", "column_mean"], required=True)
    p_ev.add_argument("--output-csv", type=Path, default=None)
    p_ev.add_argument("--predictions-dir", type=Path, default=None,
                      help="Optional directory to save per-draw predictions as .npy files.")
    p_ev.add_argument("--seed", type=int, default=1)
    p_ev.add_argument("--cv-folds", type=int, default=5)
    p_ev.add_argument("--lambda-grid-size", type=int, default=10)
    p_ev.add_argument("--max-iters", type=int, default=200)
    p_ev.add_argument("--tol", type=float, default=1e-5)

    return parser.parse_args()


# ----------------------------
# Data loading and generation
# ----------------------------

def load_smoking_matrix(smoking_csv: Optional[Path], outcome_col: str) -> tuple[np.ndarray, list[str], list[int], pd.DataFrame]:
    if smoking_csv is None:
        df = pd.read_csv(SMOKING_RDATASET_URL)
    else:
        df = pd.read_csv(smoking_csv)

    cols_lower = {c.lower(): c for c in df.columns}
    if outcome_col.lower() not in cols_lower:
        raise ValueError(f"Outcome column '{outcome_col}' not found in columns: {list(df.columns)}")
    outcome_col_real = cols_lower[outcome_col.lower()]

    state_col = cols_lower.get("state") or cols_lower.get("name")
    year_col = cols_lower.get("year")
    if state_col is None or year_col is None:
        raise ValueError("Expected columns like 'state' and 'year' in the smoking dataset.")

    # Drop California so all remaining entries are observed, exactly as in the paper's evaluation.
    state_series = df[state_col].astype(str).str.lower()
    df = df.loc[~state_series.eq("california")].copy()

    # Keep only the outcome needed for the benchmark.
    panel = (
        df.pivot_table(index=state_col, columns=year_col, values=outcome_col_real, aggfunc="mean")
          .sort_index(axis=0)
          .sort_index(axis=1)
    )
    if panel.isna().any().any():
        raise ValueError("Smoking panel has missing values after pivoting; cannot build a fully observed benchmark.")

    y = panel.to_numpy(dtype=np.float64)
    states = list(panel.index.astype(str))
    years = [int(x) for x in panel.columns]
    return y, states, years, df


def coerce_numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    # If the first column is non-numeric, treat it as an index column.
    df2 = df.copy()
    if df2.shape[1] >= 2 and not np.issubdtype(df2.iloc[:, 0].dtype, np.number):
        df2 = df2.iloc[:, 1:]
    # If the first row looks non-numeric, pandas would have used it as header already.
    arr = df2.to_numpy()
    try:
        return arr.astype(np.float64)
    except ValueError as e:
        raise ValueError("Could not coerce the provided CSV into a numeric matrix.") from e


def load_stock_matrix(stock_csv: Path, orient: str) -> np.ndarray:
    df = pd.read_csv(stock_csv)
    mat = coerce_numeric_matrix(df)
    if orient == "cols_are_units":
        mat = mat.T
    if mat.ndim != 2:
        raise ValueError("Stock matrix must be two-dimensional.")
    if np.isnan(mat).any():
        raise ValueError("Stock matrix contains NaNs. The paper's benchmark starts from a fully observed matrix.")
    return mat


def generate_synthetic_stock_matrix(n_units: int, n_periods: int, rank: int, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Low-rank factor model plus idiosyncratic noise, then standardize each unit roughly like returns.
    U = rng.normal(size=(n_units, rank))
    V = rng.normal(size=(rank, n_periods))
    E = noise * rng.normal(size=(n_units, n_periods))
    y = U @ V + E
    y = y - y.mean(axis=1, keepdims=True)
    denom = y.std(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    y = y / denom
    return y


# ----------------------------
# Mask generators
# ----------------------------

def make_simultaneous_mask(n_units: int, n_periods: int, nt: int, t0: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, object]]:
    if not (1 <= t0 < n_periods):
        raise ValueError(f"t0 must satisfy 1 <= t0 < T; got t0={t0}, T={n_periods}")
    if not (1 <= nt <= n_units):
        raise ValueError(f"nt must satisfy 1 <= nt <= N; got nt={nt}, N={n_units}")

    treated = rng.choice(n_units, size=nt, replace=False)
    w = np.zeros((n_units, n_periods), dtype=np.int8)
    w[treated[:, None], np.arange(t0, n_periods)] = 1
    meta = {
        "assignment": "simultaneous",
        "treated_units": treated.tolist(),
        "adoption_time": t0,
    }
    return w, meta


def make_staggered_mask(n_units: int, n_periods: int, nt: int, t0: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, object]]:
    if not (1 <= t0 < n_periods):
        raise ValueError(f"t0 must satisfy 1 <= t0 < T; got t0={t0}, T={n_periods}")
    if not (1 <= nt <= n_units):
        raise ValueError(f"nt must satisfy 1 <= nt <= N; got nt={nt}, N={n_units}")

    treated = rng.choice(n_units, size=nt, replace=False)
    adoption = {}
    w = np.zeros((n_units, n_periods), dtype=np.int8)
    for i in treated:
        adopt = int(rng.integers(low=t0, high=n_periods))
        w[i, adopt:] = 1
        adoption[int(i)] = adopt
    meta = {
        "assignment": "staggered",
        "treated_units": treated.tolist(),
        "adoption_times": adoption,
    }
    return w, meta


def make_stock_block_mask(n_units: int, n_periods: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, object]]:
    # Paper: second half of periods missing for a random half of units.
    n_treated_units = n_units // 2
    start = n_periods // 2
    treated = rng.choice(n_units, size=n_treated_units, replace=False)
    w = np.zeros((n_units, n_periods), dtype=np.int8)
    w[treated[:, None], np.arange(start, n_periods)] = 1
    meta = {
        "assignment": "stock_block_second_half",
        "treated_units": treated.tolist(),
        "post_col_start": start,
    }
    return w, meta


# ----------------------------
# Benchmark writers
# ----------------------------

def ensure_clean_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(f"Output directory already exists: {path}. Use --force to overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_draw_files(base_dir: Path, draw_name: str, y: np.ndarray, w: np.ndarray, metadata: dict[str, object]) -> None:
    pd.DataFrame(y).to_csv(base_dir / f"{draw_name}_Y.csv", index=False)
    pd.DataFrame(w).to_csv(base_dir / f"{draw_name}_W.csv", index=False)
    with (base_dir / f"{draw_name}_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def build_california_benchmark(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    y, states, years, _ = load_smoking_matrix(args.smoking_csv, args.outcome_col)
    n_units, n_periods = y.shape

    ensure_clean_dir(args.output_dir, args.force)
    pd.DataFrame({"state": states}).to_csv(args.output_dir / "row_index.csv", index=False)
    pd.DataFrame({"period": years}).to_csv(args.output_dir / "col_index.csv", index=False)

    rows = []
    draw_id = 0
    for ratio in args.t0_ratios:
        t0 = int(round(float(ratio) * n_periods))
        t0 = min(max(t0, 1), n_periods - 1)
        for rep in range(args.reps):
            draw_name = f"draw_{draw_id:04d}"
            if args.design == "simultaneous":
                w, extra = make_simultaneous_mask(n_units, n_periods, args.nt, t0, rng)
            else:
                w, extra = make_staggered_mask(n_units, n_periods, args.nt, t0, rng)

            meta = {
                "draw_id": draw_id,
                "draw_name": draw_name,
                "dataset": "california_smoking_controls",
                "design": args.design,
                "outcome": args.outcome_col,
                "shape": {"n_rows": n_units, "n_cols": n_periods},
                "N": n_units,
                "T": n_periods,
                "Nt": args.nt,
                "T0": t0,
                "t0_ratio": float(t0 / n_periods),
                "n_hidden": int(w.sum()),
                **extra,
            }
            write_draw_files(args.output_dir, draw_name, y, w, meta)
            rows.append(meta)
            draw_id += 1

    pd.DataFrame(rows).to_csv(args.output_dir / "manifest.csv", index=False)
    print(f"Wrote {len(rows)} California-smoking benchmark draws to {args.output_dir}")


def parse_pairs(pairs: Iterable[str]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for item in pairs:
        try:
            n_str, t_str = item.split(",")
            out.append((int(n_str), int(t_str)))
        except Exception as e:
            raise ValueError(f"Could not parse pair '{item}'. Use the form N,T, e.g. 70,70") from e
    return out


def build_stock_benchmark(args: argparse.Namespace) -> None:
    if args.stock_csv is not None:
        y_full = load_stock_matrix(args.stock_csv, args.orient)
        source = "user_stock_csv"
    else:
        if args.synthetic_stock_units is None or args.synthetic_stock_periods is None:
            raise SystemExit("Provide either --stock-csv or both --synthetic-stock-units and --synthetic-stock-periods.")
        y_full = generate_synthetic_stock_matrix(
            n_units=args.synthetic_stock_units,
            n_periods=args.synthetic_stock_periods,
            rank=args.synthetic_rank,
            noise=args.synthetic_noise,
            seed=args.seed,
        )
        source = "synthetic_factor_stock_matrix"

    n_units_full, n_periods_full = y_full.shape
    pairs = parse_pairs(args.pairs)
    rng = np.random.default_rng(args.seed)

    ensure_clean_dir(args.output_dir, args.force)
    rows = []
    draw_id = 0

    for n_units, n_periods in pairs:
        if n_units > n_units_full or n_periods > n_periods_full:
            raise ValueError(
                f"Requested (N,T)=({n_units},{n_periods}) but source matrix has only "
                f"({n_units_full},{n_periods_full})."
            )
        for rep in range(args.reps):
            draw_name = f"draw_{draw_id:04d}"
            unit_idx = np.sort(rng.choice(n_units_full, size=n_units, replace=False))
            y = y_full[unit_idx, :n_periods].copy()
            w, extra = make_stock_block_mask(n_units, n_periods, rng)
            meta = {
                "draw_id": draw_id,
                "draw_name": draw_name,
                "dataset": "stock_returns_block_mask",
                "source": source,
                "shape": {"n_rows": n_units, "n_cols": n_periods},
                "N": n_units,
                "T": n_periods,
                "n_hidden": int(w.sum()),
                **extra,
            }
            write_draw_files(args.output_dir, draw_name, y, w, meta)
            rows.append(meta)
            draw_id += 1

    pd.DataFrame(rows).to_csv(args.output_dir / "manifest.csv", index=False)
    print(f"Wrote {len(rows)} stock-style benchmark draws to {args.output_dir}")


# ----------------------------
# Imputers / estimators
# ----------------------------

def nanmean0(x: np.ndarray, axis: int) -> np.ndarray:
    out = np.nanmean(x, axis=axis)
    if np.isscalar(out):
        return np.array(0.0 if np.isnan(out) else out)
    out = np.asarray(out)
    out[np.isnan(out)] = 0.0
    return out


def two_way_fixed_effects_impute(y_obs: np.ndarray, observed: np.ndarray, n_sweeps: int = 20) -> np.ndarray:
    row = np.zeros(y_obs.shape[0], dtype=np.float64)
    col = np.zeros(y_obs.shape[1], dtype=np.float64)
    for _ in range(n_sweeps):
        resid = np.where(observed, y_obs - col[None, :], np.nan)
        row = nanmean0(resid, axis=1)
        resid = np.where(observed, y_obs - row[:, None], np.nan)
        col = nanmean0(resid, axis=0)
        m = float(np.mean(row))
        row -= m
        col += m
    return row[:, None] + col[None, :]


def column_mean_impute(y_obs: np.ndarray, observed: np.ndarray) -> np.ndarray:
    col_mean = nanmean0(np.where(observed, y_obs, np.nan), axis=0)
    return np.broadcast_to(col_mean[None, :], y_obs.shape).copy()


def soft_impute_with_fixed_effects(
    y_obs: np.ndarray,
    observed: np.ndarray,
    lam: float,
    max_iters: int = 200,
    tol: float = 1e-5,
    n_fe_sweeps: int = 3,
) -> np.ndarray:
    """
    Practical MC-NNM implementation with unregularized row/column fixed effects.

    Objective (up to scaling):
        min_{L, a, b}  1/|O| || P_O(Y - L - a1' - 1b') ||_F^2 + lam ||L||_*

    We use the paper's shrinkage logic, where the singular-value threshold is lam*|O|/2.
    """
    y_obs = np.asarray(y_obs, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)

    n_rows, n_cols = y_obs.shape
    row = np.zeros(n_rows, dtype=np.float64)
    col = np.zeros(n_cols, dtype=np.float64)
    lmat = np.zeros_like(y_obs)

    threshold = lam * max(int(observed.sum()), 1) / 2.0

    for _ in range(max_iters):
        row_old = row.copy()
        col_old = col.copy()
        l_old = lmat.copy()

        # Update fixed effects a few times on observed entries.
        for _ in range(n_fe_sweeps):
            resid = np.where(observed, y_obs - lmat - col[None, :], np.nan)
            row = nanmean0(resid, axis=1)
            resid = np.where(observed, y_obs - lmat - row[:, None], np.nan)
            col = nanmean0(resid, axis=0)
            m = float(np.mean(row))
            row -= m
            col += m

        centered_obs = np.where(observed, y_obs - row[:, None] - col[None, :], 0.0)
        filled = centered_obs + np.where(~observed, lmat, 0.0)

        try:
            u, s, vt = np.linalg.svd(filled, full_matrices=False)
        except np.linalg.LinAlgError:
            # Rare SVD non-convergence; nudge and retry once.
            jitter = 1e-8 * np.random.default_rng(0).normal(size=filled.shape)
            u, s, vt = np.linalg.svd(filled + jitter, full_matrices=False)
        s_shrunk = np.maximum(s - threshold, 0.0)
        rank = int(np.sum(s_shrunk > 0.0))
        if rank == 0:
            lmat = np.zeros_like(filled)
        else:
            lmat = (u[:, :rank] * s_shrunk[:rank]) @ vt[:rank, :]

        denom = np.linalg.norm(l_old) + np.linalg.norm(row_old) + np.linalg.norm(col_old) + 1e-8
        delta = (
            np.linalg.norm(lmat - l_old)
            + np.linalg.norm(row - row_old)
            + np.linalg.norm(col - col_old)
        ) / denom
        if delta < tol:
            break

    return lmat + row[:, None] + col[None, :]


def make_lambda_grid(y_obs: np.ndarray, observed: np.ndarray, size: int) -> np.ndarray:
    baseline = two_way_fixed_effects_impute(y_obs, observed)
    centered = np.where(observed, y_obs - baseline, 0.0)
    try:
        sigma_max = float(np.linalg.svd(centered, compute_uv=False)[0])
    except np.linalg.LinAlgError:
        sigma_max = float(np.linalg.norm(centered, ord=2))
    if sigma_max <= 0:
        return np.array([0.0])
    lam_max = 2.0 * sigma_max / max(int(observed.sum()), 1)
    if size <= 1:
        return np.array([lam_max])
    grid = np.geomspace(lam_max, lam_max * 1e-3, num=size - 1)
    grid = np.unique(np.r_[grid, 0.0])[::-1]
    return grid


def cross_validated_mcnnm(
    y_true: np.ndarray,
    observed: np.ndarray,
    cv_folds: int = 5,
    lambda_grid_size: int = 10,
    max_iters: int = 200,
    tol: float = 1e-5,
    seed: int = 1,
) -> tuple[np.ndarray, float]:
    """
    Cross-validation matching the paper's recipe:
    choose K random subsets Ok subset O with |Ok| ≈ |O|^2/(NT),
    fit on Ok, validate on O minus Ok, and pick lambda minimizing mean validation MSE.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)

    obs_idx = np.argwhere(observed)
    n_obs = obs_idx.shape[0]
    if n_obs == 0:
        raise ValueError("No observed entries available for cross-validation.")

    target_train_size = max(1, int(math.floor((n_obs ** 2) / (y_true.shape[0] * y_true.shape[1]))))
    lambda_grid = make_lambda_grid(np.where(observed, y_true, np.nan), observed, lambda_grid_size)

    scores = np.zeros(len(lambda_grid), dtype=np.float64)
    counts = np.zeros(len(lambda_grid), dtype=np.int64)

    for _ in range(cv_folds):
        perm = rng.permutation(n_obs)
        train_idx = obs_idx[perm[:target_train_size]]
        train_mask = np.zeros_like(observed, dtype=bool)
        train_mask[train_idx[:, 0], train_idx[:, 1]] = True
        valid_mask = observed & ~train_mask
        if not valid_mask.any():
            continue

        y_train = np.where(train_mask, y_true, np.nan)
        for j, lam in enumerate(lambda_grid):
            pred = soft_impute_with_fixed_effects(y_train, train_mask, lam=lam, max_iters=max_iters, tol=tol)
            err = (pred[valid_mask] - y_true[valid_mask]) ** 2
            scores[j] += float(np.mean(err))
            counts[j] += 1

    valid = counts > 0
    if not np.any(valid):
        best_lam = 0.0
    else:
        avg_scores = np.full_like(scores, fill_value=np.inf, dtype=np.float64)
        avg_scores[valid] = scores[valid] / counts[valid]
        best_lam = float(lambda_grid[int(np.argmin(avg_scores))])

    pred = soft_impute_with_fixed_effects(
        np.where(observed, y_true, np.nan),
        observed,
        lam=best_lam,
        max_iters=max_iters,
        tol=tol,
    )
    return pred, best_lam


# ----------------------------
# Evaluation
# ----------------------------

def rmse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    idx = mask.astype(bool)
    if not np.any(idx):
        return float("nan")
    return float(np.sqrt(np.mean((pred[idx] - truth[idx]) ** 2)))


def normal_ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    m = float(np.mean(values))
    if values.size == 1:
        return m, m
    se = float(np.std(values, ddof=1) / np.sqrt(values.size))
    half = 1.96 * se
    return m - half, m + half


def evaluate_benchmark(args: argparse.Namespace) -> None:
    manifest_path = args.input_dir / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest.csv in {args.input_dir}")
    manifest = pd.read_csv(manifest_path)

    if args.predictions_dir is not None:
        args.predictions_dir.mkdir(parents=True, exist_ok=True)

    per_draw = []
    for rec in manifest.to_dict(orient="records"):
        draw_name = str(rec["draw_name"])
        y = pd.read_csv(args.input_dir / f"{draw_name}_Y.csv").to_numpy(dtype=np.float64)
        w = pd.read_csv(args.input_dir / f"{draw_name}_W.csv").to_numpy(dtype=np.int8)
        observed = (w == 0)
        y_obs = np.where(observed, y, np.nan)

        if args.method == "column_mean":
            pred = column_mean_impute(y_obs, observed)
            best_lam = np.nan
        elif args.method == "two_way_fe":
            pred = two_way_fixed_effects_impute(y_obs, observed)
            best_lam = np.nan
        elif args.method == "mcnnm":
            pred, best_lam = cross_validated_mcnnm(
                y_true=y,
                observed=observed,
                cv_folds=args.cv_folds,
                lambda_grid_size=args.lambda_grid_size,
                max_iters=args.max_iters,
                tol=args.tol,
                seed=args.seed + int(rec["draw_id"]),
            )
        else:
            raise ValueError(f"Unknown method: {args.method}")

        score = rmse(pred, y, w)
        result = {**rec, "method": args.method, "rmse": score, "best_lambda": best_lam}
        per_draw.append(result)

        if args.predictions_dir is not None:
            np.save(args.predictions_dir / f"{draw_name}_{args.method}_pred.npy", pred)

    per_draw_df = pd.DataFrame(per_draw)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        per_draw_df.to_csv(args.output_csv, index=False)

    # Aggregate the way the paper summarizes results: average RMSE + 95% intervals.
    group_cols = [c for c in ["dataset", "design", "Nt", "t0_ratio", "N", "T", "method"] if c in per_draw_df.columns]
    summary_rows = []
    for keys, group in per_draw_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        vals = group["rmse"].to_numpy(dtype=np.float64)
        row["n_draws"] = int(vals.size)
        row["avg_rmse"] = float(np.mean(vals))
        lo, hi = normal_ci95(vals)
        row["ci95_low"] = lo
        row["ci95_high"] = hi
        if "best_lambda" in group.columns:
            finite_lam = group["best_lambda"].replace([np.inf, -np.inf], np.nan).dropna()
            if not finite_lam.empty:
                row["avg_best_lambda"] = float(finite_lam.mean())
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print("\nPer-draw results (head):")
    print(per_draw_df.head().to_string(index=False))
    print("\nSummary:")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No summary rows produced.")

    if args.output_csv is not None:
        summary_path = args.output_csv.with_name(args.output_csv.stem + "_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved per-draw results to {args.output_csv}")
        print(f"Saved summary results to {summary_path}")


def main() -> None:
    args = parse_args()
    if args.command == "build-california":
        build_california_benchmark(args)
    elif args.command == "build-stocks":
        build_stock_benchmark(args)
    elif args.command == "evaluate":
        evaluate_benchmark(args)
    else:
        raise RuntimeError(f"Unknown command {args.command!r}")


if __name__ == "__main__":
    main()
