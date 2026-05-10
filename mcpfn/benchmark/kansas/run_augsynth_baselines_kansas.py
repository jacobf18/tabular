#!/usr/bin/env python3
"""Augmented Synthetic Control (ASCM) baseline for the Kansas benchmark.

Implements the ridge-augmented SCM estimator from
Ben-Michael, Feller, and Rothstein (2021, JASA), "The Augmented Synthetic
Control Method", using the R `augsynth` package. One R subprocess processes all
draws in a single pass to amortize package-load overhead.

For each draw:
  - write a long-format CSV (unit, time, outcome, treatment)
  - R fits augsynth(outcome ~ treatment, progfunc="Ridge", scm=TRUE) and writes
    the treated-row counterfactual trajectory Y_hat(0)
  - Python loads the trajectory, builds `completed.npy` preserving observed
    cells, and computes the standard metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

BENCH_DIR = Path(__file__).resolve().parents[1]
CPS_DIR = BENCH_DIR / "CPS"
if str(CPS_DIR) not in sys.path:
    sys.path.insert(0, str(CPS_DIR))

from eval_cps_causal import compute_metrics  # noqa: E402


R_SCRIPT = r'''
options(warn = 1)
suppressPackageStartupMessages(library(augsynth))

work_dir <- Sys.getenv("AUGSYNTH_WORK_DIR")
draws_csv <- Sys.getenv("AUGSYNTH_DRAWS_CSV")
progfunc <- Sys.getenv("AUGSYNTH_PROGFUNC")
scm_flag <- tolower(Sys.getenv("AUGSYNTH_SCM")) %in% c("1","true","yes")
if (!nzchar(progfunc)) progfunc <- "Ridge"

draws <- read.csv(draws_csv, stringsAsFactors = FALSE)

for (i in seq_len(nrow(draws))) {
  draw_name <- draws$draw_name[i]
  in_path <- file.path(work_dir, paste0(draw_name, "_long.csv"))
  out_path <- file.path(work_dir, paste0(draw_name, "_yhat.csv"))
  if (file.exists(out_path)) next

  df <- read.csv(in_path, stringsAsFactors = FALSE)
  df$unit <- as.integer(df$unit)
  df$time <- as.integer(df$time)
  df$treatment <- as.integer(df$treatment)

  fit_err <- NULL
  yhat <- tryCatch({
    obj <- augsynth::augsynth(
      outcome ~ treatment,
      unit = unit, time = time, data = df,
      progfunc = progfunc, scm = scm_flag
    )
    as.numeric(predict(obj, att = FALSE))
  }, error = function(e) {
    fit_err <<- conditionMessage(e)
    NULL
  })

  if (is.null(yhat)) {
    out <- data.frame(time = integer(0), yhat = numeric(0))
    err_path <- file.path(work_dir, paste0(draw_name, "_err.txt"))
    err_msg <- if (is.null(fit_err)) "unknown augsynth failure" else fit_err
    writeLines(err_msg, err_path)
  } else {
    out <- data.frame(time = sort(unique(df$time)), yhat = yhat)
  }
  write.csv(out, out_path, row.names = FALSE)
}

cat(sprintf("Wrote augsynth predictions for %d draws\n", nrow(draws)))
'''


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-dir", type=Path, required=True)
    p.add_argument("--method-name", type=str, default="augsynth",
                   help="Key used in per_draw_metrics.csv and the saved .npy file.")
    p.add_argument("--progfunc", type=str, default="Ridge",
                   choices=["None", "Ridge", "EN", "RF", "GSYN", "MCP"],
                   help="augsynth::augsynth outcome-model augmentation ('Ridge' = ASCM in BFR 2021).")
    p.add_argument("--no-scm", action="store_true",
                   help="Set scm=FALSE in augsynth (pure outcome model, no synthetic control).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--rscript", type=str, default=shutil.which("Rscript") or "Rscript")
    p.add_argument("--keep-workdir", action="store_true",
                   help="Keep temporary R working directory for debugging.")
    return p.parse_args()


def write_long_csv(true: np.ndarray, treatment_mask: np.ndarray, path: Path) -> None:
    N, T = true.shape
    rows = []
    for i in range(N):
        for t in range(T):
            rows.append({
                "unit": i + 1,
                "time": t + 1,
                "outcome": float(true[i, t]),
                "treatment": int(treatment_mask[i, t]),
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def build_completed(
    observed: np.ndarray,
    treatment_mask: np.ndarray,
    treated_rows: np.ndarray,
    T0: int,
    yhat_post: np.ndarray,
) -> np.ndarray:
    missing = np.isnan(observed)
    completed = observed.astype(np.float64).copy()
    for i in treated_rows:
        completed[i, T0:] = yhat_post
    completed[~missing] = observed[~missing]
    return completed.astype(np.float32)


def main() -> None:
    args = parse_args()
    if not args.benchmark_dir.exists():
        raise SystemExit(f"Benchmark directory does not exist: {args.benchmark_dir}")

    draw_dirs = sorted(
        p for p in args.benchmark_dir.iterdir()
        if p.is_dir() and p.name.startswith("draw_")
    )
    if args.limit is not None:
        draw_dirs = draw_dirs[: args.limit]
    if not draw_dirs:
        raise SystemExit(f"No draw_* directories under {args.benchmark_dir}")

    results_dir = args.benchmark_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    method_name = args.method_name

    work_dir = Path(tempfile.mkdtemp(prefix="kansas_augsynth_"))
    r_script_path = work_dir / "run_augsynth.R"
    r_script_path.write_text(R_SCRIPT, encoding="utf-8")

    per_draw_info: list[dict] = []
    to_fit: list[str] = []

    for draw_dir in draw_dirs:
        true = np.load(draw_dir / "true.npy")
        treatment_mask = np.load(draw_dir / "treatment_mask.npy")
        long_path = work_dir / f"{draw_dir.name}_long.csv"
        out_path = work_dir / f"{draw_dir.name}_yhat.csv"

        completed_path = draw_dir / f"{method_name}.npy"
        if completed_path.exists() and not args.force_rerun:
            per_draw_info.append({"draw_dir": draw_dir, "cached": True})
            continue

        write_long_csv(true, treatment_mask, long_path)
        per_draw_info.append({"draw_dir": draw_dir, "cached": False, "yhat_path": out_path})
        to_fit.append(draw_dir.name)

    if to_fit:
        draws_csv = work_dir / "draws.csv"
        pd.DataFrame({"draw_name": to_fit}).to_csv(draws_csv, index=False)

        env = os.environ.copy()
        env.update({
            "AUGSYNTH_WORK_DIR": str(work_dir),
            "AUGSYNTH_DRAWS_CSV": str(draws_csv),
            "AUGSYNTH_PROGFUNC": args.progfunc,
            "AUGSYNTH_SCM": "0" if args.no_scm else "1",
        })

        print(f"Fitting augsynth ({args.progfunc}, scm={'TRUE' if not args.no_scm else 'FALSE'}) "
              f"on {len(to_fit)} draws via {args.rscript}…")
        start = time.time()
        proc = subprocess.run([args.rscript, str(r_script_path)], env=env, check=False)
        elapsed = time.time() - start
        if proc.returncode != 0:
            raise SystemExit(f"R augsynth script failed with exit {proc.returncode}.")
        print(f"R batch finished in {elapsed:.1f}s")

    metrics_rows: list[dict] = []
    failures: list[dict] = []

    for info in tqdm(per_draw_info, desc="augsynth metrics", unit="draw"):
        draw_dir = info["draw_dir"]
        metadata = json.loads((draw_dir / "metadata.json").read_text(encoding="utf-8"))
        true = np.load(draw_dir / "true.npy")
        observed = np.load(draw_dir / "observed.npy")
        treatment_mask = np.load(draw_dir / "treatment_mask.npy")
        treated_rows = np.flatnonzero(treatment_mask.any(axis=1))
        T0 = int(metadata["T0"])

        completed_path = draw_dir / f"{method_name}.npy"
        if info["cached"]:
            completed = np.load(completed_path)
            runtime_seconds = float("nan")
        else:
            yhat_path = info["yhat_path"]
            if not yhat_path.exists():
                failures.append({"draw_name": draw_dir.name, "method": method_name,
                                 "error": f"missing {yhat_path}"})
                continue
            yhat_df = pd.read_csv(yhat_path)
            if yhat_df.empty:
                err_path = work_dir / f"{draw_dir.name}_err.txt"
                err_msg = err_path.read_text(encoding="utf-8").strip() if err_path.exists() else "empty predictions"
                failures.append({"draw_name": draw_dir.name, "method": method_name, "error": err_msg})
                continue
            yhat = yhat_df["yhat"].to_numpy(dtype=np.float64)
            yhat_post = yhat[T0:]
            completed = build_completed(observed, treatment_mask, treated_rows, T0, yhat_post)
            np.save(completed_path, completed)
            runtime_seconds = float("nan")

        metrics = compute_metrics(true.astype(np.float32), completed, treatment_mask)
        metrics_rows.append({
            "draw_name": draw_dir.name,
            "draw_id": int(metadata["draw_id"]),
            "method": method_name,
            "runtime_seconds": runtime_seconds,
            **metrics,
        })

    per_draw_path = results_dir / "per_draw_metrics.csv"
    new_df = pd.DataFrame(metrics_rows)
    if per_draw_path.exists() and not new_df.empty:
        old_df = pd.read_csv(per_draw_path)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["draw_name", "method"], keep="last")
        merged.to_csv(per_draw_path, index=False)
    else:
        new_df.to_csv(per_draw_path, index=False)

    if failures:
        failures_path = results_dir / f"failures_{method_name}.csv"
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"Encountered {len(failures)} augsynth failures; see {failures_path}")

    if args.keep_workdir:
        print(f"Kept temporary R workdir: {work_dir}")
    else:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"Wrote {len(metrics_rows)} augsynth rows to {per_draw_path}")


if __name__ == "__main__":
    main()
