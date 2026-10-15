#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/tabular/mcpfn}"
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/scripts:${PYTHONPATH:-}"

RUN_DIR="${REPO_DIR}/src/tabimpute/workdir/tabimpute-direct-rowcol-latentattn768-rank10-p02-cdf2-balanced120k-lr2e5-fromlarge80k"
OUT_PREFIX="${REPO_DIR}/runs/figures_by_setup/04_p02_gaussian_factors_rankscale_rowcol_latent_5seeds/interface_train_level_chunks_p02_rankscale_5seeds_rowcol_cdf2_balanced120k_phase_checkpoints"

python "${REPO_DIR}/scripts/eval_train_level_chunks_interface.py" \
  --sizes 20 50 75 100 150 200 300 400 500 \
  --seeds 0 1 2 3 4 \
  --rank 10 \
  --missing-rate 0.2 \
  --generator gaussian_factors \
  --scale rank \
  --skip-softimpute \
  --direct-names \
  --extra-direct "RowCol CDF w2 balanced120 phase1 20k|${RUN_DIR}/checkpoint_phase1_20000.pth|500|rowcol_latent_attention" \
  --extra-direct "RowCol CDF w2 balanced120 phase2 50k|${RUN_DIR}/checkpoint_phase2_50000.pth|500|rowcol_latent_attention" \
  --extra-direct "RowCol CDF w2 balanced120 phase3 85k|${RUN_DIR}/checkpoint_phase3_85000.pth|500|rowcol_latent_attention" \
  --extra-direct "RowCol CDF w2 balanced120 phase4 120k|${RUN_DIR}/checkpoint_phase4_120000.pth|500|rowcol_latent_attention" \
  --extra-direct "RowCol CDF w2 balanced120 trainloss best|${RUN_DIR}/best_checkpoint.pth|500|rowcol_latent_attention" \
  --output-prefix "${OUT_PREFIX}"

python - <<'PY'
import csv
from collections import defaultdict
from pathlib import Path

path = Path("/root/tabular/mcpfn/runs/figures_by_setup/04_p02_gaussian_factors_rankscale_rowcol_latent_5seeds/interface_train_level_chunks_p02_rankscale_5seeds_rowcol_cdf2_balanced120k_phase_checkpoints_summary.csv")
rows = list(csv.DictReader(path.open()))
by_size = defaultdict(list)
for row in rows:
    by_size[int(row["size"])].append(row)

out = path.with_name(path.stem + "_best_by_size.csv")
with out.open("w", newline="", encoding="utf-8") as f:
    fieldnames = ["size", "method", "rmse", "rmse_sem", "mae", "nrmse"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for size in sorted(by_size):
        best = min(by_size[size], key=lambda r: float(r["rmse"]))
        writer.writerow({key: best[key] for key in fieldnames})
print(f"Wrote {out}")
PY
