"""Run TabImpute on the benchmark matrices (paper subsamples or full).

Input modes:
  --input subsamples   — iterate over the 100 paper subsamples (default)
  --input full         — evaluate on the full processed dataset matrix. This
                         is the ablation/extension mode. Only ``chunked`` or
                         ``auto`` eval modes are sensible here; the script
                         refuses ``full`` for whole-dataset matrices.

Output: outputs/metrics/<dataset>_<size>_tabimpute.csv (appending).

Examples:
    python src/eval/run_tabimpute_eval.py --dataset movielens --size 1m \\
        --model dwip-v2-1 --eval-mode auto

    python src/eval/run_tabimpute_eval.py --dataset movielens --size 1m \\
        --model dwip-v1 --input full --eval-mode chunked \\
        --row-chunk 50 --col-chunk 10
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
sys.path.insert(0, str(_ROOT))

from src.methods.tabimpute_adapter import build, load_tabimpute_model_entry  # noqa
from src.utils.config import load_experiments_config, load_tabimpute_models_config  # noqa
from src.utils.io import load_json, load_sparse  # noqa
from src.utils.logging_utils import get_logger  # noqa
from src.utils.paths import METRICS_DIR, dataset_root  # noqa
from src.utils.splits import rmse, withhold_observed  # noqa

LOGGER = get_logger("run_tabimpute_eval")


def _rating_scale(dataset: str) -> tuple[float, float]:
    return (1.0, 5.0)


def _metrics_csv_path(dataset: str, size: str) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    return METRICS_DIR / f"{dataset}_{size}_tabimpute.csv"


def _completed_keys(csv_path: Path) -> set[tuple]:
    if not csv_path.exists():
        return set()
    done = set()
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            done.add((row["model"], row["eval_mode_requested"], row["input"], int(row["rep"]), float(row["withhold"])))
    return done


def _append(csv_path: Path, row: Dict):
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_subsamples(args, model_entry, experiments) -> int:
    sub_dir = dataset_root(args.dataset, args.size) / "processed" / "paper_subsamples"
    idx_path = sub_dir / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"{idx_path} not found — run build_paper_subsamples.py")
    index = load_json(idx_path)
    num_reps = args.num_reps or index["num_reps"]
    withhold_list = args.withhold or experiments["paper_subsamples"]["withhold_fractions"]
    csv_path = _metrics_csv_path(args.dataset, args.size)
    done = set() if args.force else _completed_keys(csv_path)
    rating_scale = _rating_scale(args.dataset)

    ti_cfg = experiments["tabimpute"]
    adapter = build(
        model_entry=model_entry,
        eval_mode=args.eval_mode,
        row_chunk=args.row_chunk,
        col_chunk=args.col_chunk,
        device=args.device,
        checkpoint_override=args.checkpoint_override,
        full_rows_limit=ti_cfg["full_rows_limit"],
        full_cols_limit=ti_cfg["full_cols_limit"],
    )

    for rep_meta in index["replicates"][:num_reps]:
        r = rep_meta["rep"]
        matrix = load_sparse(Path(rep_meta["path"]) / "matrix.npz")
        for withhold in withhold_list:
            key = (model_entry["name"], args.eval_mode, "subsamples", r, float(withhold))
            if key in done:
                LOGGER.info("[skip cached] %s", key)
                continue
            split = withhold_observed(matrix, float(withhold), seed=int(rep_meta["seed"]) + int(round(withhold * 1000)))
            if split.test_values.size == 0:
                continue
            t0 = time.time()
            try:
                result = adapter.fit_predict(
                    split.train_matrix,
                    split.test_rows,
                    split.test_cols,
                    rating_scale=rating_scale,
                    dataset=args.dataset,
                    withhold_fraction=float(withhold),
                )
                elapsed = time.time() - t0
                score = rmse(result.predictions, split.test_values)
                used_mode = result.extra.get("eval_mode_used", args.eval_mode)
                status = "ok"
                err = ""
                fallback = result.fallback_count
            except Exception as exc:
                elapsed = time.time() - t0
                score = float("nan")
                used_mode = "error"
                status = "error"
                err = repr(exc)
                fallback = 0
                LOGGER.exception("model=%s rep=%d withhold=%.2f errored", model_entry["name"], r, withhold)

            _append(
                csv_path,
                {
                    "dataset": args.dataset,
                    "size": args.size,
                    "model": model_entry["name"],
                    "input": "subsamples",
                    "eval_mode_requested": args.eval_mode,
                    "eval_mode_used": used_mode,
                    "row_chunk": args.row_chunk,
                    "col_chunk": args.col_chunk,
                    "rep": r,
                    "withhold": float(withhold),
                    "n_test": int(split.test_values.size),
                    "n_train": int(split.train_values.size),
                    "rmse": float(score),
                    "fallback_count": int(fallback),
                    "elapsed_sec": float(elapsed),
                    "checkpoint": str(model_entry.get("checkpoint_path", "")),
                    "status": status,
                    "error": err,
                },
            )
            LOGGER.info(
                "model=%s rep=%d withhold=%.2f eval=%s rmse=%.4f elapsed=%.2fs",
                model_entry["name"], r, withhold, used_mode, score, elapsed,
            )
            done.add(key)

    LOGGER.info("Wrote %s", csv_path)
    return 0


def run_full(args, model_entry, experiments) -> int:
    """Run TabImpute on the full processed matrix using chunked/auto mode."""
    if args.eval_mode == "full":
        raise SystemExit(
            "--eval-mode full is not supported for --input full; use 'chunked' or 'auto'."
        )
    full_dir = dataset_root(args.dataset, args.size) / "processed" / "full"
    matrix = load_sparse(full_dir / "matrix.npz")
    LOGGER.info("Full matrix shape=%s nnz=%d", matrix.shape, matrix.nnz)

    csv_path = _metrics_csv_path(args.dataset, args.size)
    withhold = float(args.withhold[0]) if args.withhold else 0.10
    seed = int(args.seed)
    split = withhold_observed(matrix, withhold, seed=seed)
    LOGGER.info(
        "Train nnz=%d test nnz=%d; running model %s eval_mode=%s chunks=%dx%d",
        split.train_values.size, split.test_values.size,
        model_entry["name"], args.eval_mode, args.row_chunk, args.col_chunk,
    )

    ti_cfg = experiments["tabimpute"]
    adapter = build(
        model_entry=model_entry,
        eval_mode=args.eval_mode,
        row_chunk=args.row_chunk,
        col_chunk=args.col_chunk,
        device=args.device,
        checkpoint_override=args.checkpoint_override,
        full_rows_limit=ti_cfg["full_rows_limit"],
        full_cols_limit=ti_cfg["full_cols_limit"],
    )
    t0 = time.time()
    result = adapter.fit_predict(
        split.train_matrix,
        split.test_rows,
        split.test_cols,
        rating_scale=_rating_scale(args.dataset),
        dataset=args.dataset,
        withhold_fraction=withhold,
    )
    elapsed = time.time() - t0
    score = rmse(result.predictions, split.test_values)
    LOGGER.info("full model=%s eval=%s rmse=%.4f elapsed=%.1fs",
                model_entry["name"], result.extra.get("eval_mode_used"), score, elapsed)
    _append(
        csv_path,
        {
            "dataset": args.dataset,
            "size": args.size,
            "model": model_entry["name"],
            "input": "full",
            "eval_mode_requested": args.eval_mode,
            "eval_mode_used": result.extra.get("eval_mode_used", args.eval_mode),
            "row_chunk": args.row_chunk,
            "col_chunk": args.col_chunk,
            "rep": 0,
            "withhold": withhold,
            "n_test": int(split.test_values.size),
            "n_train": int(split.train_values.size),
            "rmse": float(score),
            "fallback_count": int(result.fallback_count),
            "elapsed_sec": float(elapsed),
            "checkpoint": str(model_entry.get("checkpoint_path", "")),
            "status": "ok",
            "error": "",
        },
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["movielens", "netflix"])
    parser.add_argument("--size", required=True)
    parser.add_argument("--model", required=True, help="Normalized model name (dwip-v2-1 / dwip-v2-2 / dwip-v1)")
    parser.add_argument("--input", default="subsamples", choices=["subsamples", "full"])
    parser.add_argument("--eval-mode", default="auto", choices=["full", "chunked", "auto"])
    parser.add_argument("--row-chunk", type=int, default=50)
    parser.add_argument("--col-chunk", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-override", default=None)
    parser.add_argument("--num-reps", type=int, default=None)
    parser.add_argument("--withhold", type=float, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=123, help="Seed for --input full split")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    experiments = load_experiments_config()
    models_cfg = load_tabimpute_models_config()
    entry = load_tabimpute_model_entry(args.model, models_cfg)

    if args.input == "subsamples":
        return run_subsamples(args, entry, experiments)
    return run_full(args, entry, experiments)


if __name__ == "__main__":
    sys.exit(main())
