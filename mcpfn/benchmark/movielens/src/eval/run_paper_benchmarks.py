"""Run the paper-reproduction baselines on subsampled matrices.

For each paper_subsample replicate x withhold_fraction x method, this:
  1. Splits the replicate into an observed-train / withheld-test partition.
  2. Runs each requested method to predict the withheld entries.
  3. Writes per-run metrics (RMSE, fallback count, timing) to
     outputs/metrics/<dataset>_<size>_paper.csv (appending).

Methods default to the four covered in the paper section:
  paper_gaussian_kernel, user_user_cf, item_item_cf, softimpute

Caching: a per-run key is written into a JSON side file and subsequent runs
skip completed (method, rep, withhold) triples unless --force is passed.

Usage:
    python src/eval/run_paper_benchmarks.py --dataset movielens --size 1m
    python src/eval/run_paper_benchmarks.py --dataset netflix --size prize \\
        --methods paper_gaussian_kernel,softimpute --num-reps 10
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
sys.path.insert(0, str(_ROOT))

from src.methods import user_user_cf, item_item_cf, paper_gaussian_kernel, softimpute_wrapper  # noqa
from src.utils.config import load_experiments_config  # noqa
from src.utils.io import load_json, load_sparse, save_json  # noqa
from src.utils.logging_utils import get_logger  # noqa
from src.utils.paths import METRICS_DIR, dataset_root  # noqa
from src.utils.splits import rmse, withhold_observed  # noqa

LOGGER = get_logger("run_paper_benchmarks")

METHODS = {
    "paper_gaussian_kernel": paper_gaussian_kernel.build,
    "user_user_cf": user_user_cf.build,
    "item_item_cf": item_item_cf.build,
    "softimpute": softimpute_wrapper.build,
}


def _rating_scale(dataset: str) -> tuple[float, float]:
    return (1.0, 5.0)


def _method_kwargs(
    method: str,
    experiments: Dict,
    dataset: str,
    withhold: float,
    cli_lambda: float | None,
) -> Dict:
    if method == "paper_gaussian_kernel":
        beta = experiments["paper_gaussian_kernel"]["beta"]
        lam_map = experiments["paper_gaussian_kernel"]["lambdas"][dataset]
        key = f"{withhold:.2f}"
        lmbda = float(cli_lambda) if cli_lambda is not None else float(lam_map[key])
        return {"beta": beta, "lmbda": lmbda, "k_neighbors": 100, "min_overlap": 2}
    if method in ("user_user_cf", "item_item_cf"):
        cfg = experiments["collaborative_filtering"]
        return {"k_neighbors": cfg["k_neighbors"], "shrinkage": cfg["shrinkage"]}
    if method == "softimpute":
        cfg = experiments["softimpute"]
        return {
            "lmbda": cfg["lambda_grid"][0],
            "max_rank": cfg["max_rank"],
            "max_iters": cfg["max_iters"],
            "tol": cfg["convergence_tol"],
            "prefer_fancyimpute": False,
        }
    return {}


def _load_index(dataset: str, size: str) -> Dict:
    sub_dir = dataset_root(dataset, size) / "processed" / "paper_subsamples"
    idx = sub_dir / "index.json"
    if not idx.exists():
        raise FileNotFoundError(
            f"{idx} not found. Run build_paper_subsamples.py first."
        )
    return load_json(idx)


def _metrics_csv_path(dataset: str, size: str) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    return METRICS_DIR / f"{dataset}_{size}_paper.csv"


def _completed_keys(csv_path: Path) -> set[tuple]:
    if not csv_path.exists():
        return set()
    done = set()
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            done.add((row["method"], int(row["rep"]), float(row["withhold"])))
    return done


def _append_metric(csv_path: Path, row: Dict):
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_benchmarks(args) -> int:
    experiments = load_experiments_config()
    index = _load_index(args.dataset, args.size)
    num_reps = args.num_reps or index["num_reps"]
    withhold_list = args.withhold or experiments["paper_subsamples"]["withhold_fractions"]
    methods = args.methods.split(",") if args.methods else list(METHODS.keys())

    rating_scale = _rating_scale(args.dataset)
    csv_path = _metrics_csv_path(args.dataset, args.size)
    done = set() if args.force else _completed_keys(csv_path)

    reps = index["replicates"][:num_reps]
    for rep_meta in reps:
        r = rep_meta["rep"]
        rep_dir = Path(rep_meta["path"])
        matrix = load_sparse(rep_dir / "matrix.npz")
        for withhold in withhold_list:
            split_seed = int(rep_meta["seed"]) + int(round(withhold * 1000))
            split = withhold_observed(matrix, float(withhold), seed=split_seed)
            if split.test_values.size == 0:
                LOGGER.warning("rep %d withhold %.2f: no test entries (nnz=%d)", r, withhold, matrix.nnz)
                continue
            for method in methods:
                key = (method, r, float(withhold))
                if key in done:
                    LOGGER.info("[skip cached] %s", key)
                    continue
                if method not in METHODS:
                    LOGGER.warning("Unknown method %s, skipping", method)
                    continue
                kwargs = _method_kwargs(method, experiments, args.dataset, float(withhold), args.lambda_override)
                completer = METHODS[method](**kwargs)
                t0 = time.time()
                try:
                    result = completer.fit_predict(
                        split.train_matrix,
                        split.test_rows,
                        split.test_cols,
                        rating_scale=rating_scale,
                        dataset=args.dataset,
                        withhold_fraction=float(withhold),
                    )
                    elapsed = time.time() - t0
                    score = rmse(result.predictions, split.test_values)
                    status = "ok"
                    err = ""
                except Exception as exc:  # pragma: no cover
                    elapsed = time.time() - t0
                    score = float("nan")
                    status = "error"
                    err = repr(exc)
                    LOGGER.exception("rep %d withhold %.2f method %s errored", r, withhold, method)
                    result = None

                fallback = getattr(result, "fallback_count", 0) if result is not None else 0
                extra = getattr(result, "extra", {}) if result is not None else {}
                row = {
                    "dataset": args.dataset,
                    "size": args.size,
                    "method": method,
                    "rep": r,
                    "withhold": float(withhold),
                    "n_test": int(split.test_values.size),
                    "n_train": int(split.train_values.size),
                    "rmse": float(score),
                    "fallback_count": int(fallback),
                    "elapsed_sec": float(elapsed),
                    "status": status,
                    "error": err,
                    "extra_json": _jsonify(extra),
                }
                _append_metric(csv_path, row)
                LOGGER.info(
                    "rep=%d withhold=%.2f method=%-26s rmse=%.4f fallback=%d elapsed=%.2fs",
                    r, withhold, method, score, fallback, elapsed,
                )
                done.add(key)

    LOGGER.info("Wrote metrics to %s", csv_path)
    return 0


def _jsonify(d: Dict) -> str:
    import json
    safe = {}
    for k, v in d.items():
        try:
            json.dumps({k: v})
            safe[k] = v
        except TypeError:
            safe[k] = str(v)
    return json.dumps(safe)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["movielens", "netflix"])
    parser.add_argument("--size", required=True)
    parser.add_argument("--num-reps", type=int, default=None)
    parser.add_argument(
        "--withhold",
        type=float,
        nargs="+",
        default=None,
        help="Withhold fractions to evaluate (default: all from experiments.yaml)",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help=f"Comma-separated subset of {list(METHODS.keys())}. Default: all",
    )
    parser.add_argument(
        "--lambda-override",
        type=float,
        default=None,
        help="Override paper-schedule lambda for paper_gaussian_kernel",
    )
    parser.add_argument("--force", action="store_true", help="Ignore cached completions")
    args = parser.parse_args()
    return run_benchmarks(args)


if __name__ == "__main__":
    sys.exit(main())
