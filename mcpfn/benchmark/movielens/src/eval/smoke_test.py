"""End-to-end smoke test against a small synthetic rating matrix.

Runs the full pipeline without downloading MovieLens/Netflix: it fabricates
a low-rank rating matrix, writes it to the processed/full and
processed/paper_subsamples locations, then calls the paper benchmarks and
(optionally) the TabImpute adapter on a tiny model config.

Usage:
    python src/eval/smoke_test.py
    python src/eval/smoke_test.py --include-tabimpute --model dwip-v2-1
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
sys.path.insert(0, str(_ROOT))

from src.data.build_paper_subsamples import build_subsamples  # noqa
from src.eval.aggregation import run as aggregate_run  # noqa
from src.eval.run_paper_benchmarks import run_benchmarks as run_paper  # noqa
from src.eval.run_tabimpute_eval import run_subsamples as run_ti_subsamples  # noqa
from src.methods.tabimpute_adapter import load_tabimpute_model_entry  # noqa
from src.utils.config import load_experiments_config, load_tabimpute_models_config  # noqa
from src.utils.io import save_json, save_sparse  # noqa
from src.utils.logging_utils import get_logger  # noqa
from src.utils.paths import METRICS_DIR, PLOTS_DIR, dataset_root  # noqa

LOGGER = get_logger("smoke_test")


def _synthesize(n_users: int, n_items: int, rank: int, density: float, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_users, rank)).astype(np.float32)
    V = rng.normal(size=(rank, n_items)).astype(np.float32)
    dense = U @ V
    # Map to rating scale 1-5
    dense = (dense - dense.min()) / (dense.max() - dense.min() + 1e-6) * 4.0 + 1.0
    dense = np.round(dense).astype(np.float32)
    mask = rng.random(size=dense.shape) < density
    dense[~mask] = 0.0
    return sp.csr_matrix(dense)


def stage_synthetic_dataset(dataset: str = "movielens", size: str = "1m_smoke",
                            n_users: int = 500, n_items: int = 500,
                            rank: int = 5, density: float = 0.3, seed: int = 17) -> None:
    """Write a synthetic full + subsamples layout under data/<dataset>/<size>/."""
    full_dir = dataset_root(dataset, size) / "processed" / "full"
    manifests = dataset_root(dataset, size) / "manifests"
    full_dir.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    matrix = _synthesize(n_users, n_items, rank, density, seed)
    save_sparse(full_dir / "matrix.npz", matrix)
    np.save(full_dir / "user_ids.npy", np.arange(n_users, dtype=np.int64))
    np.save(full_dir / "item_ids.npy", np.arange(n_items, dtype=np.int64))
    save_json(
        manifests / "full.json",
        {
            "dataset": dataset,
            "size": size,
            "num_users": n_users,
            "num_items": n_items,
            "num_ratings": int(matrix.nnz),
            "sparsity": 1.0 - float(matrix.nnz) / (n_users * n_items),
            "rating_scale": [1, 5],
            "synthetic": True,
            "seed": seed,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="movielens")
    parser.add_argument("--size", default="1m_smoke")
    parser.add_argument("--num-reps", type=int, default=2)
    parser.add_argument("--rows", type=int, default=150)
    parser.add_argument("--cols", type=int, default=150)
    parser.add_argument("--include-tabimpute", action="store_true")
    parser.add_argument("--model", default="dwip-v2-1")
    parser.add_argument("--eval-mode", default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-override", default=None)
    parser.add_argument("--clean", action="store_true", help="Wipe the synthetic dataset before running")
    args = parser.parse_args()

    dataset_dir = dataset_root(args.dataset, args.size)
    if args.clean and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    LOGGER.info("Staging synthetic dataset at %s", dataset_dir)
    stage_synthetic_dataset(args.dataset, args.size)

    LOGGER.info("Building subsamples")
    build_subsamples(
        dataset=args.dataset,
        size=args.size,
        num_reps=args.num_reps,
        rows=args.rows,
        cols=args.cols,
        seed=42,
        force=True,
    )

    LOGGER.info("Running paper benchmarks")
    ns = argparse.Namespace(
        dataset=args.dataset,
        size=args.size,
        num_reps=args.num_reps,
        withhold=[0.1, 0.3, 0.5, 0.7],
        methods="paper_gaussian_kernel,user_user_cf,item_item_cf,softimpute",
        lambda_override=None,
        force=True,
    )
    run_paper(ns)

    if args.include_tabimpute:
        LOGGER.info("Running TabImpute eval on subsamples")
        models_cfg = load_tabimpute_models_config()
        entry = load_tabimpute_model_entry(args.model, models_cfg)
        ns_ti = argparse.Namespace(
            dataset=args.dataset,
            size=args.size,
            model=args.model,
            input="subsamples",
            eval_mode=args.eval_mode,
            row_chunk=50,
            col_chunk=10,
            device=args.device,
            checkpoint_override=args.checkpoint_override,
            num_reps=args.num_reps,
            withhold=[0.1, 0.3, 0.5, 0.7],
            seed=123,
            force=True,
        )
        experiments = load_experiments_config()
        run_ti_subsamples(ns_ti, entry, experiments)

    LOGGER.info("Aggregating + plotting")
    ns_agg = argparse.Namespace(all=False, datasets=[(args.dataset, args.size)])
    aggregate_run(ns_agg)
    LOGGER.info("Smoke test done. Metrics in %s, plots in %s", METRICS_DIR, PLOTS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
