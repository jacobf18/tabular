"""
Runtime scaling benchmark: imputation time on random matrices of growing size.

By default compares TabImpute (entry-wise featurization) with TabImpute V2 on
matrices with 10 columns and an increasing number of rows. Each size is run
`num_runs` times after one warm-up run.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .experiment import resolve_device
from .methods import Method

DEFAULT_ROWS = (10, 25, 50, 100, 250, 500, 1000)
DEFAULT_COLS = 10


def make_matrix(num_rows: int, num_cols: int, missing_rate: float = 0.1, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((num_rows, num_cols))
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-16)
    X[rng.random(X.shape) < missing_rate] = np.nan
    return X.astype(np.float32)


def _sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run(methods: list[Method], device: str = "auto", rows=DEFAULT_ROWS, num_cols: int = DEFAULT_COLS,
        num_runs: int = 5) -> pd.DataFrame:
    device = resolve_device(device)
    records = []
    for method in methods:
        method_device = "cpu" if method.cpu_only else device
        impute = method.build(method_device, 0)
        for num_rows in rows:
            X = make_matrix(num_rows, num_cols)
            impute(X.copy())  # warm-up
            for run_number in range(1, num_runs + 1):
                _sync()
                start = time.time()
                impute(X.copy())
                _sync()
                seconds = time.time() - start
                records.append({"method": method.key, "device": method_device, "num_rows": num_rows,
                                "num_cols": num_cols, "run": run_number, "seconds": seconds})
            mean = np.mean([r["seconds"] for r in records[-num_runs:]])
            print(f"  {method.key:<15} {num_rows:>5} x {num_cols}: {mean:.4f} s")
    return pd.DataFrame(records)
