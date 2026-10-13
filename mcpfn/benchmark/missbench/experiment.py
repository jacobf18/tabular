"""
Benchmark experiments: create missingness, run imputers, and score the results.

On-disk layout (the same layout the paper results were produced with):

    <data_dir>/<suite>/<dataset>/dataset.pkl                     complete data
    <data_dir>/<suite>/<dataset>/<PATTERN>_<p>/missing.npy       standardized data with missing values
    <data_dir>/<suite>/<dataset>/<PATTERN>_<p>/true.npy          standardized ground truth
    <data_dir>/<suite>/<dataset>/<PATTERN>_<p>/<method>.npy      imputation from <method>
    <data_dir>/<suite>/<dataset>/<PATTERN>_<p>/<method>.json     runtime and device
    <data_dir>/<suite>/<dataset>/<PATTERN>_<p>/repeats/<r>/...   repeats r >= 1
"""
from __future__ import annotations

import importlib
import itertools
import json
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import datasets as ds
from .methods import METHODS, Method, get_method
from .metrics import METRICS
from .patterns import DEFAULT_P, MCAR_SWEEP_P, PATTERNS, Pattern, induce_missingness, pattern_seed


@dataclass(frozen=True)
class Experiment:
    name: str
    suite: str  # dataset suite, also the data sub-directory
    patterns: tuple[str, ...]
    ps: tuple[float, ...]
    repeats: int
    description: str


EXPERIMENTS: dict[str, Experiment] = {
    e.name: e
    for e in [
        Experiment("openml", "openml", tuple(PATTERNS), (DEFAULT_P,), 1,
                   "MissBench: OpenML datasets x 11 missingness patterns at p=0.4"),
        Experiment("openml_mcar_sweep", "openml", ("mcar",), tuple(MCAR_SWEEP_P), 1,
                   "OpenML datasets under MCAR with p from 0.2 to 0.8 (paper Fig. 7)"),
        Experiment("uci", "uci", ("mcar",), (0.1, 0.2, 0.3, 0.4, 0.5), 10,
                   "HyperImpute-paper UCI datasets under MCAR, 10 repeats per level"),
    ]
}


@dataclass(frozen=True)
class Cell:
    """One benchmark instance: a dataset with one sampled missingness mask."""
    suite: str
    dataset: str
    pattern: Pattern
    p: float
    repeat: int

    def dir(self, data_dir: Path) -> Path:
        d = Path(data_dir) / self.suite / self.dataset / f"{self.pattern.code}_{self.p}"
        return d if self.repeat == 0 else d / "repeats" / str(self.repeat)


def cells(exp: Experiment, datasets: list[str], patterns: list[str] | None = None,
          ps: list[float] | None = None, repeats: int | None = None) -> list[Cell]:
    patterns = patterns or list(exp.patterns)
    ps = ps or list(exp.ps)
    repeats = exp.repeats if repeats is None else repeats
    return [
        Cell(exp.suite, d, PATTERNS[k], p, r)
        for d, k, p, r in itertools.product(datasets, patterns, ps, range(repeats))
    ]


# --- Missingness ---------------------------------------------------------------

def create_missingness(data_dir: Path, todo: list[Cell], seed: int = 0, overwrite: bool = False) -> None:
    for cell in tqdm(todo, desc="Creating missingness"):
        out = cell.dir(data_dir)
        if (out / "missing.npy").exists() and not overwrite:
            continue
        X = ds.load_dataset(data_dir, cell.suite, cell.dataset)
        X_missing, X_true = induce_missingness(
            X, cell.pattern, cell.p, pattern_seed(cell.dataset, cell.pattern, cell.p, cell.repeat, seed)
        )
        out.mkdir(parents=True, exist_ok=True)
        # Imputations of a previous mask are no longer valid
        for old in [*out.glob("*.npy"), *out.glob("*.json")]:
            old.unlink()
        np.save(out / "missing.npy", X_missing)
        np.save(out / "true.npy", X_true)


# --- Imputation ----------------------------------------------------------------

def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def run_imputers(data_dir: Path, todo: list[Cell], methods: list[tuple[str, str | None]], device: str = "auto",
                 overwrite: bool = False, fail_fast: bool = False) -> None:
    """Run each (method key, checkpoint path or None) on every cell in `todo`.

    Each method runs in its own worker process. This keeps one method's native
    libraries (e.g. OpenMP runtimes of PyTorch and xgboost) and crashes from affecting
    the others, and frees GPU memory between methods.
    """
    device = resolve_device(device)
    missing_inputs = [c for c in todo if not (c.dir(data_dir) / "missing.npy").exists()]
    if missing_inputs:
        raise SystemExit(f"{len(missing_inputs)} datasets have no missingness yet "
                         f"(e.g. {missing_inputs[0].dir(data_dir)}). Run the 'missingness' step first.")

    context = mp.get_context("spawn")
    for key, checkpoint in methods:
        pending = [c for c in todo if overwrite or not (c.dir(data_dir) / f"{key}.npy").exists()]
        if not pending:
            print(f"{key}: all {len(todo)} imputations already exist")
            continue
        worker = context.Process(target=_impute_worker,
                                 args=(Path(data_dir), pending, key, checkpoint, device, fail_fast))
        worker.start()
        worker.join()
        if worker.exitcode != 0:
            message = f"{key}: worker exited with code {worker.exitcode}"
            if fail_fast:
                raise SystemExit(message)
            print(message)


def _impute_worker(data_dir: Path, pending: list[Cell], key: str, checkpoint: str | None, device: str,
                   fail_fast: bool) -> None:
    method = get_method(key, checkpoint)
    for module in method.preload:
        importlib.import_module(module)
    method_device = "cpu" if method.cpu_only else device
    print(f"{key}: imputing {len(pending)} datasets on {method_device}")
    try:
        impute = method.build(method_device, 0)
    except ImportError as e:
        hint = f" Install it with: {method.requires}" if method.requires else ""
        raise SystemExit(f"Could not load {key}: {e}.{hint}") from e

    failures = 0
    for cell in tqdm(pending, desc=key):
        cfg_dir = cell.dir(data_dir)
        X_missing = np.load(cfg_dir / "missing.npy")
        _seed_everything(cell.repeat)
        info = {"method": key, "device": method_device}
        start = time.time()
        try:
            X_imputed = np.asarray(impute(X_missing.copy()), dtype=np.float64)
        except Exception as e:
            if fail_fast:
                raise
            failures += 1
            info["error"] = repr(e)
            tqdm.write(f"  {key} failed on {cfg_dir}: {e}")
            tqdm.write(traceback.format_exc(limit=3))
            _write_json(cfg_dir / f"{key}.json", info)
            continue
        info["seconds"] = time.time() - start
        np.save(cfg_dir / f"{key}.npy", X_imputed)
        _write_json(cfg_dir / f"{key}.json", info)
    if failures:
        print(f"{key}: {failures} of {len(pending)} imputations failed (see messages above)")


def _seed_everything(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_json(path: Path, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


# --- Evaluation ----------------------------------------------------------------

def _find_imputation(cfg_dir: Path, method: Method) -> Path | None:
    for name in (method.key, *method.legacy_files):
        path = cfg_dir / f"{name}.npy"
        if path.exists():
            return path
    return None


def _find_runtime(cfg_dir: Path, method: Method) -> tuple[float, str | None]:
    path = cfg_dir / f"{method.key}.json"
    if path.exists():
        with open(path) as f:
            info = json.load(f)
        return info.get("seconds", np.nan), info.get("device")
    for name in method.legacy_time_files:
        path = cfg_dir / f"{name}_imputation_time.txt"
        if path.exists():
            lines = [line for line in path.read_text().split() if line.strip()]
            if lines:
                return float(lines[0]), None
    return np.nan, None


def evaluate(data_dir: Path, todo: list[Cell], methods: list[Method] | None = None) -> pd.DataFrame:
    """Score every imputation found for `todo` and return one row per (cell, method)."""
    methods = methods or list(METHODS.values())
    rows = []
    for cell in tqdm(todo, desc="Scoring"):
        cfg_dir = cell.dir(data_dir)
        if not (cfg_dir / "missing.npy").exists():
            continue
        X_missing = np.load(cfg_dir / "missing.npy")
        X_true = np.load(cfg_dir / "true.npy")
        mask = np.isnan(X_missing)
        for method in methods:
            path = _find_imputation(cfg_dir, method)
            seconds, device = _find_runtime(cfg_dir, method)
            if path is None and np.isnan(seconds):
                continue
            row = {
                "suite": cell.suite,
                "dataset": cell.dataset,
                "pattern": cell.pattern.key,
                "p": cell.p,
                "repeat": cell.repeat,
                "method": method.key,
                "n_rows": X_true.shape[0],
                "n_cols": X_true.shape[1],
                "n_missing": int(mask.sum()),
                "seconds": seconds,
                "device": device,
            }
            # A method can have a recorded runtime but no imputation (e.g. runtime-only runs)
            X_imputed = np.load(path) if path is not None else None
            for name, metric in METRICS.items():
                row[name] = metric(X_true, X_imputed, mask) if X_imputed is not None else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def results_path(results_dir: Path, experiment: str) -> Path:
    return Path(results_dir) / f"{experiment}.csv"


def save_results(df: pd.DataFrame, path: Path) -> None:
    os.makedirs(Path(path).parent, exist_ok=True)
    df.to_csv(path, index=False)
