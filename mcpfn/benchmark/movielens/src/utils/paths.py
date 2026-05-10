"""Central path constants for the benchmark harness."""
from __future__ import annotations

from pathlib import Path

# benchmark/movielens/ root (the package root)
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PACKAGE_ROOT / "data"
CONFIGS_DIR = PACKAGE_ROOT / "configs"
OUTPUTS_DIR = PACKAGE_ROOT / "outputs"

METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
PLOTS_DIR = OUTPUTS_DIR / "plots"
LOGS_DIR = OUTPUTS_DIR / "logs"


def dataset_root(dataset: str, size: str) -> Path:
    return DATA_DIR / dataset / size


def dataset_subdir(dataset: str, size: str, name: str) -> Path:
    return dataset_root(dataset, size) / name


def ensure_dirs() -> None:
    for d in (DATA_DIR, CONFIGS_DIR, OUTPUTS_DIR, METRICS_DIR, PREDICTIONS_DIR, PLOTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
