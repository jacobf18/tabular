"""Configurable paths for release experiment scripts.

By default, scripts use the public release layout:
  data/          input matrices and covariance caches
  figures/      paper figures
  figures_logit/ logit-space appendix figures
  logs/         experiment logs
  logs_logit/   logit-space logs

Set BENCHSELECT_DATA_DIR and/or BENCHSELECT_EXPERIMENT_ROOT to redirect reads
and writes for scratch runs.
"""

import os
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent


def _resolve(path):
    return Path(path).expanduser().resolve()


def _from_env(name, default):
    value = os.environ.get(name)
    return _resolve(value) if value else _resolve(default)


def data_dir():
    return _from_env("BENCHSELECT_DATA_DIR", ROOT / "data")


def _experiment_default(dirname, legacy_default):
    root = os.environ.get("BENCHSELECT_EXPERIMENT_ROOT")
    return _resolve(root) / dirname if root else legacy_default


def figures_dir():
    default = _experiment_default("figures", ROOT / "figures")
    return _from_env("BENCHSELECT_FIGURES_DIR", default)


def logit_figures_dir():
    default = _experiment_default("figures_logit", ROOT / "figures_logit")
    return _from_env("BENCHSELECT_LOGIT_FIGURES_DIR", default)


def logs_dir():
    default = _experiment_default("logs", ROOT / "logs")
    return _from_env("BENCHSELECT_LOGS_DIR", default)


def logit_logs_dir():
    default = _experiment_default("logs_logit", ROOT / "logs_logit")
    return _from_env("BENCHSELECT_LOGIT_LOGS_DIR", default)


def covariance_dir():
    return _from_env("BENCHSELECT_COV_DIR", data_dir())
