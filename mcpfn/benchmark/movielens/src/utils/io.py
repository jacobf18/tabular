"""Sparse matrix I/O helpers for the benchmark harness."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import scipy.sparse as sp


def save_sparse(path: Path, matrix: sp.spmatrix) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(str(path), matrix.tocsr())


def load_sparse(path: Path) -> sp.csr_matrix:
    return sp.load_npz(str(path)).tocsr()


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
