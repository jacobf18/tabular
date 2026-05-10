"""Base class every matrix-completion method must expose."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import scipy.sparse as sp


@dataclass
class MethodResult:
    predictions: np.ndarray
    fallback_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseMatrixCompleter:
    """Abstract matrix completer.

    Implementations fit on ``train_matrix`` (a sparse CSR where zero means
    "unobserved") and must return one predicted rating per (row, col) pair in
    ``test_rows``/``test_cols``.
    """

    name: str = "base"

    def __init__(self, **kwargs: Any):
        self._kwargs = kwargs

    def fit_predict(
        self,
        train_matrix: sp.csr_matrix,
        test_rows: np.ndarray,
        test_cols: np.ndarray,
        *,
        rating_scale: Optional[tuple[float, float]] = None,
        dataset: Optional[str] = None,
        withhold_fraction: Optional[float] = None,
    ) -> MethodResult:
        raise NotImplementedError

    @staticmethod
    def _global_mean(train_matrix: sp.csr_matrix) -> float:
        if train_matrix.nnz == 0:
            return 0.0
        return float(train_matrix.data.mean())

    @staticmethod
    def _clip_scale(preds: np.ndarray, rating_scale: Optional[tuple[float, float]]) -> np.ndarray:
        if rating_scale is None:
            return preds
        lo, hi = rating_scale
        return np.clip(preds, lo, hi)
