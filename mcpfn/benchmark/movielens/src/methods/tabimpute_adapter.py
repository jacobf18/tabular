"""TabImpute adapter for the matrix-completion benchmark.

Wraps ``tabimpute.interface.ImputePFN`` in the ``entry_wise_features=False``
path using a local checkpoint. Supports:

* ``full``    — pass the whole matrix to TabImpute in one forward.
* ``chunked`` — split into non-overlapping ``row_chunk`` x ``col_chunk``
                chunks (default 50 x 10) and run each independently.
* ``auto``    — try ``full`` (subject to shape guards) and fall back to
                ``chunked`` on CUDA OOM.

Missing entries are encoded as NaN on input. Observed entries keep their
rating value. The adapter returns predictions at the requested ``test_rows``
/ ``test_cols``.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from .base import BaseMatrixCompleter, MethodResult

# Ensure the tabimpute package is importable. The repo layout puts it at
# /root/tabular/mcpfn/src/tabimpute.
_REPO_SRC = Path("/root/tabular/mcpfn/src")
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))


def _resolve_architecture_file(path: str, basename_hint: Optional[str]) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    # Search the local repo for the basename.
    hint = basename_hint or p.name
    for candidate in _REPO_SRC.rglob(hint):
        return str(candidate)
    raise FileNotFoundError(
        f"architecture_file not found: tried {path} and a local search for {hint}. "
        "Update configs/tabimpute_models.json."
    )


def _resolve_checkpoint(path: str, override: Optional[str]) -> Path:
    if override is not None:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"--checkpoint-override {p} does not exist")
        return p
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Checkpoint path {p} does not exist. Re-run with "
            f"--checkpoint-override <local_path> or update "
            "configs/tabimpute_models.json."
        )
    return p


def _find_training_config(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """Locate a training-time config.json next to the checkpoint, if any."""
    candidate = checkpoint_path.parent / "config.json"
    if candidate.exists():
        return json.loads(candidate.read_text())
    # Fall back to hard-coded BEST_TRANSFER / sensible default.
    return None


def _default_model_config(family: str) -> Dict[str, Any]:
    """Fallback model hyperparameters used when no config.json is present."""
    if family == "new_stable":
        return {
            "embedding_size": 768,
            "num_attention_heads": 16,
            "mlp_hidden_size": 1536,
            "num_layers": 8,
            "num_outputs": 5000,
            "num_cls": 12,
            "use_rope": True,
            "rope_base": 10000.0,
            "rope_fraction": 0.5,
            "use_absolute_positional_embeddings": False,
            "positional_damping_factor": 0.1,
            "attention_dropout": 0.0,
            "ffn_dropout": 0.0,
            "drop_path_rate": 0.0,
            "residual_scale_init": 1.0,
            "rms_norm_eps": 1e-6,
            "embedding_dropout": 0.0,
        }
    # "new"
    return {
        "embedding_size": 768,
        "num_attention_heads": 16,
        "mlp_hidden_size": 1536,
        "num_layers": 8,
        "num_outputs": 5000,
        "num_cls": 12,
        "use_rope": True,
        "rope_base": 10000.0,
        "rope_fraction": 0.5,
        "use_absolute_positional_embeddings": False,
        "positional_damping_factor": 0.1,
    }


class TabImputeAdapter(BaseMatrixCompleter):
    name = "tabimpute"

    def __init__(
        self,
        model_entry: Dict[str, Any],
        eval_mode: str = "auto",
        row_chunk: int = 50,
        col_chunk: int = 10,
        device: str = "cuda",
        checkpoint_override: Optional[str] = None,
        full_rows_limit: int = 400,
        full_cols_limit: int = 400,
    ):
        super().__init__(
            eval_mode=eval_mode, row_chunk=row_chunk, col_chunk=col_chunk, device=device
        )
        if eval_mode not in {"full", "chunked", "auto"}:
            raise ValueError(f"eval_mode must be one of full/chunked/auto, got {eval_mode}")

        self.model_entry = model_entry
        self.eval_mode = eval_mode
        self.row_chunk = int(row_chunk)
        self.col_chunk = int(col_chunk)
        self.device = device
        self.full_rows_limit = int(full_rows_limit)
        self.full_cols_limit = int(full_cols_limit)

        self._ckpt_path = _resolve_checkpoint(model_entry["checkpoint_path"], checkpoint_override)
        self._arch_path = _resolve_architecture_file(
            model_entry.get("architecture_file", ""),
            model_entry.get("architecture_basename"),
        )
        self._family = model_entry.get("model_family", "new_stable")

        # Lazily build the imputer the first time fit_predict is called — this
        # way the adapter can be instantiated cheaply before confirming that
        # torch / CUDA / checkpoint all work.
        self._imputer_full = None
        self._imputer_chunked = None

    # ----------------------------------------------------------- imputers
    def _build_imputer(self, chunked: bool):
        from tabimpute.interface import ImputePFN  # imported here so tests w/o torch still import

        # Priority: manifest model_config > sibling config.json > built-in default.
        model_cfg = self.model_entry.get("model_config")
        if not model_cfg:
            training_cfg = _find_training_config(self._ckpt_path) or {}
            model_cfg = training_cfg.get("model") or _default_model_config(self._family)
        json_config = {"config": model_cfg}

        return ImputePFN(
            device=self.device,
            entry_wise_features=False,
            json_config=json_config,
            model_family=self._family,
            checkpoint_path=str(self._ckpt_path),
            max_num_rows=self.row_chunk if chunked else None,
            max_num_cols=self.col_chunk if chunked else None,
            verbose=False,
        )

    def _get_full(self):
        if self._imputer_full is None:
            self._imputer_full = self._build_imputer(chunked=False)
        return self._imputer_full

    def _get_chunked(self):
        if self._imputer_chunked is None:
            self._imputer_chunked = self._build_imputer(chunked=True)
        return self._imputer_chunked

    # ------------------------------------------------------- fit/predict
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
        X = np.asarray(train_matrix.toarray(), dtype=np.float32)
        M = (X != 0.0)
        X_in = np.where(M, X, np.nan).astype(np.float32)

        # Decide mode
        mode = self.eval_mode
        if mode == "auto":
            if (
                X_in.shape[0] <= self.full_rows_limit
                and X_in.shape[1] <= self.full_cols_limit
            ):
                mode = "full"
            else:
                mode = "chunked"

        used_mode = mode
        X_out, fallback = self._run_with_mode(mode, X_in)
        if X_out is None and self.eval_mode == "auto":
            # OOM fallback path
            used_mode = "chunked"
            X_out, fallback = self._run_with_mode("chunked", X_in)

        if X_out is None:
            raise RuntimeError("TabImpute inference failed for both full and chunked modes")

        preds = X_out[test_rows, test_cols].astype(np.float32)
        # Replace lingering NaNs with global mean fallback.
        global_mean = self._global_mean(train_matrix)
        residual_nan = int(np.isnan(preds).sum())
        preds = np.where(np.isnan(preds), global_mean, preds)
        preds = self._clip_scale(preds, rating_scale)

        return MethodResult(
            predictions=preds,
            fallback_count=fallback + residual_nan,
            extra={
                "eval_mode_requested": self.eval_mode,
                "eval_mode_used": used_mode,
                "row_chunk": self.row_chunk,
                "col_chunk": self.col_chunk,
                "checkpoint": str(self._ckpt_path),
                "model_family": self._family,
            },
        )

    def _run_with_mode(self, mode: str, X_in: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        import torch

        try:
            imputer = self._get_chunked() if mode == "chunked" else self._get_full()
            with torch.no_grad():
                X_imputed = imputer.impute(X_in.copy())
            return X_imputed, 0
        except RuntimeError as err:
            # CUDA OOM gets surfaced as a RuntimeError; fall back in auto.
            msg = str(err).lower()
            if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                if self.eval_mode == "auto" and mode == "full":
                    torch.cuda.empty_cache()
                    return None, 0
            raise


def load_tabimpute_model_entry(name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Look up a model by its normalized name or any of its aliases."""
    normalized = name.strip().lower()
    for entry in config["models"]:
        if entry["name"].lower() == normalized:
            return copy.deepcopy(entry)
        for alias in entry.get("aliases", []) or []:
            if alias.strip().lower() == normalized:
                return copy.deepcopy(entry)
    raise KeyError(
        f"Unknown TabImpute model '{name}'. Known: "
        + ", ".join(e["name"] for e in config["models"])
    )


def build(
    *,
    model_entry: Dict[str, Any],
    eval_mode: str = "auto",
    row_chunk: int = 50,
    col_chunk: int = 10,
    device: str = "cuda",
    checkpoint_override: Optional[str] = None,
    full_rows_limit: int = 400,
    full_cols_limit: int = 400,
    **_: object,
) -> TabImputeAdapter:
    return TabImputeAdapter(
        model_entry=model_entry,
        eval_mode=eval_mode,
        row_chunk=row_chunk,
        col_chunk=col_chunk,
        device=device,
        checkpoint_override=checkpoint_override,
        full_rows_limit=full_rows_limit,
        full_cols_limit=full_cols_limit,
    )
