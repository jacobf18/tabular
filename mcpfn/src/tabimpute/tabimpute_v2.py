from __future__ import annotations

import importlib.resources as resources
import numpy as np
import torch
from typing import Any, Callable, Optional
from huggingface_hub import hf_hub_download

from tabimpute.interface import ImputePFN
from tabimpute.model.bar_distribution import FullSupportBarDistribution
# from tabimpute.model.model import TabImputeModel
from tabimpute.model.model_new_stable import TabImputeModel
from tqdm import tqdm


def get_v2_model_from_huggingface() -> str:
    repo_id = "Tabimpute/TabImpute"
    filename = "tabimputev2.ckpt"
    return hf_hub_download(repo_id=repo_id, filename=filename)


class TabImputeV2(ImputePFN):
    """TabImpute V2 using the non-entry-wise-features model."""

    def __init__(
        self,
        device: str = "cpu",
        nhead: int = 2,
        preprocessors: list[Any] = None,
        postprocessor: Optional[Callable[..., torch.Tensor]] = None,
        postprocessor_kwargs: Optional[dict[str, Any]] = None,
        checkpoint_path: str = None,
        max_num_rows: int = None,
        max_num_cols: int = None,
        max_num_chunks: int = None,
        verbose: bool = False,
    ):
        if checkpoint_path is None:
            checkpoint_path = get_v2_model_from_huggingface()

        self.device = device
        
        model_cfg = {
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
            "ffn_dropout": 0.03,
            "drop_path_rate": 0.0,
            "residual_scale_init": 0.7,
            "embedding_dropout": 0.0,
            "rms_norm_eps": 1e-6,
        }

        self.model = TabImputeModel(**model_cfg).to(self.device).to(torch.bfloat16)
        self.model.eval()

        with resources.files("tabimpute.data").joinpath("borders.pt").open("rb") as f:
            self.borders = torch.load(f, map_location=self.device)

        self.checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(self.checkpoint["model"], strict=False)

        self.preprocessors = preprocessors
        self.postprocessor = postprocessor
        self.postprocessor_kwargs = postprocessor_kwargs or {}
        self.max_num_rows = max_num_rows
        self.max_num_cols = max_num_cols
        if max_num_chunks is None:
            max_num_chunks = float("inf")
        self.max_num_chunks = max_num_chunks
        borders = self.borders.to(self.device)
        self.bar_distribution = FullSupportBarDistribution(borders=borders)
        self.verbose = verbose

    def impute(
        self,
        X: np.ndarray,
        return_full: bool = False,
        num_repeats: int = 1,
        means: np.ndarray | None = None,
        stds: np.ndarray | None = None,
    ) -> np.ndarray:
        return super().impute(
            X,
            return_full=return_full,
            num_repeats=num_repeats,
            means=means,
            stds=stds,
        )

    def _postprocess_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.postprocessor is None:
            return logits

        return self.postprocessor(
            logits=logits,
            borders=self.borders,
            **self.postprocessor_kwargs,
        )

    def get_imputation(
        self,
        X_normalized: np.ndarray,
        num_repeats: int = 1,
        return_full: bool = True,
    ) -> np.ndarray:
        if self.max_num_rows is None and self.max_num_cols is None:
            return self._get_imputation_single(
                X_normalized,
                num_repeats=num_repeats,
                return_full=return_full,
            )

        X_full = X_normalized.copy() if return_full else None
        start_index = 0
        if self.max_num_rows is None or np.isinf(self.max_num_chunks):
            row_window_size = X_normalized.shape[0]
        else:
            row_window_size = self.max_num_rows * int(self.max_num_chunks)

        if self.verbose:
            pbar = tqdm(total=X_normalized.shape[0], desc="Processed rows")

        while start_index < X_normalized.shape[0]:
            end_index = min(start_index + row_window_size, X_normalized.shape[0])
            if self.verbose:
                pbar.update(end_index - start_index)

            X_normalized_chunk = X_normalized[start_index:end_index, :]
            chunk_result = self._get_imputation_chunk(
                X_normalized_chunk,
                num_repeats=num_repeats,
                return_full=return_full,
            )
            if return_full:
                X_normalized_chunk, X_full_chunk = chunk_result
                X_full[start_index:end_index, :] = X_full_chunk
            else:
                X_normalized_chunk = chunk_result
            X_normalized[start_index:end_index, :] = X_normalized_chunk
            start_index = end_index

        if return_full:
            return X_normalized, X_full
        return X_normalized

    def _get_imputation_single(
        self,
        X_normalized: np.ndarray,
        num_repeats: int = 1,
        return_full: bool = True,
        return_logits: bool = False,
    ) -> np.ndarray:
        if num_repeats > 1:
            raise ValueError("`num_repeats > 1` is not supported in TabImputeV2.")

        X_normalized_tensor = torch.from_numpy(X_normalized).to(self.device)
        with torch.no_grad():
            X_normalized_tensor = X_normalized_tensor.unsqueeze(0).to(torch.bfloat16)
            preds = self.model(X_normalized_tensor)
            preds = self._postprocess_logits(preds)
            medians = self.bar_distribution.median(logits=preds).squeeze(0)

        X_imputed = medians.cpu().detach().numpy()
        missing_mask = np.isnan(X_normalized)
        X_normalized[missing_mask] = X_imputed[missing_mask]
        if return_full and return_logits:
            return X_normalized, X_imputed, preds
        elif return_full:
            return X_normalized, X_imputed
        elif return_logits:
            return preds
        else:
            return X_normalized
        
    def get_embeddings(self, X_normalized: np.ndarray) -> np.ndarray:
        X_normalized_tensor = torch.from_numpy(X_normalized).to(self.device)
        with torch.no_grad():
            X_normalized_tensor = X_normalized_tensor.unsqueeze(0).to(torch.bfloat16)
            embeddings = self.model.embeddings(X_normalized_tensor)
            return embeddings.cpu().detach().numpy()

    def _predict_chunks(self, chunks: list[np.ndarray]) -> list[np.ndarray]:
        if len(chunks) == 0:
            return []

        chunk_batch = torch.stack(
            [torch.from_numpy(chunk) for chunk in chunks], dim=0
        ).to(self.device)

        with torch.no_grad():
            chunk_batch = chunk_batch.to(torch.bfloat16)
            preds = self.model(chunk_batch)
            preds = self._postprocess_logits(preds)
            medians = self.bar_distribution.median(logits=preds)

        return [median.cpu().detach().numpy() for median in medians]

    def _get_axis_ranges(
        self, axis_length: int, chunk_size: Optional[int]
    ) -> list[tuple[int, int]]:
        if axis_length <= 0:
            return []
        if chunk_size is None or chunk_size >= axis_length:
            return [(0, axis_length)]
        return [
            (start_index, min(start_index + chunk_size, axis_length))
            for start_index in range(0, axis_length, chunk_size)
        ]

    def _apply_chunk_prediction(
        self,
        X_normalized: np.ndarray,
        X_full: Optional[np.ndarray],
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        full_chunk: np.ndarray,
    ) -> None:
        chunk_slice = X_normalized[row_start:row_end, col_start:col_end]
        missing_mask = np.isnan(chunk_slice)
        chunk_slice[missing_mask] = full_chunk[missing_mask]
        if X_full is not None:
            X_full[row_start:row_end, col_start:col_end] = full_chunk

    def _get_imputation_chunk(
        self, X_normalized: np.ndarray, num_repeats: int = 1, return_full: bool = True,
    ) -> np.ndarray:
        if num_repeats > 1:
            raise ValueError("`num_repeats > 1` is not supported in TabImputeV2.")

        row_chunk_size = self.max_num_rows or X_normalized.shape[0]
        row_chunks, start_indices, end_indices = self.split_into_chunks(
            X_normalized, row_chunk_size
        )
        col_ranges = self._get_axis_ranges(X_normalized.shape[1], self.max_num_cols)
        if len(row_chunks) == 0:
            if return_full:
                return X_normalized, X_normalized.copy()
            return X_normalized
        if len(col_ranges) == 0:
            if return_full:
                return X_normalized, X_normalized.copy()
            return X_normalized

        X_full = X_normalized.copy() if return_full else None
        split_last_chunk = (
            len(row_chunks) > 1 and row_chunks[-1].shape[0] != row_chunks[-2].shape[0]
        )
        regular_chunks = row_chunks[:-1] if split_last_chunk else row_chunks
        regular_indices = list(range(len(regular_chunks)))

        for col_start, col_end in col_ranges:
            regular_tiles = [chunk[:, col_start:col_end] for chunk in regular_chunks]
            if regular_tiles:
                for chunk_idx, full_chunk in zip(
                    regular_indices, self._predict_chunks(regular_tiles)
                ):
                    self._apply_chunk_prediction(
                        X_normalized,
                        X_full,
                        start_indices[chunk_idx],
                        end_indices[chunk_idx],
                        col_start,
                        col_end,
                        full_chunk,
                    )

            if split_last_chunk:
                last_prediction = self._predict_chunks(
                    [row_chunks[-1][:, col_start:col_end]]
                )[0]
                self._apply_chunk_prediction(
                    X_normalized,
                    X_full,
                    start_indices[-1],
                    end_indices[-1],
                    col_start,
                    col_end,
                    last_prediction,
                )

        if return_full:
            if X_full is None:
                raise RuntimeError("Missing full predictions for a TabImputeV2 chunk.")
            return X_normalized, X_full
        return X_normalized

if __name__ == "__main__":
    import os
    import sys
    import time
    import numpy as np

    checkpoint_path = "/home/jacobf18/tabular/mcpfn/src/tabimpute/workdir/tabimpute-round2-t2/checkpoint_10000.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if checkpoint_path is not None:
        print(f"Using checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint provided; using Hugging Face fallback.")

    X = np.random.randn(7, 5).astype(np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    X[np.random.rand(*X.shape) < 0.2] = np.nan
    print(X)
