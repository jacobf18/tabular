from __future__ import annotations

import importlib.resources as resources
import numpy as np
import torch
from typing import Any, Callable, Optional
from huggingface_hub import hf_hub_download

from tabimpute.interface import ImputePFN, _generate_synthetic_low_rank
from tabimpute.model.bar_distribution import FullSupportBarDistribution
from tabimpute.model.model import TabImputeModel


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
        max_num_chunks: int = None,
        verbose: bool = False,
    ):
        if checkpoint_path is None:
            checkpoint_path = get_v2_model_from_huggingface()

        self.device = device

        num_attention_heads = 32
        embedding_size = 32 * num_attention_heads
        mlp_hidden_size = 1024
        num_cls = 12
        num_layers = 12
        self.model = TabImputeModel(
            embedding_size=embedding_size,
            num_attention_heads=num_attention_heads,
            mlp_hidden_size=mlp_hidden_size,
            num_layers=num_layers,
            num_cls=num_cls,
            num_outputs=5000,
        ).to(self.device).to(torch.bfloat16)
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

    def impute_with_test_time_training(
        self,
        X: np.ndarray,
        mask: Optional[np.ndarray] = None,
        k: int = 10,
        optimizer: Optional[torch.optim.Optimizer] = None,
        rank: Optional[int] = None,
        return_full: bool = False,
        means: np.ndarray | None = None,
        stds: np.ndarray | None = None,
    ) -> np.ndarray:
        return super().impute_with_test_time_training(
            X,
            mask=mask,
            k=k,
            optimizer=optimizer,
            rank=rank,
            return_full=return_full,
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
        self, X_normalized: np.ndarray, num_repeats: int = 1
    ) -> np.ndarray:
        if self.max_num_rows is None:
            return self._get_imputation_single(X_normalized, num_repeats=num_repeats)

        X_full = X_normalized.copy()
        start_index = 0

        if self.verbose:
            from tqdm import tqdm

            pbar = tqdm(total=X_normalized.shape[0], desc="Processed rows")

        while start_index < X_normalized.shape[0]:
            end_index = min(
                start_index + (self.max_num_rows * self.max_num_chunks),
                X_normalized.shape[0],
            )
            if self.verbose:
                pbar.update(end_index - start_index)

            X_normalized_chunk = X_normalized[start_index:end_index, :]
            X_normalized_chunk, X_full_chunk = self._get_imputation_chunk(
                X_normalized_chunk, num_repeats=num_repeats
            )
            X_normalized[start_index:end_index, :] = X_normalized_chunk
            X_full[start_index:end_index, :] = X_full_chunk
            start_index = end_index

        return X_normalized, X_full

    def _get_imputation_single(
        self, X_normalized: np.ndarray, num_repeats: int = 1
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
        return X_normalized, X_imputed

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

    def _get_imputation_chunk(
        self, X_normalized: np.ndarray, num_repeats: int = 1
    ) -> np.ndarray:
        if num_repeats > 1:
            raise ValueError("`num_repeats > 1` is not supported in TabImputeV2.")

        row_chunks, start_indices, end_indices = self.split_into_chunks(
            X_normalized, self.max_num_rows
        )
        if len(row_chunks) == 0:
            return X_normalized, X_normalized.copy()

        X_full = X_normalized.copy()
        full_predictions: list[Optional[np.ndarray]] = [None] * len(row_chunks)

        regular_chunks = row_chunks
        regular_indices = list(range(len(row_chunks)))
        last_chunk = None
        last_index = None

        if len(row_chunks) > 1 and row_chunks[-1].shape[0] != row_chunks[-2].shape[0]:
            regular_chunks = row_chunks[:-1]
            regular_indices = list(range(len(row_chunks) - 1))
            last_chunk = row_chunks[-1]
            last_index = len(row_chunks) - 1

        for chunk_idx, full_chunk in zip(
            regular_indices, self._predict_chunks(regular_chunks)
        ):
            full_predictions[chunk_idx] = full_chunk

        if last_chunk is not None and last_index is not None:
            full_predictions[last_index] = self._predict_chunks([last_chunk])[0]

        for i, (start_index, end_index) in enumerate(zip(start_indices, end_indices)):
            full_chunk = full_predictions[i]
            if full_chunk is None:
                raise RuntimeError("Missing full predictions for a TabImputeV2 chunk.")

            missing_mask = np.isnan(X_normalized[start_index:end_index, :])
            X_normalized[start_index:end_index, :][missing_mask] = full_chunk[
                missing_mask
            ]
            X_full[start_index:end_index, :] = full_chunk

        return X_normalized, X_full

    def _get_imputation_single_ttt(
        self,
        X_normalized: np.ndarray,
        mask: Optional[np.ndarray] = None,
        k: int = 10,
        optimizer: Optional[torch.optim.Optimizer] = None,
        rank: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single imputation with test-time training on synthetic low-rank data.

        TabImputeV2 version: uses matrix-based forward (B, R, C) instead of
        train/test set format.
        """
        n_rows, n_cols = X_normalized.shape
        if mask is None:
            mask = ~np.isnan(X_normalized)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (n_rows, n_cols):
                raise ValueError(
                    f"mask shape {mask.shape} must match X shape ({n_rows}, {n_cols})"
                )

        if rank is None:
            rank = min(n_rows, n_cols, 10)

        n_missing = np.sum(~mask)
        if n_missing == 0:
            X_full = X_normalized.copy()
            return X_normalized.copy(), X_full

        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        try:
            self.model.train()
            for _ in range(k):
                Y_syn = _generate_synthetic_low_rank(n_rows, n_cols, rank, self.borders)
                X_syn = Y_syn.copy()
                X_syn[~mask] = np.nan

                X_syn_tensor = (
                    torch.from_numpy(X_syn).to(self.device).unsqueeze(0).to(torch.bfloat16)
                )
                Y_syn_tensor = (
                    torch.from_numpy(Y_syn).to(self.device).unsqueeze(0).to(torch.bfloat16)
                )
                # Loss only on missing: set observed to NaN so bar_distribution ignores them
                mask_t = torch.from_numpy(mask).to(self.device)
                loss_y = torch.where(
                    mask_t.unsqueeze(0).expand_as(Y_syn_tensor),
                    torch.full_like(Y_syn_tensor, float("nan")),
                    Y_syn_tensor,
                )

                optimizer.zero_grad()
                preds = self.model(X_syn_tensor)
                preds = self._postprocess_logits(preds)
                loss = self.bar_distribution(logits=preds, y=loss_y)
                missing_mask_t = ~mask_t.unsqueeze(0).expand(1, n_rows, n_cols)
                missing_loss = loss[missing_mask_t].mean()
                missing_loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                X_orig_tensor = (
                    torch.from_numpy(X_normalized.copy())
                    .to(self.device)
                    .unsqueeze(0)
                    .to(torch.bfloat16)
                )
                preds = self.model(X_orig_tensor)
                preds = self._postprocess_logits(preds)
                medians = self.bar_distribution.median(logits=preds).squeeze(0)
                X_imputed = medians.cpu().detach().numpy()

            X_normalized_out = X_normalized.copy()
            X_normalized_out[~mask] = X_imputed[~mask]
            X_full = X_imputed.copy()
            X_full[mask] = X_normalized[mask]

            return X_normalized_out, X_full
        finally:
            self.model.load_state_dict(self.checkpoint["model"], strict=False)


if __name__ == "__main__":
    import os
    import sys
    import time
    import numpy as np

    checkpoint_path = (
        os.environ.get("TABIMPUTE_V2_CHECKPOINT")
        or (sys.argv[1] if len(sys.argv) > 1 else None)
    )

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
    
    tabimpute_v2 = TabImputeV2(device=device, checkpoint_path=checkpoint_path)
    X_imputed = tabimpute_v2.impute(X.copy())
    print(X_imputed)
