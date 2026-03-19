from __future__ import annotations

import importlib.resources as resources
import numpy as np
import torch
from typing import Any, Callable, Optional
from huggingface_hub import hf_hub_download

from tabimpute.interface import ImputePFN, _generate_synthetic_low_rank
from tabimpute.model.bar_distribution import FullSupportBarDistribution
# from tabimpute.model.model import TabImputeModel
from tabimpute.model.model_new_stable import TabImputeModel


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
        self,
        X_normalized: np.ndarray,
        num_repeats: int = 1,
        return_full: bool = True,
    ) -> np.ndarray:
        if self.max_num_rows is None:
            return self._get_imputation_single(
                X_normalized,
                num_repeats=num_repeats,
                return_full=return_full,
            )

        X_full = X_normalized.copy() if return_full else None
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
        if return_full:
            return X_normalized, X_imputed
        return X_normalized

    def _predict_chunks(self, chunks: list[np.ndarray]) -> torch.Tensor:
        if len(chunks) == 0:
            return torch.empty(0)

        chunk_batch = torch.stack(
            [torch.from_numpy(chunk) for chunk in chunks], dim=0
        ).to(self.device)

        with torch.no_grad():
            chunk_batch = chunk_batch.to(torch.bfloat16)
            preds = self.model(chunk_batch)
            preds = self._postprocess_logits(preds)
            medians = self.bar_distribution.median(logits=preds)

        return medians.cpu()

    def _get_imputation_chunk(
        self,
        X_normalized: np.ndarray,
        num_repeats: int = 1,
        return_full: bool = True,
    ) -> np.ndarray:
        if num_repeats > 1:
            raise ValueError("`num_repeats > 1` is not supported in TabImputeV2.")

        row_chunks, start_indices, end_indices = self.split_into_chunks(
            X_normalized, self.max_num_rows
        )
        if len(row_chunks) == 0:
            if return_full:
                return X_normalized, X_normalized.copy()
            return X_normalized

        X_full = X_normalized.copy() if return_full else None
        split_last_chunk = (
            len(row_chunks) > 1 and row_chunks[-1].shape[0] != row_chunks[-2].shape[0]
        )
        regular_chunk_count = len(row_chunks) - int(split_last_chunk)

        if regular_chunk_count > 0:
            regular_predictions = self._predict_chunks(row_chunks[:regular_chunk_count])
            for full_chunk, start_index, end_index in zip(
                regular_predictions,
                start_indices[:regular_chunk_count],
                end_indices[:regular_chunk_count],
            ):
                chunk_slice = X_normalized[start_index:end_index, :]
                full_chunk_np = full_chunk.numpy()
                missing_mask = np.isnan(chunk_slice)
                chunk_slice[missing_mask] = full_chunk_np[missing_mask]
                if X_full is not None:
                    X_full[start_index:end_index, :] = full_chunk_np

        if split_last_chunk:
            last_prediction = self._predict_chunks([row_chunks[-1]])[0].numpy()
            chunk_slice = X_normalized[start_indices[-1] : end_indices[-1], :]
            missing_mask = np.isnan(chunk_slice)
            chunk_slice[missing_mask] = last_prediction[missing_mask]
            if X_full is not None:
                X_full[start_indices[-1] : end_indices[-1], :] = last_prediction

        if return_full:
            return X_normalized, X_full
        return X_normalized

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

    def _get_imputation_single_acs_ttt(
        self,
        X_normalized: np.ndarray,
        mask: Optional[np.ndarray] = None,
        k: int = 10,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """TTT using the ACS region (fully-observed rows) as real training data.

        Instead of synthetic low-rank matrices, adapts the model to the actual
        data distribution using the ACS patches — the rows where every feature
        is observed.  For each gradient step a random column-wise (feature-wise)
        missing mask is sampled whose per-feature drop probability matches the
        actual missingness rate seen in non-ACS rows.  This teaches the model to
        recover masked features from the joint ACS distribution before it is
        asked to impute the full matrix.

        Args:
            X_normalized: Normalized matrix of shape (n_patches, n_features)
                with NaN for missing entries.  Rows = patches, cols = S-matrix
                features (i.e. already transposed for tab_impute).
            mask: Boolean array (True = observed).  Inferred from X_normalized
                if None.
            k: Number of gradient steps.
            optimizer: Optimizer for fine-tuning.  Defaults to AdamW lr=1e-5.

        Returns:
            (X_imputed, X_full) both of shape (n_patches, n_features).
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

        # Identify ACS rows: every feature is observed.
        acs_row_mask = np.all(mask, axis=1)
        X_acs = X_normalized[acs_row_mask].astype(np.float32)
        n_acs = X_acs.shape[0]

        n_missing = np.sum(~mask)
        if n_missing == 0 or n_acs == 0:
            X_full = X_normalized.copy()
            return X_normalized.copy(), X_full

        # Per-feature missing probability from non-ACS rows.
        non_acs_mask = mask[~acs_row_mask]  # (n_non_acs, n_cols)
        if non_acs_mask.shape[0] > 0:
            feature_missing_prob = 1.0 - non_acs_mask.mean(axis=0)  # (n_cols,)
        else:
            feature_missing_prob = np.full(n_cols, 0.3, dtype=np.float64)

        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        # Determine training batch size so we stay within max_num_rows.
        batch_rows = n_acs
        if self.max_num_rows is not None and n_acs > self.max_num_rows:
            batch_rows = self.max_num_rows

        try:
            self.model.train()
            for _ in range(k):
                # Subsample ACS rows if needed.
                if batch_rows < n_acs:
                    row_idx = np.random.choice(n_acs, size=batch_rows, replace=False)
                    Y_batch = X_acs[row_idx]
                else:
                    Y_batch = X_acs

                # Sample a column-wise missing mask matching the actual drop rate.
                feature_missing = np.random.rand(n_cols) < feature_missing_prob
                # Guard: need at least one missing and one observed column.
                if not feature_missing.any():
                    feature_missing[np.random.randint(n_cols)] = True
                if feature_missing.all():
                    feature_missing[np.random.randint(n_cols)] = False

                X_batch = Y_batch.copy()
                X_batch[:, feature_missing] = np.nan

                X_batch_t = (
                    torch.from_numpy(X_batch).to(self.device).unsqueeze(0).to(torch.bfloat16)
                )
                Y_batch_t = (
                    torch.from_numpy(Y_batch).to(self.device).unsqueeze(0).to(torch.bfloat16)
                )

                # Compute loss only on the artificially masked (missing) features.
                feat_miss_t = (
                    torch.from_numpy(feature_missing)
                    .to(self.device)
                    .view(1, 1, n_cols)
                    .expand_as(Y_batch_t)
                )
                loss_y = torch.where(
                    feat_miss_t,
                    Y_batch_t,
                    torch.full_like(Y_batch_t, float("nan")),
                )

                optimizer.zero_grad()
                preds = self.model(X_batch_t)
                preds = self._postprocess_logits(preds)
                loss = self.bar_distribution(logits=preds, y=loss_y)
                missing_loss = loss[feat_miss_t].mean()
                missing_loss.backward()
                optimizer.step()

            # Inference on the full matrix.
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

    def impute_with_acs_ttt(
        self,
        X: np.ndarray,
        mask: Optional[np.ndarray] = None,
        k: int = 10,
        optimizer: Optional[torch.optim.Optimizer] = None,
        return_full: bool = False,
        means: np.ndarray | None = None,
        stds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Impute with test-time training conditioned on the ACS region.

        Fine-tunes the model for ``k`` steps on the fully-observed rows of
        ``X`` (the ACS patches) before imputing the full matrix.  Each training
        step masks a random subset of features according to the actual per-feature
        missingness rate observed in non-ACS rows, so the model learns to
        recover exactly the kind of missing structure it will face at inference.

        Args:
            X: Input matrix of shape (n_patches, n_features) with NaN for
                missing entries (already transposed for tab_impute convention).
            mask: Boolean mask; True = observed.  Inferred from ~np.isnan(X)
                if None.
            k: Number of gradient steps for test-time training.
            optimizer: Optimizer for fine-tuning.  Defaults to AdamW lr=1e-5.
            return_full: If True, return (X_imputed, X_full); else X_imputed.
            means: Optional per-column means for normalization.
            stds: Optional per-column standard deviations for normalization.

        Returns:
            Imputed matrix, or (X_imputed, X_full) if return_full is True.
        """
        if X.ndim != 2:
            raise ValueError("Input matrix must be 2-dimensional.")

        means, stds = self._resolve_normalization_stats(X, means=means, stds=stds)
        X_normalized = (X - means) / (stds + 1e-16)

        X_imputed, X_full = self._get_imputation_single_acs_ttt(
            X_normalized, mask=mask, k=k, optimizer=optimizer
        )

        X_imputed = X_imputed * (stds + 1e-16) + means
        X_full = X_full * (stds + 1e-16) + means

        if return_full:
            return X_imputed, X_full
        return X_imputed


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
