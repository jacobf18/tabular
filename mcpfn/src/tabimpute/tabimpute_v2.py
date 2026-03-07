from __future__ import annotations

import importlib.resources as resources
import numpy as np
import torch
from typing import Any, Optional
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
        self.max_num_rows = max_num_rows
        if max_num_chunks is None:
            max_num_chunks = float("inf")
        self.max_num_chunks = max_num_chunks
        borders = self.borders.to(self.device)
        self.bar_distribution = FullSupportBarDistribution(borders=borders)
        self.verbose = verbose

    def get_imputation(
        self, X_normalized: np.ndarray, num_repeats: int = 1
    ) -> np.ndarray:
        if self.max_num_rows is not None:
            raise ValueError("`max_num_rows` is not supported in TabImputeV2.")
        return self._get_imputation_single(X_normalized, num_repeats=num_repeats)

    def _get_imputation_single(
        self, X_normalized: np.ndarray, num_repeats: int = 1
    ) -> np.ndarray:
        if num_repeats > 1:
            raise ValueError("`num_repeats > 1` is not supported in TabImputeV2.")

        X_normalized_tensor = torch.from_numpy(X_normalized).to(self.device)
        with torch.no_grad():
            X_normalized_tensor = X_normalized_tensor.unsqueeze(0).to(torch.bfloat16)
            preds = self.model(X_normalized_tensor)
            medians = self.bar_distribution.median(logits=preds).squeeze(0)

        X_imputed = medians.cpu().detach().numpy()
        missing_mask = np.isnan(X_normalized)
        X_normalized[missing_mask] = X_imputed[missing_mask]
        return X_normalized, X_imputed

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
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

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

    checkpoint_path = (
        os.environ.get("TABIMPUTE_V2_CHECKPOINT")
        or (sys.argv[1] if len(sys.argv) > 1 else None)
        or os.path.join(
            os.path.dirname(__file__),
            "workdir",
            "tabimpute-mcar_p0.4-num_cls_12-rank_1_11",
            "checkpoint_85000.pth",
        )
    )
    if not os.path.exists(checkpoint_path):
        print(
            f"TabImputeV2 requires a checkpoint. Set TABIMPUTE_V2_CHECKPOINT env var "
            f"or pass path as first arg: python -m tabimpute.tabimpute_v2 <path>"
        )
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    imputer = TabImputeV2(device=device, checkpoint_path=checkpoint_path)
    print(f"Model size: {sum(p.numel() for p in imputer.model.parameters()):,}")

    X = np.random.randn(50, 8)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X[np.random.rand(*X.shape) < 0.1] = np.nan

    start_time = time.time()
    out, full = imputer.impute(X, return_full=True)
    end_time = time.time()
    print(f"Standard impute time: {end_time - start_time:.4f} seconds")

    X_ttt = np.random.randn(40, 6)
    X_ttt = (X_ttt - X_ttt.mean(axis=0)) / X_ttt.std(axis=0)
    X_ttt[np.random.rand(*X_ttt.shape) < 0.15] = np.nan
    n_missing = np.isnan(X_ttt).sum()
    print(
        f"\nTTT test: {X_ttt.shape[0]}x{X_ttt.shape[1]} matrix, {n_missing} missing values"
    )
    start_time = time.time()
    out_ttt, full_ttt = imputer.impute_with_test_time_training(
        X_ttt, k=5, return_full=True
    )
    end_time = time.time()
    print(f"TTT impute time: {end_time - start_time:.4f} seconds")
    print(f"TTT output shape: {out_ttt.shape}, any NaN: {np.isnan(out_ttt).any()}")

    # Test TTT with preprocessors
    from tabimpute.prepreocess import RandomRowColumnPermutation

    preprocessors = [RandomRowColumnPermutation(), RandomRowColumnPermutation()]
    imputer_pp = TabImputeV2(
        device=device, checkpoint_path=checkpoint_path, preprocessors=preprocessors
    )
    X_ttt_pp = np.random.randn(35, 6)
    X_ttt_pp = (X_ttt_pp - X_ttt_pp.mean(axis=0)) / X_ttt_pp.std(axis=0)
    X_ttt_pp[np.random.rand(*X_ttt_pp.shape) < 0.12] = np.nan
    n_missing_pp = np.isnan(X_ttt_pp).sum()
    print(
        f"\nTTT with preprocessors: {X_ttt_pp.shape[0]}x{X_ttt_pp.shape[1]} matrix, "
        f"{n_missing_pp} missing, {len(preprocessors)} preprocessors"
    )
    start_time = time.time()
    out_ttt_pp, full_ttt_pp = imputer_pp.impute_with_test_time_training(
        X_ttt_pp, k=3, return_full=True
    )
    end_time = time.time()
    print(f"TTT+preprocessors time: {end_time - start_time:.4f} seconds")
    print(
        f"TTT+preprocessors output shape: {out_ttt_pp.shape}, "
        f"any NaN: {np.isnan(out_ttt_pp).any()}"
    )

    # Test TTT with matrix larger than default max_len (100) to exercise positional
    # extrapolation and verify load_state_dict restore works (previously failed with
    # "size mismatch for feature_encoder.row_embedding.pe" when pe was mutated)
    X_large = np.random.randn(214, 8)
    X_large = (X_large - X_large.mean(axis=0)) / (X_large.std(axis=0) + 1e-8)
    X_large[np.random.rand(*X_large.shape) < 0.2] = np.nan
    print(
        f"\nTTT large (>{100} rows): {X_large.shape[0]}x{X_large.shape[1]} matrix, "
        f"{np.isnan(X_large).sum()} missing"
    )
    out_large, _ = imputer.impute_with_test_time_training(
        X_large.copy(), k=3, return_full=True
    )
    print(f"TTT large output shape: {out_large.shape}, any NaN: {np.isnan(out_large).any()}")
