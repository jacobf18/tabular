from __future__ import annotations

import torch

from tabimpute.prior.training_set_generation import create_train_test_sets

import numpy as np
from tabimpute.model.mcpfn import MCPFN
import argparse
from tabimpute.model.bar_distribution import FullSupportBarDistribution

# from tabpfn import TabPFNRegressor
from tabimpute.prepreocess import (
    RandomRowPermutation,
    RandomColumnPermutation,
    RandomRowColumnPermutation,
    Preprocess,
    PowerTransform,
    standardize_excluding_outliers_torch,
)
from sklearn.linear_model import LinearRegression
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised
import importlib.resources as resources
from huggingface_hub import hf_hub_download
from scipy.optimize import minimize
from scipy.special import softmax
from tabimpute.model.encoders import normalize_data
from tabimpute.model.model_new import TabImputeModel as TabImputeModelNew
from tabimpute.model.model_new_stable import (
    TabImputeModel as TabImputeModelStable,
    build_best_transfer_model,
)
from tabimpute.model.model_new_stable_memory_efficient import (
    TabImputeModel as TabImputeModelMemoryEfficient,
)
from tabimpute.model.model_compressed_cross import (
    ColumnCompressionModel,
    CompressedCrossTabImputeModel,
    RowCompressionModel,
    build_compressor_config_from_predictor,
)
from tabimpute.model.model_compressed_cross_rabel import (
    ColumnCompressionModelRaBEL,
    CompressedCrossTabImputeModelRaBEL,
    RowCompressionModelRaBEL,
    TabImputeModelRaBEL,
)
from tabimpute.model.model_compressed_cross_pure_rabel import (
    ColumnCompressionModelPureRaBEL,
    CompressedCrossTabImputeModelPureRaBEL,
    RowCompressionModelPureRaBEL,
)
from tabimpute.model.model_new_stable_rabel import (
    TabImputeModelPureRaBEL,
)

import matplotlib.pyplot as plt


def _load_json_config_from_checkpoint(checkpoint_path: str) -> dict:
    from pathlib import Path
    import json

    cfg_path = Path(checkpoint_path).parent / "config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "model" in cfg:
        return dict(cfg["model"])
    if "config" in cfg:
        return dict(cfg["config"])
    return dict(cfg)

def get_model_from_huggingface() -> str:
    repo_id = "Tabimpute/TabImpute"
    filename = "tabimpute_001.ckpt"
    return hf_hub_download(repo_id=repo_id, filename=filename)


def calibrate_predictions(
    y_val, y_pred_val, pred_sigma_val, y_pred_test, pred_sigma_test
):
    """
    Calibrate mean and std of predictive distribution.
    Returns calibrated test means and stds.
    """
    # ---- Variance calibration ----
    resid = y_val - y_pred_val
    std_resid = np.std(resid)
    mean_sigma = np.mean(pred_sigma_val)
    alpha = std_resid / (mean_sigma + 1e-12)

    sigma_cal_test = pred_sigma_test * alpha

    # ---- Linear correction for means ----
    reg = LinearRegression().fit(y_pred_val.reshape(-1, 1), y_val)
    a, b = reg.coef_[0], reg.intercept_
    mu_cal_test = a * y_pred_test + b

    return mu_cal_test, sigma_cal_test, (a, b, alpha)


class ImputePFN:
    """A Tabular Foundation Model for Matrix Completion.

    MCPFN is a transformer-based architecture for matrix completion on tabular data.

    Parameters
    ----------
    """

    def __init__(
        self,
        device: str = "cpu",
        nhead: int = 2,
        preprocessors: list[Preprocess] = None,
        checkpoint_path: str = None,
        max_num_rows: int = None,
        max_num_cols: int = None,
        max_num_chunks: int = None,
        verbose: bool = False,
        entry_wise_features: bool = True,
        json_config: dict = None,
        model_family: str = "new_stable",
        use_ema: bool = False,
    ):
        self.device = device
        self.model_family = model_family

        # Build model
        if entry_wise_features:
            self.model = MCPFN(nhead=nhead).to(self.device).to(torch.bfloat16)
        else:
            self.model = self._build_direct_model(json_config=json_config).to(
                self.device, dtype=torch.bfloat16
            )
            self.model.eval()
        # torch.compile(self.model)

        # Load borders tensor for outputting continuous values
        with resources.files("tabimpute.data").joinpath("borders.pt").open("rb") as f:
            self.borders = torch.load(f, map_location=self.device)

        if entry_wise_features:
            if checkpoint_path is None:
                checkpoint_path = get_model_from_huggingface()

            # Load model state dict
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )

            state_dict = self._select_state_dict(
                checkpoint, use_ema=use_ema, checkpoint_path=checkpoint_path
            )
            self.model.load_state_dict(state_dict)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model'], strict=True)

        self.preprocessors = preprocessors
        self.max_num_rows = max_num_rows
        self.max_num_cols = max_num_cols
        if max_num_chunks is None:
            max_num_chunks = float('inf')
        self.max_num_chunks = max_num_chunks
        # Get the median predictions
        borders = self.borders.to(self.device)
        self.bar_distribution = FullSupportBarDistribution(borders=borders)
        self.verbose = verbose
        self.entry_wise_features = entry_wise_features

    def set_attention_capture(
        self, enabled: bool = True, store_weights: bool = True
    ) -> None:
        if not hasattr(self.model, "set_attention_capture"):
            raise NotImplementedError(
                "Attention capture is implemented for direct TabImpute models "
                "(entry_wise_features=False)."
            )
        self.model.set_attention_capture(enabled, store_weights=store_weights)

    def get_attention_diagnostics(self) -> list[dict[str, object]]:
        if not hasattr(self.model, "get_attention_diagnostics"):
            raise NotImplementedError(
                "Attention diagnostics are implemented for direct TabImpute models "
                "(entry_wise_features=False)."
            )
        return self.model.get_attention_diagnostics()

    def diagnose_attention(
        self,
        X: np.ndarray,
        store_weights: bool = False,
        means: np.ndarray | None = None,
        stds: np.ndarray | None = None,
    ) -> list[dict[str, object]]:
        """Run one direct-model pass and return per-layer attention weights/stats."""
        if self.entry_wise_features:
            raise NotImplementedError(
                "diagnose_attention is intended for the direct mat-to-mat model. "
                "Construct ImputePFN with entry_wise_features=False."
            )
        if X.ndim != 2:
            raise ValueError("Input matrix must be 2-dimensional")

        if means is None:
            means = np.nanmean(X, axis=0)
        else:
            means = np.asarray(means, dtype=np.float64)
            if means.shape != (X.shape[1],):
                raise ValueError(
                    f"means must have shape ({X.shape[1]},), got {means.shape}"
                )

        if stds is None:
            stds = np.nanstd(X, axis=0)
        else:
            stds = np.asarray(stds, dtype=np.float64)
            if stds.shape != (X.shape[1],):
                raise ValueError(
                    f"stds must have shape ({X.shape[1]},), got {stds.shape}"
                )

        stds = np.where(np.isnan(stds), 1, stds)
        means = np.where(np.isnan(means), 0, means)
        X_normalized = (X - means) / (stds + 1e-16)
        X_tensor = (
            torch.from_numpy(X_normalized)
            .unsqueeze(0)
            .to(self.device)
            .to(torch.bfloat16)
        )

        self.set_attention_capture(True, store_weights=store_weights)
        with torch.no_grad():
            self.model(X_tensor)
        diagnostics = self.get_attention_diagnostics()
        self.set_attention_capture(False)
        return diagnostics

    @staticmethod
    def _select_state_dict(
        checkpoint: dict, use_ema: bool, checkpoint_path: str
    ) -> dict:
        """Pick the parameter dict to load, preferring EMA weights when requested.

        Handles three checkpoint shapes:
          - best.ckpt with `ema_applied=True` and `state_dict` already = EMA weights.
          - step-*.ckpt with `state_dict` = live weights and
            `ema_state["shadow"]` = the EMA shadow keyed by parameter name.
          - any checkpoint with no EMA info (falls back to `state_dict`).

        For (2), the EMA shadow is overlaid onto the live state_dict so that
        non-parameter buffers are preserved.
        """
        if "state_dict" not in checkpoint:
            raise ValueError(
                f"Checkpoint at {checkpoint_path} has no 'state_dict' key."
            )
        live_state_dict = checkpoint["state_dict"]

        if not use_ema:
            return live_state_dict

        if checkpoint.get("ema_applied"):
            # state_dict already contains EMA weights (best.ckpt path).
            return live_state_dict

        ema_state = checkpoint.get("ema_state")
        if ema_state is None or "shadow" not in ema_state:
            raise ValueError(
                f"use_ema=True but checkpoint at {checkpoint_path} has no EMA "
                "shadow (no 'ema_state' or 'ema_applied' marker). Train with "
                "--ema_enabled true or pass use_ema=False."
            )

        shadow = ema_state["shadow"]
        merged = {**live_state_dict}
        for name, tensor in shadow.items():
            if name in merged:
                merged[name] = tensor.to(dtype=merged[name].dtype)
            else:
                merged[name] = tensor
        return merged

    def _get_direct_model_config(self, json_config: dict | None) -> dict:
        if json_config is None:
            raise ValueError(
                "json_config must be provided when entry_wise_features=False"
            )

        if "config" in json_config:
            return dict(json_config["config"])
        return dict(json_config)

    def _build_direct_model(self, json_config: dict | None):
        config = self._get_direct_model_config(json_config)
        model_family = self.model_family.lower()

        common_kwargs = {
            "embedding_size": config["embedding_size"],
            "num_attention_heads": config["num_attention_heads"],
            "mlp_hidden_size": config["mlp_hidden_size"],
            "num_layers": config["num_layers"],
            "num_outputs": config.get("num_outputs", 5000),
            "num_cls": config["num_cls"],
            "use_rope": config.get("use_rope", True),
            "rope_base": config.get("rope_base", 10000.0),
            "rope_fraction": config.get("rope_fraction", 1.0),
            "use_absolute_positional_embeddings": config.get(
                "use_absolute_positional_embeddings", False
            ),
            "positional_damping_factor": config.get(
                "positional_damping_factor", 0.1
            ),
        }

        if model_family in {"new", "train_new", "legacy"}:
            return TabImputeModelNew(**common_kwargs)

        if model_family in {"new_stable", "stable", "train_new_stable"}:
            stable_kwargs = {
                **common_kwargs,
                "attention_dropout": config.get("attention_dropout", 0.0),
                "ffn_dropout": config.get("ffn_dropout", 0.0),
                "drop_path_rate": config.get("drop_path_rate", 0.0),
                "residual_scale_init": config.get("residual_scale_init", 1.0),
                "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
                "embedding_dropout": config.get("embedding_dropout", 0.0),
                "attention_batch_chunk_size": config.get(
                    "attention_batch_chunk_size", None
                ),
            }
            return TabImputeModelStable(**stable_kwargs)

        if model_family in {
            "rabel",
            "new_stable_rabel",
            "stable_rabel",
        }:
            rabel_kwargs = {
                **common_kwargs,
                "attention_dropout": config.get("attention_dropout", 0.0),
                "ffn_dropout": config.get("ffn_dropout", 0.0),
                "drop_path_rate": config.get("drop_path_rate", 0.0),
                "residual_scale_init": config.get("residual_scale_init", 1.0),
                "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
                "embedding_dropout": config.get("embedding_dropout", 0.0),
                "attention_batch_chunk_size": config.get(
                    "attention_batch_chunk_size", None
                ),
                "rabel_num_kernels": config.get("rabel_num_kernels", 64),
                "rabel_sigma": config.get("rabel_sigma", 1.0),
                "rabel_exponent_digits": config.get(
                    "rabel_exponent_digits", 1
                ),
                "rabel_exponent_embedding_size": config.get(
                    "rabel_exponent_embedding_size", 32
                ),
                "rabel_initial_scale": config.get(
                    "rabel_initial_scale", 0.05
                ),
            }
            return TabImputeModelRaBEL(**rabel_kwargs)

        if model_family in {
            "pure_rabel",
            "pure_rabel_memory_efficient",
            "new_stable_pure_rabel",
        }:
            rabel_kwargs = {
                **common_kwargs,
                "attention_dropout": config.get("attention_dropout", 0.0),
                "ffn_dropout": config.get("ffn_dropout", 0.0),
                "drop_path_rate": config.get("drop_path_rate", 0.0),
                "residual_scale_init": config.get("residual_scale_init", 1.0),
                "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
                "embedding_dropout": config.get("embedding_dropout", 0.0),
                "attention_batch_chunk_size": config.get(
                    "attention_batch_chunk_size", None
                ),
                "use_activation_checkpointing": config.get(
                    "use_activation_checkpointing", True
                ),
                "checkpoint_use_reentrant": config.get(
                    "checkpoint_use_reentrant", False
                ),
                "rabel_num_kernels": config.get("rabel_num_kernels", 64),
                "rabel_sigma": config.get("rabel_sigma", 1.0),
                "rabel_exponent_digits": config.get(
                    "rabel_exponent_digits", 1
                ),
                "rabel_exponent_embedding_size": config.get(
                    "rabel_exponent_embedding_size", 32
                ),
            }
            return TabImputeModelPureRaBEL(**rabel_kwargs)

        if model_family in {
            "new_stable_memory_efficient",
            "stable_memory_efficient",
            "memory_efficient",
            "train_new_stable_memory_efficient",
        }:
            stable_kwargs = {
                **common_kwargs,
                "attention_dropout": config.get("attention_dropout", 0.0),
                "ffn_dropout": config.get("ffn_dropout", 0.0),
                "drop_path_rate": config.get("drop_path_rate", 0.0),
                "residual_scale_init": config.get("residual_scale_init", 1.0),
                "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
                "embedding_dropout": config.get("embedding_dropout", 0.0),
                "attention_batch_chunk_size": config.get(
                    "attention_batch_chunk_size", None
                ),
                "use_activation_checkpointing": config.get(
                    "use_activation_checkpointing", True
                ),
            }
            return TabImputeModelMemoryEfficient(**stable_kwargs)

        if model_family in {"best_transfer", "best_transfer_stable"}:
            stable_overrides = {
                **common_kwargs,
                "attention_dropout": config.get("attention_dropout", 0.0),
                "ffn_dropout": config.get("ffn_dropout", 0.0),
                "drop_path_rate": config.get("drop_path_rate", 0.0),
                "residual_scale_init": config.get("residual_scale_init", 1.0),
                "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
                "embedding_dropout": config.get("embedding_dropout", 0.0),
                "attention_batch_chunk_size": config.get(
                    "attention_batch_chunk_size", None
                ),
            }
            return build_best_transfer_model(**stable_overrides)

        raise ValueError(
            f"Unsupported model_family '{self.model_family}'. "
            "Expected one of: 'new', 'new_stable', "
            "'new_stable_memory_efficient', 'rabel', 'pure_rabel', "
            "'best_transfer'."
        )


    def impute(
        self,
        X: np.ndarray,
        return_full: bool = False,
        num_repeats: int = 1,
        means: np.ndarray | None = None,
        stds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Impute missing values in the input matrix.
        Imputes the missing values in place.

        Args:
            X (np.ndarray): Input matrix of shape (T, H) where:
             - T is the number of samples (rows)
             - H is the number of features (columns)

        Returns:
            np.ndarray: Imputed matrix of shape (T, H)
        """
        # Verify that the input matrix is valid
        if X.ndim != 2:
            raise ValueError("Input matrix must be 2-dimensional")

        # Get means and stds per column unless they are supplied manually.
        if means is None:
            means = np.nanmean(X, axis=0)
        else:
            means = np.asarray(means, dtype=np.float64)
            if means.shape != (X.shape[1],):
                raise ValueError(
                    f"means must have shape ({X.shape[1]},), got {means.shape}"
                )

        if stds is None:
            stds = np.nanstd(X, axis=0)
        else:
            stds = np.asarray(stds, dtype=np.float64)
            if stds.shape != (X.shape[1],):
                raise ValueError(
                    f"stds must have shape ({X.shape[1]},), got {stds.shape}"
                )

        # set stds to 1 if they are nan
        stds = np.where(np.isnan(stds), 1, stds)

        # set means to 0 if they are nan
        means = np.where(np.isnan(means), 0, means)

        # Normalize the input matrix
        X_normalized = (X - means) / (stds + 1e-16)
        # Add a small epsilon to avoid division by zero

        # If any preprocessors, do ensemble of them
        X_imputed = np.zeros_like(X_normalized)
        X_full = np.zeros_like(X_normalized)
        X_full_list = [np.zeros_like(X_normalized) for _ in range(num_repeats)]
        X_imputed_list = [np.zeros_like(X_normalized) for _ in range(num_repeats)]
        if self.preprocessors is not None:
            for preprocessor in self.preprocessors:
                X_preprocessed = preprocessor.fit_transform(X_normalized)
                imput, X_full = self.get_imputation(X_preprocessed, num_repeats=num_repeats)
                if num_repeats > 1:
                    for i in range(num_repeats):
                        X_imputed_list[i] += preprocessor.inverse_transform(imput[i])
                        X_full_list[i] += preprocessor.inverse_transform(X_full[i])
                else:
                    X_imputed += preprocessor.inverse_transform(imput)
                    X_full += preprocessor.inverse_transform(X_full)
            if num_repeats > 1:
                for i in range(num_repeats):
                    X_imputed_list[i] /= num_repeats
                    X_full_list[i] /= num_repeats
            else:
                X_imputed /= len(self.preprocessors)
                X_full /= len(self.preprocessors)
        else:
            imput, X_full_ = self.get_imputation(X_normalized, num_repeats=num_repeats)
            if num_repeats > 1:
                for i in range(num_repeats):
                    X_imputed_list[i] += imput[i]
                    X_full_list[i] += X_full_[i]
            else:
                X_imputed += imput
                X_full += X_full_
        torch.cuda.empty_cache()
        
        # Add back the means and stds
        if num_repeats > 1:
            for i in range(num_repeats):
                X_imputed_list[i] = X_imputed_list[i] * (stds + 1e-16) + means
                X_full_list[i] = X_full_list[i] * (stds + 1e-16) + means
            if return_full:
                return X_imputed_list, X_full_list
            else:
                return X_imputed_list
        else:
            X_imputed = X_imputed * (stds + 1e-16) + means
            X_full = X_full * (stds + 1e-16) + means

        if return_full:
            return X_imputed, X_full
        else:
            return X_imputed
        
    def get_imputation(self, X_normalized: np.ndarray, num_repeats: int = 1) -> np.ndarray:
        """Get the imputation for the input matrix.
        If ``max_num_rows`` and/or ``max_num_cols`` is set, the input matrix is
        split into row/column chunks and the imputation is performed on each
        submatrix.

        Args:
            X_normalized (np.ndarray): Input matrix of shape (T, H) where:
             - T is the number of samples (rows)
             - H is the number of features (columns)
            num_repeats (int, optional): Number of times to repeat the imputation. Defaults to 1.
            max_num_rows (int, optional): Maximum number of rows to impute at once. Defaults to None.
            max_num_cols (int, optional): Maximum number of columns to impute at once. Defaults to None.

        Returns:
            np.ndarray: Imputed matrix of shape (T, H)
        """
        if self.max_num_rows is None and self.max_num_cols is None:
            return self._get_imputation_single(X_normalized, num_repeats=num_repeats)

        row_chunk_size = self._get_row_window_size(X_normalized.shape[0])
        row_spans = self._get_chunk_spans(X_normalized.shape[0], row_chunk_size)
        col_spans = self._get_chunk_spans(X_normalized.shape[1], self.max_num_cols)
        X_imputed_sum = np.zeros_like(X_normalized, dtype=np.float64)
        X_full_sum = np.zeros_like(X_normalized, dtype=np.float64)
        prediction_counts = np.zeros_like(X_normalized, dtype=np.float64)

        if self.verbose:
            from tqdm import tqdm

            total_chunks = len(row_spans) * len(col_spans)
            pbar = tqdm(total=total_chunks, desc="Processed chunks")

        for row_start, row_end in row_spans:
            for col_start, col_end in col_spans:
                row_slice = slice(row_start, row_end)
                col_slice = slice(col_start, col_end)
                X_chunk = X_normalized[row_slice, col_slice].copy()
                X_chunk_imputed, X_chunk_full = self._get_imputation_chunk(
                    X_chunk, num_repeats=num_repeats
                )
                X_imputed_sum[row_slice, col_slice] += X_chunk_imputed
                X_full_sum[row_slice, col_slice] += X_chunk_full
                prediction_counts[row_slice, col_slice] += 1.0
                if self.verbose:
                    pbar.update(1)

        if self.verbose:
            pbar.close()

        has_predictions = prediction_counts > 0
        X_imputed = X_normalized.copy()
        X_full = X_normalized.copy()
        X_imputed[has_predictions] = (
            X_imputed_sum[has_predictions] / prediction_counts[has_predictions]
        )
        X_full[has_predictions] = (
            X_full_sum[has_predictions] / prediction_counts[has_predictions]
        )

        return X_imputed, X_full


    def _get_input_tensors(self, X_normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the input tensors for the input matrix.
        Args:
            X_normalized (torch.Tensor): Input matrix of shape (T, H) where:
             - T is the number of samples (rows) to impute
             - H is the number of features (columns)
        Returns:
            tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]: Input tensors, input y, missing indices, non-missing indices.
        """
        train_X, train_y, test_X, test_y = create_train_test_sets(
            X_normalized, X_normalized
        )
        
        input_y = torch.cat(
            (train_y, torch.full_like(test_y, torch.nan, device=self.device)), dim=0
        )
        
        missing_indices = np.where(np.isnan(X_normalized.cpu().numpy()))
        non_missing_indices = np.where(~np.isnan(X_normalized.cpu().numpy()))
        
        # Move tensors to device
        train_X = train_X.to(self.device)
        input_y = input_y.to(self.device)
        test_X = test_X.to(self.device)
        
        X_input = torch.cat((train_X, test_X), dim=0)
        
        # X_input = X_input.float()
        # input_y = input_y.float()
        X_input = X_input.to(torch.bfloat16)
        input_y = input_y.to(torch.bfloat16)
        
        train_size = train_y.shape[0]
        
        return X_input, input_y, missing_indices, non_missing_indices, train_size
    
    def _get_row_window_size(self, num_rows: int) -> int | None:
        if self.max_num_rows is None:
            return None
        if self.max_num_chunks == float("inf"):
            return num_rows
        return max(1, int(self.max_num_rows * self.max_num_chunks))

    def _get_chunk_spans(
        self, size: int, chunk_size: int | None
    ) -> list[tuple[int, int]]:
        if chunk_size is None or chunk_size >= size:
            return [(0, size)]
        spans = []
        start_index = 0
        while start_index + chunk_size < size:
            spans.append((start_index, start_index + chunk_size))
            start_index += chunk_size
        if start_index < size:
            final_span = (max(0, size - chunk_size), size)
            if not spans or spans[-1] != final_span:
                spans.append(final_span)
        return spans

    def split_into_chunks(self, arr: np.ndarray, chunk_size: int | None) -> list[np.ndarray]:
        """
        Split a 2D array into row chunks of size `chunk_size`.
        The final chunk overlaps the previous chunk when needed so it can keep
        a full-size context window.
        """
        if chunk_size is None or chunk_size >= arr.shape[0]:
            return [arr], [0], [arr.shape[0]]

        spans = self._get_chunk_spans(arr.shape[0], chunk_size)
        start_indices = [start_index for start_index, _ in spans]
        end_indices = [end_index for _, end_index in spans]
        chunks = [arr[start_index:end_index] for start_index, end_index in spans]
        return chunks, start_indices, end_indices
    
    def _get_imputation_chunk(self, X_normalized: np.ndarray, num_repeats: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Get the imputation for a chunk of the input matrix.
        Dispatches to entry-wise or direct implementation based on entry_wise_features.
        Args:
            X_normalized (np.ndarray): Input matrix of shape (T, H) where:
             - T is the number of samples (rows)
             - H is the number of features (columns)
            num_repeats (int, optional): Number of times to repeat the imputation. Defaults to 1.
        Returns:
            tuple[np.ndarray, np.ndarray]: (X_imputed, X_full)
        """
        if self.entry_wise_features:
            return self._get_imputation_chunk_entry_wise(X_normalized, num_repeats=num_repeats)
        else:
            return self._get_imputation_chunk_direct(X_normalized, num_repeats=num_repeats)

    def _get_imputation_chunk_direct(self, X_normalized: np.ndarray, num_repeats: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Get imputation for a chunk when entry_wise_features=False.
        Passes data directly to model (no train/test split), similar to _get_imputation_single.
        """
        row_chunks, start_indices, end_indices = self.split_into_chunks(X_normalized, self.max_num_rows)
        chunk_tensors = [
            torch.from_numpy(chunk).to(self.device).to(torch.bfloat16)
            for chunk in row_chunks
        ]
        X_batch_chunks, X_last_chunk = self._stack_chunks_for_batch(chunk_tensors)
        medians_chunks, medians_last_chunk = self._run_model_direct(X_batch_chunks, X_last_chunk)

        X_full = X_normalized.copy()
        for i, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):
            medians = (
                medians_last_chunk[0].cpu().numpy()
                if i == len(start_indices) - 1 and medians_last_chunk is not None
                else medians_chunks[i].cpu().numpy()
            )
            chunk_slice = X_normalized[start_idx:end_idx, :]
            missing_mask = np.isnan(chunk_slice)
            chunk_slice[missing_mask] = medians[missing_mask]
            X_full[start_idx:end_idx, :] = medians

        return X_normalized, X_full

    def _stack_chunks_for_batch(
        self, chunk_tensors: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Stack chunk tensors for batched inference, handling variable-sized last chunk."""
        if len(chunk_tensors) == 1:
            return torch.stack(chunk_tensors, dim=0), None
        if chunk_tensors[-1].shape[0] != chunk_tensors[-2].shape[0]:
            return torch.stack(chunk_tensors[:-1], dim=0), chunk_tensors[-1].unsqueeze(0)
        return torch.stack(chunk_tensors, dim=0), None

    def _stack_input_and_y_for_batch(
        self, X_list: list[torch.Tensor], y_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Stack (X, y) chunk pairs for batched inference, handling variable-sized last chunk."""
        if len(X_list) > 1 and X_list[-1].shape[0] != X_list[-2].shape[0]:
            X_batch = torch.stack(X_list[:-1], dim=0)
            y_batch = torch.stack(y_list[:-1], dim=0)
            return X_batch, y_batch, X_list[-1], y_list[-1]
        return (
            torch.stack(X_list, dim=0),
            torch.stack(y_list, dim=0),
            None,
            None,
        )

    def _run_model_direct(
        self, X_batch: torch.Tensor, X_last: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run model (entry_wise_features=False) on batched chunks and return medians."""
        with torch.no_grad():
            preds = self.model(X_batch)
            medians_chunks = self.bar_distribution.median(logits=preds)
            medians_last = None
            if X_last is not None:
                preds_last = self.model(X_last)
                medians_last = self.bar_distribution.median(logits=preds_last)
        return medians_chunks, medians_last

    def _get_imputation_chunk_entry_wise(self, X_normalized: np.ndarray, num_repeats: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Get imputation for a chunk when entry_wise_features=True (MCPFN with train/test split)."""
        row_chunks, start_indices, end_indices = self.split_into_chunks(X_normalized, self.max_num_rows)
        row_chunks_normalized = []
        means = []
        stds = []
        X_input_list = []
        input_y_list = []
        missing_indices_list = []
        non_missing_indices_list = []
        train_size_list = []
        for chunk in row_chunks:
            chunk_tensor = torch.from_numpy(chunk).to(self.device)
            
            # normalize the chunk
            chunk_normalized, (mean, std) = normalize_data(chunk_tensor, return_scaling=True)
            row_chunks_normalized.append(chunk_normalized)
            means.append(mean.cpu().detach().numpy())
            stds.append(std.cpu().detach().numpy())
            
            X_input, input_y, missing_indices, non_missing_indices, train_size = self._get_input_tensors(chunk_normalized)
            
            X_input_list.append(X_input)
            input_y_list.append(input_y)
            missing_indices_list.append(missing_indices)
            non_missing_indices_list.append(non_missing_indices)
            train_size_list.append(train_size)
            
        # Stack chunks for batched inference (handle variable-sized last chunk)
        X_batch_chunks, input_y_batch_chunks, X_last_chunk, input_y_last_chunk = (
            self._stack_input_and_y_for_batch(X_input_list, input_y_list)
        )
            
        medians_chunks = None
        medians_last_chunk = None
        
        with torch.no_grad():
            preds = self.model(X_batch_chunks, input_y_batch_chunks) # shape: (num_chunks, T, H)
            # Get the median predictions
            medians_chunks = self.bar_distribution.median(logits=preds)
            
            if X_last_chunk is not None:
                preds_last = self.model(
                    X_last_chunk.unsqueeze(0), input_y_last_chunk.unsqueeze(0)
                )
                medians_last_chunk = self.bar_distribution.median(logits=preds_last)
                
        X_full = X_normalized.copy()
        
        for i, (
            start_index, 
            end_index, 
            missing_indices, 
            non_missing_indices, 
            train_size,
            mean,
            std
        ) in enumerate(zip(start_indices, end_indices, missing_indices_list, non_missing_indices_list, train_size_list, means, stds)):
            if i == len(start_indices) - 1 and X_last_chunk is not None:
                medians = medians_last_chunk[0]
            else:
                medians = medians_chunks[i]
                
            medians_train = medians[: train_size].cpu().detach().numpy()
            medians_test = medians[train_size:].cpu().detach().numpy()
            
            X_normalized[start_index + missing_indices[0], missing_indices[1]] = medians_test
            X_normalized[start_index:end_index, :] = X_normalized[start_index:end_index, :] * std + mean
            
            # Reset the non-missing indices to the original values in the full matrix
            X_normalized[start_index + non_missing_indices[0], non_missing_indices[1]] = X_full[start_index + non_missing_indices[0], non_missing_indices[1]]
            
            X_full[start_index + missing_indices[0], missing_indices[1]] = medians_test
            X_full[start_index + non_missing_indices[0], non_missing_indices[1]] = medians_train
            X_full[start_index:end_index, :] = X_full[start_index:end_index, :] * std + mean
            
        return X_normalized, X_full
        

    def _get_imputation_single(self, X_normalized: np.ndarray, num_repeats: int = 1) -> np.ndarray:
        X_normalized_tensor = torch.from_numpy(X_normalized).to(self.device)
        
        if self.entry_wise_features:
            X_input, input_y, missing_indices, non_missing_indices, train_size = self._get_input_tensors(X_normalized_tensor)
            
            X_input = X_input.unsqueeze(0) # batch size 1
            input_y = input_y.unsqueeze(0) # batch size 1

            with torch.no_grad():
                preds = self.model(X_input, input_y)

                medians = self.bar_distribution.median(logits=preds).flatten()
                
                if num_repeats > 1:
                    out_full = []
                    out_normalized = []
                    
                    for _ in range(num_repeats):
                        sample = self.bar_distribution.sample(logits=preds.squeeze(0))
                        X_full = X_normalized.copy()
                        X_imputed = X_normalized.copy()
                        
                        sampls_train = sample[:train_size]
                        sampls_test = sample[train_size:]
                        X_full[missing_indices] = sampls_test.cpu().detach().numpy()
                        X_full[non_missing_indices] = sampls_train.cpu().detach().numpy()
                        X_imputed[missing_indices] = sampls_test.cpu().detach().numpy()
                        out_full.append(X_full)
                        out_normalized.append(X_imputed)
                    
                    return out_normalized, out_full

            # Denormalize the predictions with train y mean and std
            # medians = medians * (train_y.std() + 1e-8) + train_y.mean()

            X_full = X_normalized.copy()

            medians_train = medians[: train_size]
            medians_test = medians[train_size :]

            # Impute the missing values with the median predictions
            X_normalized[missing_indices] = medians_test.cpu().detach().numpy()

            X_full[missing_indices] = medians_test.cpu().detach().numpy()
            X_full[non_missing_indices] = medians_train.cpu().detach().numpy()

            return X_normalized, X_full
        else:
            with torch.no_grad():
                X_normalized_tensor = X_normalized_tensor.unsqueeze(0).to(torch.bfloat16)
                preds = self.model(X_normalized_tensor)
                medians = self.bar_distribution.median(logits=preds).squeeze(0)
                # print(medians)
                # preds_0 = preds[0,0,0,:]
                # # print the softmax of the predictions
                # softmax = torch.softmax(preds_0, dim=-1)
                # plt.bar(range(softmax.shape[0]), softmax.cpu().detach().to(torch.float32).numpy())
                # plt.savefig('softmax.png')
            X_imputed = medians.cpu().detach().numpy()
            missing_mask = np.isnan(X_normalized)
            X_normalized[missing_mask] = X_imputed[missing_mask]
            return X_normalized, X_imputed
	            


class CompressedCrossImputePFN:
    """Chunked cross-compression interface for direct TabImpute checkpoints."""

    def __init__(
        self,
        predictor_checkpoint_path: str | None = None,
        row_compressor_checkpoint_path: str | None = None,
        column_compressor_checkpoint_path: str | None = None,
        compressed_checkpoint_path: str | None = None,
        predictor_config: dict | None = None,
        compressor_config: dict | None = None,
        chunk_rows: int | None = None,
        chunk_cols: int | None = None,
        k_row: int = 4,
        k_col: int = 4,
        device: str = "cpu",
        model_family: str = "new_stable_memory_efficient",
    ):
        self.device = device
        self.chunk_rows = 50 if chunk_rows is None else int(chunk_rows)
        self.chunk_cols = 50 if chunk_cols is None else int(chunk_cols)
        self.k_row = k_row
        self.k_col = k_col

        compressed_config = None
        if compressed_checkpoint_path is not None:
            compressed_config = self._load_compressed_run_config(
                compressed_checkpoint_path
            )
            model_config = dict(compressed_config.get("model", {}))
            training_config = dict(compressed_config.get("training", {}))
            predictor_config = predictor_config or model_config.get("predictor")
            compressor_config = compressor_config or model_config.get("compressor")
            k_row = int(model_config.get("k_row", training_config.get("k_row", k_row)))
            k_col = int(model_config.get("k_col", training_config.get("k_col", k_col)))
            self.k_row = k_row
            self.k_col = k_col
            if chunk_rows is None:
                self.chunk_rows = int(training_config.get("chunk_rows", 50))
            if chunk_cols is None:
                self.chunk_cols = int(training_config.get("chunk_cols", 50))
            model_family = str(
                model_config.get(
                    "model_family",
                    training_config.get("model_family", model_family),
                )
            )

        if predictor_config is None:
            if predictor_checkpoint_path is None:
                raise ValueError(
                    "Either compressed_checkpoint_path with a config.json or "
                    "predictor_checkpoint_path/predictor_config must be provided."
                )
            predictor_config = _load_json_config_from_checkpoint(
                predictor_checkpoint_path
            )
        if "model" in predictor_config:
            predictor_config = dict(predictor_config["model"])
        self.predictor_config = dict(predictor_config)

        if predictor_checkpoint_path is None:
            predictor = self._build_predictor_from_config(
                self.predictor_config,
                model_family=model_family,
            ).to(device=device, dtype=torch.bfloat16)
        else:
            predictor_builder = ImputePFN(
                device=device,
                checkpoint_path=predictor_checkpoint_path,
                entry_wise_features=False,
                json_config={"config": self.predictor_config},
                model_family=model_family,
            )
            predictor = predictor_builder.model

        if compressor_config is None:
            compressor_config = build_compressor_config_from_predictor(
                self.predictor_config,
                k_row=k_row,
                k_col=k_col,
            )
        self.compressor_config = dict(compressor_config)

        use_rabel = model_family.lower() in {
            "rabel",
            "new_stable_rabel",
            "stable_rabel",
        }
        use_pure_rabel = model_family.lower() in {
            "pure_rabel",
            "pure_rabel_memory_efficient",
            "new_stable_pure_rabel",
        }
        if use_rabel:
            row_compressor_class = RowCompressionModelRaBEL
            column_compressor_class = ColumnCompressionModelRaBEL
            compressed_model_class = CompressedCrossTabImputeModelRaBEL
        elif use_pure_rabel:
            row_compressor_class = RowCompressionModelPureRaBEL
            column_compressor_class = ColumnCompressionModelPureRaBEL
            compressed_model_class = CompressedCrossTabImputeModelPureRaBEL
        else:
            row_compressor_class = RowCompressionModel
            column_compressor_class = ColumnCompressionModel
            compressed_model_class = CompressedCrossTabImputeModel
        rabel_compressor_kwargs = (
            {"rabel_config": self.compressor_config}
            if use_rabel or use_pure_rabel
            else {}
        )

        row_compressor = row_compressor_class(
            embedding_size=self.compressor_config["embedding_size"],
            num_attention_heads=self.compressor_config["num_attention_heads"],
            mlp_hidden_size=self.compressor_config["mlp_hidden_size"],
            num_layers=self.compressor_config["num_layers"],
            k_row=k_row,
            num_cls=self.compressor_config.get(
                "num_cls",
                self.predictor_config.get("num_cls", 12),
            ),
            attention_dropout=self.compressor_config.get("attention_dropout", 0.0),
            ffn_dropout=self.compressor_config.get("ffn_dropout", 0.0),
            drop_path_rate=self.compressor_config.get("drop_path_rate", 0.0),
            residual_scale_init=self.compressor_config.get("residual_scale_init", 1.0),
            rms_norm_eps=self.compressor_config.get("rms_norm_eps", 1e-6),
            use_rope=self.compressor_config.get("use_rope", True),
            rope_base=self.compressor_config.get("rope_base", 10000.0),
            rope_fraction=self.compressor_config.get("rope_fraction", 1.0),
            use_activation_checkpointing=self.compressor_config.get(
                "use_activation_checkpointing",
                True,
            ),
            **rabel_compressor_kwargs,
        )
        column_compressor = column_compressor_class(
            embedding_size=self.compressor_config["embedding_size"],
            num_attention_heads=self.compressor_config["num_attention_heads"],
            mlp_hidden_size=self.compressor_config["mlp_hidden_size"],
            num_layers=self.compressor_config["num_layers"],
            k_col=k_col,
            num_cls=self.compressor_config.get(
                "num_cls",
                self.predictor_config.get("num_cls", 12),
            ),
            attention_dropout=self.compressor_config.get("attention_dropout", 0.0),
            ffn_dropout=self.compressor_config.get("ffn_dropout", 0.0),
            drop_path_rate=self.compressor_config.get("drop_path_rate", 0.0),
            residual_scale_init=self.compressor_config.get("residual_scale_init", 1.0),
            rms_norm_eps=self.compressor_config.get("rms_norm_eps", 1e-6),
            use_rope=self.compressor_config.get("use_rope", True),
            rope_base=self.compressor_config.get("rope_base", 10000.0),
            rope_fraction=self.compressor_config.get("rope_fraction", 1.0),
            use_activation_checkpointing=self.compressor_config.get(
                "use_activation_checkpointing",
                True,
            ),
            **rabel_compressor_kwargs,
        )

        self.model = compressed_model_class(
            predictor=predictor,
            row_compressor=row_compressor,
            column_compressor=column_compressor,
            k_row=k_row,
            k_col=k_col,
        ).to(device=device, dtype=torch.bfloat16)
        self.model.initialize_compressor_embedders_from_predictor()

        self._load_compressor_checkpoint(
            row_compressor_checkpoint_path,
            module_name="row_compressor",
        )
        self._load_compressor_checkpoint(
            column_compressor_checkpoint_path,
            module_name="column_compressor",
        )
        if compressed_checkpoint_path is not None:
            self._load_full_compressed_checkpoint(compressed_checkpoint_path)
        self.model.eval()

        with resources.files("tabimpute.data").joinpath("borders.pt").open("rb") as f:
            self.borders = torch.load(f, map_location=self.device)
        self.bar_distribution = FullSupportBarDistribution(
            borders=self.borders.to(self.device)
        )

    @staticmethod
    def _load_compressed_run_config(checkpoint_path: str) -> dict:
        from pathlib import Path
        import json

        cfg_path = Path(checkpoint_path).parent / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"Compressed checkpoint config not found: {cfg_path}"
            )
        with cfg_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _build_predictor_from_config(config: dict, model_family: str):
        builder = ImputePFN.__new__(ImputePFN)
        builder.model_family = model_family
        return builder._build_direct_model(json_config={"config": config})

    def _load_compressor_checkpoint(
        self,
        checkpoint_path: str | None,
        module_name: str,
    ) -> None:
        if checkpoint_path is None:
            return
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        module = getattr(self.model, module_name)
        if module_name in checkpoint:
            state_dict = checkpoint[module_name]
        elif "model" in checkpoint:
            prefix = f"{module_name}."
            state_dict = {
                key.removeprefix(prefix): value
                for key, value in checkpoint["model"].items()
                if key.startswith(prefix)
            }
        else:
            state_dict = checkpoint
        module.load_state_dict(state_dict, strict=True)

    def _load_full_compressed_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=True)

    def impute(
        self,
        X: np.ndarray,
        return_full: bool = False,
        means: np.ndarray | None = None,
        stds: np.ndarray | None = None,
    ) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("Input matrix must be 2-dimensional")

        if means is None:
            means = np.nanmean(X, axis=0)
        else:
            means = np.asarray(means, dtype=np.float64)
            if means.shape != (X.shape[1],):
                raise ValueError(
                    f"means must have shape ({X.shape[1]},), got {means.shape}"
                )

        if stds is None:
            stds = np.nanstd(X, axis=0)
        else:
            stds = np.asarray(stds, dtype=np.float64)
            if stds.shape != (X.shape[1],):
                raise ValueError(
                    f"stds must have shape ({X.shape[1]},), got {stds.shape}"
                )

        stds = np.where(np.isnan(stds), 1, stds)
        means = np.where(np.isnan(means), 0, means)
        X_normalized = (X - means) / (stds + 1e-16)
        X_tensor = (
            torch.from_numpy(X_normalized)
            .unsqueeze(0)
            .to(self.device)
            .to(torch.bfloat16)
        )

        with torch.no_grad():
            imputed = self.model.impute_normalized(
                X_tensor,
                chunk_rows=self.chunk_rows,
                chunk_cols=self.chunk_cols,
                bar_distribution=self.bar_distribution,
            )[0]

        X_full = imputed.float().cpu().numpy() * stds + means
        X_imputed = X.copy()
        missing_mask = np.isnan(X_imputed)
        X_imputed[missing_mask] = X_full[missing_mask]
        if return_full:
            return X_imputed, X_full
        return X_imputed


class TabPFNImputer:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.reg = TabPFNRegressor(device=device, n_estimators=8)

    def impute(self, X: np.ndarray, return_full: bool = False, num_repeats: int = 1) -> np.ndarray:
        """Impute missing values in the input matrix.
        Imputes the missing values in place.
        """
        X_tensor = torch.from_numpy(X).to(self.device)

        # Impute the missing values
        train_X, train_y, test_X, _ = create_train_test_sets(X_tensor, X_tensor)

        train_X_npy = train_X.cpu().numpy()
        train_y_npy = train_y.cpu().numpy()
        test_X_npy = test_X.cpu().numpy()
        
        missing_indices = np.where(np.isnan(X))
        non_missing_indices = np.where(~np.isnan(X))

        self.reg.fit(train_X_npy, train_y_npy)

        # self.reg.model_ = mcpfn.model # override the model with the mcpfn model
        
        if num_repeats > 1:
            out_full = self.reg.predict(test_X_npy, output_type='full')
            bar_dist = out_full['criterion']
            logits = out_full['logits']
            
            out_normalized = []
            # out_full = []
            
            for _ in range(num_repeats):
                sample = bar_dist.sample(logits=logits)
                X_imputed = X.copy()
                
                # sampls_train = sample[:train_y.shape[0]]
                # sampls_test = sample[train_y.shape[0]:]
                X_imputed[missing_indices] = sample.cpu().detach().numpy()
                # out_full.append(X_full)
                out_normalized.append(X_imputed)
            
            return out_normalized

        preds = self.reg.predict(test_X_npy)
        preds_train = self.reg.predict(train_X_npy)

        X_full = X.copy()
        mask = np.isnan(X)
        X[mask] = preds
        X_full[mask] = preds
        X_full[~mask] = preds_train

        # Clean up memory
        del train_X, train_y, test_X, X_tensor
        torch.cuda.empty_cache()

        if return_full:
            return X, X_full
        else:
            return X


class TabPFNUnsupervisedModel:
    def __init__(self, device: str = "cuda"):
        self.device = device
        clf = TabPFNClassifier(device=device, n_estimators=3)
        reg = TabPFNRegressor(device=device, n_estimators=3)
        self.model = unsupervised.TabPFNUnsupervisedModel(
            tabpfn_clf=clf,
            tabpfn_reg=reg,
        )

    def impute(self, X, n_permutations=10):
        self.model.fit(X)
        return self.model.impute(X, n_permutations=n_permutations).cpu().numpy()


class MCTabPFNEnsemble:
    def __init__(
        self,
        device: str = "cpu",
        nhead: int = 2,
        preprocessors: list[Preprocess] = None,
    ):
        self.device = device
        self.tabpfn_imputer = TabPFNImputer(device=device)
        self.mcpfn_imputer = ImputePFN(
            device=device, nhead=nhead, preprocessors=preprocessors
        )

    def impute(self, X):
        missing_mask = np.isnan(X)
        y_observed = X[~missing_mask]

        X_tabpfn, X_full_tabpfn = self.tabpfn_imputer.impute(X, return_full=True)
        X_mcpfn, X_full_mcpfn = self.mcpfn_imputer.impute(X, return_full=True)

        y_tabpfn = X_full_tabpfn[~missing_mask]
        y_mcpfn = X_full_mcpfn[~missing_mask]

        w_tabpfn, w_mcpfn = self.optimal_weights(y_tabpfn, y_mcpfn, y_observed)

        X[missing_mask] = (w_tabpfn * X_tabpfn[missing_mask]) + (
            w_mcpfn * X_mcpfn[missing_mask]
        )

        return X

    def optimal_weights(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """
        Optimal weights for a convex combination of x and y to minimize the distance to z.
        Returns the weights for x and y.
        """
        d = x - y
        numerator = np.dot(z - y, d)
        denominator = np.dot(d, d)

        if denominator == 0:
            # x and y are identical, any convex combo works
            return 0.5, 0.5

        w = numerator / denominator
        w = np.clip(w, 0, 1)  # enforce nonnegativity and sum-to-1
        return w, 1 - w
    
class TabImputeRouter:
    def __init__(
        self,
        device: str = "cpu",
        nhead: int = 2,
        preprocessors: list[Preprocess] = None,
        checkpoint_paths: list[str] = None,
    ):
        self.tabimpute_models = []
        self.routing_models = []
        for checkpoint_path in checkpoint_paths:
            self.tabimpute_models.append(
                ImputePFN(device=device, nhead=nhead, preprocessors=preprocessors, checkpoint_path=checkpoint_path)
            )
            self.routing_models.append(
                ImputePFN(device=device, nhead=nhead, preprocessors=None, checkpoint_path=checkpoint_path)
            )
        self.device = device

    def impute(self, X):
        missing_mask = np.isnan(X)
        y_observed = X[~missing_mask]
        
        X_full_list = []
        X_imputed_list = []
        best_model_mse = float('inf')
        best_model_index = None
        
        names = ['mcar', 'mar', 'mnar']
        
        for i, model in enumerate(self.routing_models):
            X_imputed, X_full = model.impute(X.copy(), return_full=True)
            X_full_list.append(X_full)
            X_imputed_list.append(X_imputed)
            mse = np.mean((X_full[~missing_mask] - y_observed) ** 2)
            mae = np.mean(np.abs(X_full[~missing_mask] - y_observed))
            
            if mse < best_model_mse:
                best_model_mse = mse
                best_model_index = i
            print(f"Routing model {names[i]}: MSE: {mse}, MAE: {mae}, Best MSE: {best_model_mse}")
            
        # Calculate the imputation for the best model
        X_imputed = self.tabimpute_models[best_model_index].impute(X.copy())
        
        X[missing_mask] = X_imputed[missing_mask]
        
        return X
    
class TabImputeEnsemble:
    def __init__(
        self,
        device: str = "cpu",
        nhead: int = 2,
        preprocessors: list[Preprocess] = None,
        checkpoint_paths: list[str] = None,
    ):
        self.tabimpute_models = [
            ImputePFN(device=device, nhead=nhead, preprocessors=preprocessors, checkpoint_path=checkpoint_path) for checkpoint_path in checkpoint_paths
        ]
        self.device = device

    def impute(self, X):
        missing_mask = np.isnan(X)
        y_observed = X[~missing_mask]
        
        X_full_list = []
        X_imputed_list = []
        for model in self.tabimpute_models:
            X_imputed, X_full = model.impute(X.copy(), return_full=True)
            X_full_list.append(X_full)
            X_imputed_list.append(X_imputed)
        
        # Create list of prediction values
        y_pred_list = [X_full[~missing_mask] for X_full in X_full_list]
        
        w_optimal = self.find_optimal_weights(y_pred_list, y_observed)
        
        # Impute the missing values as a weighted sum of the imputed values
        imputed_vals = np.zeros_like(X[missing_mask])
        
        for i, X_imputed in enumerate(X_imputed_list):
            imputed_vals += w_optimal[i] * X_imputed[missing_mask]
            
        X[missing_mask] = imputed_vals
        
        return X

    def find_optimal_weights(self, x_list: list[np.ndarray], y: np.ndarray):
        """
        Find optimal nonnegative weights that sum to 1 and minimize
        the squared distance between the weighted sum of x_list and target y.
        
        Args:
            x_list: list of numpy arrays (each shape (n,))
            y: numpy array of shape (n,)
        
        Returns:
            w_opt: numpy array of optimal weights (shape (m,))
        """
        m = len(x_list)
        X = np.column_stack(x_list)  # shape: (n, m)

        # # Objective: ||Xw - y||^2
        # def objective(w):
        #     return np.sum((X @ w - y)**2)

        # # Constraints: sum(w) = 1, w >= 0
        # constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        # bounds = [(0, None)] * m
        # w0 = np.ones(m) / m

        # Solve using SLSQP
        # result = minimize(objective, w0, bounds=bounds, constraints=constraints)
        # return result.x
        reg = LinearRegression(fit_intercept=False).fit(X, y)
        
        return reg.coef_
    
    
    
class TabImputeCategorical:
    def __init__(self, 
                 device: str = "cuda",
                 nhead: int = 2,
                 preprocessors: list[Preprocess] = None,
                 checkpoint_path: str = None):
        self.device = device
        self.imputer = ImputePFN(device=device, nhead=nhead, preprocessors=preprocessors, checkpoint_path=checkpoint_path)
        
        
    def impute(self, X, categorical_columns: list[int] | None = None, ordered_categorical_columns: list[int] | None = None):
        """
        Impute missing values in a matrix with categorical columns.
        
        Args:
            X: Input matrix of shape (n_samples, n_features)
            categorical_columns: List of column indices that are categorical.
                                If None, treats all columns as numerical.
        
        Returns:
            Imputed matrix of shape (n_samples, n_features) with categorical
            values restored from one-hot encodings.
        """
        def _isnan_or_none(x):
            if isinstance(x, float) and np.isnan(x):
                return True
            elif x is None:
                return True
            else:
                return False

        my_isnan = np.vectorize(_isnan_or_none)
        
        if categorical_columns is None:
            categorical_columns = []
        if ordered_categorical_columns is None:
            ordered_categorical_columns = []
        X = X.copy()
        n_samples, n_features = X.shape
        
        # Track mappings for categorical columns
        cat_mappings = {}  # col_idx -> (categories, one_hot_start_idx, one_hot_end_idx)
        # Track mappings for numerical columns
        num_mappings = {}  # col_idx -> one_hot_col_idx
        X_onehot_list = []
        current_col_idx = 0
        
        # Convert categorical columns to one-hot encodings
        for col_idx in range(n_features):
            if col_idx in categorical_columns:
                # Get unique categories from non-NaN values
                col_data = X[:, col_idx]
                # non_nan_mask = ~np.isnan(col_data)
                non_nan_mask = ~my_isnan(col_data)
                
                if not np.all(my_isnan(col_data)):
                    unique_cats = np.unique(col_data[non_nan_mask])
                    n_categories = len(unique_cats)
                    
                    # Create one-hot encoding
                    onehot = np.zeros((n_samples, n_categories))
                    for i, cat_val in enumerate(unique_cats):
                        onehot[col_data == cat_val, i] = 1.0
                    
                    # Set NaN values to NaN in all one-hot columns
                    nan_mask = my_isnan(col_data)
                    onehot[nan_mask, :] = np.nan
                    
                    # Store mapping
                    onehot_start = current_col_idx
                    onehot_end = current_col_idx + n_categories
                    cat_mappings[col_idx] = (unique_cats, onehot_start, onehot_end)
                    
                    # Add one-hot columns
                    X_onehot_list.append(onehot)
                    current_col_idx += n_categories
                else:
                    # All values are NaN, create a single column with all NaN
                    onehot = np.full((n_samples, 1), np.nan)
                    onehot_start = current_col_idx
                    onehot_end = current_col_idx + 1
                    cat_mappings[col_idx] = (np.array([]), onehot_start, onehot_end)
                    X_onehot_list.append(onehot)
                    current_col_idx += 1
            else:
                # Numerical column, keep as is
                num_mappings[col_idx] = current_col_idx
                X_onehot_list.append(X[:, col_idx:col_idx+1])
                current_col_idx += 1
        
        # Combine all columns into one matrix
        X_onehot = np.hstack(X_onehot_list).astype(np.float32)
        
        # Run imputation on the one-hot encoded matrix
        X_onehot_imputed = self.imputer.impute(X_onehot)
        
        # Convert back to original format
        X_imputed = np.zeros_like(X)
        
        for col_idx in range(n_features):
            if col_idx in categorical_columns:
                # Extract one-hot columns for this categorical column
                unique_cats, onehot_start, onehot_end = cat_mappings[col_idx]
                
                if len(unique_cats) > 0:
                    # Extract the one-hot columns
                    onehot_cols = X_onehot_imputed[:, onehot_start:onehot_end]
                    
                    # Apply softmax to get probabilities
                    # Handle NaN values: set them to 0 before softmax, then normalize
                    onehot_cols_clean = np.where(my_isnan(onehot_cols), 0.0, onehot_cols)
                    
                    # Softmax: numerically stable implementation from scipy
                    probs = softmax(onehot_cols_clean, axis=1)
                    
                    # Choose the class with highest probability
                    predicted_indices = np.argmax(probs, axis=1)
                    
                    # Convert back to original categorical values
                    X_imputed[:, col_idx] = unique_cats[predicted_indices]
                else:
                    # No categories found, keep as NaN
                    X_imputed[:, col_idx] = np.nan
            else:
                # Numerical column, copy directly using the mapping
                onehot_col_idx = num_mappings[col_idx]
                X_imputed[:, col_idx] = X_onehot_imputed[:, onehot_col_idx]
        
        return X_imputed


# How to use:
"""
from mcpfn.model.interface import ImputePFN

imputer = ImputePFN(device='cpu') # cuda if you have a GPU

X = np.random.rand(10, 10) # Test matrix of size 10 x 10
X[np.random.rand(*X.shape) < 0.1] = np.nan # Set 10% of values to NaN

out = imputer.impute(X) # Impute the missing values
print(out)
"""

import time
if __name__ == "__main__":
    # --- Test new chunking (entry_wise_features=True, no checkpoint needed) ---
    print("Testing chunking path (entry_wise_features=True)...")
    np.random.seed(42)
    X_test = np.random.randn(80, 8)
    X_test = (X_test - X_test.mean(axis=0)) / (X_test.std(axis=0) + 1e-8)
    X_test[np.random.rand(*X_test.shape) < 0.15] = np.nan

    imputer_no_chunk = ImputePFN(device="cpu", entry_wise_features=True, max_num_rows=None)
    imputer_chunk = ImputePFN(
        device="cpu",
        entry_wise_features=True,
        max_num_rows=25,
        max_num_cols=4,
    )

    out_no_chunk, full_no_chunk = imputer_no_chunk.impute(X_test.copy(), return_full=True)
    out_chunk, full_chunk = imputer_chunk.impute(X_test.copy(), return_full=True)

    assert out_no_chunk.shape == out_chunk.shape == X_test.shape
    assert full_no_chunk.shape == full_chunk.shape == X_test.shape
    assert not np.any(np.isnan(out_chunk)), "Chunked output should have no NaN"
    assert not np.any(np.isnan(full_chunk)), "Chunked full should have no NaN"
    print("  Chunking test passed: shapes correct, no NaN in output.")
    
    exit()

    # --- Benchmark (requires checkpoint for entry_wise_features=False) ---
    try:
        imputer = ImputePFN(device='cuda', 
                            entry_wise_features=False, 
                            checkpoint_path='/home/jacobf18/tabular/mcpfn/src/tabimpute/workdir/tabimpute-mcar_p0.4-num_cls_8-rank_1_11/checkpoint_60000.pth')
    except FileNotFoundError:
        print("Skipping benchmark: checkpoint not found.")
        exit(0)

    imputer_old = ImputePFN(device='cuda', entry_wise_features=True)
    
    print(f"New Model size: {sum(p.numel() for p in imputer.model.parameters()):,}")
    print(f"Old Model size: {sum(p.numel() for p in imputer_old.model.parameters()):,}")
    
    # X = np.arange(25).reshape(5, 5) # Test matrix of size 10 x 10
    X = np.random.randn(100,10)
    # X = np.random.randn(5,5)
    X = (X - X.mean(axis=0) )/ X.std(axis=0)
    # print(X)
    # X[list(range(5)), list(range(5))] = np.nan # Set 10% of values to NaN
    X[np.random.rand(*X.shape) < 0.1] = np.nan # Set 10% of values to NaN
    # print(X)
    start_time = time.time()
    out, full = imputer.impute(X, return_full=True) # Impute the missing values
    end_time = time.time()
    speed_new = end_time - start_time
    print(f"New Model Time taken: {end_time - start_time:.4f} seconds")
    
    start_time = time.time()
    out_old, full_old = imputer_old.impute(X, return_full=True)
    end_time = time.time()
    speed_old = end_time - start_time
    print(f"Old Model Time taken: {end_time - start_time:.4f} seconds")
    
    print(f"Speedup: {speed_old / speed_new:.4f} x")
    
    exit()
    
    print(np.nanmean(X, axis=0))
    print(out)
    print(full)
    
    
    attn_weights = imputer.model.transformer_blocks[0].get_attention_weights(attention_type="datapoints")

    # Look at the weights for a [MASK] token
    # Attention weights shape: (batch*rows, num_heads, seq_len, seq_len) or (batch*rows, seq_len, seq_len)
    if attn_weights is not None:
        print(f"Attention weights shape: {attn_weights.shape}")
        # If multi-head, average across heads or select first head
        if attn_weights.dim() == 4:
            # Multi-head: (batch*rows, num_heads, seq_len, seq_len)
            attn_weights = attn_weights.mean(dim=1)  # Average across heads
        # Now shape is (batch*rows, seq_len, seq_len)
        print(attn_weights[0, 0, :])  # First row, first query position, attention to all keys
    else:
        print("Attention weights not available. Run a forward pass first.")
