from __future__ import annotations

import torch

from tabimpute.prior.splits import create_train_test_sets
from tabimpute.prior.training_set_generation import LatentFactorPrior

import numpy as np
from tabimpute.model.mcpfn import MCPFN
from tabimpute.model.bar_distribution import FullSupportBarDistribution

import importlib.resources as resources
from huggingface_hub import hf_hub_download
from tabimpute.model.encoders import normalize_data
# from tabimpute.model.model_new import TabImputeModel
# from tabimpute.model.model import TabImputeModel
from tabimpute.model.model_new_stable import TabImputeModel


def get_model_from_huggingface() -> str:
    repo_id = "Tabimpute/TabImpute"
    filename = "tabimpute_001.ckpt"
    return hf_hub_download(repo_id=repo_id, filename=filename)


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
        preprocessors: list[Any] = None,
        checkpoint_path: str = None,
        max_num_rows: int = None,
        max_num_chunks: int = None,
        verbose: bool = False,
        entry_wise_features: bool = True,
        json_config: dict = None,
    ):
        self.device = device

        # Build model
        if entry_wise_features:
            self.model = MCPFN(nhead=nhead).to(self.device).to(torch.bfloat16)
        else:
            # num_attention_heads = 32
            # embedding_size = 32 * num_attention_heads
            # mlp_hidden_size = 1024
            # num_cls = 12
            # num_layers = 12
            # self.model = TabImputeModel(embedding_size=embedding_size, 
            #                             num_attention_heads=num_attention_heads, 
            #                             mlp_hidden_size=mlp_hidden_size, 
            #                             num_layers=num_layers,
            #                             num_cls=num_cls,
            #                             num_outputs=5000).to(self.device).to(torch.bfloat16)
            num_attention_heads = json_config["config"]["num_attention_heads"]
            embedding_size = json_config["config"]["embedding_size"]
            mlp_hidden_size = json_config["config"]["mlp_hidden_size"]
            num_cls = json_config["config"]["num_cls"]
            num_layers = json_config["config"]["num_layers"]
            rope_base = 10000.0
            rope_fraction = json_config["config"]["rope_fraction"]
            attention_dropout = json_config["config"]["attention_dropout"]
            ffn_dropout = json_config["config"]["ffn_dropout"]
            drop_path_rate = json_config["config"]["drop_path_rate"]
            residual_scale_init = json_config["config"]["residual_scale_init"]
            embedding_dropout = json_config["config"]["embedding_dropout"]
            
            # self.model = TabImputeModel(
            #                 embedding_size=embedding_size,
            #                 num_attention_heads=num_attention_heads,
            #                 mlp_hidden_size=mlp_hidden_size,
            #                 num_layers=num_layers,
            #                 num_outputs=5000,
            #                 num_cls=num_cls,
            #                 use_rope=True,
            #                 rope_base=10000.0,
            #                 rope_fraction=json_config["config"]["rope_fraction"],
            #                 use_absolute_positional_embeddings=False,
            #                 positional_damping_factor=0.1,
            #             ).to('cuda').to(torch.bfloat16)

            self.model = TabImputeModel(
                embedding_size=embedding_size,
                num_attention_heads=num_attention_heads,
                mlp_hidden_size=mlp_hidden_size,
                num_layers=num_layers,
                num_outputs=5000,
                num_cls=num_cls,
                use_rope=True,
                rope_base=rope_base,
                rope_fraction=rope_fraction,
                use_absolute_positional_embeddings=False,
                positional_damping_factor=0.1,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                drop_path_rate=drop_path_rate,
                residual_scale_init=residual_scale_init,
                rms_norm_eps=1e-6,
                embedding_dropout=embedding_dropout,
            ).to("cuda", dtype=torch.bfloat16)
            self.model.eval()
        # torch.compile(self.model)

        # Load borders tensor for outputting continuous values
        with resources.files("tabimpute.data").joinpath("borders.pt").open("rb") as f:
            self.borders = torch.load(f, map_location=self.device)

        if checkpoint_path is None:
            checkpoint_path = get_model_from_huggingface()

            # Load model state dict
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model'], strict=True)

        self.preprocessors = preprocessors
        self.max_num_rows = max_num_rows
        if max_num_chunks is None:
            max_num_chunks = float('inf')
        self.max_num_chunks = max_num_chunks
        # Get the median predictions
        borders = self.borders.to(self.device)
        self.bar_distribution = FullSupportBarDistribution(borders=borders)
        self.verbose = verbose

    def _resolve_normalization_stats(
        self,
        X: np.ndarray,
        means: np.ndarray | None = None,
        stds: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        num_features = X.shape[1]

        computed_means = np.nanmean(X, axis=0)
        computed_stds = np.nanstd(X, axis=0)
        computed_stds = np.where(np.isnan(computed_stds), 1, computed_stds)
        computed_means = np.where(np.isnan(computed_means), 0, computed_means)

        def _validate_stats(name: str, values: np.ndarray | None) -> np.ndarray | None:
            if values is None:
                return None

            arr = np.asarray(values)
            if arr.shape != (num_features,):
                raise ValueError(
                    f"`{name}` must have shape ({num_features},), got {arr.shape}."
                )
            return arr

        means_arr = _validate_stats("means", means)
        stds_arr = _validate_stats("stds", stds)

        if means_arr is None:
            resolved_means = computed_means
        else:
            resolved_means = np.where(np.isnan(means_arr), computed_means, means_arr)

        if stds_arr is None:
            resolved_stds = computed_stds
        else:
            resolved_stds = np.where(np.isnan(stds_arr), computed_stds, stds_arr)
            resolved_stds = np.where(np.isnan(resolved_stds), 1, resolved_stds)

        return resolved_means, resolved_stds
        
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
            means (np.ndarray | None): Optional per-column means to use for
                normalization. If None, computed from `X`.
            stds (np.ndarray | None): Optional per-column standard deviations to
                use for normalization. If None, computed from `X`.

        Returns:
            np.ndarray: Imputed matrix of shape (T, H)
        """
        # Verify that the input matrix is valid
        if X.ndim != 2:
            raise ValueError("Input matrix must be 2-dimensional")

        means, stds = self._resolve_normalization_stats(X, means=means, stds=stds)

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
        """Impute missing values using test-time training on synthetic low-rank data.

        Performs test-time training by: (1) using the given mask, (2) generating
        synthetic low-rank data, (3) training for k steps with the given optimizer,
        (4) outputting predictions from the fine-tuned model.

        When preprocessors are set, runs TTT on each preprocessed view and averages
        the inverse-transformed results (same as standard impute).

        Args:
            X: Input matrix of shape (T, H) with NaN for missing values.
            mask: Boolean mask; True = observed, False = missing. If None, inferred
                from ~np.isnan(X). Ignored when preprocessors are used (mask inferred
                from preprocessed data).
            k: Number of gradient steps for test-time training.
            optimizer: Optimizer for fine-tuning. If None, uses AdamW with lr=1e-4.
            rank: Rank for synthetic low-rank data. If None, min(n_rows, n_cols, 10).
            return_full: If True, return (X_imputed, X_full); else return X_imputed.
            means: Optional per-column means to use for normalization. If None,
                computed from `X`.
            stds: Optional per-column standard deviations to use for normalization.
                If None, computed from `X`.

        Returns:
            Imputed matrix, or (X_imputed, X_full) if return_full.
        """
        if X.ndim != 2:
            raise ValueError("Input matrix must be 2-dimensional")

        means, stds = self._resolve_normalization_stats(X, means=means, stds=stds)

        X_normalized = (X - means) / (stds + 1e-16)

        if self.preprocessors is not None:
            X_imputed = np.zeros_like(X_normalized)
            X_full = np.zeros_like(X_normalized)
            for preprocessor in self.preprocessors:
                X_preprocessed = preprocessor.fit_transform(X_normalized)
                imput, full = self._get_imputation_single_ttt(
                    X_preprocessed,
                    mask=None,  # infer from preprocessed data
                    k=k,
                    optimizer=optimizer,
                    rank=rank,
                )
                X_imputed += preprocessor.inverse_transform(imput)
                X_full += preprocessor.inverse_transform(full)
            X_imputed /= len(self.preprocessors)
            X_full /= len(self.preprocessors)
        else:
            X_imputed, X_full = self._get_imputation_single_ttt(
                X_normalized, mask=mask, k=k, optimizer=optimizer, rank=rank
            )

        X_imputed = X_imputed * (stds + 1e-16) + means
        X_full = X_full * (stds + 1e-16) + means

        if return_full:
            return X_imputed, X_full
        return X_imputed

    def get_imputation(self, X_normalized: np.ndarray, num_repeats: int = 1) -> np.ndarray:
        """Get the imputation for the input matrix.
        If max_num_rows is not None, the input matrix is split into chunks of max_num_rows rows, 
        and the imputation is performed on each chunk.

        Args:
            X_normalized (np.ndarray): Input matrix of shape (T, H) where:
             - T is the number of samples (rows)
             - H is the number of features (columns)
            num_repeats (int, optional): Number of times to repeat the imputation. Defaults to 1.
            max_num_rows (int, optional): Maximum number of rows to impute at once. Defaults to None.

        Returns:
            np.ndarray: Imputed matrix of shape (T, H)
        """
        if self.max_num_rows is None:
            return self._get_imputation_single(X_normalized, num_repeats=num_repeats)
        
        else:
            X_full = X_normalized.copy()
            start_index = 0
            
            if self.verbose:
                from tqdm import tqdm
                pbar = tqdm(total=X_normalized.shape[0], desc="Processed rows")
                
            while start_index < X_normalized.shape[0]:
                end_index = min(start_index + (self.max_num_rows * self.max_num_chunks), X_normalized.shape[0])
                if self.verbose:
                    pbar.update(end_index - start_index)
                X_normalized_chunk = X_normalized[start_index:end_index, :]
                X_normalized_chunk, X_full_chunk = self._get_imputation_chunk(X_normalized_chunk, num_repeats=num_repeats)
                X_normalized[start_index:end_index, :] = X_normalized_chunk
                X_full[start_index:end_index, :] = X_full_chunk
                start_index = end_index
            return X_normalized, X_full
            
        
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
    
    def split_into_chunks(self, arr: np.ndarray, chunk_size: int) -> list[np.ndarray]:
        """
        Split a 2D array into row chunks of size `chunk_size`.
        The final chunk may have fewer rows.
        """
        start_indices = []
        end_indices = []
        chunks = []
        for i in range(0, arr.shape[0], chunk_size):
            start_indices.append(i)
            end_indices.append(min(i + chunk_size, arr.shape[0]))
            chunks.append(arr[start_indices[-1]:end_indices[-1]])
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

        X_full = X_normalized.copy()

        medians_train = medians[: train_size]
        medians_test = medians[train_size :]

        X_normalized[missing_indices] = medians_test.cpu().detach().numpy()
        X_full[missing_indices] = medians_test.cpu().detach().numpy()
        X_full[non_missing_indices] = medians_train.cpu().detach().numpy()

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

        Performs test-time training by: (1) using the given mask to define
        observed/missing pattern, (2) generating synthetic low-rank data,
        (3) training the model for k steps on this data, (4) outputting
        predictions from the fine-tuned model.

        Args:
            X_normalized: Input matrix of shape (T, H), normalized. NaN indicates
                missing values if mask is not provided.
            mask: Boolean mask of shape (T, H). True = observed, False = missing.
                If None, inferred from ~np.isnan(X_normalized).
            k: Number of gradient steps for test-time training.
            optimizer: Optimizer for fine-tuning. If None, uses AdamW with lr=1e-4.
            rank: Rank for synthetic low-rank data. If None, uses
                min(n_rows, n_cols, 10).

        Returns:
            Tuple of (X_imputed, X_full) where X_imputed has missing values filled,
            and X_full has both observed and imputed values (observed at their
            original positions, imputed at missing).
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
            # No missing values, return as-is
            X_full = X_normalized.copy()
            return X_normalized.copy(), X_full

        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        # Save original weights to restore after inference
        state_before = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        try:
            self.model.train()
            for _ in range(k):
                # Generate new synthetic low-rank matrix each step; mask stays the same
                Y_syn = _generate_synthetic_low_rank(n_rows, n_cols, rank, self.borders)
                X_syn = Y_syn.copy()
                X_syn[~mask] = np.nan  # Apply mask: observed = mask True, missing = mask False

                X_syn_tensor = torch.from_numpy(X_syn).to(self.device)
                Y_syn_tensor = torch.from_numpy(Y_syn).to(self.device)

                X_input, input_y, missing_indices, non_missing_indices, train_size = (
                    self._get_input_tensors(X_syn_tensor)
                )
                _, train_y, _, test_y = create_train_test_sets(X_syn_tensor, Y_syn_tensor)
                full_y = torch.cat((train_y, test_y), dim=0)

                X_input = X_input.unsqueeze(0)
                input_y = input_y.unsqueeze(0)
                full_y = full_y.unsqueeze(0).to(torch.bfloat16)

                obs_mask = torch.zeros(1, full_y.shape[1], dtype=torch.bool, device=self.device)
                obs_mask[0, :train_size] = True

                optimizer.zero_grad()
                preds = self.model(X_input, input_y)
                loss = self.bar_distribution(logits=preds, y=full_y)
                missing_loss = loss[~obs_mask].mean()
                missing_loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                # Run inference on the original sample (with its mask)
                X_orig_tensor = torch.from_numpy(X_normalized.copy()).to(self.device)
                (
                    X_input_orig,
                    input_y_orig,
                    missing_indices_orig,
                    non_missing_indices_orig,
                    train_size_orig,
                ) = self._get_input_tensors(X_orig_tensor)
                X_input_orig = X_input_orig.unsqueeze(0)
                input_y_orig = input_y_orig.unsqueeze(0)

                preds = self.model(X_input_orig, input_y_orig)
                medians = self.bar_distribution.median(logits=preds).flatten()

            X_full = X_normalized.copy()
            medians_train = medians[:train_size_orig].cpu().detach().numpy()
            medians_test = medians[train_size_orig:].cpu().detach().numpy()

            X_normalized_out = X_normalized.copy()
            X_normalized_out[missing_indices_orig] = medians_test
            X_full[missing_indices_orig] = medians_test
            X_full[non_missing_indices_orig] = medians_train

            return X_normalized_out, X_full
        finally:
            # Restore original weights so model is unchanged for subsequent calls
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in state_before.items()}
            )

_TABPFN_EXTENSIONS_IMPORT_ERROR = None
try:
    from tabimpute.tabpfn_extensions_interface import (
        TabPFNImputer,
        TabPFNUnsupervisedModel,
    )
except ModuleNotFoundError as exc:
    if exc.name != "tabpfn_extensions":
        raise
    _TABPFN_EXTENSIONS_IMPORT_ERROR = exc

    def _raise_tabpfn_extensions_missing() -> None:
        raise ModuleNotFoundError(
            "tabpfn_extensions is not installed. Install it with "
            "`pip install tabpfn_extensions` or `pip install 'tabimpute[tabpfn_extensions]'` "
            "to use TabPFN extension imputers."
        ) from _TABPFN_EXTENSIONS_IMPORT_ERROR

    class TabPFNImputer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _raise_tabpfn_extensions_missing()

    class TabPFNUnsupervisedModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _raise_tabpfn_extensions_missing()
    
class TabImputeCategorical:
    def __init__(self, 
                 device: str = "cuda",
                 nhead: int = 2,
                 preprocessors: list[Any] = None,
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
        from scipy.special import softmax
        
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
    imputer_chunk = ImputePFN(device="cpu", entry_wise_features=True, max_num_rows=25)

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
    out, full = imputer.impute(X, return_full=True)  # Impute the missing values
    end_time = time.time()
    print(f"Standard impute time: {end_time - start_time:.4f} seconds")

    # Test TTT (test-time training) imputation
    X_ttt = np.random.randn(50, 8)
    X_ttt = (X_ttt - X_ttt.mean(axis=0)) / X_ttt.std(axis=0)
    X_ttt[np.random.rand(*X_ttt.shape) < 0.15] = np.nan
    n_missing = np.isnan(X_ttt).sum()
    print(f"\nTTT test: {X_ttt.shape[0]}x{X_ttt.shape[1]} matrix, {n_missing} missing values")
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
    imputer_pp = ImputePFN(device="cuda", preprocessors=preprocessors)
    X_ttt_pp = np.random.randn(40, 7)
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

    print(np.nanmean(X, axis=0))
    print(out)
    print(full)
    
