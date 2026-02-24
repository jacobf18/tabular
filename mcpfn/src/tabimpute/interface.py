from __future__ import annotations

import torch

from tabimpute.prior.splits import create_train_test_sets

import numpy as np
from tabimpute.model.mcpfn import MCPFN
from tabimpute.model.bar_distribution import FullSupportBarDistribution

import importlib.resources as resources
from huggingface_hub import hf_hub_download
from tabimpute.model.encoders import normalize_data
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tabimpute.prepreocess import Preprocess

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
    ):
        self.device = device

        self.model = MCPFN(nhead=nhead).to(self.device).to(torch.bfloat16)
        self.model.eval()
        # torch.compile(self.model)

        # Load borders tensor for outputting continuous values
        with resources.files("tabimpute.data").joinpath("borders.pt").open("rb") as f:
            self.borders = torch.load(f, map_location=self.device)

        if checkpoint_path is None:
            checkpoint_path = get_model_from_huggingface()

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(checkpoint["state_dict"])

        self.preprocessors = preprocessors
        self.max_num_rows = max_num_rows
        if max_num_chunks is None:
            max_num_chunks = float('inf')
        self.max_num_chunks = max_num_chunks
        # Get the median predictions
        borders = self.borders.to(self.device)
        self.bar_distribution = FullSupportBarDistribution(borders=borders)
        self.verbose = verbose
        
    def impute(self, X: np.ndarray, return_full: bool = False, num_repeats: int = 1) -> np.ndarray:
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

        # Get means and stds per column
        means = np.nanmean(X, axis=0)
        stds = np.nanstd(X, axis=0)
        
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
    
    def _get_imputation_chunk(self, X_normalized: np.ndarray, num_repeats: int = 1) -> np.ndarray:
        """Get the imputation for a chunk of the input matrix.
        Args:
            X_normalized (np.ndarray): Input matrix of shape (T, H) where:
             - T is the number of samples (rows)
             - H is the number of features (columns)
            num_repeats (int, optional): Number of times to repeat the imputation. Defaults to 1.
        Returns:
            np.ndarray: Imputed matrix of shape (T, H)
        """
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
            
        # All of the row chunks should be the same size except for maybe the last one if the number of rows is not a multiple of the chunk size
        X_batch_chunks = None
        input_y_batch_chunks = None
        X_last_chunk = None
        input_y_last_chunk = None
        
        # parallelize the imputation with batching
        if len(X_input_list) > 1:
            if X_input_list[-1].shape[0] != X_input_list[-2].shape[0]:
                # Run this one separately
                X_last_chunk = X_input_list[-1]
                input_y_last_chunk = input_y_list[-1]
                X_batch_chunks = torch.stack(X_input_list[:-1], dim=0) # shape: (num_chunks, T, H)
                input_y_batch_chunks = torch.stack(input_y_list[:-1], dim=0)
            else:
                X_batch_chunks = torch.stack(X_input_list, dim=0) # shape: (num_chunks, T, H)
                input_y_batch_chunks = torch.stack(input_y_list, dim=0)
        else:
            X_batch_chunks = torch.stack(X_input_list, dim=0) # shape: (num_chunks, T, H)
            input_y_batch_chunks = torch.stack(input_y_list, dim=0)
            
        medians_chunks = None
        medians_last_chunk = None
        
        with torch.no_grad():
            preds = self.model(X_batch_chunks, input_y_batch_chunks) # shape: (num_chunks, T, H)
            # Get the median predictions
            medians_chunks = self.bar_distribution.median(logits=preds)
            
            if X_last_chunk is not None:
                preds_last = self.model(X_last_chunk.unsqueeze(0), input_y_list[-1].unsqueeze(0)) # shape: (1, T, H)
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
    imputer = ImputePFN(device='cuda')
    print(f"Model size: {sum(p.numel() for p in imputer.model.parameters()):,}")
    
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
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    
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
