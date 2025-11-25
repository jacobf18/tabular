from __future__ import annotations

import torch

from .prior.training_set_generation import create_train_test_sets

import numpy as np
from .model.mcpfn import MCPFN
import argparse
from .model.bar_distribution import FullSupportBarDistribution

# from tabpfn import TabPFNRegressor
from .prepreocess import (
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

import einops

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
        checkpoint_path: str = None
    ):
        self.device = device

        # Build model
        self.model = MCPFN(nhead=nhead).to(self.device)
        self.model.eval()
        torch.compile(self.model)

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
        # self.model.model.load_state_dict(torch.load('/root/tabular/mcpfn/data/tabpfn_model.pt', weights_only=True))

        self.preprocessors = preprocessors

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
        X_normalized_tensor = torch.from_numpy(X_normalized).to(self.device)

        # Impute the missing values
        train_X, train_y, test_X, test_y = create_train_test_sets(
            X_normalized_tensor, X_normalized_tensor
        )

        # Normalize the train_y and test_y
        # train_y = (train_y - train_y.mean()) / (train_y.std() + 1e-8)
        # train_y, mean, std = standardize_excluding_outliers_torch(train_y)

        input_y = torch.cat(
            (train_y, torch.full_like(test_y, torch.nan, device=self.device)), dim=0
        )

        missing_indices = np.where(
            np.isnan(X_normalized)
        )  # This will always be the same order as the calculated train_X and test_X
        non_missing_indices = np.where(~np.isnan(X_normalized))

        # Move tensors to device
        train_X = train_X.to(self.device)
        input_y = input_y.to(self.device)
        test_X = test_X.to(self.device)

        # batch = (train_X.unsqueeze(0), train_y.unsqueeze(0), test_X.unsqueeze(0), None)

        # Impute missing entries with means
        X_input = torch.cat((train_X, test_X), dim=0)
        X_input = X_input.unsqueeze(0)

        X_input = X_input.float()
        input_y = input_y.float()

        with torch.no_grad():
            preds = self.model(X_input, input_y.unsqueeze(0))

            # Get the median predictions
            borders = self.borders.to(self.device)
            bar_distribution = FullSupportBarDistribution(borders=borders)

            medians = bar_distribution.median(logits=preds).flatten()
            
            if num_repeats > 1:
                out_full = []
                out_normalized = []
                
                for _ in range(num_repeats):
                    sample = bar_distribution.sample(logits=preds.squeeze(0))
                    X_full = X_normalized.copy()
                    X_imputed = X_normalized.copy()
                    
                    sampls_train = sample[:train_y.shape[0]]
                    sampls_test = sample[train_y.shape[0]:]
                    X_full[missing_indices] = sampls_test.cpu().detach().numpy()
                    X_full[non_missing_indices] = sampls_train.cpu().detach().numpy()
                    X_imputed[missing_indices] = sampls_test.cpu().detach().numpy()
                    out_full.append(X_full)
                    out_normalized.append(X_imputed)
                
                return out_normalized, out_full

        # Denormalize the predictions with train y mean and std
        # medians = medians * (train_y.std() + 1e-8) + train_y.mean()

        X_full = X_normalized.copy()

        medians_train = medians[: train_y.shape[0]]
        medians_test = medians[train_y.shape[0] :]

        # Impute the missing values with the median predictions
        X_normalized[missing_indices] = medians_test.cpu().detach().numpy()

        X_full[missing_indices] = medians_test.cpu().detach().numpy()
        X_full[non_missing_indices] = medians_train.cpu().detach().numpy()

        return X_normalized, X_full
    
class TabImpute(ImputePFN):
    def __init__(self, device: str = "cpu", nhead: int = 2, preprocessors: list[Preprocess] = None, checkpoint_path: str = None):
        super().__init__(device=device, nhead=nhead, preprocessors=preprocessors, checkpoint_path=checkpoint_path)
        
    def impute(self, X: np.ndarray, return_full: bool = False, num_repeats: int = 1) -> np.ndarray:
        """Impute missing values in the input matrix.
        Imputes the missing values in place.
        """
        return super().impute(X, return_full=return_full, num_repeats=num_repeats)


class TabPFNImputer:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.reg = TabPFNRegressor(device=device, n_estimators=8)

    def impute(self, X: np.ndarray, return_full: bool = False) -> np.ndarray:
        """Impute missing values in the input matrix.
        Imputes the missing values in place.
        """
        X_tensor = torch.from_numpy(X).to(self.device)

        # Impute the missing values
        train_X, train_y, test_X, _ = create_train_test_sets(X_tensor, X_tensor)

        train_X_npy = train_X.cpu().numpy()
        train_y_npy = train_y.cpu().numpy()
        test_X_npy = test_X.cpu().numpy()

        self.reg.fit(train_X_npy, train_y_npy)

        # self.reg.model_ = mcpfn.model # override the model with the mcpfn model

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
