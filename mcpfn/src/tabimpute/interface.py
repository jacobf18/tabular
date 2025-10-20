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
    ):
        self.device = device

        # Build model
        self.model = MCPFN(nhead=nhead).to(self.device)

        # Load borders tensor for outputting continuous values
        with resources.files("tabimpute.data").joinpath("borders.pt").open("rb") as f:
            self.borders = torch.load(f, map_location=self.device)

        checkpoint_path = get_model_from_huggingface()

        # Load model state dict
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        # self.model.model.load_state_dict(torch.load('/root/tabular/mcpfn/data/tabpfn_model.pt', weights_only=True))

        self.preprocessors = preprocessors

    def impute(self, X: np.ndarray, return_full: bool = False) -> np.ndarray:
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

        # Normalize the input matrix
        X_normalized = (X - means) / (stds + 1e-16)
        # Add a small epsilon to avoid division by zero

        # If any preprocessors, do ensemble of them
        X_imputed = np.zeros_like(X_normalized)
        X_full = np.zeros_like(X_normalized)
        if self.preprocessors is not None:
            for preprocessor in self.preprocessors:
                X_preprocessed = preprocessor.fit_transform(X_normalized)
                imput, X_full = self.get_imputation(X_preprocessed)
                X_imputed += preprocessor.inverse_transform(imput)
                X_full += preprocessor.inverse_transform(X_full)
            X_imputed /= len(self.preprocessors)
        else:
            X_imputed, X_full = self.get_imputation(X_normalized)

        torch.cuda.empty_cache()
        
        # Add back the means and stds
        X_imputed = X_imputed * (stds + 1e-16) + means
        X_full = X_full * (stds + 1e-16) + means

        if return_full:
            return X_imputed, X_full
        else:
            return X_imputed

    def get_imputation(self, X_normalized: np.ndarray) -> np.ndarray:
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
            stds = torch.sqrt(bar_distribution.variance(logits=preds).flatten())

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
        y_missing = X[missing_mask]
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


# How to use:
"""
from mcpfn.model.interface import ImputePFN

imputer = ImputePFN(device='cpu') # cuda if you have a GPU

X = np.random.rand(10, 10) # Test matrix of size 10 x 10
X[np.random.rand(*X.shape) < 0.1] = np.nan # Set 10% of values to NaN

out = imputer.impute(X) # Impute the missing values
print(out)
"""
