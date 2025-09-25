import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings


class Preprocess:
    def __init__(self):
        pass

    def transform(self, X):
        pass

    def inverse_transform(self, X):
        pass

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class SequentialPreprocess(Preprocess):
    def __init__(self, preprocessors):
        super().__init__()
        self.preprocessors = preprocessors

    def fit_transform(self, X):
        for preprocessor in self.preprocessors:
            X = preprocessor.fit_transform(X)
        return X

    def inverse_transform(self, X):
        for preprocessor in self.preprocessors[::-1]:
            X = preprocessor.inverse_transform(X)
        return X

    def fit(self, X):
        raise NotImplementedError("Fit method not implemented for this class.")

    def transform(self, X):
        raise NotImplementedError("Transform method not implemented for this class.")


class RandomRowPermutation(Preprocess):
    def __init__(self):
        super().__init__()
        self.perm = None

    def transform(self, X):
        return X[self.perm]

    def fit(self, X):
        self.perm = np.random.permutation(X.shape[0])
        return self

    def inverse_transform(self, X):
        return X[np.argsort(self.perm)]


class RandomColumnPermutation(Preprocess):
    def __init__(self):
        super().__init__()
        self.perm = None

    def transform(self, X):
        return X[:, self.perm]

    def fit(self, X):
        self.perm = np.random.permutation(X.shape[1])
        return self

    def inverse_transform(self, X):
        return X[:, np.argsort(self.perm)]


class RandomRowColumnPermutation(Preprocess):
    def __init__(self):
        super().__init__()
        self.row_perm = None
        self.col_perm = None

    def transform(self, X):
        return X[self.row_perm, :][:, self.col_perm]

    def fit(self, X):
        self.row_perm = np.random.permutation(X.shape[0])
        self.col_perm = np.random.permutation(X.shape[1])
        return self

    def inverse_transform(self, X):
        return X[np.argsort(self.row_perm), :][:, np.argsort(self.col_perm)]


import numpy as np


class StandardizeWhiten(Preprocess):
    def __init__(self, whiten=False, eps=1e-8):
        super().__init__()
        self.whiten = whiten
        self.eps = eps
        # Params learned in fit
        self.means = None
        self.stds = None
        self.W = None
        self.W_inv = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n, m = X.shape

        # Column means/stds ignoring NaNs
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0) + self.eps

        # Default whitening matrix = identity
        self.W = np.eye(m)
        self.W_inv = np.eye(m)

        if self.whiten:
            # Standardize first for covariance estimation
            X_std = (X - self.means) / self.stds
            X_filled = np.nan_to_num(X_std, nan=0.0)

            # Column covariance
            cov = np.cov(X_filled, rowvar=False, bias=True)

            # Eigen-decomposition for whitening
            eigvals, eigvecs = np.linalg.eigh(cov)
            self.W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + self.eps)) @ eigvecs.T
            self.W_inv = np.linalg.inv(self.W)

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_std = (X - self.means) / self.stds
        X_std_filled = np.nan_to_num(X_std, nan=0.0)  # NaN -> 0
        X_trans = X_std_filled @ self.W
        X_trans[np.isnan(X)] = np.nan
        return X_trans

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float)
        X_filled = np.nan_to_num(X, nan=0.0)
        X_unwhiten = X_filled @ self.W_inv
        X_orig = X_unwhiten * self.stds + self.means
        X_orig[np.isnan(X)] = np.nan
        return X_orig


class PowerTransform(Preprocess):
    """
    Power transform class implementing Box-Cox and Yeo-Johnson transformations.

    Parameters:
    -----------
    method : str, default='yeo-johnson'
        The power transform method to use. Options are 'box-cox' and 'yeo-johnson'.
    standardize : bool, default=True
        Whether to standardize the data after transformation.
    copy : bool, default=True
        Whether to create a copy of the input data.
    """

    def __init__(self, method="yeo-johnson", standardize=True, copy=True):
        super().__init__()
        self.method = method
        self.standardize = standardize
        self.copy = copy
        self.lambdas_ = None
        self.mean_ = None
        self.std_ = None
        self.fitted_ = False

        if method not in ["box-cox", "yeo-johnson"]:
            raise ValueError("Method must be 'box-cox' or 'yeo-johnson'")

    def _box_cox_transform(self, X, lmbda):
        """Apply Box-Cox transformation."""
        X = np.asarray(X)
        result = np.zeros_like(X)

        # Handle NaN values
        nan_mask = np.isnan(X)
        valid_mask = ~nan_mask

        if np.any(valid_mask):
            if lmbda == 0:
                result[valid_mask] = np.log(X[valid_mask])
            else:
                result[valid_mask] = (np.power(X[valid_mask], lmbda) - 1) / lmbda

        # Preserve NaN values
        result[nan_mask] = np.nan

        return result

    def _box_cox_inverse(self, X, lmbda):
        """Apply inverse Box-Cox transformation."""
        X = np.asarray(X)
        result = np.zeros_like(X)

        # Handle NaN values
        nan_mask = np.isnan(X)
        valid_mask = ~nan_mask

        if np.any(valid_mask):
            if lmbda == 0:
                result[valid_mask] = np.exp(X[valid_mask])
            else:
                result[valid_mask] = np.power(lmbda * X[valid_mask] + 1, 1 / lmbda)

        # Preserve NaN values
        result[nan_mask] = np.nan

        return result

    def _yeo_johnson_transform(self, X, lmbda):
        """Apply Yeo-Johnson transformation."""
        X = np.asarray(X)
        result = np.zeros_like(X)

        # Handle NaN values
        nan_mask = np.isnan(X)
        valid_mask = ~nan_mask

        if np.any(valid_mask):
            X_valid = X[valid_mask]
            result_valid = np.zeros_like(X_valid)

            # For positive values
            pos_mask = X_valid >= 0
            if np.any(pos_mask):
                if lmbda == 0:
                    result_valid[pos_mask] = np.log(X_valid[pos_mask] + 1)
                else:
                    result_valid[pos_mask] = (
                        np.power(X_valid[pos_mask] + 1, lmbda) - 1
                    ) / lmbda

            # For negative values
            neg_mask = X_valid < 0
            if np.any(neg_mask):
                if lmbda == 2:
                    result_valid[neg_mask] = -np.log(-X_valid[neg_mask] + 1)
                else:
                    result_valid[neg_mask] = -(
                        np.power(-X_valid[neg_mask] + 1, 2 - lmbda) - 1
                    ) / (2 - lmbda)

            result[valid_mask] = result_valid

        # Preserve NaN values
        result[nan_mask] = np.nan

        return result

    def _yeo_johnson_inverse(self, X, lmbda):
        """Apply inverse Yeo-Johnson transformation."""
        X = np.asarray(X)
        result = np.zeros_like(X)

        # Handle NaN values
        nan_mask = np.isnan(X)
        valid_mask = ~nan_mask

        if np.any(valid_mask):
            X_valid = X[valid_mask]
            result_valid = np.zeros_like(X_valid)

            # For positive values
            pos_mask = X_valid >= 0
            if np.any(pos_mask):
                if lmbda == 0:
                    result_valid[pos_mask] = np.exp(X_valid[pos_mask]) - 1
                else:
                    result_valid[pos_mask] = (
                        np.power(lmbda * X_valid[pos_mask] + 1, 1 / lmbda) - 1
                    )

            # For negative values
            neg_mask = X_valid < 0
            if np.any(neg_mask):
                if lmbda == 2:
                    result_valid[neg_mask] = 1 - np.exp(-X_valid[neg_mask])
                else:
                    result_valid[neg_mask] = 1 - np.power(
                        -(2 - lmbda) * X_valid[neg_mask] + 1, 1 / (2 - lmbda)
                    )

            result[valid_mask] = result_valid

        # Preserve NaN values
        result[nan_mask] = np.nan

        return result

    def _log_likelihood_box_cox(self, lmbda, X):
        """Compute log-likelihood for Box-Cox transformation."""
        if lmbda == 0:
            transformed = np.log(X)
        else:
            transformed = (np.power(X, lmbda) - 1) / lmbda

        # Log-likelihood
        n = len(X)
        log_likelihood = -n / 2 * np.log(np.var(transformed)) + (lmbda - 1) * np.sum(
            np.log(X)
        )
        return -log_likelihood  # Minimize negative log-likelihood

    def _log_likelihood_yeo_johnson(self, lmbda, X):
        """Compute log-likelihood for Yeo-Johnson transformation."""
        transformed = self._yeo_johnson_transform(X, lmbda)

        # Log-likelihood
        n = len(X)
        log_likelihood = -n / 2 * np.log(np.var(transformed))

        # Add Jacobian term
        pos_mask = X >= 0
        neg_mask = X < 0

        if np.any(pos_mask):
            if lmbda == 0:
                log_likelihood += (lmbda - 1) * np.sum(np.log(X[pos_mask] + 1))
            else:
                log_likelihood += (lmbda - 1) * np.sum(np.log(X[pos_mask] + 1))

        if np.any(neg_mask):
            if lmbda == 2:
                log_likelihood += (lmbda - 1) * np.sum(np.log(-X[neg_mask] + 1))
            else:
                log_likelihood += (lmbda - 1) * np.sum(np.log(-X[neg_mask] + 1))

        return -log_likelihood  # Minimize negative log-likelihood

    def fit(self, X):
        """
        Fit the power transform to the data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The data to fit the transform to.

        Returns:
        --------
        self : PowerTransform
            Returns self for method chaining.
        """
        X = np.asarray(X)
        if self.copy:
            X = X.copy()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.lambdas_ = np.zeros(n_features)

        for i in range(n_features):
            column = X[:, i]

            # Remove any NaN or infinite values
            valid_mask = np.isfinite(column)
            if not np.any(valid_mask):
                warnings.warn(
                    f"Column {i} contains no finite values. Setting lambda to 1."
                )
                self.lambdas_[i] = 1.0
                continue

            column = column[valid_mask]

            if self.method == "box-cox":
                # Box-Cox requires positive values
                if np.any(column <= 0):
                    warnings.warn(
                        f"Column {i} contains non-positive values. "
                        "Box-Cox requires positive values. Using Yeo-Johnson instead."
                    )
                    # Fall back to Yeo-Johnson for this column
                    result = minimize_scalar(
                        self._log_likelihood_yeo_johnson,
                        args=(column,),
                        bounds=(-2, 2),
                        method="bounded",
                    )
                else:
                    result = minimize_scalar(
                        self._log_likelihood_box_cox,
                        args=(column,),
                        bounds=(-2, 2),
                        method="bounded",
                    )
            else:  # yeo-johnson
                result = minimize_scalar(
                    self._log_likelihood_yeo_johnson,
                    args=(column,),
                    bounds=(-2, 2),
                    method="bounded",
                )

            self.lambdas_[i] = result.x

        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Apply the power transform to the data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The data to transform.

        Returns:
        --------
        X_transformed : array-like of shape (n_samples, n_features)
            The transformed data.
        """
        if not self.fitted_:
            raise ValueError("PowerTransform must be fitted before transform")

        X = np.asarray(X)
        if self.copy:
            X = X.copy()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        if n_features != len(self.lambdas_):
            raise ValueError(
                f"Number of features ({n_features}) doesn't match "
                f"number of fitted lambdas ({len(self.lambdas_)})"
            )

        X_transformed = np.zeros_like(X)

        for i in range(n_features):
            if self.method == "box-cox":
                X_transformed[:, i] = self._box_cox_transform(X[:, i], self.lambdas_[i])
            else:  # yeo-johnson
                X_transformed[:, i] = self._yeo_johnson_transform(
                    X[:, i], self.lambdas_[i]
                )

        # Standardize if requested
        if self.standardize:
            if self.mean_ is None or self.std_ is None:
                # Use nanmean and nanstd to handle NaN values
                self.mean_ = np.nanmean(X_transformed, axis=0)
                self.std_ = np.nanstd(X_transformed, axis=0)
                # Avoid division by zero
                self.std_[self.std_ == 0] = 1.0

            X_transformed = (X_transformed - self.mean_) / self.std_

        return X_transformed

    def inverse_transform(self, X):
        """
        Apply the inverse power transform to the data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The transformed data to inverse transform.

        Returns:
        --------
        X_original : array-like of shape (n_samples, n_features)
            The original data.
        """
        if not self.fitted_:
            raise ValueError("PowerTransform must be fitted before inverse_transform")

        X = np.asarray(X)
        if self.copy:
            X = X.copy()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        if n_features != len(self.lambdas_):
            raise ValueError(
                f"Number of features ({n_features}) doesn't match "
                f"number of fitted lambdas ({len(self.lambdas_)})"
            )

        # Reverse standardization if it was applied
        if self.standardize and self.mean_ is not None and self.std_ is not None:
            X = X * self.std_ + self.mean_

        X_original = np.zeros_like(X)

        for i in range(n_features):
            if self.method == "box-cox":
                X_original[:, i] = self._box_cox_inverse(X[:, i], self.lambdas_[i])
            else:  # yeo-johnson
                X_original[:, i] = self._yeo_johnson_inverse(X[:, i], self.lambdas_[i])

        return X_original

    def fit_transform(self, X):
        """
        Fit the power transform and apply it to the data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The data to fit and transform.

        Returns:
        --------
        X_transformed : array-like of shape (n_samples, n_features)
            The transformed data.
        """
        return self.fit(X).transform(X)


import torch


def standardize_excluding_outliers_torch(x, iqr_factor=1.5):
    """
    Standardize a 1D torch tensor while excluding outliers
    from mean and std calculation. Outliers are still transformed
    using the same scaling, not removed.

    Args:
        x (Tensor): 1D tensor of data (can include NaNs).
        iqr_factor (float): Multiplier for IQR when defining outliers.

    Returns:
        standardized (Tensor): Standardized vector (same shape as x).
        mask (Tensor): Boolean mask indicating which values were used
                       in computing mean/std (True = non-outlier).
    """
    x = x.clone().float()

    # Handle NaNs by masking
    nan_mask = ~torch.isnan(x)
    x_valid = x[nan_mask]

    # Quartiles
    q1 = torch.quantile(x_valid, 0.25)
    q3 = torch.quantile(x_valid, 0.75)
    iqr = q3 - q1

    lower = q1 - iqr_factor * iqr
    upper = q3 + iqr_factor * iqr

    # Outlier mask (excluding NaNs)
    mask = (x >= lower) & (x <= upper) & nan_mask

    # Mean/std excluding outliers
    if mask.sum() > 0:
        mean = torch.mean(x[mask])
        std = torch.std(x[mask], unbiased=False)
    else:
        mean, std = torch.tensor(0.0), torch.tensor(1.0)

    # Standardize (keep NaNs in place)
    standardized = (x - mean) / (std if std > 0 else 1.0)
    standardized[~nan_mask] = float("nan")

    return standardized, mean, std
