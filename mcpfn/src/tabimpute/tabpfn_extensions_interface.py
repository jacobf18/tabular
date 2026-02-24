from __future__ import annotations

import numpy as np
import torch
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised

from tabimpute.prior.splits import create_train_test_sets


class TabPFNImputer:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.reg = TabPFNRegressor(device=device, n_estimators=8)

    def impute(
        self, X: np.ndarray, return_full: bool = False, num_repeats: int = 1
    ) -> np.ndarray:
        """Impute missing values in the input matrix."""
        X_tensor = torch.from_numpy(X).to(self.device)

        train_X, train_y, test_X, _ = create_train_test_sets(X_tensor, X_tensor)
        train_X_npy = train_X.cpu().numpy()
        train_y_npy = train_y.cpu().numpy()
        test_X_npy = test_X.cpu().numpy()

        missing_indices = np.where(np.isnan(X))

        self.reg.fit(train_X_npy, train_y_npy)

        if num_repeats > 1:
            out_full = self.reg.predict(test_X_npy, output_type="full")
            bar_dist = out_full["criterion"]
            logits = out_full["logits"]
            out_normalized = []

            for _ in range(num_repeats):
                sample = bar_dist.sample(logits=logits)
                X_imputed = X.copy()
                X_imputed[missing_indices] = sample.cpu().detach().numpy()
                out_normalized.append(X_imputed)

            return out_normalized

        preds = self.reg.predict(test_X_npy)
        preds_train = self.reg.predict(train_X_npy)

        X_full = X.copy()
        mask = np.isnan(X)
        X[mask] = preds
        X_full[mask] = preds
        X_full[~mask] = preds_train

        del train_X, train_y, test_X, X_tensor
        torch.cuda.empty_cache()

        if return_full:
            return X, X_full
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
