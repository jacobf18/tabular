from __future__ import annotations

import torch

from mcpfn.prior.training_set_generation import create_train_test_sets

import numpy as np
from mcpfn.model.mcpfn import MCPFN
import argparse
from mcpfn.model.bar_distribution import FullSupportBarDistribution
from tabpfn import TabPFNRegressor


class ImputePFN:
    """A Tabular Foundation Model for Matrix Completion.

    MCPFN is a transformer-based architecture for matrix completion on tabular data.

    Parameters
    ----------

    """

    def __init__(
        self,
        device: str = "cpu",
        encoder_path: str = "encoder.pth",
        borders_path: str = "borders.pt",
        checkpoint_path: str = "test.ckpt",
        nhead: int = 2,
    ):
        self.device = device

        # Build model
        self.model = MCPFN(encoder_path=encoder_path, nhead=nhead).to(self.device)

        # Load borders tensor for outputting continuous values
        self.borders_path = borders_path

        # Load model state dict
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(checkpoint["state_dict"])

    def impute(self, X: np.ndarray) -> np.ndarray:
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

        X_normalized_tensor = torch.from_numpy(X_normalized).to(self.device)

        # Impute the missing values
        train_X, train_y, test_X, test_y = create_train_test_sets(
            X_normalized_tensor, X_normalized_tensor
        )
        
        input_y = torch.cat((train_y, 
                             torch.full_like(test_y, torch.nan)), 
                            dim=0)

        missing_indices = np.where(
            np.isnan(X)
        )  # This will always be the same order as the calculated train_X and test_X

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
            preds = preds[:, train_y.shape[0]:] # Only keep the predictions for the test set

            # Get the median predictions
            borders = torch.load(self.borders_path).to(self.device)
            bar_distribution = FullSupportBarDistribution(borders=borders)

            medians = bar_distribution.median(logits=preds)

        # Impute the missing values with the median predictions
        X_normalized[missing_indices] = medians.cpu().detach().numpy()

        # Denormalize the imputed matrix
        X_imputed = X_normalized * (stds + 1e-8) + means

        # Clean up memory
        del train_X, train_y, test_X, X_normalized_tensor, X_normalized, X_input, input_y, preds, borders, medians
        torch.cuda.empty_cache()

        return X_imputed  # Return the imputed matrix
    
class TabPFNImputer:
    def __init__(self, device: str = "cpu", encoder_path: str = "encoder.pth"):
        self.device = device
        self.reg = TabPFNRegressor(device=device, n_estimators=8)
        self.encoder_path = encoder_path
        
    def impute(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values in the input matrix.
        Imputes the missing values in place.
        """
        X_tensor = torch.from_numpy(X).to(self.device)
        
        # Impute the missing values
        train_X, train_y, test_X, _ = create_train_test_sets(
            X_tensor, X_tensor
        )
        
        train_X_npy = train_X.cpu().numpy()
        train_y_npy = train_y.cpu().numpy()
        test_X_npy = test_X.cpu().numpy()
        # test_y_npy = test_y.cpu().numpy()
        
        
        mcpfn = MCPFN(encoder_path=self.encoder_path, nhead=2).to(self.device)
        checkpoint = torch.load('/mnt/mcpfn_data/checkpoints/mixed_adaptive/step-49000.ckpt', map_location=self.device, weights_only=True)
        # mcpfn.model.load_state_dict(torch.load('/root/tabular/mcpfn/src/mcpfn/model/tabpfn_model.pt', weights_only=True))
        mcpfn.load_state_dict(checkpoint["state_dict"])
        
        self.reg.fit(train_X_npy, train_y_npy, model=mcpfn.model)
        
        # self.reg.model_ = mcpfn.model # override the model with the mcpfn model
        
        preds = self.reg.predict(test_X_npy)
        
        X[np.where(np.isnan(X))] = preds[train_y.shape[0]:]
        
        # Clean up memory
        del train_X, train_y, test_X, X_tensor
        torch.cuda.empty_cache()
        
        return X


# How to use:
"""
from mcpfn.model.interface import ImputePFN

imputer = ImputePFN(device='cpu', # 'cuda' if you have a GPU
                    encoder_path='./src/mcpfn/model/encoder.pth', # Path to the encoder model
                    borders_path='./borders.pt', # Path to the borders model
                    checkpoint_path='./test.ckpt') 

X = np.random.rand(10, 10) # Test matrix of size 10 x 10
X[np.random.rand(*X.shape) < 0.1] = np.nan # Set 10% of values to NaN

out = imputer.impute(X) # Impute the missing values
print(out)
"""
