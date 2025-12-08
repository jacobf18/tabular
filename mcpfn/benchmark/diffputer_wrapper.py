"""
DiffPuter wrapper for tabimpute benchmark.
Adapted from https://github.com/hengruizhang98/DiffPuter
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings

# Add DiffPuter to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DiffPuter'))

from model import MLPDiffusion, Model
from diffusion_utils import impute_mask

warnings.filterwarnings('ignore')


class DiffPuterImputer:
    """
    DiffPuter imputer adapted for tabimpute benchmark.
    Uses diffusion models for missing value imputation with EM algorithm.
    """
    
    def __init__(
        self,
        hid_dim=1024,
        max_iter=1,  # Reduced from 10 for faster benchmarking
        num_trials=10,  # Reduced from 20 for faster benchmarking
        num_steps=50,
        num_epochs=3000,  # Reduced from 10000 for faster benchmarking
        early_stop_patience=500,
        batch_size=4096,
        lr=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize DiffPuter imputer.
        
        Args:
            hid_dim: Hidden dimension for MLP diffusion model
            max_iter: Maximum EM iterations
            num_trials: Number of sampling trials for imputation
            num_steps: Number of diffusion steps
            num_epochs: Number of training epochs per EM iteration
            early_stop_patience: Early stopping patience
            batch_size: Batch size for training
            lr: Learning rate
            device: Device to use ('cuda' or 'cpu')
        """
        self.hid_dim = hid_dim
        self.max_iter = max_iter
        self.num_trials = num_trials
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.early_stop_patience = early_stop_patience
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        
        self.mean_X = None
        self.std_X = None
        self.in_dim = None

    def _compute_mean_std(self, X, mask):
        """Compute mean and std from observed values only."""
        mean_X = np.zeros(X.shape[1])
        std_X = np.ones(X.shape[1])
        
        for i in range(X.shape[1]):
            observed = X[mask[:, i] == 1, i]
            if len(observed) > 0:
                mean_X[i] = np.mean(observed)
                std_X[i] = np.std(observed)
                if std_X[i] == 0:
                    std_X[i] = 1.0
        
        return mean_X, std_X

    def fit_transform(self, X):
        """
        Fit and transform in one step using EM algorithm.
        
        Args:
            X: Input data with missing values (numpy array or pandas DataFrame)
            
        Returns:
            Imputed data (numpy array)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Create mask (1 = observed, 0 = missing)
        mask = ~np.isnan(X)
        
        # Compute normalization parameters
        self.mean_X, self.std_X = self._compute_mean_std(X, mask)
        
        # Normalize
        X_normalized = (X - self.mean_X) / self.std_X / 2
        X_normalized = np.nan_to_num(X_normalized, nan=0.0)
        
        X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        
        self.in_dim = X.shape[1]
        
        # EM Algorithm
        X_imputed = X_tensor.clone()
        
        for iteration in range(self.max_iter):
            # print(f"DiffPuter EM iteration {iteration + 1}/{self.max_iter}")
            
            # M-Step: Train diffusion model
            if iteration == 0:
                train_data = ((1 - mask_tensor) * X_tensor).numpy()
            else:
                train_data = X_imputed.numpy()
            
            train_loader = DataLoader(
                train_data,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,  # Set to 0 for compatibility
            )
            
            # Build model
            denoise_fn = MLPDiffusion(self.in_dim, self.hid_dim).to(self.device)
            model = Model(denoise_fn=denoise_fn, hid_dim=self.in_dim).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0)
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=0.9, patience=50
            )
            
            model.train()
            best_loss = float('inf')
            patience = 0
            
            # Training loop
            for epoch in range(self.num_epochs):
                batch_loss = 0.0
                len_input = 0
                
                for batch in train_loader:
                    inputs = batch.float().to(self.device)
                    loss = model(inputs)
                    loss = loss.mean()
                    
                    batch_loss += loss.item() * len(inputs)
                    len_input += len(inputs)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                curr_loss = batch_loss / len_input
                scheduler.step(curr_loss)
                
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    patience = 0
                    # Save best model state
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= self.early_stop_patience:
                        print(f'Early stopping at epoch {epoch}')
                        break
                
                if epoch % 500 == 0:
                    print(f'  Epoch {epoch}, Loss: {curr_loss:.6f}')
            
        # Load best model
        model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
        
        # E-Step: Impute missing values
        model.eval()
        rec_Xs = []
        
        for trial in range(self.num_trials):
            X_miss = ((1 - mask_tensor) * X_tensor).to(self.device)
            
            with torch.no_grad():
                rec_X = impute_mask(
                    model.denoise_fn_D,
                    X_miss,
                    mask_tensor,
                    X_tensor.shape[0],
                    X_tensor.shape[1],
                    self.num_steps,
                    self.device
                )
            
            # Keep observed values
            mask_device = mask_tensor.to(self.device)
            rec_X = rec_X * mask_device + X_miss * (1 - mask_device)
            rec_Xs.append(rec_X.cpu())
        
        # Average over trials
        X_imputed = torch.stack(rec_Xs, dim=-1).mean(dim=-1)
        print(X_imputed)
        
        # Clean up
        del model, denoise_fn, optimizer, scheduler
        torch.cuda.empty_cache()
        
        # Denormalize
        X_imputed = X_imputed.numpy() * 2
        X_imputed = X_imputed * self.std_X + self.mean_X
        
        return X_imputed


def impute_diffputer(X_missing: np.ndarray, **kwargs) -> np.ndarray:
    """
    Convenience function for DiffPuter imputation.
    
    Args:
        X_missing: numpy array with NaN for missing values
        **kwargs: Additional arguments for DiffPuterImputer
        
    Returns:
        Imputed numpy array
    """
    imputer = DiffPuterImputer(**kwargs)
    return imputer.fit_transform(X_missing)

if __name__ == "__main__":
    X = np.random.rand(5, 5)
    X[np.random.rand(5) < 0.5] = np.nan
    print(X)
    imputer = DiffPuterImputer()
    X_imputed = imputer.fit_transform(X)
    print(X_imputed)
