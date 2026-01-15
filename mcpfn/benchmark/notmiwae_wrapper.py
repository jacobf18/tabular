"""
NotMIWAE wrapper for tabimpute benchmark.
Adapted from notmiwae_pytorch library.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from notmiwae_pytorch import NotMIWAE, Trainer
from notmiwae_pytorch.utils import set_seed, impute
import warnings
import contextlib
import io

warnings.filterwarnings('ignore')


class NotMIWAEImputer:
    """
    NotMIWAE imputer adapted for tabimpute benchmark.
    Uses not-MIWAE (Missing Not At Random) for missing value imputation.
    """
    
    def __init__(
        self,
        latent_dim=50,
        hidden_dim=128,
        n_samples=20,
        out_dist='gauss',  # 'gauss', 'bern', 'student_t'
        missing_process='selfmasking',  # 'selfmasking', 'selfmasking_known_signs', 'linear', 'nonlinear'
        n_epochs=100,
        batch_size=64,
        lr=1e-3,
        n_impute_samples=1000,
        impute_solver='l2',  # 'l2' or 'l1'
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42,
        verbose=False
    ):
        """
        Initialize NotMIWAE imputer.
        
        Args:
            latent_dim: Dimension of the latent space
            hidden_dim: Dimension of hidden layers
            n_samples: Number of importance samples for training
            out_dist: Output distribution ('gauss', 'student_t', or 'bern')
            missing_process: Missing process type ('selfmasking', 'selfmasking_known_signs', 'linear', 'nonlinear')
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            n_impute_samples: Number of importance samples for imputation
            impute_solver: Loss function for imputation ('l2' or 'l1')
            device: Device to use ('cuda' or 'cpu')
            seed: Random seed for reproducibility
            verbose: Whether to print training progress
        """
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_samples = n_samples
        self.out_dist = out_dist
        self.missing_process = missing_process
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_impute_samples = n_impute_samples
        self.impute_solver = impute_solver
        self.device = device
        self.seed = seed
        self.verbose = verbose
        
        self.model = None
        self.input_dim = None
        
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
        Fit and transform in one step.
        
        Args:
            X: Input data with missing values (numpy array or pandas DataFrame)
                Missing values should be represented as np.nan
                
        Returns:
            Imputed data (numpy array)
        """
        # Convert to numpy array and ensure float type
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Ensure X is a numpy array of float type
        X = np.asarray(X, dtype=np.float64)
        
        # Set seed for reproducibility
        set_seed(self.seed)
        
        # Create mask (1 = observed, 0 = missing)
        mask = (~np.isnan(X)).astype(np.float32)
        
        # Fill missing values with 0 (will be replaced during imputation)
        X_filled = np.nan_to_num(X, nan=0.0).astype(np.float32)
        
        # Get input dimension
        self.input_dim = X_filled.shape[1]
        
        # Convert to tensors
        X_tensor = torch.tensor(X_filled, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        
        # Create model
        self.model = NotMIWAE(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            n_samples=self.n_samples,
            missing_process=self.missing_process,
            out_dist=self.out_dist
        ).to(self.device)
        
        # Create dataset and dataloader
        train_dataset = TensorDataset(X_tensor, mask_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Train model
        trainer = Trainer(
            self.model,
            device=self.device,
            lr=self.lr,
            original_data_available=False  # We don't have original data for evaluation
        )
        
        if self.verbose:
            print(f"Training NotMIWAE model on {X.shape[0]} samples with {X.shape[1]} features...")
        
        if self.verbose:
            trainer.train(train_loader, n_epochs=self.n_epochs)
        else:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                trainer.train(train_loader, n_epochs=self.n_epochs, log_interval=10**9)
        
        # Impute missing values
        if self.verbose:
            print(f"Imputing missing values with {self.n_impute_samples} samples...")
        
        X_imputed = impute(
            self.model,
            train_dataset,
            n_samples=self.n_impute_samples,
            batch_size=self.batch_size,
            device=self.device,
            verbose=self.verbose,
            solver=self.impute_solver
        )
        
        return X_imputed


def impute_notmiwae(
    X,
    latent_dim=50,
    hidden_dim=128,
    n_samples=20,
    out_dist='gauss',
    missing_process='selfmasking',
    n_epochs=100,
    batch_size=64,
    lr=1e-3,
    n_impute_samples=1000,
    impute_solver='l2',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    seed=42,
    verbose=False
):
    """
    Convenience function for imputing missing values using NotMIWAE.
    
    Args:
        X: Input data with missing values (numpy array or pandas DataFrame)
           Missing values should be represented as np.nan
        latent_dim: Dimension of the latent space
        hidden_dim: Dimension of hidden layers
        n_samples: Number of importance samples for training
        out_dist: Output distribution ('gauss', 'student_t', or 'bern')
        missing_process: Missing process type ('selfmasking', 'selfmasking_known_signs', 'linear', 'nonlinear')
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        n_impute_samples: Number of importance samples for imputation
        impute_solver: Loss function for imputation ('l2' or 'l1')
        device: Device to use ('cuda' or 'cpu')
        seed: Random seed for reproducibility
        verbose: Whether to print training progress
        
    Returns:
        Imputed data (numpy array)
        
    Example:
        >>> import numpy as np
        >>> from notmiwae_wrapper import impute_notmiwae
        >>> 
        >>> # Create data with missing values
        >>> X = np.random.randn(100, 10)
        >>> X[np.random.rand(*X.shape) < 0.2] = np.nan
        >>> 
        >>> # Impute missing values
        >>> X_imputed = impute_notmiwae(X, n_epochs=50, verbose=True)
    """
    imputer = NotMIWAEImputer(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_samples=n_samples,
        out_dist=out_dist,
        missing_process=missing_process,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        n_impute_samples=n_impute_samples,
        impute_solver=impute_solver,
        device=device,
        seed=seed,
        verbose=verbose
    )
    
    return imputer.fit_transform(X)


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from tabimpute.interface import ImputePFN
    
    # Create sample data with missing values
    np.random.seed(42)
    X = np.random.randn(100, 10)
    X_original = X.copy()
    mask = np.random.rand(*X.shape) < 0.2
    X[mask] = np.nan
    
    print("Original data shape:", X.shape)
    print("Number of missing values:", np.isnan(X).sum())
    
    # Impute missing values with NotMIWAE
    X_imputed = impute_notmiwae(X, n_epochs=50, verbose=False)
    
    print("Imputed data shape:", X_imputed.shape)
    print("Number of missing values after imputation:", np.isnan(X_imputed).sum())
    
    # Compute RMSE for NotMIWAE imputation
    notmiwae_rmse = np.sqrt(np.mean((X_imputed[mask] - X_original[mask])**2))
    print(f'NotMIWAE RMSE: {notmiwae_rmse:.4f}')
    
    # Baseline: impute with observed column means
    X_baseline = X.copy()
    for col_idx in range(X.shape[1]):
        col_data = X[:, col_idx]
        observed_values = col_data[~np.isnan(col_data)]
        if len(observed_values) > 0:
            col_mean = np.mean(observed_values)
            X_baseline[np.isnan(X_baseline[:, col_idx]), col_idx] = col_mean
    
    baseline_rmse = np.sqrt(np.mean((X_baseline[mask] - X_original[mask])**2))
    print(f'Baseline (column means) RMSE: {baseline_rmse:.4f}')
    
    # TabImpute with entry_wise_features=True
    print("\nImputing with TabImpute (entry_wise_features=True)...")
    tabimpute_imputer = ImputePFN(device='cuda' if torch.cuda.is_available() else 'cpu', 
                                    entry_wise_features=True)
    X_tabimpute = tabimpute_imputer.impute(X.copy())
    
    tabimpute_rmse = np.sqrt(np.mean((X_tabimpute[mask] - X_original[mask])**2))
    print(f'TabImpute (entry_wise_features=True) RMSE: {tabimpute_rmse:.4f}')
    
