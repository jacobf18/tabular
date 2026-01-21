"""
CACTI wrapper for tabimpute benchmark.
Adapted from https://github.com/.../CACTI
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import warnings

# Add CACTI to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CACTI'))

from train import load_model, setup_environment
from src.utils.training_helpers import _getscales, _get_ctxembed_size

warnings.filterwarnings('ignore')


class CACTIImputer:
    """
    CACTI imputer adapted for tabimpute benchmark.
    Uses context-aware masked autoencoder for missing value imputation.
    """
    
    def __init__(
        self,
        model='CMAE',  # Default to CMAE (CACTI without context)
        embeddings=None,
        mask_ratio=0.9,
        batch_size=128,
        epochs=300,
        warmup_epochs=50,
        lr=1e-3,
        min_lr=None,
        embed_dim=64,
        nencoder=10,
        ndecoder=4,
        cembed_size=None,
        grad_clip=5.0,
        weight_decay=0.001,
        num_workers=0,
        gpus='0',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        """
        Initialize CACTI imputer.
        
        Args:
            model: Model type ('CACTI', 'CACTIabl', 'RCTXMAE', 'CMAE', 'RMAE')
                  - 'CMAE' (default) and 'RMAE' do NOT require embeddings
                  - 'CACTI', 'CACTIabl', 'RCTXMAE' require embeddings
            embeddings: Path to .npz file containing column embeddings 
                       (required only for CACTI, CACTIabl, RCTXMAE)
            mask_ratio: Masking ratio during training
            batch_size: Batch size for training
            epochs: Number of training epochs
            warmup_epochs: Number of warmup epochs
            lr: Learning rate
            min_lr: Minimum learning rate
            embed_dim: Embedding dimension
            nencoder: Number of encoder layers
            ndecoder: Number of decoder layers
            cembed_size: Context embedding size (auto-detected if None and embeddings provided)
            grad_clip: Gradient clipping value
            weight_decay: Weight decay for optimizer
            num_workers: Number of data loader workers
            gpus: GPU IDs to use
            device: Device to use ('cuda' or 'cpu')
            **kwargs: Additional arguments passed to model
        """
        self.model_type = model
        self.embeddings = embeddings
        self.mask_ratio = mask_ratio
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.min_lr = min_lr
        self.embed_dim = embed_dim
        self.nencoder = nencoder
        self.ndecoder = ndecoder
        self.cembed_size = cembed_size
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.gpus = gpus
        self.device = device
        self.kwargs = kwargs
        
        self.model = None
        self.colnames = None
        
        # Models that require embeddings
        models_requiring_embeddings = ['CACTI', 'CACTIabl', 'RCTXMAE']
        
        # Validate embeddings requirement only for models that need it
        if model in models_requiring_embeddings:
            if embeddings is None:
                raise ValueError(f'embeddings are required to run {model}! '
                               f'Use RMAE or CMAE if you do not have embeddings.')
            # Auto-detect cembed_size if not provided
            if cembed_size is None:
                self.cembed_size = _get_ctxembed_size(embeddings)
        
        # Validate mask_ratio requirement
        if mask_ratio is None:
            raise ValueError('mask_ratio is required!')

    def _create_args_object(self, colnames):
        """
        Create an args object compatible with CACTI's argument parser.
        
        Args:
            colnames: List of column names
            
        Returns:
            Namespace object with all required arguments
        """
        class Args:
            pass
        
        args = Args()
        args.model = self.model_type
        args.tabular = None  # Not used in wrapper
        args.tabularcm = None
        args.tabular_infer = None
        args.finfer = None
        args.batch_size = self.batch_size
        args.epochs = self.epochs
        args.warmup_epochs = self.warmup_epochs
        args.lr = self.lr
        args.min_lr = self.min_lr
        args.splits = None
        args.num_workers = self.num_workers
        args.gpus = self.gpus
        args.grad_clip = self.grad_clip
        args.weight_decay = self.weight_decay
        args.enable_wandb = False
        args.table_size = len(colnames)
        args.resume_checkpoint = None
        args.train_only = False
        args.log_path = None
        args.checkpoint_id = None
        args.binary_map = None
        args.context_size = None
        args.mask_ratio = self.mask_ratio
        args.embed_dim = self.embed_dim
        args.nencoder = self.nencoder
        args.ndecoder = self.ndecoder
        args.embeddings = self.embeddings
        args.cembed_size = self.cembed_size
        args.loss_type = self.kwargs.get('loss_type', None)
        args.ctx_prop = self.kwargs.get('ctx_prop', None)
        args.precision = '32-true'
        args.save_path = None
        args.description = f'{self.model_type}'
        
        # Set device
        args.device = torch.device(self.device)
        
        return args

    def fit_transform(self, X):
        """
        Fit and transform in one step.
        
        Args:
            X: Input data with missing values (numpy array or pandas DataFrame)
            
        Returns:
            Imputed data (numpy array)
        """
        if isinstance(X, pd.DataFrame):
            colnames = X.columns.tolist()
            X = X.values
        else:
            colnames = [f'col_{i}' for i in range(X.shape[1])]
        
        # Create mask (1 = observed, 0 = missing)
        mask = ~np.isnan(X)
        
        # Compute normalization parameters from observed values only
        minscale, maxscale = _getscales(X)
        
        # Create data_dict compatible with CACTI
        data_dict = {
            'data': X,
            'minscale': minscale,
            'maxscale': maxscale,
            'colnames': colnames,
            'fname': 'temp_data'
        }
        
        # Create args object
        args = self._create_args_object(colnames)
        
        # Setup environment (sets CUDA, device, etc.)
        # Note: setup_environment requires CUDA, so we handle CPU mode separately
        try:
            if self.device == 'cuda':
                setup_environment(args)
            else:
                # For CPU mode, manually set device and skip CUDA setup
                args.device = torch.device('cpu')
                args.checkpoint_path = None
        except RuntimeError as e:
            # If CUDA is required but not available, fall back to CPU if requested
            if self.device == 'cpu':
                args.device = torch.device('cpu')
                args.checkpoint_path = None
            else:
                raise RuntimeError(f"CACTI requires CUDA but it's not available. Error: {e}")
        
        # Load model
        self.model = load_model(args, data_dict)
        
        # Train and impute
        torch.set_float32_matmul_precision('medium')
        imputed_data = self.model.fit_transform(data_dict)
        
        # Clean up
        del self.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return imputed_data


def impute_cacti(X_missing: np.ndarray, **kwargs) -> np.ndarray:
    """
    Convenience function for CACTI imputation.
    
    Args:
        X_missing: numpy array with NaN for missing values
        **kwargs: Additional arguments for CACTIImputer
        
    Returns:
        Imputed numpy array
    """
    imputer = CACTIImputer(**kwargs)
    return imputer.fit_transform(X_missing)


if __name__ == "__main__":
    # Test with random data
    base_dir = '/home/jacobf18/tabular/mcpfn/benchmark/datasets/openml/SolarPower/MNARSoftPolarizationPattern_0.4'
    X_missing = np.load(f"{base_dir}/missing.npy")
    X_true = np.load(f"{base_dir}/true.npy")
    
    mask = np.isnan(X_missing)
    
    imputer = CACTIImputer(device="cuda", model="CMAE", mask_ratio=0.9, epochs=100)
    X_imputed = imputer.fit_transform(np.copy(X_missing))
    
    print(X_missing)
    print(X_imputed)
