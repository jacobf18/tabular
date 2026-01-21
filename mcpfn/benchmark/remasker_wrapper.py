"""
ReMasker wrapper for tabimpute benchmark.
Adapted from https://github.com/tydusky/remasker
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from functools import partial
import math

# Add remasker to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'remasker'))

import model_mae as model_mae
from utils import NativeScaler, MAEDataset, adjust_learning_rate
from torch.utils.data import DataLoader, RandomSampler

eps = 1e-8

class ReMaskerImputer:
    """
    ReMasker imputer adapted for tabimpute benchmark.
    Uses masked autoencoder for missing value imputation.
    """
    
    def __init__(
        self,
        embed_dim=64,
        depth=8,
        decoder_depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        max_epochs=600,
        mask_ratio=0.75,
        batch_size=128,
        lr=None,
        blr=1e-3,
        min_lr=0.0,
        weight_decay=0.05,
        warmup_epochs=5,
        accum_iter=1,
        norm_field_loss=True,
        encode_func='linear',
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize ReMasker imputer.
        
        Args:
            embed_dim: Embedding dimension
            depth: Number of encoder layers
            decoder_depth: Number of decoder layers
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            max_epochs: Maximum training epochs
            mask_ratio: Masking ratio during training
            batch_size: Batch size for training
            lr: Learning rate (if None, computed from blr)
            blr: Base learning rate
            min_lr: Minimum learning rate
            weight_decay: Weight decay for optimizer
            warmup_epochs: Warmup epochs
            accum_iter: Gradient accumulation iterations
            norm_field_loss: Whether to normalize field loss
            encode_func: Encoding function ('linear' or 'active')
        """
        self.embed_dim = embed_dim
        self.depth = depth
        self.decoder_depth = decoder_depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.max_epochs = max_epochs
        self.mask_ratio = mask_ratio
        self.batch_size = batch_size
        self.lr = lr
        self.blr = blr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.accum_iter = accum_iter
        self.norm_field_loss = norm_field_loss
        self.encode_func = encode_func
        self.device = device
        self.model = None
        self.norm_parameters = None

    def fit(self, X_raw):
        """
        Fit the ReMasker model on the data.
        
        Args:
            X_raw: Input data (numpy array or torch tensor)
        
        Returns:
            self
        """
        if isinstance(X_raw, np.ndarray):
            X_raw = torch.from_numpy(X_raw).float()
        
        X = X_raw.clone()
        
        # Parameters
        no = len(X)
        dim = len(X[0, :])
        
        X = X.cpu()

        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        
        # MinMax normalization
        for i in range(dim):
            min_val[i] = np.nanmin(X[:, i])
            max_val[i] = np.nanmax(X[:, i])
            X[:, i] = (X[:, i] - min_val[i]) / (max_val[i] - min_val[i] + eps)
            
        self.norm_parameters = {"min": min_val, "max": max_val}

        # Set missing mask
        M = 1 - (1 * (np.isnan(X)))
        M = M.float().to(self.device)

        X = torch.nan_to_num(X)
        X = X.to(self.device)
        
        # Build model
        self.model = model_mae.MaskedAutoencoder(
            rec_len=dim,
            embed_dim=self.embed_dim, 
            depth=self.depth, 
            num_heads=self.num_heads,
            decoder_embed_dim=self.embed_dim, 
            decoder_depth=self.decoder_depth, 
            decoder_num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=eps),
            norm_field_loss=self.norm_field_loss,
            encode_func=self.encode_func
        )

        self.model.to(self.device)

        # Set optimizer
        eff_batch_size = self.batch_size * self.accum_iter 
        if self.lr is None:
            self.lr = self.blr * eff_batch_size / 64
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr / 10, 
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay
        )
        loss_scaler = NativeScaler()

        dataset = MAEDataset(X, M)
        dataloader = DataLoader(
            dataset, 
            sampler=RandomSampler(dataset),
            batch_size=self.batch_size,
        )

        self.model.train()

        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            total_loss = 0

            iter_count = 0
            for iter_count, (samples, masks) in enumerate(dataloader):
                # Learning rate scheduler
                if iter_count % self.accum_iter == 0:
                    adjust_learning_rate(
                        optimizer, 
                        iter_count / len(dataloader) + epoch, 
                        self.lr, 
                        self.min_lr, 
                        self.max_epochs, 
                        self.warmup_epochs
                    )

                samples = samples.unsqueeze(dim=1)
                samples = samples.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    loss, _, _, _ = self.model(samples, masks, mask_ratio=self.mask_ratio)
                    loss_value = loss.item()
                    total_loss += loss_value

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    break

                loss /= self.accum_iter
                loss_scaler(
                    loss, 
                    optimizer, 
                    parameters=self.model.parameters(),
                    update_grad=(iter_count + 1) % self.accum_iter == 0
                )
                
                if (iter_count + 1) % self.accum_iter == 0:
                    optimizer.zero_grad()
            print(f"Loss: {total_loss / (iter_count + 1)}")

        return self

    def transform(self, X_raw):
        """
        Impute missing values.
        
        Args:
            X_raw: Input data with missing values (numpy array or torch tensor)
            
        Returns:
            Imputed data (numpy array)
        """
        if isinstance(X_raw, np.ndarray):
            X_raw = torch.from_numpy(X_raw).float()
            
        X = X_raw.clone()

        min_val = self.norm_parameters["min"]
        max_val = self.norm_parameters["max"]
        
        no, dim = X.shape
        X = X.cpu()

        # MinMaxScaler normalization
        for i in range(dim):
            X[:, i] = (X[:, i] - min_val[i]) / (max_val[i] - min_val[i] + eps)

        # Set missing mask
        M = 1 - (1 * (np.isnan(X)))
        X = np.nan_to_num(X)

        X = torch.from_numpy(X).to(self.device)
        M = M.to(self.device)
        
        self.model.eval()

        # Imputed data        
        with torch.no_grad():
            for i in range(no):
                sample = torch.reshape(X[i], (1, 1, -1))
                mask = torch.reshape(M[i], (1, -1))
                _, pred, _, _ = self.model(sample, mask)
                pred = pred.squeeze(dim=2)
                if i == 0:
                    imputed_data = pred
                else:
                    imputed_data = torch.cat((imputed_data, pred), 0) 
            
        # Renormalize
        for i in range(dim):
            imputed_data[:, i] = imputed_data[:, i] * (max_val[i] - min_val[i] + eps) + min_val[i]

        if np.all(np.isnan(imputed_data.detach().cpu().numpy())):
            raise RuntimeError("The imputed result contains nan. This is a bug.")

        M = M.cpu()
        imputed_data = imputed_data.detach().cpu()
        
        # Return imputed values (keep original observed values)
        result = M * np.nan_to_num(X_raw.cpu()) + (1 - M) * imputed_data
        
        return result.numpy()

    def fit_transform(self, X):
        """
        Fit and transform in one step.
        
        Args:
            X: Input data with missing values (numpy array or pandas DataFrame)
            
        Returns:
            Imputed data (numpy array)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.fit(X).transform(X)


def impute_remasker(X_missing: np.ndarray, **kwargs) -> np.ndarray:
    """
    Convenience function for ReMasker imputation.
    
    Args:
        X_missing: numpy array with NaN for missing values
        **kwargs: Additional arguments for ReMaskerImputer
        
    Returns:
        Imputed numpy array
    """
    imputer = ReMaskerImputer(**kwargs)
    return imputer.fit_transform(X_missing)

if __name__ == "__main__":
    X = np.random.rand(5, 5)
    # X[:,0] = np.nan
    X[np.random.rand(5) < 0.5] = np.nan
    print(X)
    imputer = ReMaskerImputer(device="cuda")
    X_imputed = imputer.fit_transform(X)
    print(type(X_imputed))