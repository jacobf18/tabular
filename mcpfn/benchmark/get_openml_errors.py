import torch
import numpy as np
from mcpfn.prior.training_set_generation import (
    MCARPattern, MARPattern, MNARPattern
)
from mcpfn.interface import ImputePFN, TabPFNImputer
from hyperimpute.plugins.imputers import Imputers
import warnings
from mcpfn.model.encoders import normalize_data
import pandas as pd
import os
from tqdm import tqdm

# --- Suppress warnings ---
warnings.filterwarnings("ignore")

# --- Load imputer classes ---
mcpfn = ImputePFN(
    device='cuda',
    encoder_path='/root/tabular/mcpfn/src/mcpfn/model/encoder.pth',
    borders_path='/root/tabular/mcpfn/borders.pt',
    checkpoint_path='/mnt/mcpfn_data/checkpoints/full_batch_size_64/step-199900.ckpt'
)
tabpfn = TabPFNImputer(device='cuda')

# --- Store all results ---
base_path = "datasets/openml"

# Get dataset names from base path
datasets = os.listdir(base_path)

imputers = set([
    "mcpfn", 
    "hyperimpute", 
    "softimpute", 
    "column_mean", 
    "tabpfn"
])

# --- Run benchmark ---
pbar = tqdm(datasets)
for name in pbar:
    pbar.set_description(f"Running {name}")
    # Get filenames
    configs = os.listdir(f"{base_path}/{name}")
    for config in configs:
        # print(f"Running {name} | {config}")
        pattern_name = config.split("_")[0]
        p = config.split("_")[1]
        X_missing = np.load(f"{base_path}/{name}/{config}/missing.npy")
        X_normalized = np.load(f"{base_path}/{name}/{config}/true.npy")
        print(X_normalized.shape)
    
        mask = np.isnan(X_missing)

        # MCPFN
        if "mcpfn" in imputers:
            X_mcpfn = mcpfn.impute(X_missing.copy())
            np.save(f"{base_path}/{name}/{config}/mcpfn.npy", X_mcpfn)
        
        # TabPFN
        if "tabpfn" in imputers:
            X_tabpfn = tabpfn.impute(X_missing.copy())
            np.save(f"{base_path}/{name}/{config}/tabpfn.npy", X_tabpfn)
        
        # SoftImpute
        if "softimpute" in imputers:
            plugin = Imputers().get("softimpute")
            out = plugin.fit_transform(X_missing.copy()).to_numpy()
            np.save(f"{base_path}/{name}/{config}/softimpute.npy", out)
        
        # Column Mean
        if "column_mean" in imputers:
            plugin = Imputers().get("mean")
            out = plugin.fit_transform(X_missing.copy()).to_numpy()
            np.save(f"{base_path}/{name}/{config}/column_mean.npy", out)
        
        # HyperImpute
        if "hyperimpute" in imputers:
            plugin = Imputers().get("hyperimpute")
            out = plugin.fit_transform(X_missing.copy()).to_numpy()
            np.save(f"{base_path}/{name}/{config}/hyperimpute.npy", out)
            
        print(f"Total allocated memory: {torch.cuda.memory_allocated(device='cuda'):.2f}")
        
        # Empty cache
        torch.cuda.empty_cache()