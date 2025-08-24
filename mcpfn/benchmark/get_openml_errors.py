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

# --- Suppress warnings ---
warnings.filterwarnings("ignore")

# --- compute absolute errors on missing entries ---
def compute_abs_errors(X_true: torch.Tensor, X_missing: torch.Tensor, X_imputed: np.ndarray) -> np.ndarray:
    mask = torch.isnan(X_missing)
    true_vals = X_true[mask]
    imputed_vals = torch.tensor(X_imputed)[mask]
    return torch.abs(true_vals - imputed_vals).numpy()

# --- Load imputer classes ---
mcpfn = ImputePFN(
    device='cuda',
    encoder_path='/root/tabular/mcpfn/src/mcpfn/model/encoder.pth',
    borders_path='/root/tabular/mcpfn/borders.pt',
    checkpoint_path='/mnt/mcpfn_data/checkpoints/full_batch_size_64/step-170000.ckpt'
)
tabpfn = TabPFNImputer(device='cuda')

# --- Store all results ---
rows = []

base_path = "datasets/openml"

# Get dataset names from base path
datasets = os.listdir(base_path)

# --- Run benchmark ---
for name in datasets:
    # Get filenames
    configs = os.listdir(f"{base_path}/{name}")
    for config in configs:
        print(f"Running {name} | {config}")
        pattern_name = config.split("_")[0]
        p = config.split("_")[1]
        X_missing = np.load(f"{base_path}/{name}/{config}/missing.npy")
        X_normalized = np.load(f"{base_path}/{name}/{config}/true.npy")
    
        mask = np.isnan(X_missing)

        # MCPFN
        X_mcpfn = mcpfn.impute(X_missing.copy())
        np.save(f"{base_path}/{name}/{config}/mcpfn.npy", X_mcpfn)
        
        # # TabPFN
        # X_tabpfn = tabpfn.impute(X_missing.copy())
        # np.save(f"{base_path}/{name}/{config}/tabpfn.npy", X_tabpfn)
        
        # SoftImpute
        plugin = Imputers().get("softimpute")
        out = plugin.fit_transform(X_missing.copy()).to_numpy()
        np.save(f"{base_path}/{name}/{config}/softimpute.npy", out)
        
        # Column Mean
        plugin = Imputers().get("mean")
        out = plugin.fit_transform(X_missing.copy()).to_numpy()
        np.save(f"{base_path}/{name}/{config}/column_mean.npy", out)
        
        # HyperImpute
        # plugin = Imputers().get("hyperimpute")
        # out = plugin.fit_transform(X_missing.copy()).to_numpy()
        # add_rows(rows, name, pattern_name, "HyperImpute", X_normalized[mask], out[mask])
        
        # torch.cuda.empty_cache()
        
# Output results to a csv file
columns = ["dataset_name", "pattern_name", "imputer_name", "true_value", "imputed_value"]
df = pd.DataFrame(rows, columns=columns)

# If the file already exists, append to it
import os
if os.path.exists("results.csv"):
    df.to_csv("out/errors/results.csv", mode='a', header=False, index=False)
else:
    df.to_csv("out/errors/results.csv", index=False)