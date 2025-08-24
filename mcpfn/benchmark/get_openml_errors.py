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

def add_rows(rows_list, dataset_name, pattern_name, imputer_name, true_vals, imputed_vals):
    for true_val, imputed_val in zip(true_vals, imputed_vals):
        rows_list.append([dataset_name, pattern_name, imputer_name, true_val, imputed_val])

# --- Define missingness patterns ---
patterns = {
    "MCAR": MCARPattern(config={"p_missing": 0.5}),
    "MAR": MARPattern(config={"p_missing": 0.5}),
    "MNAR": MNARPattern(config={"p_missing": 0.5}),
}

# --- Load imputer classes ---
mcpfn = ImputePFN(
    device='cuda',
    encoder_path='/root/tabular/mcpfn/src/mcpfn/model/encoder.pth',
    borders_path='/root/tabular/mcpfn/borders.pt',
    checkpoint_path='/mnt/mcpfn_data/checkpoints/full_batch_size_64/step-170000.ckpt'
)
tabpfn = TabPFNImputer(device='cuda')

# --- Store all results ---
results = {}

rows = []

base_path = "datasets/openml"

# Get dataset names from base path
datasets = os.listdir(base_path)

# --- Run benchmark ---
for name in datasets:
    # Get filenames
    filenames = os.listdir(f"{base_path}/{name}")
    for filename in filenames:
        if filename.endswith(".npy"):
            pattern_name = filename.split("_")[0]
            p = filename.split("_")[1]
            X_missing = np.load(f"{base_path}/{name}/{filename}")
            X_normalized = np.load(f"{base_path}/{name}/{pattern_name}_{p}_true.npy")
    
    print(filenames)
    exit()
    
    mask = torch.isnan(X_missing)

    # MCPFN
    X_mcpfn = mcpfn.impute(X_missing.numpy().copy())
    imputer_errors["MCPFN"] = compute_abs_errors(X_normalized, X_missing, X_mcpfn)
    add_rows(rows, name, pattern_name, "MCPFN", X_normalized[mask], X_mcpfn[mask])
    
    # TabPFN
    # X_tabpfn = tabpfn.impute(X_missing.numpy().copy())
    # imputer_errors["TabPFN"] = compute_abs_errors(X_normalized, X_missing, X_tabpfn)
    # add_rows(rows, name, pattern_name, "TabPFN", X_normalized[mask], X_tabpfn[mask])

    # SoftImpute
    plugin = Imputers().get("softimpute")
    out = plugin.fit_transform(X_missing.numpy().copy()).to_numpy()
    imputer_errors["SoftImpute"] = compute_abs_errors(X_normalized, X_missing, out)
    add_rows(rows, name, pattern_name, "SoftImpute", X_normalized[mask], out[mask])
    
    # Column Mean
    plugin = Imputers().get("mean")
    out = plugin.fit_transform(X_missing.numpy().copy()).to_numpy()
    imputer_errors["Column Mean"] = compute_abs_errors(X_normalized, X_missing, out)
    add_rows(rows, name, pattern_name, "Column Mean", X_normalized[mask], out[mask])
    
    # HyperImpute
    plugin = Imputers().get("hyperimpute")
    out = plugin.fit_transform(X_missing.numpy().copy()).to_numpy()
    imputer_errors["HyperImpute"] = compute_abs_errors(X_normalized, X_missing, out)
    add_rows(rows, name, pattern_name, "HyperImpute", X_normalized[mask], out[mask])
    
    # Store results in dictionary
    results[(name, pattern_name)] = imputer_errors
    print(f"✅ {name} | {pattern_name} done.")
    
    torch.cuda.empty_cache()
        
# Output results to a csv file
columns = ["dataset_name", "pattern_name", "imputer_name", "true_value", "imputed_value"]
df = pd.DataFrame(rows, columns=columns)

# If the file already exists, append to it
import os
if os.path.exists("results.csv"):
    df.to_csv("out/errors/results.csv", mode='a', header=False, index=False)
else:
    df.to_csv("out/errors/results.csv", index=False)