import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mcpfn.prior.training_set_generation import (
    MCARPattern, MARPattern, MNARPattern
)
from mcpfn.interface import ImputePFN, TabPFNImputer
from download_openml import fetch_clean_openml_datasets
from fancyimpute import SoftImpute
from hyperimpute.plugins.imputers import Imputers
import warnings
from mcpfn.model.encoders import normalize_data
import pandas as pd

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

# --- Column-wise scaling while ignoring NaNs for softimpute ---
def scale_columns_ignoring_nans(X: np.ndarray) -> np.ndarray:
    X_scaled = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        non_nan_mask = ~np.isnan(col)
        if np.any(non_nan_mask):
            mean = np.mean(col[non_nan_mask])
            std = np.std(col[non_nan_mask])
            if std > 0:
                X_scaled[non_nan_mask, j] = (col[non_nan_mask] - mean) / std
            else:
                X_scaled[non_nan_mask, j] = 0.0
    return X_scaled

# --- Fetch datasets ---
datasets = fetch_clean_openml_datasets(num_datasets=4)

# Load diabetes dataset
# from sklearn.datasets import load_diabetes
# X, y = load_diabetes(return_X_y=True)
# X = torch.tensor(X, dtype=torch.float32)

# datasets = [(X, "diabetes", 61)]

# --- Define missingness patterns ---
# add the one I said there was a bug in: MCARFixedPattern
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

# --- Run benchmark ---
for X, name, did in datasets:
    for pattern_name, pattern in patterns.items():
        print(f"Running {name} | {pattern_name}")
        torch.cuda.empty_cache()
        X_missing = pattern._induce_missingness(X.clone())
        
        # Normalize the data (after inducing missingness)
        X_missing, (mean, std) = normalize_data(X_missing, return_scaling=True)
        # std is set to 1 if all values are the same
        X_normalized = (X - mean) / std
        imputer_errors = {}
        
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


# --- Plotting ---
# sns.set(style="whitegrid")

# for (dataset_name, pattern_name), imputer_errors in results.items():
#     imputer_names = list(imputer_errors.keys())
#     error_data = [imputer_errors[imp] for imp in imputer_names]

#     plt.figure(figsize=(8, 5))
#     ax = sns.boxplot(data=error_data)
#     ax.set_xticks(range(len(imputer_names)))
#     ax.set_xticklabels(imputer_names)
#     ax.set_ylabel("Absolute Error")
#     ax.set_title(f"Dataset: {dataset_name} | Pattern: {pattern_name}")
#     ax.set_ylim(bottom=0.0)
#     plt.xticks(rotation=20)
#     plt.tight_layout()
#     plt.savefig(f"boxplot_{dataset_name}_{pattern_name}.png")
#     plt.close()
    
#     plt.figure(figsize=(8, 5))
#     ax = sns.violinplot(data=error_data, cut=0)
#     ax.set_xticks(range(len(imputer_names)))
#     ax.set_xticklabels(imputer_names)
#     ax.set_ylabel("Absolute Error")
#     ax.set_title(f"Dataset: {dataset_name} | Pattern: {pattern_name}")
#     ax.set_ylim(bottom=0.0)
#     plt.xticks(rotation=20)
#     plt.tight_layout()
#     plt.savefig(f"violinplot_{dataset_name}_{pattern_name}.png")
#     plt.close()