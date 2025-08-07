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

# --- Suppress warnings ---
warnings.filterwarnings("ignore")

# --- compute absolute errors on missing entries ---
def compute_abs_errors(X_true: torch.Tensor, X_missing: torch.Tensor, X_imputed: np.ndarray) -> np.ndarray:
    mask = torch.isnan(X_missing)
    true_vals = X_true[mask]
    imputed_vals = torch.tensor(X_imputed)[mask]
    return torch.abs(true_vals - imputed_vals).numpy()

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
# datasets = fetch_clean_openml_datasets(num_datasets=4)

# Load diabetes dataset
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True)
X = torch.tensor(X, dtype=torch.float32)

datasets = [(X, "diabetes", 61)]

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
    checkpoint_path='/mnt/mcpfn_data_tor/checkpoints/all_data/config_27_2e-4.ckpt'
)
tabpfn = TabPFNImputer(device='cuda')

# --- Store all results ---
results = {}

# --- Run benchmark ---
for X, name, did in datasets:
    # Normalize the data
    X_normalized = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-16)
    X_normalized = X_normalized[:10,:5]
    
    for pattern_name, pattern in patterns.items():
        X_missing = pattern._induce_missingness(X_normalized.clone())
        imputer_errors = {}

        # MCPFN
        X_mcpfn = mcpfn.impute(X_missing.numpy().copy())
        imputer_errors["MCPFN"] = compute_abs_errors(X_normalized, X_missing, X_mcpfn)

        # TabPFN; right now this is not working unsure why
        X_tabpfn = tabpfn.impute(X_missing.numpy().copy())
        imputer_errors["TabPFN"] = compute_abs_errors(X_normalized, X_missing, X_tabpfn)

        # # SoftImpute with safe scaling
        # X_np = X_missing.numpy()
        # X_scaled = scale_columns_ignoring_nans(X_np)
        # X_soft = SoftImpute().fit_transform(X_scaled)
        # imputer_errors["SoftImpute"] = compute_abs_errors(X_normalized, X_missing, X_soft)
        
        # Mean imputation
        plugin = Imputers().get("mean")
        out = plugin.fit_transform(X_missing.numpy().copy()).to_numpy()
        imputer_errors["Column Mean"] = compute_abs_errors(X_normalized, X_missing, out)
        
        # HyperImpute imputation
        plugin = Imputers().get("hyperimpute")
        out = plugin.fit_transform(X_missing.numpy().copy()).to_numpy()
        imputer_errors["HyperImpute"] = compute_abs_errors(X_normalized, X_missing, out)

        # Store results
        results[(name, pattern_name)] = imputer_errors
        print(f"✅ {name} | {pattern_name} done.")

# --- Plotting ---
sns.set(style="whitegrid")

for (dataset_name, pattern_name), imputer_errors in results.items():
    imputer_names = list(imputer_errors.keys())
    error_data = [imputer_errors[imp] for imp in imputer_names]

    plt.figure(figsize=(8, 5))
    ax = sns.boxplot(data=error_data)
    ax.set_xticks(range(len(imputer_names)))
    ax.set_xticklabels(imputer_names)
    ax.set_ylabel("Absolute Error")
    ax.set_title(f"Dataset: {dataset_name} | Pattern: {pattern_name}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"boxplot_{dataset_name}_{pattern_name}.png")
    plt.close()