import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings

# --- Plotting ---
sns.set(style="whitegrid")

base_path = "datasets/openml"

datasets = os.listdir(base_path)

methods = [
    "softimpute", 
    "column_mean", 
    "hyperimpute", 
    # "mcpfn_mixed_linear",
    # "mcpfn_mixed_random",
    # "mcpfn_mixed_linear_fixed",
    "mcpfn_mixed_adaptive",
    "mcpfn_tabpfn",
    # "mcpfn_mcar_linear",
    # "mcpfn_mar_linear",
    "tabpfn",
    "mcpfn_tabpfn_with_preprocessing"
]

negative_rmse = {}

def compute_negative_rmse(X_true, X_imputed, mask):
    return -np.sqrt(np.mean((X_true[mask] - X_imputed[mask]) ** 2))

for dataset in datasets:
    configs = os.listdir(f"{base_path}/{dataset}")
    for config in configs:
        config_split = config.split("_")
        p = config_split[-1]
        # remove p from config_split
        config_split = config_split[:-1]
        pattern_name = "_".join(config_split)
        X_missing = np.load(f"{base_path}/{dataset}/{pattern_name}_{p}/missing.npy")
        X_true = np.load(f"{base_path}/{dataset}/{pattern_name}_{p}/true.npy")
        
        mask = np.isnan(X_missing)
        
        for method in methods:
            X_imputed = np.load(f"{base_path}/{dataset}/{pattern_name}_{p}/{method}.npy")
            
            negative_rmse[(dataset, pattern_name, method)] = compute_negative_rmse(X_true, X_imputed, mask)
            
df = pd.Series(negative_rmse).unstack()

for pattern_name in ["MCAR", "MAR", "MNAR", "MAR_Neural", "MAR_BlockNeural", "MAR_Sequential"]:
    # Get dataframe for 1 pattern
    df2 = df[df.index.get_level_values(1) == pattern_name]
    df_norm = (df2 - df2.min(axis=1).values[:, None]) / (df2.max(axis=1) - df2.min(axis=1)).values[:, None]
        
    # Average across datasets
    # --- Barplot ---
    plt.figure(figsize=(7,5))
    ax = sns.barplot(data=df_norm)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")

    plt.ylabel("Normalized Negative RMSE (0–1)")
    plt.xlabel("Algorithm")
    plt.title(f"Comparison of Imputation Algorithms | {pattern_name}")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"figures/negative_rmse_{pattern_name}.png", dpi=300)
    plt.close()