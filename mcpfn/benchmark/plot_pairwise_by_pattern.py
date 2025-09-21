import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings
from scipy import stats

# --- Plotting ---
sns.set(style="whitegrid")

base_path = "datasets/openml"

datasets = os.listdir(base_path)

methods = [
    "softimpute", 
    "column_mean", 
    "hyperimpute", 
    "ot_sinkhorn",
    "missforest",
    "ice",
    "mice",
    "gain",
    "miwae",
    # "mcpfn_mixed_linear",
    # "mcpfn_mixed_random",
    # "mcpfn_mixed_linear_fixed",
    # "mcpfn_mixed_adaptive",
    # "mcpfn_tabpfn",
    # "mixed_adaptive_more_heads",
    # "mixed_adaptive_permuted_3",
    # "mixed_perm_both_row_col",
    # "mixed_nonlinear",
    # "mixed_adaptive",
    "mcpfn_ensemble",
    # "mixed_more_heads",
    # "mixed_perm_all_row_col_whiten",
    # "mixed_adaptive_row_column_permutation_8",
    # "mcpfn_mcar_linear",
    # "mcpfn_mar_linear",
    # "tabpfn",
    "tabpfn_impute",
    # "mcpfn_tabpfn_with_preprocessing",
    # "forestdiffusion",
]

patterns = {
    "MCAR",
    "MAR",
    "MAR_Neural",
    "MAR_BlockNeural",
    "MAR_Sequential",
    "MNAR",
}

method_names = {
    "mixed_nonlinear": "MCPFN (Nonlinear Permuted)",
    "mixed_more": "MCPFN (Adaptive Permuted + More Training)",
    "mcpfn_ensemble": "MCPFN Ensemble",
    "mixed_adaptive": "MCPFN",
    "mixed_perm_both_row_col": "MCPFN (Linear Permuted)",
    "tabpfn_impute": "TabPFN",
    "tabpfn": "MC-TabPFN",
    "column_mean": "Col Mean",
    "hyperimpute": "HyperImpute",
    "ot_sinkhorn": "OT",
    "missforest": "MissForest",
    "softimpute": "SoftImpute",
    "ice": "ICE",
    "mice": "MICE",
    "gain": "GAIN",
    "miwae": "MIWAE",
    "forestdiffusion": "ForestDiffusion",
}

def compute_negative_rmse(X_true, X_imputed, mask):
    return -np.sqrt(np.mean((X_true[mask] - X_imputed[mask]) ** 2))

negative_rmse = {}

for dataset in datasets:
    configs = os.listdir(f"{base_path}/{dataset}")
    for config in configs:
        config_split = config.split("_")
        p = config_split[-1]
        if p != str(0.4):
            continue
        
        # remove p from config_split
        config_split = config_split[:-1]
        pattern_name = "_".join(config_split)
        
        if pattern_name not in patterns:
            continue
        
        X_missing = np.load(f"{base_path}/{dataset}/{pattern_name}_{p}/missing.npy")
        X_true = np.load(f"{base_path}/{dataset}/{pattern_name}_{p}/true.npy")
        
        mask = np.isnan(X_missing)
        
        for method in methods:
            X_imputed = np.load(f"{base_path}/{dataset}/{pattern_name}_{p}/{method}.npy")
            name = method_names[method]
            negative_rmse[(dataset, pattern_name, name)] = compute_negative_rmse(X_true, X_imputed, mask)
            
df = pd.Series(negative_rmse).unstack()

method1 = "MCPFN Ensemble"
method2 = "Col Mean"

# Plot for each individual pattern
for pattern_name in patterns:
    # Get dataframe for 1 pattern
    df2 = df[df.index.get_level_values(1) == pattern_name]
    df_norm = (df2 - df2.min(axis=1).values[:, None]) / (df2.max(axis=1) - df2.min(axis=1)).values[:, None]
    
    plt.figure(figsize=(5,6))
    
    plt.scatter(df_norm[method1], df_norm[method2])
    
    plt.xlabel(method1)
    plt.ylabel(method2)
    plt.title(f"Comparison of Imputation Algorithms on {pattern_name} data")
    plt.ylim(0, 1.05)
    plt.savefig(f"figures/pairwise_{pattern_name}.png")
    plt.close()