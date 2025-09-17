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
    "mixed_adaptive",
    # "mixed_more_heads",
    # "mixed_perm_all_row_col_whiten",
    # "mixed_adaptive_row_column_permutation_8",
    # "mcpfn_mcar_linear",
    # "mcpfn_mar_linear",
    "tabpfn",
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
    "mixed_adaptive": "MCPFN",
    "mixed_perm_both_row_col": "MCPFN (Linear Permuted)",
    "tabpfn_impute": "TabPFN",
    "tabpfn": "MC-TabPFN",
    "column_mean": "Column Mean",
    "hyperimpute": "HyperImpute",
    "ot_sinkhorn": "Optimal Transport",
    "missforest": "MissForest",
    "softimpute": "SoftImpute",
    "ice": "ICE",
    "mice": "MICE",
    "gain": "GAIN",
    "miwae": "MIWAE",
    "forestdiffusion": "ForestDiffusion",
}

negative_rmse = {}

def compute_negative_rmse(X_true, X_imputed, mask):
    return -np.sqrt(np.mean((X_true[mask] - X_imputed[mask]) ** 2))

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

# Plot for each individual pattern
for pattern_name in patterns:
    # Get dataframe for 1 pattern
    df2 = df[df.index.get_level_values(1) == pattern_name]
    df_norm = (df2 - df2.min(axis=1).values[:, None]) / (df2.max(axis=1) - df2.min(axis=1)).values[:, None]
    
    # Average across datasets
    # --- Barplot ---
    plt.figure(figsize=(7,5))
    # sort df_norm by the mean of the rows
    sorted_methods = df_norm.mean(axis=0).sort_values(ascending=True).index
    ax = sns.barplot(data=df_norm, order=sorted_methods)
    
    # Remove x-axis labels
    ax.set_xticklabels([])
    ax.set_xlabel("")
    
    # Add method names inside bars
    for i, method in enumerate(sorted_methods):
        # Get the bar height (mean value across datasets)
        bar_height = df_norm[method].mean()
        # Position the text at the bottom of the bar
        # Use larger font size and position at bottom for better readability
        ax.text(i, 0.05, method, ha='center', va='bottom', 
                fontsize=11.0, rotation=90, color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.3))

    plt.ylabel("Normalized Negative RMSE (0–1)")
    plt.title(f"Comparison of Imputation Algorithms | {pattern_name}")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"figures/negative_rmse_{pattern_name}.png", dpi=300)
    plt.close()

# Plot for all patterns combined
# Get dataframe for all patterns
df_all = df.copy()
df_norm_all = (df_all - df_all.min(axis=1).values[:, None]) / (df_all.max(axis=1) - df_all.min(axis=1)).values[:, None]

# Average across datasets and patterns
plt.figure(figsize=(7,5))
# sort df_norm_all by the mean of the rows
sorted_methods_all = df_norm_all.mean(axis=0).sort_values(ascending=True).index
ax = sns.barplot(data=df_norm_all, order=sorted_methods_all)

# Remove x-axis labels
ax.set_xticklabels([])
ax.set_xlabel("")

# Add method names inside bars
for i, method in enumerate(sorted_methods_all):
    # Get the bar height (mean value across datasets and patterns)
    bar_height = df_norm_all[method].mean()
    # Position the text at the bottom of the bar
    ax.text(i, 0.05, method, ha='center', va='bottom', 
            fontsize=11.0, rotation=90, color='white', weight='bold',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.3))

plt.ylabel("Normalized Negative RMSE (0–1)")
plt.title("Comparison of Imputation Algorithms | All Patterns")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(f"figures/negative_rmse_overall.png", dpi=300)
plt.close()