import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# --- Plotting ---
sns.set(style="whitegrid")

base_path = "datasets/openml"

datasets = os.listdir(base_path)

methods = ["softimpute", "column_mean", "hyperimpute", "mcpfn", "tabpfn"]

negative_mae = {}

def compute_negative_mae(X_true, X_imputed, mask):
    return -np.mean(np.abs(X_true[mask] - X_imputed[mask]))

for dataset in datasets:
    configs = os.listdir(f"{base_path}/{dataset}")
    for config in configs:
        pattern_name = config.split("_")[0]
        p = config.split("_")[1]
        X_missing = np.load(f"{base_path}/{dataset}/{pattern_name}_{p}/missing.npy")
        X_true = np.load(f"{base_path}/{dataset}/{pattern_name}_{p}/true.npy")
        
        mask = np.isnan(X_missing)
        
        for method in methods:
            X_imputed = np.load(f"{base_path}/{dataset}/{pattern_name}_{p}/{method}.npy")
            negative_mae[(dataset, pattern_name, method)] = compute_negative_mae(X_true, X_imputed, mask)
            
df = pd.Series(negative_mae).unstack()

for pattern_name in ["MCAR", "MAR", "MNAR"]:
    # Get dataframe for 1 pattern
    df2 = df[df.index.get_level_values(1) == pattern_name]
    df_norm = (df2 - df2.min(axis=1).values[:, None]) / (df2.max(axis=1) - df2.min(axis=1)).values[:, None]
        
    # Average across datasets
    # --- Barplot ---
    plt.figure(figsize=(7,5))
    sns.barplot(data=df_norm)

    plt.ylabel("Normalized Negative MAE (0–1)")
    plt.xlabel("Algorithm")
    plt.title(f"Comparison of Imputation Algorithms on {pattern_name} data")
    plt.ylim(0, 1.05)
    plt.savefig(f"figures/negative_mae_{pattern_name}.png")
    plt.close()