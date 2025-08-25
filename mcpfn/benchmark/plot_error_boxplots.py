import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Plotting ---
sns.set(style="whitegrid")

base_path = "datasets/openml"

datasets = os.listdir(base_path)

for dataset in datasets:
    configs = os.listdir(f"{base_path}/{dataset}")
    for config in configs:
        pattern_name = config.split("_")[0]
        p = config.split("_")[1]
        X_missing = np.load(f"{base_path}/{dataset}/{pattern_name}_{p}/missing.npy")
        mask = np.isnan(X_missing)
        
        # Load imputed values
        X_true = np.load(f"{base_path}/{dataset}/{pattern_name}_{p}/true.npy")
        X_mcpfn = np.load(f"{base_path}/{dataset}/{pattern_name}_{p}/mcpfn.npy")
        # X_tabpfn = np.load(f"{base_path}/{dataset}/{pattern_name}_{p}/tabpfn.npy")
        X_softimpute = np.load(f"{base_path}/{dataset}/{pattern_name}_{p}/softimpute.npy")
        X_column_mean = np.load(f"{base_path}/{dataset}/{pattern_name}_{p}/column_mean.npy")
        
        # Compute errors
        errors = {
            "MCPFN": np.abs(X_mcpfn[mask] - X_true[mask]),
            # "TabPFN": np.abs(X_tabpfn[mask] - X_true[mask]),
            "SoftImpute": np.abs(X_softimpute[mask] - X_true[mask]),
            "Column Mean": np.abs(X_column_mean[mask] - X_true[mask])
        }
        
        # Plot boxplot
        plt.figure(figsize=(8, 5))
        ax = sns.boxplot(data=errors)
        ax.set_xticks(range(len(errors)))
        ax.set_xticklabels(errors.keys())
        ax.set_ylabel("Absolute Error")
        ax.set_title(f"Dataset: {dataset} | Pattern: {pattern_name}")
        ax.set_ylim(bottom=0.0)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(f"figures/boxplot_{dataset}_{pattern_name}.png")
        plt.close()
    
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