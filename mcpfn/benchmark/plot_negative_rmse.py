import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


# --- Plotting ---
sns.set(style="whitegrid")

base_path = "datasets/openml"

datasets = os.listdir(base_path)

methods = ["softimpute", "column_mean", "mcpfn"]

negative_rmse = {}

def compute_negative_rmse(X_true, X_imputed, mask):
    return -np.sqrt(np.mean((X_true[mask] - X_imputed[mask]) ** 2))

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
            negative_rmse[(dataset, pattern_name, method)] = compute_negative_rmse(X_true, X_imputed, mask)
            
# Normalize negative rmse by the max and min per dataset
# Save max and min per dataset
# max_rmse = {}
# min_rmse = {}
# for key in negative_rmse.keys():
#     dataset, pattern_name, method = key
#     max_rmse[dataset] = max(negative_rmse[key] for key in negative_rmse.keys() if key[0] == dataset)
#     min_rmse[dataset] = min(negative_rmse[key] for key in negative_rmse.keys() if key[0] == dataset)
    
# for key in negative_rmse.keys():
#     dataset, pattern_name, method = key
#     negative_rmse[key] = (negative_rmse[key] - min_rmse[dataset]) / (max_rmse[dataset] - min_rmse[dataset])
    
negative_rmse_per_method = {}
for key in negative_rmse.keys():
    dataset, pattern_name, method = key
    if pattern_name != "MCAR":
        continue
    if method not in negative_rmse_per_method:
        negative_rmse_per_method[method] = []
    negative_rmse_per_method[method].append(negative_rmse[key])
    
# Plot negative RMSE per method across datasets and patterns as a bar plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=negative_rmse_per_method, capsize=.4,
    err_kws={"color": ".5", "linewidth": 2.5})
ax.set_xticks(range(len(negative_rmse_per_method)))
ax.set_xticklabels(negative_rmse_per_method.keys())
ax.set_ylabel("Normalized Negative RMSE")
ax.set_title("Negative RMSE Comparison")
plt.savefig("figures/negative_rmse.png")
plt.close()