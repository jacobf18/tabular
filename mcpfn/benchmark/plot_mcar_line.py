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
    # # "ot_sinkhorn",
    "missforest",
    # "ice",
    # "mice",
    # "gain",
    # "miwae",
    "mcpfn_ensemble",
    # # "tabpfn_impute",
]

method_names = {
    "mcpfn_ensemble": "TabImpute++",
    "tabpfn_impute": "TabPFN",
    "column_mean": "Col Mean",
    "hyperimpute": "HyperImpute",
    "ot_sinkhorn": "OT",
    "missforest": "MissForest",
    "softimpute": "SoftImpute",
    "ice": "ICE",
    "mice": "MICE",
    "gain": "GAIN",
    "miwae": "MIWAE",
}

# Only focus on MCAR pattern
pattern_name = "MCAR"

# p values from 0.1 to 0.9
p_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

negative_rmse = {}

def compute_negative_rmse(X_true, X_imputed, mask):
    return -np.sqrt(np.mean((X_true[mask] - X_imputed[mask]) ** 2))

def compute_percent_correct(X_true, X_imputed, mask):
    return np.mean(np.isclose(X_true[mask], X_imputed[mask], atol=0.01)) * 100

# Collect data for all p values and MCAR pattern
for dataset in datasets:
    configs = os.listdir(f"{base_path}/{dataset}")
    for config in configs:
        config_split = config.split("_")
        p = config_split[-1]
        
        # Check if this is one of our target p values
        if float(p) not in p_values:
            continue
        
        # remove p from config_split
        config_split = config_split[:-1]
        current_pattern = "_".join(config_split)
        
        # Only process MCAR pattern
        if current_pattern != pattern_name:
            continue
        
        X_missing = np.load(f"{base_path}/{dataset}/{current_pattern}_{p}/missing.npy")
        X_true = np.load(f"{base_path}/{dataset}/{current_pattern}_{p}/true.npy")
        
        mask = np.isnan(X_missing)
        
        for method in methods:
            try:
                X_imputed = np.load(f"{base_path}/{dataset}/{current_pattern}_{p}/{method}.npy")
                name = method_names[method]
                negative_rmse[(dataset, float(p), name)] = compute_negative_rmse(X_true, X_imputed, mask)
                # percent_correct[(dataset, float(p), name)] = compute_percent_correct(X_true, X_imputed, mask)
            except FileNotFoundError:
                print(f"Warning: File not found for {dataset}/{current_pattern}_{p}/{method}.npy")
                continue

data = []
for (dataset, p, method), value in negative_rmse.items():
    data.append({
        'dataset': dataset,
        'p': p,
        'method': method,
        'value': value
    })
df = pd.DataFrame(data)
df.sort_values(by="p", inplace=True)

# # Remove outliers from the value column
# df["value"] = df.groupby(["dataset", "p"])["value"].transform(
#     lambda x: x.clip(lower=x.quantile(0.05), upper=x.quantile(0.95))
# )

# Normalize the value column within each dataset and p value, but leave the method column
df["value_norm"] = df.groupby(["dataset"])["value"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)
print(df)
include_methods = ["TabImpute++", "HyperImpute", "Col Mean", "MissForest"]

# Plot the values with a separate line for each method
plt.figure(figsize=(10, 6))
sns.lineplot(x="p", y="value_norm", hue="method", data=df[df["method"].isin(include_methods)], hue_order=include_methods)
plt.ylabel("1 - Normalized RMSE (0–1)")
plt.xlabel("Fraction of Missing Values")
# Remove legend title
plt.legend(title="")
plt.savefig('figures/mcar_normalized_performance_vs_p.pdf', dpi=300)
plt.close()