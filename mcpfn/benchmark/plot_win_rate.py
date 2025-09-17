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
    # "softimpute", 
    # "column_mean", 
    "hyperimpute", 
    # "ot_sinkhorn",
    "missforest",
    # "ice",
    # "mice",
    # "gain",
    # "miwae",
    # "mixed_perm_both_row_col",
    # "mixed_nonlinear",
    "mixed_adaptive",
    "tabpfn",
    # "tabpfn_impute",
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
    "mixed_adaptive": "MCPFN",
    "mixed_perm_both_row_col": "MCPFN (Linear Permuted)",
    "tabpfn_impute": "TabPFN (Unsupervised)",
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
}

def compute_rmse(X_true, X_imputed, mask):
    """Compute RMSE for imputed values"""
    return np.sqrt(np.mean((X_true[mask] - X_imputed[mask]) ** 2))

# Dictionary to store RMSE results for each (dataset, pattern, method) combination
rmse_results = {}

# Load all RMSE results
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
            rmse = compute_rmse(X_true, X_imputed, mask)
            rmse_results[(dataset, pattern_name, name)] = rmse

# Convert to DataFrame
df = pd.Series(rmse_results).unstack()

# Calculate win rates for each pattern separately
pattern_win_rates = {}

for pattern_name in patterns:
    # Get dataframe for this pattern
    df_pattern = df[df.index.get_level_values(1) == pattern_name]
    
    if df_pattern.empty:
        print(f"No data found for pattern: {pattern_name}")
        continue
    
    # Calculate win counts for this pattern
    win_counts = {}
    total_datasets = 0
    
    # For each dataset in this pattern, find the method with lowest RMSE (best performance)
    for dataset_idx in df_pattern.index:
        dataset_name = dataset_idx[0]
        total_datasets += 1
        
        # Get RMSE values for this dataset
        dataset_rmse = df_pattern.loc[dataset_idx]
        
        # Find the method with minimum RMSE (best performance)
        best_method = dataset_rmse.idxmin()
        
        # Count wins
        if best_method not in win_counts:
            win_counts[best_method] = 0
        win_counts[best_method] += 1
    
    # Calculate win rates as percentages for this pattern
    win_rates = {}
    for method in win_counts:
        win_rates[method] = (win_counts[method] / total_datasets) * 100
    
    pattern_win_rates[pattern_name] = win_rates
    
    # Create DataFrame for plotting this pattern
    win_rate_df = pd.DataFrame(list(win_rates.items()), columns=['Method', 'Win Rate (%)'])
    win_rate_df = win_rate_df.sort_values('Win Rate (%)', ascending=True)
    
    # Create the win-rate plot for this pattern
    plt.figure(figsize=(6.5, 4))
    ax = sns.barplot(data=win_rate_df, x='Win Rate (%)', y='Method', hue='Method', palette='viridis', legend=False)
    
    plt.xlabel("Win Rate (%)")
    plt.ylabel("")
    plt.title(f"Win Rate by Algorithm - {pattern_name} Pattern")
    plt.xlim(0, max(win_rates.values()) * 1.1 if win_rates else 0)
    
    # Add value labels on bars
    for i, v in enumerate(win_rate_df['Win Rate (%)']):
        ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"figures/win_rate_{pattern_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Calculate overall win rates across all patterns
overall_win_counts = {}
total_all_datasets = 0

for pattern_name in patterns:
    if pattern_name not in pattern_win_rates:
        continue
    
    df_pattern = df[df.index.get_level_values(1) == pattern_name]
    total_all_datasets += len(df_pattern)
    
    for dataset_idx in df_pattern.index:
        dataset_rmse = df_pattern.loc[dataset_idx]
        best_method = dataset_rmse.idxmin()
        
        if best_method not in overall_win_counts:
            overall_win_counts[best_method] = 0
        overall_win_counts[best_method] += 1

# Calculate overall win rates
overall_win_rates = {}
for method in overall_win_counts:
    overall_win_rates[method] = (overall_win_counts[method] / total_all_datasets) * 100

# Create overall summary plot
overall_df = pd.DataFrame(list(overall_win_rates.items()), columns=['Method', 'Win Rate (%)'])
overall_df = overall_df.sort_values('Win Rate (%)', ascending=True)

plt.figure(figsize=(6.5, 4))
ax = sns.barplot(data=overall_df, x='Win Rate (%)', y='Method', hue='Method', palette='viridis', legend=False)

plt.xlabel("Win Rate (%)")
plt.ylabel("")
plt.title("Overall Win Rate by Algorithm Across All Patterns")
plt.xlim(0, max(overall_win_rates.values()) * 1.1 if overall_win_rates else 0)

# Add value labels on bars
for i, v in enumerate(overall_df['Win Rate (%)']):
    ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig("figures/win_rate_overall.png", dpi=300, bbox_inches='tight')
plt.close()
