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
    # "mixed_perm_both_row_col",
    # "mixed_nonlinear",
    "mixed_more",
    "tabpfn",
    "tabpfn_impute",
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
    return np.sqrt(np.nanmean((X_true[mask] - X_imputed[mask]) ** 2))

# Dictionary to store RMSE results for each (dataset, pattern, method) combination
rmse_results = {}

# Load all RMSE results
for dataset in datasets:
    configs = os.listdir(f"{base_path}/{dataset}")
    for config in configs:
        config_split = config.split("_")
        p = config_split[-1]
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

# Calculate ranks for each pattern separately
pattern_ranks = {}

for pattern_name in patterns:
    # Get dataframe for this pattern
    df_pattern = df[df.index.get_level_values(1) == pattern_name]
    
    if df_pattern.empty:
        print(f"No data found for pattern: {pattern_name}")
        continue
    
    # Calculate ranks for each dataset in this pattern
    # Lower RMSE = better rank (rank 1 is best)
    ranks_data = []
    
    for dataset_idx in df_pattern.index:
        dataset_name = dataset_idx[0]
        
        # Get RMSE values for this dataset
        dataset_rmse = df_pattern.loc[dataset_idx]
        
        # Calculate ranks (1 = best, higher number = worse)
        # Using method='min' to handle ties by assigning minimum rank
        ranks = dataset_rmse.rank(method='min', ascending=True)
        
        # Store ranks for this dataset
        for method, rank in ranks.items():
            ranks_data.append({
                'Dataset': dataset_name,
                'Method': method,
                'Rank': rank,
                'RMSE': dataset_rmse[method]
            })
    
    # Convert to DataFrame for this pattern
    pattern_df = pd.DataFrame(ranks_data)
    pattern_ranks[pattern_name] = pattern_df
    
    # Calculate average rank for each method in this pattern
    avg_ranks = pattern_df.groupby('Method')['Rank'].agg(['mean', 'std', 'count']).reset_index()
    avg_ranks = avg_ranks.sort_values('mean', ascending=True)
    
    # Create rank plot for this pattern
    plt.figure(figsize=(12, 8))
    
    # Create box plot of ranks
    ax = sns.boxplot(data=pattern_df, x='Rank', y='Method', order=avg_ranks['Method'].tolist())
    
    # Add mean rank as text
    for i, method in enumerate(avg_ranks['Method']):
        mean_rank = avg_ranks[avg_ranks['Method'] == method]['mean'].iloc[0]
        std_rank = avg_ranks[avg_ranks['Method'] == method]['std'].iloc[0]
        count = avg_ranks[avg_ranks['Method'] == method]['count'].iloc[0]
        
        ax.text(mean_rank, i, f'{mean_rank:.2f}±{std_rank:.2f}\n(n={count})', 
                va='center', ha='left', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.xlabel("Rank (1 = Best)")
    plt.ylabel("Algorithm")
    plt.title(f"Algorithm Ranks by RMSE - {pattern_name} Pattern\n(Lower RMSE = Better Rank)")
    plt.xlim(0.5, len(methods) + 0.5)
    
    # Add vertical line at rank 1
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='Best Rank')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"figures/ranks_{pattern_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Combine all patterns for overall analysis
all_ranks_data = []
for pattern_name, pattern_df in pattern_ranks.items():
    all_ranks_data.append(pattern_df)

if all_ranks_data:
    overall_df = pd.concat(all_ranks_data, ignore_index=True)
    
    # Calculate overall average ranks
    overall_avg_ranks = overall_df.groupby('Method')['Rank'].agg(['mean', 'std', 'count']).reset_index()
    overall_avg_ranks = overall_avg_ranks.sort_values('mean', ascending=True)
    
    # Create overall rank plot
    plt.figure(figsize=(12, 8))
    
    # Create box plot of ranks
    ax = sns.boxplot(data=overall_df, x='Rank', y='Method', order=overall_avg_ranks['Method'].tolist())
    
    # Add mean rank as text
    for i, method in enumerate(overall_avg_ranks['Method']):
        mean_rank = overall_avg_ranks[overall_avg_ranks['Method'] == method]['mean'].iloc[0]
        std_rank = overall_avg_ranks[overall_avg_ranks['Method'] == method]['std'].iloc[0]
        count = overall_avg_ranks[overall_avg_ranks['Method'] == method]['count'].iloc[0]
        
        ax.text(mean_rank, i, f'{mean_rank:.2f}±{std_rank:.2f}\n(n={count})', 
                va='center', ha='left', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.xlabel("Rank (1 = Best)")
    plt.ylabel("Algorithm")
    plt.title("Overall Algorithm Ranks by RMSE Across All Patterns\n(Lower RMSE = Better Rank)")
    plt.xlim(0.5, len(methods) + 0.5)
    
    # Add vertical line at rank 1
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='Best Rank')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("figures/ranks_overall.png", dpi=300, bbox_inches='tight')
    plt.close()
    
if all_ranks_data:
    summary_data = []
    for pattern_name, pattern_df in pattern_ranks.items():
        pattern_avg_ranks = pattern_df.groupby('Method')['Rank'].mean().reset_index()
        pattern_avg_ranks['Pattern'] = pattern_name
        summary_data.append(pattern_avg_ranks)
    
    # Add overall ranks
    overall_avg_ranks['Pattern'] = 'Overall'
    summary_data.append(overall_avg_ranks)
    
    summary_df = pd.concat(summary_data, ignore_index=True)
    summary_pivot = summary_df.pivot(index='Method', columns='Pattern', values='Rank')
    summary_pivot = summary_pivot.sort_values('Overall', ascending=True)
