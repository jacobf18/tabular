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
    "mcpfn_ensemble",
    # "mixed_perm_both_row_col",
    # "mixed_nonlinear",
    # "mixed_adaptive",
    # "tabpfn",
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
    "mixed_adaptive": "MCPFN",
    "mcpfn_ensemble": "MCPFN Ensemble",
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
    
    
table_methods = {
    "MCPFN Ensemble",
    "HyperImpute",
    # "TabPFN",
    # "Column Mean",
    # "Optimal Transport",
    "MissForest",
    "SoftImpute",
    # "ICE",
}
if all_ranks_data:
    summary_data = []
    # Remove methods not in table_methods
    for pattern_name, pattern_df in pattern_ranks.items():
        # Filter to only include methods in table_methods
        filtered_pattern_df = pattern_df[pattern_df['Method'].isin(table_methods)]
        pattern_avg_ranks = filtered_pattern_df.groupby('Method')['Rank'].mean().reset_index()
        pattern_avg_ranks['Pattern'] = pattern_name
        summary_data.append(pattern_avg_ranks)
    
    # Add overall ranks (filtered to table_methods)
    overall_avg_ranks_filtered = overall_avg_ranks[overall_avg_ranks['Method'].isin(table_methods)]
    overall_avg_ranks_filtered['Pattern'] = 'Overall'
    summary_data.append(overall_avg_ranks_filtered)
    
    summary_df = pd.concat(summary_data, ignore_index=True)
    summary_pivot = summary_df.pivot(index='Method', columns='Pattern', values='Rank')
    summary_pivot = summary_pivot.sort_values('Overall', ascending=True)
    
    # Create LaTeX table with mean rank ± std
    print("Creating LaTeX table...")
    
    # Get all patterns including Overall
    all_patterns = ['MCAR', 
                'MAR', 
                'MAR_Neural', 
                'MAR_BlockNeural', 
                'MAR_Sequential', 
                'MNAR', 
                "MNARPanelPattern",
                "MNARPolarizationPattern",
                "MNARSoftPolarizationPattern",
                "MNARLatentFactorPattern",
                "MNARClusterLevelPattern",
                "MNARTwoPhaseSubsetPattern",
                'Overall']
    
    # Create a comprehensive table with mean ± std for each pattern
    latex_table_data = []
    
    for pattern in all_patterns:
        if pattern in pattern_ranks:
            # Individual pattern - filter to table_methods
            pattern_df = pattern_ranks[pattern]
            filtered_pattern_df = pattern_df[pattern_df['Method'].isin(table_methods)]
            pattern_stats = filtered_pattern_df.groupby('Method')['Rank'].agg(['mean', 'std']).reset_index()
            pattern_stats['Pattern'] = pattern
        else:
            # Overall pattern - filter to table_methods
            filtered_overall_df = overall_df[overall_df['Method'].isin(table_methods)]
            pattern_stats = filtered_overall_df.groupby('Method')['Rank'].agg(['mean', 'std']).reset_index()
            pattern_stats['Pattern'] = 'Overall'
        
        # Add to table data
        for _, row in pattern_stats.iterrows():
            latex_table_data.append({
                'Pattern': pattern,
                'Method': row['Method'],
                'Mean_Rank': row['mean'],
                'Std_Rank': row['std']
            })
    
    # Convert to DataFrame and pivot (patterns as rows, methods as columns)
    latex_df = pd.DataFrame(latex_table_data)
    latex_pivot = latex_df.pivot(index='Pattern', columns='Method', values=['Mean_Rank', 'Std_Rank'])
    
    # Flatten column names
    latex_pivot.columns = [f"{col[1]}_{col[0]}" for col in latex_pivot.columns]
    
    # Get all methods and sort by overall performance (using filtered methods)
    all_methods = []
    for method in overall_avg_ranks_filtered['Method']:
        if f"{method}_Mean_Rank" in latex_pivot.columns:
            all_methods.append(method)
    
    # Generate LaTeX table with maximum 4 columns
    max_cols = 4
    num_tables = (len(all_methods) + max_cols - 1) // max_cols
    
    for table_idx in range(num_tables):
        start_col = table_idx * max_cols
        end_col = min(start_col + max_cols, len(all_methods))
        table_methods = all_methods[start_col:end_col]
        
        latex_content = "\\begin{table}[h]\n"
        latex_content += "\\centering\n"
        if num_tables > 1:
            latex_content += f"\\caption{{Mean Rank ± Standard Deviation by Missingness Pattern (Table {table_idx + 1}/{num_tables})}}\n"
        else:
            latex_content += "\\caption{Mean Rank ± Standard Deviation by Missingness Pattern}\n"
        latex_content += f"\\label{{tab:ranks_by_pattern_{table_idx + 1}}}\n"
        
        # Create column specification
        num_methods_table = len(table_methods)
        col_spec = "l" + "c" * num_methods_table  # l for pattern names, c for each method
        
        latex_content += f"\\begin{{tabular}}{{{col_spec}}}\n"
        latex_content += "\\toprule\n"
        
        # Header row
        header = "Pattern"
        for method in table_methods:
            method_name = method.replace('_', '\\_')  # Escape underscores
            header += f" & {method_name}"
        header += " \\\\\n"
        latex_content += header
        latex_content += "\\midrule\n"
        
        # Data rows (patterns as rows)
        for i, pattern in enumerate(all_patterns):
            row = pattern.replace('_', '\\_')  # Escape underscores in pattern names
            
            # Add midrule before Overall performance
            if pattern == 'Overall' and i > 0:
                latex_content += "\\midrule\n"
            
            # Find the minimum mean rank for this pattern to bold it
            min_mean = float('inf')
            for method in table_methods:
                if f"{method}_Mean_Rank" in latex_pivot.columns:
                    mean_val = latex_pivot.loc[pattern, f"{method}_Mean_Rank"]
                    if pd.notna(mean_val) and mean_val < min_mean:
                        min_mean = mean_val
            
            for method in table_methods:
                if f"{method}_Mean_Rank" in latex_pivot.columns:
                    mean_val = latex_pivot.loc[pattern, f"{method}_Mean_Rank"]
                    std_val = latex_pivot.loc[pattern, f"{method}_Std_Rank"]
                    if pd.notna(mean_val) and pd.notna(std_val):
                        # Bold if this is the minimum mean rank for this pattern
                        if abs(mean_val - min_mean) < 1e-6:  # Use small epsilon for float comparison
                            row += f" & \\textbf{{{mean_val:.2f} ± {std_val:.2f}}}"
                        else:
                            row += f" & {mean_val:.2f} ± {std_val:.2f}"
                    else:
                        row += " & --"
            row += " \\\\\n"
            latex_content += row
        
        latex_content += "\\bottomrule\n"
        latex_content += "\\end{tabular}\n"
        latex_content += "\\end{table}\n"
        
        # Save each table to a separate file
        filename = f"figures/ranks_table_{table_idx + 1}.txt" if num_tables > 1 else "figures/ranks_table.txt"
        with open(filename, "w") as f:
            f.write(latex_content)
        
        print(f"LaTeX table {table_idx + 1} saved to {filename}")