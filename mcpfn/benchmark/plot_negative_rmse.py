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
    # "MAR_Diffusion",
    "MNARPanelPattern",
    "MNARPolarizationPattern",
    "MNARSoftPolarizationPattern",
    "MNARLatentFactorPattern",
    "MNARClusterLevelPattern",
    "MNARTwoPhaseSubsetPattern",
}

method_names = {
    "mixed_nonlinear": "MCPFN (Nonlinear Permuted)",
    "mixed_more": "MCPFN (Adaptive Permuted + More Training)",
    "mcpfn_ensemble": "TabImpute++",
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
    plt.figure(figsize=(6,5))
    # sort df_norm by the mean of the rows
    sorted_methods = df_norm.mean(axis=0).sort_values(ascending=True).index
    
    ax = sns.barplot(data=df_norm, order=sorted_methods, capsize=0.2, err_kws={'color': 'dimgray'})
    
    # Remove x-axis labels
    ax.set_xticklabels([])
    ax.set_xlabel("")
    
    # Add method names inside bars
    for i, method in enumerate(sorted_methods):
        # Get the bar height (mean value across datasets)
        bar_height = df_norm[method].mean()
        # Position the text at the bottom of the bar
        # Use larger font size and position at bottom for better readability
        ax.text(i, 0.02, method, ha='center', va='bottom', 
            fontsize=15.0, rotation=90, color='white', weight='bold',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.0))

    plt.ylabel("Normalized Negative RMSE (0–1)", fontsize=15)
    # plt.title(f"Comparison of Imputation Algorithms | {pattern_name}")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(f"figures/negative_rmse_{pattern_name}.png", dpi=300)
    plt.close()

# Plot for all patterns combined
# Get dataframe for all patterns
df_all = df.copy()
df_norm_all = (df_all - df_all.min(axis=1).values[:, None]) / (df_all.max(axis=1) - df_all.min(axis=1)).values[:, None]

# Average across datasets and patterns
plt.figure(figsize=(6,5))
# sort df_norm_all by the mean of the rows
sorted_methods_all = df_norm_all.mean(axis=0).sort_values(ascending=True).index

ax = sns.barplot(data=df_norm_all, order=sorted_methods_all, capsize=0.2, err_kws={'color': 'dimgray'})

# Remove x-axis labels
ax.set_xticklabels([])
ax.set_xlabel("")

# Add method names inside bars
for i, method in enumerate(sorted_methods_all):
    # Get the bar height (mean value across datasets and patterns)
    bar_height = df_norm_all[method].mean()
    # Position the text at the bottom of the bar
    ax.text(i, 0.02, method, ha='center', va='bottom', 
            fontsize=15.0, rotation=90, color='white', weight='bold',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.0))

plt.ylabel("Normalized Negative RMSE (0–1)", fontsize=18)
# plt.title("Comparison of Imputation Algorithms | All Patterns")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(f"figures/negative_rmse_overall.pdf", dpi=300)
plt.close()

# Generate LaTeX table for normalized negative RMSE
print("Creating LaTeX table for normalized negative RMSE...")

# Define methods to include in the table (subset of all methods)
table_methods = [
    "TabImpute++",
    "HyperImpute",
    "MissForest",
    "OT",
    "Col Mean",
    "SoftImpute",
    "ICE",
    "MICE",
    "GAIN",
    "MIWAE",
    "TabPFN",
    # "TabPFN",
]

# Calculate normalized negative RMSE for each pattern
pattern_normalized_rmse = {}

for pattern_name in patterns:
    # Get dataframe for this pattern
    df_pattern = df[df.index.get_level_values(1) == pattern_name]
    
    if df_pattern.empty:
        print(f"No data found for pattern: {pattern_name}")
        continue
    
    # Normalize negative RMSE for this pattern
    df_norm = (df_pattern - df_pattern.min(axis=1).values[:, None]) / (df_pattern.max(axis=1) - df_pattern.min(axis=1)).values[:, None]
    
    # Calculate mean and std for each method across datasets
    pattern_means = df_norm.mean(axis=0)
    pattern_stds = df_norm.std(axis=0)
    
    # Create summary data for this pattern
    pattern_data = []
    for method in pattern_means.index:
        pattern_data.append({
            'Method': method,
            'mean': pattern_means[method],
            'std': pattern_stds[method],
            'Pattern': pattern_name
        })
    
    pattern_normalized_rmse[pattern_name] = pd.DataFrame(pattern_data)

# Calculate overall normalized negative RMSE
df_all_norm = (df_all - df_all.min(axis=1).values[:, None]) / (df_all.max(axis=1) - df_all.min(axis=1)).values[:, None]
overall_means = df_all_norm.mean(axis=0)
overall_stds = df_all_norm.std(axis=0)

# Create summary data for overall
overall_data = []
for method in overall_means.index:
    overall_data.append({
        'Method': method,
        'mean': overall_means[method],
        'std': overall_stds[method],
        'Pattern': 'Overall'
    })

pattern_normalized_rmse['Overall'] = pd.DataFrame(overall_data)

# Create summary data for LaTeX table
summary_data = []
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

for pattern in all_patterns:
    if pattern in pattern_normalized_rmse:
        pattern_df = pattern_normalized_rmse[pattern]
        # Filter to only include methods in table_methods
        filtered_pattern_df = pattern_df[pattern_df['Method'].isin(table_methods)]
        summary_data.append(filtered_pattern_df)

if summary_data:
    summary_df = pd.concat(summary_data, ignore_index=True)
    summary_pivot = summary_df.pivot(index='Pattern', columns='Method', values=['mean', 'std'])
    
    # Flatten column names
    summary_pivot.columns = [f"{col[1]}_{col[0]}" for col in summary_pivot.columns]
    
    # Get all methods and sort by overall performance
    all_methods = []
    for method in table_methods:
        if f"{method}_mean" in summary_pivot.columns:
            all_methods.append(method)
    
    # Sort methods by overall performance (higher normalized negative RMSE is better)
    if 'Overall' in summary_pivot.index:
        overall_means = summary_pivot.loc['Overall', [f"{method}_mean" for method in all_methods]]
        all_methods = [method for _, method in sorted(zip(overall_means, all_methods), reverse=True)]
    
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
            latex_content += f"\\caption{{Mean Normalized Negative RMSE ± Standard Deviation by Missingness Pattern (Table {table_idx + 1}/{num_tables})}}\n"
        else:
            latex_content += "\\caption{Mean Normalized Negative RMSE ± Standard Deviation by Missingness Pattern}\n"
        latex_content += f"\\label{{tab:normalized_negative_rmse_by_pattern_{table_idx + 1}}}\n"
        
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
            
            # Find the maximum mean value for this pattern to bold it
            max_mean = float('-inf')
            for method in table_methods:
                if f"{method}_mean" in summary_pivot.columns:
                    mean_val = summary_pivot.loc[pattern, f"{method}_mean"]
                    if pd.notna(mean_val) and mean_val > max_mean:
                        max_mean = mean_val
            
            for method in table_methods:
                if f"{method}_mean" in summary_pivot.columns:
                    mean_val = summary_pivot.loc[pattern, f"{method}_mean"]
                    std_val = summary_pivot.loc[pattern, f"{method}_std"]
                    if pd.notna(mean_val) and pd.notna(std_val):
                        # Bold if this is the maximum mean value for this pattern
                        if abs(mean_val - max_mean) < 1e-6:  # Use small epsilon for float comparison
                            row += f" & \\textbf{{{mean_val:.3f} ± {std_val:.3f}}}"
                        else:
                            row += f" & {mean_val:.3f} ± {std_val:.3f}"
                    else:
                        row += " & --"
            row += " \\\\\n"
            latex_content += row
        
        latex_content += "\\bottomrule\n"
        latex_content += "\\end{tabular}\n"
        latex_content += "\\end{table}\n"
        
        # Save each table to a separate file
        filename = f"figures/normalized_negative_rmse_table_{table_idx + 1}.txt" if num_tables > 1 else "figures/normalized_negative_rmse_table.txt"
        with open(filename, "w") as f:
            f.write(latex_content)
        
        print(f"LaTeX table {table_idx + 1} saved to {filename}")