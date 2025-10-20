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
    "ot_sinkhorn",
    "missforest",
    # "ice",
    # "mice",
    # "gain",
    # "miwae",
    # "mcpfn_mixed_linear",
    # "mcpfn_mixed_random",
    # "mcpfn_mixed_linear_fixed",
    # "mcpfn_mixed_adaptive",
    # "tabpfn_no_proprocessing",
    # # "mixed_adaptive_more_heads",
    # # "mixed_adaptive_permuted_3",
    # "mixed_perm_both_row_col",
    # "mixed_nonlinear",
    # "mcpfn_mixed_adaptive",
    "mcpfn_ensemble",
    # "mixed_more_heads",
    # # # "mixed_perm_all_row_col_whiten",
    # # # "mixed_adaptive_row_column_permutation_8",
    # # # "mcpfn_mcar_linear",
    # "mcpfn_mar_linear",
    # "tabpfn",
    # "tabpfn_impute",
    # "knn",
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
    # "MNARPanelPattern",
    # "MNARPolarizationPattern",
    # "MNARSoftPolarizationPattern",
    # "MNARLatentFactorPattern",
    # "MNARClusterLevelPattern",
    # "MNARTwoPhaseSubsetPattern",
    # "MNARCensoringPattern",
}

method_names = {
    "mixed_nonlinear": "TabImpute (Nonlinear FM)",
    "mcpfn_ensemble": "TabImpute+",
    "mcpfn_mixed_adaptive": "TabImpute",
    "mcpfn_mar_linear": "TabImpute (MCAR then MAR)",
    "mixed_more_heads": "TabImpute (More Heads)",
    "tabpfn_no_proprocessing": "TabPFN Fine-Tuned No Preprocessing",
    # "mixed_perm_both_row_col": "TabImpute",
    "tabpfn_impute": "TabPFN",
    "tabpfn": "EWF-TabPFN",
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
    "knn": "K-Nearest Neighbors",
}

# Define consistent color mapping for methods (using display names as they appear in the DataFrame)
method_colors = {
    "TabImpute+": "#2f88a8",  # Blue
    "TabImpute": "#2f88a8",  # Sea Green (distinct from GPU)
    "HyperImpute": "#ff7f0e",  # Orange
    "MissForest": "#2ca02c",   # Green
    "OT": "#591942",           # Red
    "Col Mean": "#9467bd",     # Purple
    "SoftImpute": "#8c564b",   # Brown
    "ICE": "#a14d88",          # Pink
    "MICE": "#7f7f7f",         # Gray
    "GAIN": "#286b33",         # Dark Green
    "MIWAE": "#17becf",        # Cyan
    "TabPFN": "#3e3b6e",       # Blue
    "K-Nearest Neighbors": "#a36424",  # Orange
    "ForestDiffusion": "#98df8a",      # Light Green
    "MCPFN": "#ff9896",        # Light Red
    "MCPFN (Linear Permuted)": "#c5b0d5",  # Light Purple
    "MCPFN (Nonlinear Permuted)": "#c49c94",  # Light Brown
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

plot_pattern = False

# Plot for all patterns combined
# Get dataframe for all patterns
df_all = df.copy()
df_norm_all = (df_all - df_all.min(axis=1).values[:, None]) / (df_all.max(axis=1) - df_all.min(axis=1)).values[:, None]


if plot_pattern:
    for pattern_name in patterns:
        # Get dataframe for 1 pattern
        df2 = df[df.index.get_level_values(1) == pattern_name]
        df_norm = (df2 - df2.min(axis=1).values[:, None]) / (
            df2.max(axis=1) - df2.min(axis=1)
        ).values[:, None]
        
        # Average across datasets
        # --- Barplot ---
        plt.figure(figsize=(6.5,4.5))
        
        # sort methods by mean performance
        sorted_methods = df_norm.mean(axis=0).sort_values(ascending=True).index
        
        # Melt into long format
        df_long = df_norm.melt(var_name="method", value_name="score")
        
        # Use your method_colors dictionary for consistent mapping
        ax = sns.barplot(
            data=df_long,
            x="method",
            y="score",
            order=sorted_methods,
            palette=method_colors,   # <- consistent colors
            capsize=0.2,
            err_kws={"color": "#999999"},
        )
        
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

    # Average across datasets and patterns
    fig = plt.figure(figsize=(6.5,4.5))
    # sort df_norm_all by the mean of the rows
    sorted_methods_all = df_norm_all.mean(axis=0).sort_values(ascending=True).index

    # Melt into long format
    df_long = df_norm_all.melt(var_name="method", value_name="score")

    # Use your method_colors dictionary for consistent mapping
    ax = sns.barplot(
        data=df_long,
        x="method",
        y="score",
        order=sorted_methods_all,
        palette=method_colors,   # <- consistent colors
        capsize=0.2,
        err_kws={"color": "#999999"},
    )

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

    plt.ylabel("1 - Normalized RMSE (0–1)", fontsize=18)
    # plt.title("Comparison of Imputation Algorithms | All Patterns")
    plt.ylim(0, 1.0)
    # plt.tight_layout()
    
    fig.subplots_adjust(left=0.2, right=0.95, bottom=0.05, top=0.95)
    
    plt.savefig(f"figures/negative_rmse_overall.pdf", dpi=300, bbox_inches=None)
    plt.close()
# Generate LaTeX table for normalized negative RMSE
print("Creating LaTeX table for normalized negative RMSE...")

# Define methods to include in the table (subset of all methods)
table_methods = [
    "TabImpute+",
    # "TabImpute",
    # "TabPFN Fine-Tuned No Preprocessing",
    # "EWF-TabPFN",
    # "TabPFN",
    # "TabImpute (MCAR then MAR)",
    # "TabImpute (More Heads)",
    # "TabImpute (Nonlinear FM)",
    "HyperImpute",
    "MissForest",
    "OT",
    # "Col Mean",
    # "SoftImpute",
    # "ICE",
    # "MICE",
    # "GAIN",
    # "MIWAE",
    # "TabPFN",
    # "K-Nearest Neighbors",
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
                'MAR_Neural', 
                'MNAR', 
                'MAR', 
                'MAR_BlockNeural', 
                'MAR_Sequential', 
                # "MNARPanelPattern",
                # "MNARPolarizationPattern",
                # "MNARSoftPolarizationPattern",
                # "MNARLatentFactorPattern",
                # "MNARClusterLevelPattern",
                # "MNARTwoPhaseSubsetPattern",
                # "MNARCensoringPattern",
                'Overall'
                ]

for pattern in all_patterns:
    if pattern in pattern_normalized_rmse:
        pattern_df = pattern_normalized_rmse[pattern]
        # Filter to only include methods in table_methods
        filtered_pattern_df = pattern_df[pattern_df['Method'].isin(table_methods)]
        summary_data.append(filtered_pattern_df)
        
print(summary_data)

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

# Generate transposed LaTeX table (methods as rows, patterns as columns)
print("Creating transposed LaTeX table...")

# Create transposed summary data
transposed_summary_data = []
for method in all_methods:
    method_data = {'Method': method}
    for pattern in all_patterns:
        if pattern in pattern_normalized_rmse:
            pattern_df = pattern_normalized_rmse[pattern]
            method_row = pattern_df[pattern_df['Method'] == method]
            if not method_row.empty:
                method_data[f"{pattern}_mean"] = method_row.iloc[0]['mean']
                method_data[f"{pattern}_std"] = method_row.iloc[0]['std']
            else:
                method_data[f"{pattern}_mean"] = None
                method_data[f"{pattern}_std"] = None
        else:
            method_data[f"{pattern}_mean"] = None
            method_data[f"{pattern}_std"] = None
    transposed_summary_data.append(method_data)

transposed_df = pd.DataFrame(transposed_summary_data)

# Generate transposed LaTeX table with maximum 4 columns
max_cols = 4
num_tables_transposed = (len(all_patterns) + max_cols - 1) // max_cols

# Calculate global maximum for each pattern across all methods (for consistent bolding)
pattern_global_max_means = {}
for pattern in all_patterns:
    max_mean = float('-inf')
    for method in all_methods:
        mean_col = f"{pattern}_mean"
        if mean_col in transposed_df.columns:
            mean_val = transposed_df[transposed_df['Method'] == method][mean_col].iloc[0]
            if pd.notna(mean_val) and mean_val > max_mean:
                max_mean = mean_val
    pattern_global_max_means[pattern] = max_mean

for table_idx in range(num_tables_transposed):
    start_col = table_idx * max_cols
    end_col = min(start_col + max_cols, len(all_patterns))
    table_patterns = all_patterns[start_col:end_col]
    
    latex_content_transposed = "\\begin{table}[h]\n"
    latex_content_transposed += "\\centering\n"
    if num_tables_transposed > 1:
        latex_content_transposed += f"\\caption{{Mean Normalized Negative RMSE ± Standard Deviation by Method (Transposed, Table {table_idx + 1}/{num_tables_transposed})}}\n"
    else:
        latex_content_transposed += "\\caption{Mean Normalized Negative RMSE ± Standard Deviation by Method (Transposed)}\n"
    latex_content_transposed += f"\\label{{tab:normalized_negative_rmse_by_method_transposed_{table_idx + 1}}}\n"
    
    # Create column specification
    num_patterns_table = len(table_patterns)
    col_spec = "l" + "c" * num_patterns_table  # l for method names, c for each pattern
    
    latex_content_transposed += f"\\begin{{tabular}}{{{col_spec}}}\n"
    latex_content_transposed += "\\toprule\n"
    
    # Header row
    header = "Method"
    for pattern in table_patterns:
        pattern_name = pattern.replace('_', '\\_')  # Escape underscores
        header += f" & {pattern_name}"
    header += " \\\\\n"
    latex_content_transposed += header
    latex_content_transposed += "\\midrule\n"
    
    # Data rows (methods as rows)
    for method in all_methods:
        row = method.replace('_', '\\_')  # Escape underscores in method names
        
        # Use global maximum for each pattern (calculated above)
        
        for pattern in table_patterns:
            mean_col = f"{pattern}_mean"
            std_col = f"{pattern}_std"
            
            if mean_col in transposed_df.columns and std_col in transposed_df.columns:
                method_row = transposed_df[transposed_df['Method'] == method]
                if not method_row.empty:
                    mean_val = method_row[mean_col].iloc[0]
                    std_val = method_row[std_col].iloc[0]
                    if pd.notna(mean_val) and pd.notna(std_val):
                        # Bold if this is the global maximum mean value for this pattern
                        if abs(mean_val - pattern_global_max_means[pattern]) < 1e-6:  # Use small epsilon for float comparison
                            row += f" & \\textbf{{{mean_val:.3f} ± {std_val:.3f}}}"
                        else:
                            row += f" & {mean_val:.3f} ± {std_val:.3f}"
                    else:
                        row += " & --"
                else:
                    row += " & --"
            else:
                row += " & --"
        
        row += " \\\\\n"
        latex_content_transposed += row
    
    latex_content_transposed += "\\bottomrule\n"
    latex_content_transposed += "\\end{tabular}\n"
    latex_content_transposed += "\\end{table}\n"
    
    # Save each transposed table to a separate file
    transposed_filename = f"figures/normalized_negative_rmse_table_transposed_{table_idx + 1}.txt" if num_tables_transposed > 1 else "figures/normalized_negative_rmse_table_transposed.txt"
    with open(transposed_filename, "w") as f:
        f.write(latex_content_transposed)
    
    print(f"Transposed LaTeX table {table_idx + 1} saved to {transposed_filename}")

# Generate LaTeX table for non-normalized RMSE values for MCAR with datasets as rows
print("Creating LaTeX table for non-normalized RMSE values (MCAR, datasets as rows)...")

# Get data for MCAR pattern only
mcar_data = df[df.index.get_level_values(1) == 'MCAR']

if not mcar_data.empty:
    # Filter to only include methods in table_methods
    available_methods = [method for method in table_methods if method in mcar_data.columns]
    
    if available_methods:
        # Create the table data
        mcar_table_data = []
        for dataset in mcar_data.index.get_level_values(0).unique():
            dataset_data = {'Dataset': dataset}
            for method in available_methods:
                if method in mcar_data.columns:
                    # Get the negative RMSE value and convert to positive RMSE
                    # Use .iloc[0] to get the scalar value from the Series
                    try:
                        neg_rmse = mcar_data.loc[dataset, method].iloc[0]
                        if pd.notna(neg_rmse):
                            rmse = -neg_rmse  # Convert from negative RMSE to positive RMSE
                            dataset_data[method] = rmse
                        else:
                            dataset_data[method] = None
                    except (IndexError, KeyError):
                        dataset_data[method] = None
                else:
                    dataset_data[method] = None
            mcar_table_data.append(dataset_data)
        
        mcar_df = pd.DataFrame(mcar_table_data)
        
        # Generate LaTeX table
        latex_content_mcar = "\\begin{table}[h]\n"
        latex_content_mcar += "\\centering\n"
        latex_content_mcar += "\\caption{Non-normalized RMSE Values for MCAR Pattern by Dataset}\n"
        latex_content_mcar += "\\label{tab:rmse_mcar_by_dataset}\n"
        
        # Create column specification
        num_methods = len(available_methods)
        col_spec = "l" + "c" * num_methods  # l for dataset names, c for each method
        
        latex_content_mcar += f"\\begin{{tabular}}{{{col_spec}}}\n"
        latex_content_mcar += "\\toprule\n"
        
        # Header row
        header = "Dataset"
        for method in available_methods:
            method_name = method.replace('_', '\\_')  # Escape underscores
            header += f" & {method_name}"
        header += " \\\\\n"
        latex_content_mcar += header
        latex_content_mcar += "\\midrule\n"
        
        # Data rows (datasets as rows)
        for _, row_data in mcar_df.iterrows():
            dataset_name = row_data['Dataset'].replace('_', '\\_')  # Escape underscores
            row = dataset_name
            
            # Find the minimum (best) RMSE value for this row
            valid_rmse_values = []
            for method in available_methods:
                rmse_val = row_data[method]
                if pd.notna(rmse_val):
                    valid_rmse_values.append(rmse_val)
            
            min_rmse = min(valid_rmse_values) if valid_rmse_values else None
            
            for method in available_methods:
                rmse_val = row_data[method]
                if pd.notna(rmse_val):
                    # Bold if this is the minimum (best) RMSE value for this row
                    if abs(rmse_val - min_rmse) < 1e-6:  # Use small epsilon for float comparison
                        row += f" & \\textbf{{{rmse_val:.3f}}}"
                    else:
                        row += f" & {rmse_val:.3f}"
                else:
                    row += " & --"
            
            row += " \\\\\n"
            latex_content_mcar += row
        
        latex_content_mcar += "\\bottomrule\n"
        latex_content_mcar += "\\end{tabular}\n"
        latex_content_mcar += "\\end{table}\n"
        
        # Save MCAR table to file
        mcar_filename = "figures/rmse_mcar_by_dataset.txt"
        with open(mcar_filename, "w") as f:
            f.write(latex_content_mcar)
        
        print(f"MCAR RMSE table saved to {mcar_filename}")
    else:
        print("No methods from table_methods found in MCAR data")
else:
    print("No MCAR data found")