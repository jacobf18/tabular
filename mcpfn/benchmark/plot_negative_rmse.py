import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings
from scipy import stats
from plot_options import (
    setup_latex_fonts,
    METHOD_NAMES,
    METHOD_COLORS,
    HIGHLIGHT_COLOR,
    NEUTRAL_COLOR,
    PATTERNS,
    PATTERN_LATEX_NAMES,
    FIGURE_SIZES,
    BARPLOT_STYLE,
)

# --- Plotting ---
# Configure LaTeX rendering for all text in plots
setup_latex_fonts()

# Ensure LaTeX is enabled (redundant but explicit)
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'

base_path = "datasets/openml"

datasets = os.listdir(base_path)

methods = [
    # "tabpfn_no_preprocessing",
    "softimpute", 
    # "tabimpute_mcar_lin",
    # "tabimpute_dynamic_cls",
    # "tabimpute_large_cls_8",
    "tabimpute_75_75_rank_1_11",
    "tabimpute_large_mcar_rank_1_11",
    # "column_mean", 
    "hyperimpute",
    "ot_sinkhorn",
    "missforest",
    "ice",
    "mice",
    "gain",
    "miwae",
    "masters_mcar",
    "tabpfn",
    "tabpfn_impute",
    "knn",
    "forestdiffusion",
    "remasker",
    "cacti",
]



# Use patterns from plot_options
patterns = PATTERNS

# Use method names from plot_options
method_names = METHOD_NAMES

# Use colors from plot_options
neutral_color = NEUTRAL_COLOR
highlight_color = HIGHLIGHT_COLOR
method_colors = METHOD_COLORS.copy()

method_names.update({
    "tabimpute_mcar_lin": "TabImpute (Lin. Emb.)",
    "tabimpute_large_mcar_rank_1_11": "TabImpute (50x50)",
    "tabimpute_dynamic_cls": "TabImpute (Dynamic CLs)",
    "tabimpute_large_cls_8": "TabImpute (CLS-8)",
    "tabimpute_75_75_rank_1_11": "TabImpute (75x75)",
})

method_colors.update({
    "TabImpute (Lin. Emb.)": highlight_color,
    "TabImpute (50x50)": highlight_color,
    "TabImpute (Dynamic CLs)": highlight_color,
    "TabImpute (CLS-8)": highlight_color,
    "TabImpute (75x75)": highlight_color,
})

# Add missing method colors that may appear in the data

negative_rmse = {}

def compute_negative_rmse(X_true, X_imputed, mask):
    return -np.sqrt(np.mean((X_true[mask] - X_imputed[mask]) ** 2))

def compute_normalized_rmse(X_true, X_imputed, mask):
    return np.sqrt(np.mean((X_true[mask] - X_imputed[mask]) ** 2)) / np.std(X_true[mask])

def compute_normalized_rmse_columnwise(X_true, X_imputed, mask):
    """
    Computes the Average Column-wise NRMSE.
    
    The metric is calculated for each column j as:
        NRMSE_j = RMSE_j / sigma_j
    
    Where:
        RMSE_j  = RMSE of the imputed values (masked entries only) in column j
        sigma_j = Standard deviation of the TRUE values (masked entries only) in column j
        
    The final score is the mean of NRMSE_j across all columns.

    Args:
        X_true (np.ndarray): The ground truth matrix (complete).
        X_imputed (np.ndarray): The matrix with imputed values.
        mask (np.ndarray): Boolean mask where True indicates a missing value 
                           (the value was imputed) and False indicates observed.

    Returns:
        float: The average NRMSE across all columns.
    """
    # Ensure inputs are numpy arrays
    X_true = np.asarray(X_true)
    X_imputed = np.asarray(X_imputed)
    mask = np.asarray(mask, dtype=bool)

    n_cols = X_true.shape[1]
    nrmse_list = []

    for j in range(n_cols):
        # Extract the boolean mask for the current column
        col_mask = mask[:, j]
        
        # If there are no missing values in this column, skip it
        if not np.any(col_mask):
            continue
            
        # Get the true and imputed values corresponding to the mask
        true_vals = X_true[col_mask, j]
        imp_vals = X_imputed[col_mask, j]
        
        # Calculate MSE for this column
        mse = np.mean((true_vals - imp_vals) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate Standard Deviation of the TRUE values in the missing set
        # (Matches the logic of the MissForest R code provided)
        sigma = np.std(true_vals)
        
        # Edge case: If sigma is 0 (all true values are identical), 
        # avoid division by zero.
        if sigma < 1e-9:
            # If the error is also 0, score is 0. Otherwise, it's effectively infinite error.
            if rmse < 1e-9:
                nrmse = 0.0
            else:
                # Log warning or handle as appropriate; here we skip or set to high value
                continue 
        else:
            nrmse = rmse / sigma
            
        nrmse_list.append(nrmse)

    # Return the average across all valid columns
    if not nrmse_list:
        return 0.0
    
    return np.mean(nrmse_list)

for dataset in datasets:
    configs = os.listdir(f"{base_path}/{dataset}")
    for config in configs:
        if "repeats" in config:
            continue
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
            # print(dataset, config, method, X_imputed.shape, X_true.shape, mask.shape)
            name = method_names[method]
            # negative_rmse[(dataset, pattern_name, name)] = compute_negative_rmse(X_true, X_imputed, mask)
            # negative_rmse[(dataset, pattern_name, name)] = compute_normalized_rmse(X_true, X_imputed, mask)
            negative_rmse[(dataset, pattern_name, name)] = compute_normalized_rmse_columnwise(X_true, X_imputed, mask)
            
# s = pd.Series(negative_rmse)
# s.index = pd.MultiIndex.from_tuples(s.index, names=['dataset', 'pattern', 'method'])
# df_summary = s.groupby(['pattern', 'method']).mean().unstack('method')
# print("\nMean by pattern and method:")
# print(df_summary)

# overall_row = df_summary.mean(axis=0)

# df_summary.loc['Overall'] = overall_row

# print(df_summary)

# exit()

df = pd.Series(negative_rmse).unstack()

plot_pattern = True

dont_plot_methods = [
    # "EWF-TabPFN",
]

# Plot for all patterns combined
# Get dataframe for all patterns
df_all = df.copy()
df_norm_all = 1.0 - (df_all - df_all.min(axis=1).values[:, None]) / (df_all.max(axis=1) - df_all.min(axis=1)).values[:, None]
# df_norm_all = df_all
if plot_pattern:
    for pattern_name in patterns:
        # Get dataframe for 1 pattern
        df2 = df[df.index.get_level_values(1) == pattern_name]
        df_norm = 1.0 - (df2 - df2.min(axis=1).values[:, None]) / (
            df2.max(axis=1) - df2.min(axis=1)
        ).values[:, None]
        # df_norm = df2
        
        # Average across datasets
        # --- Barplot ---
        plt.figure(figsize=FIGURE_SIZES['standard'])
        
        # sort methods by mean performance
        sorted_methods = df_norm.mean(axis=0).sort_values(ascending=False).index
        sorted_methods = [method for method in sorted_methods if method not in dont_plot_methods]
        # Melt into long format
        df_long = df_norm.melt(var_name="method", value_name="score")
        
        # Use method_colors dictionary for consistent mapping
        ax = sns.barplot(
            data=df_long,
            x="method",
            y="score",
            hue="method",
            order=sorted_methods,
            palette=method_colors,
            **BARPLOT_STYLE,
            legend=False,
        )
        
        # Set x-axis labels with 45-degree rotation
        # Bold TabImpute using LaTeX \textbf{}
        labels_with_bold = [r"\textbf{" + method + "}" if method == "TabImpute" else method for method in sorted_methods]
        ax.set_xticks(range(len(sorted_methods)))
        ax.set_xticklabels(labels_with_bold, rotation=45, ha='right')
        ax.set_xlabel("")
        
        # Set label colors to match bar colors and make TabImpute even bolder
        for i, label in enumerate(ax.get_xticklabels()):
            method_name = sorted_methods[i]
            if method_name in method_colors:
                label.set_color(method_colors[method_name])
            if method_name == "TabImpute":
                # Make TabImpute slightly larger for extra boldness
                label.set_fontsize(label.get_fontsize() * 1.1)

        # Use LaTeX-formatted label
        plt.ylabel(r"1 - Normalized RMSE", fontsize=15)
        # plt.title(f"Comparison of Imputation Algorithms | {pattern_name}")
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(f"figures/negative_rmse_{pattern_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Average across datasets and patterns
    fig = plt.figure(figsize=FIGURE_SIZES['standard'])
    # sort df_norm_all by the mean of the rows
    sorted_methods_all = df_norm_all.mean(axis=0).sort_values(ascending=False).index

    # Melt into long format
    df_long = df_norm_all.melt(var_name="method", value_name="score")
    sorted_methods_all = [method for method in sorted_methods_all if method not in dont_plot_methods]
    # Use method_colors dictionary for consistent mapping
    ax = sns.barplot(
        data=df_long,
        x="method",
        y="score",
        hue="method",
        order=sorted_methods_all,
        palette=method_colors,
        **BARPLOT_STYLE,
        legend=False,
    )

    # Set x-axis labels with 45-degree rotation
    # Bold TabImpute using LaTeX \textbf{}
    labels_with_bold = [r"\textbf{" + method + "}" if method == "TabImpute" else method for method in sorted_methods_all]
    ax.set_xticks(range(len(sorted_methods_all)))
    ax.set_xticklabels(labels_with_bold, rotation=45, ha='right', fontsize=14)
    ax.set_xlabel("")
    
    # Set label colors to match bar colors and make TabImpute even bolder
    for i, label in enumerate(ax.get_xticklabels()):
        method_name = sorted_methods_all[i]
        if method_name == "TabImpute":
            label.set_color(method_colors[method_name])
            # Make TabImpute slightly larger for extra boldness
            label.set_fontsize(label.get_fontsize() * 1.1)

    # Use LaTeX-formatted label
    plt.ylabel(r"1 - Normalized RMSE", fontsize=18)
    # plt.title("Comparison of Imputation Algorithms | All Patterns")
    plt.ylim(0.3, 0.95)
    plt.tight_layout()
    
    # fig.subplots_adjust(left=0.2, right=0.95, bottom=0.05, top=0.95)
    
    plt.savefig(f"figures/negative_rmse_overall.pdf", dpi=300, bbox_inches=None)
    plt.close()


exit()
# Count how many times TabImpute is the best method
# print(df_norm_all)
# tabimpute_best = df_norm_all[df_norm_all.index.get_level_values(2) == "TabImpute"].mean(axis=0).sort_values(ascending=True).index
# print(f"TabImpute is the best method {len(tabimpute_best)} times")

# Generate LaTeX table for normalized negative RMSE
print("Creating LaTeX table for normalized negative RMSE...")

# Define methods to include in the table (subset of all methods)
table_methods = [
    "TabImpute",
    # "EWF-TabPFN",
    "TabPFN",
    # "TabPFN Fine-Tuned No Preprocessing",
    # "TabImpute (MCAR then MAR)",
    # "TabImpute (More Heads)",
    # "TabImpute (Nonlinear)",
    "HyperImpute",
    "MissForest",
    "OT",
    # "Col Mean",
    # "SoftImpute",
    # "ICE",
    # "MICE",
    # "GAIN",
    # "MIWAE",
    "K-Nearest Neighbors",
    # "ForestDiffusion",
    # "ReMasker",
    # "CACTI",
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
    df_norm = 1.0 - (df_pattern - df_pattern.min(axis=1).values[:, None]) / (df_pattern.max(axis=1) - df_pattern.min(axis=1)).values[:, None]
    
    # Calculate mean and std for each method across datasets
    pattern_means = df_norm.mean(axis=0)
    # pattern_stds = df_norm.std(axis=0)
    pattern_stds = df_norm.sem(axis=0)
    
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
df_all_norm = 1.0 - (df_all - df_all.min(axis=1).values[:, None]) / (df_all.max(axis=1) - df_all.min(axis=1)).values[:, None]
overall_means = df_all_norm.mean(axis=0)
# overall_stds = df_all_norm.std(axis=0)
overall_stds = df_all_norm.sem(axis=0)

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
                "MNARPanelPattern",
                "MNARPolarizationPattern",
                "MNARSoftPolarizationPattern",
                "MNARLatentFactorPattern",
                "MNARClusterLevelPattern",
                "MNARTwoPhaseSubsetPattern",
                "MNARCensoringPattern",
                'Overall'
                ]

# Use pattern LaTeX names from plot_options
patern_latex_names = PATTERN_LATEX_NAMES

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
    
    # Generate LaTeX table with maximum 5 columns per tabular
    max_cols = 5
    num_tabulars = (len(all_methods) + max_cols - 1) // max_cols
    
    # Calculate global row maximums across ALL methods (for bolding across all tabulars)
    global_row_maxs = {}
    for pattern in all_patterns:
        row_max = float('-inf')
        for method in all_methods:
            if f"{method}_mean" in summary_pivot.columns:
                mean_val = summary_pivot.loc[pattern, f"{method}_mean"]
                if pd.notna(mean_val) and mean_val > row_max:
                    row_max = mean_val
        global_row_maxs[pattern] = row_max
    
    # Start single table environment
    latex_content = "\\begin{table}[h]\n"
    latex_content += "\\centering\n"
    latex_content += "\\caption{Mean Normalized Negative RMSE ± Standard Deviation by Missingness Pattern}\n"
    latex_content += "\\label{tab:normalized_negative_rmse_by_pattern}\n"
    
    # Generate each tabular environment
    for tabular_idx in range(num_tabulars):
        start_col = tabular_idx * max_cols
        end_col = min(start_col + max_cols, len(all_methods))
        this_table_methods = all_methods[start_col:end_col]
        
        # Create column specification
        num_methods_table = len(this_table_methods)
        col_spec = "l" + "c" * num_methods_table  # l for pattern names, c for each method
        
        latex_content += f"\\begin{{tabular}}{{{col_spec}}}\n"
        latex_content += "\\toprule\n"
        
        # Header row
        header = "Pattern"
        for method in this_table_methods:
            method_name = method.replace('_', '\\_')  # Escape underscores
            header += f" & {method_name}"
        header += " \\\\\n"
        latex_content += header
        latex_content += "\\midrule\n"
        
        # Data rows (patterns as rows)
        for i, pattern in enumerate(all_patterns):
            row = patern_latex_names[pattern]  # Escape underscores in pattern names
            
            # Add midrule before Overall performance
            if pattern == 'Overall' and i > 0:
                latex_content += "\\midrule\n"
            
            # Use global row maximum for this pattern (across all methods)
            row_max = global_row_maxs[pattern]
            
            for method in this_table_methods:
                if f"{method}_mean" in summary_pivot.columns:
                    mean_val = summary_pivot.loc[pattern, f"{method}_mean"]
                    std_val = summary_pivot.loc[pattern, f"{method}_std"]
                    if pd.notna(mean_val) and pd.notna(std_val):
                        # Bold if this is the global maximum value in this row (across all tabulars)
                        if abs(mean_val - row_max) < 1e-6:  # Use small epsilon for float comparison
                            row += f" & \\textbf{{{mean_val:.3f} ± {std_val:.3f}}}"
                        else:
                            row += f" & {mean_val:.3f} ± {std_val:.3f}"
                    else:
                        row += " & --"
            row += " \\\\\n"
            latex_content += row
        
        latex_content += "\\bottomrule\n"
        latex_content += "\\end{tabular}\n"
        
        # Add spacing between tabulars if not the last one
        if tabular_idx < num_tabulars - 1:
            latex_content += "\\quad\n"
    
    # End table environment
    latex_content += "\\end{table}\n"
    
    # Save single table file
    filename = "figures/normalized_negative_rmse_table.txt"
    with open(filename, "w") as f:
        f.write(latex_content)
    
    print(f"LaTeX table saved to {filename}")

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

# Generate transposed LaTeX table with maximum 4 columns per tabular
max_cols = 4
num_tabulars_transposed = (len(all_patterns) + max_cols - 1) // max_cols

# Calculate global row maximums across ALL patterns (for bolding across all tabulars)
global_row_maxs_transposed = {}
for method in all_methods:
    row_max = float('-inf')
    for pattern in all_patterns:
        mean_col = f"{pattern}_mean"
        if mean_col in transposed_df.columns:
            method_row = transposed_df[transposed_df['Method'] == method]
            if not method_row.empty:
                mean_val = method_row[mean_col].iloc[0]
                if pd.notna(mean_val) and mean_val > row_max:
                    row_max = mean_val
    global_row_maxs_transposed[method] = row_max

# Start single table environment
latex_content_transposed = "\\begin{table}[h]\n"
latex_content_transposed += "\\centering\n"
latex_content_transposed += "\\caption{Mean Normalized Negative RMSE ± Standard Deviation by Method (Transposed)}\n"
latex_content_transposed += "\\label{tab:normalized_negative_rmse_by_method_transposed}\n"

# Generate each tabular environment
for tabular_idx in range(num_tabulars_transposed):
    start_col = tabular_idx * max_cols
    end_col = min(start_col + max_cols, len(all_patterns))
    table_patterns = all_patterns[start_col:end_col]
    
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
        
        # Use global row maximum for this method (across all patterns)
        row_max = global_row_maxs_transposed[method]
        
        for pattern in table_patterns:
            mean_col = f"{pattern}_mean"
            std_col = f"{pattern}_std"
            
            if mean_col in transposed_df.columns and std_col in transposed_df.columns:
                method_row = transposed_df[transposed_df['Method'] == method]
                if not method_row.empty:
                    mean_val = method_row[mean_col].iloc[0]
                    std_val = method_row[std_col].iloc[0]
                    if pd.notna(mean_val) and pd.notna(std_val):
                        # Bold if this is the global maximum value in this row (across all tabulars)
                        if abs(mean_val - row_max) < 1e-6:  # Use small epsilon for float comparison
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
    
    # Add spacing between tabulars if not the last one
    if tabular_idx < num_tabulars_transposed - 1:
        latex_content_transposed += "\\quad\n"

# End table environment
latex_content_transposed += "\\end{table}\n"

# Save single transposed table file
transposed_filename = "figures/normalized_negative_rmse_table_transposed.txt"
with open(transposed_filename, "w") as f:
    f.write(latex_content_transposed)

print(f"Transposed LaTeX table saved to {transposed_filename}")

# Generate LaTeX table for non-normalized RMSE values for MCAR with datasets as rows
print("Creating LaTeX table for non-normalized RMSE values (MCAR, datasets as rows)...")

# Get data for MCAR pattern only
mcar_data = df[df.index.get_level_values(1) == 'MCAR']

print(table_methods)

available_methods = [
    'TabImpute',
    # 'EWF-TabPFN',
    'HyperImpute',
    'MissForest',
    'OT',
    "K-Nearest Neighbors",
    
]

if not mcar_data.empty:
    
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
                            rmse = neg_rmse  # Convert from negative RMSE to positive RMSE
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
            
            # Calculate row minimum for this dataset (across methods)
            row_min = float('inf')
            for method in available_methods:
                rmse_val = row_data[method]
                if pd.notna(rmse_val) and rmse_val < row_min:
                    row_min = rmse_val
            
            for method in available_methods:
                rmse_val = row_data[method]
                if pd.notna(rmse_val):
                    # Bold if this is the minimum value in this row
                    if abs(rmse_val - row_min) < 1e-6:  # Use small epsilon for float comparison
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
    
    
exit()
    
# Calculate p values for comparison of TabImpute and HyperImpute
print("\n" + "="*60)
print("Statistical comparison: TabImpute vs HyperImpute")
print("="*60)

# Get normalized RMSE values for both methods
# df_all_norm has MultiIndex (dataset, pattern) and columns are methods
tabimpute_values = []
hyperimpute_values = []

# Extract paired values (same dataset-pattern combination)
for idx in df_all_norm.index:
    if 'TabImpute' in df_all_norm.columns and 'HyperImpute' in df_all_norm.columns:
        tabimpute_val = df_all_norm.loc[idx, 'TabImpute']
        hyperimpute_val = df_all_norm.loc[idx, 'HyperImpute']
        
        # Only include pairs where both values are not NaN
        if pd.notna(tabimpute_val) and pd.notna(hyperimpute_val):
            tabimpute_values.append(tabimpute_val)
            hyperimpute_values.append(hyperimpute_val)

tabimpute_values = np.array(tabimpute_values)
hyperimpute_values = np.array(hyperimpute_values)

if len(tabimpute_values) > 0:
    print(f"\nNumber of valid comparisons: {len(tabimpute_values)}")
    print(f"TabImpute mean normalized negative RMSE: {np.mean(tabimpute_values):.4f} ± {np.std(tabimpute_values):.4f}")
    print(f"HyperImpute mean normalized negative RMSE: {np.mean(hyperimpute_values):.4f} ± {np.std(hyperimpute_values):.4f}")
    
    # Calculate differences (TabImpute - HyperImpute)
    # Higher normalized negative RMSE is better, so positive difference means TabImpute is better
    differences = tabimpute_values - hyperimpute_values
    print(f"\nMean difference (TabImpute - HyperImpute): {np.mean(differences):.4f}")
    print(f"  (Positive values indicate TabImpute is better)")
    
    # Paired t-test (parametric)
    # H0: mean difference = 0 (no difference)
    # H1: mean difference > 0 (TabImpute is better)
    t_stat, p_value_ttest = stats.ttest_rel(tabimpute_values, hyperimpute_values, alternative='greater')
    print(f"\nPaired t-test (one-sided, TabImpute > HyperImpute):")
    print(f"  t-statistic: {t_stat:.8f}")
    print(f"  p-value: {p_value_ttest:.8f}")
    
    # Wilcoxon signed-rank test (non-parametric)
    # H0: median difference = 0 (no difference)
    # H1: median difference > 0 (TabImpute is better)
    try:
        wilcoxon_stat, p_value_wilcoxon = stats.wilcoxon(
            tabimpute_values, hyperimpute_values, 
            alternative='greater', 
            zero_method='pratt'  # Handle zero differences
        )
        print(f"\nWilcoxon signed-rank test (one-sided, TabImpute > HyperImpute):")
        print(f"  statistic: {wilcoxon_stat:.8f}")
        print(f"  p-value: {p_value_wilcoxon:.8f}")
    except Exception as e:
        print(f"\nWilcoxon test could not be computed: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    if p_value_ttest < 0.001:
        print("Result: TabImpute is significantly better than HyperImpute (p < 0.001)")
    elif p_value_ttest < 0.01:
        print("Result: TabImpute is significantly better than HyperImpute (p < 0.01)")
    elif p_value_ttest < 0.05:
        print("Result: TabImpute is significantly better than HyperImpute (p < 0.05)")
    else:
        print("Result: No significant difference between TabImpute and HyperImpute")
    print("="*60 + "\n")
else:
    print("No valid paired comparisons found between TabImpute and HyperImpute")

