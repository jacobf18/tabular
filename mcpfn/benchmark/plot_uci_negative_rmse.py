import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings
from scipy import stats
import itertools
from plot_options import (
    METHOD_NAMES,
    METHOD_COLORS,
    PATTERNS,
    PATTERN_LATEX_NAMES,
    setup_latex_fonts,
    PLOT_PARAMS,
    BARPLOT_STYLE,
    FIGURE_SIZES,
    HIGHLIGHT_COLOR,
)

# --- Plotting ---
<<<<<<< HEAD
# setup_latex_fonts()
matplotlib.rcParams['font.family'] = 'serif'
=======
# Configure LaTeX rendering for all text in plots
setup_latex_fonts()
>>>>>>> a577979 (added test time training)

# Ensure LaTeX is enabled (redundant but explicit)
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'

# Ensure LaTeX is enabled (redundant but explicit)
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'

base_path = "datasets/uci"

methods = [
    "softimpute", 
    # "column_mean", 
    "hyperimpute",
    # "ot_sinkhorn",
    "missforest",
    "ice",
    # "mice",
    "gain",
    # "miwae",
    "masters_mcar",
    # "tabimpute_mcar_p0.4_num_cls_8_rank_1_11",
    # "tabimpute_mcar_p0.4_num_cls_12_rank_1_11",
    # "tabimpute_rope_v1",
    # "tabimpute_rope_trial_6_chunking",
    "tabimpute_rope_trial_14_no_permute",
    # "tabimpute_new_stable_v1_100k",
    # "tabimpute_new_stable_v1_100k_w_permute",
    "tabimpute_rope_trial_12_no_permute",
    # "tabimpute_new_stable_v4_27500_no_permute",
    # "tabimpute_new_stable_v3_15000",
    # "tabimpute_new_stable_v6_25000",
    # "tabimpute_new_stable_v3_15000_300_rows",
    # "tabimpute_new_stable_v3_15000_250_rows",
    # "tabimpute_new_stable_v3_15000_100_rows",
    "tabimpute_new_stable_round2_trial_3",
    "tabimpute_new_stable_round2_trial_2",
    "tabimpute_new_stable_round2_trial_6",
    "tabimpute_new_stable_round2_trial_10",
    "tabimpute_new_stable_shallow_trial_3",
    "tabimpute_new_stable_shallow_trial_6",
    "tabimpute_new_stable_shallow_trial_2",
    "tabimpute_new_stable_deep_trial_3",
    # "tabimpute_rope_trial_12",
    # "tabimpute_large_mcar",
    # "tabimpute_large_mcar_rank_1_11",
    # "tabimpute_large_mcar_mnar",
    # # "masters_mar",
    # # "masters_mnar",
    # # "masters_mcar_nonlinear",
    # "tabpfn",
    # "tabpfn_impute",
    "knn",
    # "forestdiffusion",
    # "diffputer",
    # "remasker",
]

# Use patterns from plot_options, but filter to only active ones
patterns = {
    "MCAR",
    # "MAR",
    # "MNAR",
    # "MAR_Neural",
    # "MAR_BlockNeural",
    # "MAR_Sequential",
    # "MNARPanelPattern",
    # "MNARPolarizationPattern",
    # "MNARSoftPolarizationPattern",
    # "MNARLatentFactorPattern",
    # "MNARClusterLevelPattern",
    # "MNARTwoPhaseSubsetPattern",
    # "MNARCensoringPattern",
}

missingness_levels = [
    # 0.1, 0.2, 0.3, 
    0.4, 
    # 0.5
]

num_repeats = 10

# Use method names from plot_options
method_names = METHOD_NAMES.copy()
# Add any UCI-specific method name mappings if needed
method_names["tabimpute_mcar_p0.4_num_cls_8_rank_1_11"] = "TabImpute (CLS-8)"
method_names["tabimpute_mcar_p0.4_num_cls_12_rank_1_11"] = "TabImpute (New)"
method_names["tabimpute_rope_v1"] = "TabImpute (Rope no chunks)"
method_names["tabimpute_rope_trial_6_chunking"] = "TabImpute (Rope w/ Chunks)"
method_names["tabimpute_rope_trial_14_no_permute"] = "TabImpute (Rope Trial 14 No Permute)"
method_names["tabimpute_rope_trial_12_no_permute"] = "TabImpute (Rope Trial 12 No Permute)"
method_names["tabimpute_rope_trial_12"] = "TabImpute (Rope Trial 12 w/ Permute)"
method_names["tabimpute_new_stable_v1_100k"] = "TabImpute (New Stable v1 100k)"
method_names["tabimpute_new_stable_v1_100k_w_permute"] = "TabImpute (New Stable v1 100k w/ Permute)"
method_names["tabimpute_new_stable_v4_27500_no_permute"] = "TabImpute (New Stable v4 27500 No Permute)"
method_names["tabimpute_new_stable_v3_15000"] = "TabImpute (New Stable v3 15000)"
method_names["tabimpute_new_stable_v6_25000"] = "TabImpute (New Stable v6 25000)"
method_names["tabimpute_new_stable_v3_15000_300_rows"] = "TabImpute (New Stable v3 15000 300 Rows)"
method_names["tabimpute_new_stable_v3_15000_250_rows"] = "TabImpute (New Stable v3 15000 250 Rows)"
method_names["tabimpute_new_stable_v3_15000_100_rows"] = "TabImpute (New Stable v3 15000 100 Rows)"
method_names["tabimpute_new_stable_round2_trial_3"] = "TabImpute (New Stable Round 2 Trial 3)"
method_names["tabimpute_new_stable_round2_trial_2"] = "TabImpute (New Stable Round 2 Trial 2)"
method_names["tabimpute_new_stable_round2_trial_6"] = "TabImpute (New Stable Round 2 Trial 6)"
method_names["tabimpute_new_stable_round2_trial_10"] = "TabImpute (New Stable Round 2 Trial 10)"
method_names["tabimpute_new_stable_shallow_trial_3"] = "TabImpute (New Stable Shallow Trial 3)"
method_names["tabimpute_new_stable_shallow_trial_6"] = "TabImpute (New Stable Shallow Trial 6)"
method_names["tabimpute_new_stable_shallow_trial_2"] = "TabImpute (New Stable Shallow Trial 2)"
method_names["tabimpute_new_stable_deep_trial_3"] = "TabImpute (New Stable Deep Trial 3)"
# Use method colors from plot_options
method_colors = METHOD_COLORS

for method in method_names.values():
    if "TabImpute" in method:
        method_colors[method] = HIGHLIGHT_COLOR

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

datasets = os.listdir(base_path)
for dataset in datasets:
    for pattern, missingness_level, repeat in itertools.product(patterns, missingness_levels, range(num_repeats)):
        cfg_dir = f"{base_path}/{dataset}/{pattern}/missingness-{missingness_level}/repeat-{repeat}"

        X_missing = np.load(f"{cfg_dir}/missing.npy")
        X_true = np.load(f"{cfg_dir}/true.npy")
        
        mask = np.isnan(X_missing)
        
        for method in methods:
            X_imputed = np.load(f"{cfg_dir}/{method}.npy")
            name = method_names[method]
            negative_rmse[(dataset, pattern, missingness_level, repeat, name)] = compute_normalized_rmse_columnwise(X_true, X_imputed, mask)

# Create summary dataframe with missingness_level as rows and method as columns
s = pd.Series(negative_rmse)
s.index = pd.MultiIndex.from_tuples(s.index, names=['dataset', 'pattern', 'missingness_level', 'repeat', 'method'])
df_summary = s.groupby(['missingness_level', 'method']).mean().unstack('method')
print("\nMean by missingness_level and method:")
print(df_summary.sort_values(by='missingness_level', ascending=True))

# exit()

df = pd.Series(negative_rmse).unstack()
df_norm = 1.0 - (df - df.min(axis=1).values[:, None]) / (df.max(axis=1) - df.min(axis=1)).values[:, None]

# print(df_norm)

plot_pattern = True
dont_plot_methods = []

# Plot for all patterns combined
# Get dataframe for all patterns
df_all = df.copy()
df_norm_all = 1.0 - (df_all - df_all.min(axis=1).values[:, None]) / (df_all.max(axis=1) - df_all.min(axis=1)).values[:, None]

if plot_pattern:
    for pattern_name in patterns:
        # Get dataframe for 1 pattern
        df2 = df[df.index.get_level_values(1) == pattern_name]
        df_norm = 1.0 - (df2 - df2.min(axis=1).values[:, None]) / (
            df2.max(axis=1) - df2.min(axis=1)
        ).values[:, None]
        
        # Average across datasets
        # --- Barplot ---
<<<<<<< HEAD
        plt.figure(figsize=FIGURE_SIZES['standard'])

        # sort methods by mean performance (best first)
        sorted_methods = df_norm.mean(axis=0).sort_values(ascending=False).index
        sorted_methods = [method for method in sorted_methods if method not in dont_plot_methods]

=======
        plt.figure(figsize=(4.5,4.5))
        
        # sort methods by mean performance (descending order - best first)
        sorted_methods = df_norm.mean(axis=0).sort_values(ascending=False).index
        
>>>>>>> a577979 (added test time training)
        # Melt into long format
        df_long = df_norm.melt(var_name="method", value_name="score")

        # Use method_colors from plot_options for consistent mapping
        ax = sns.barplot(
            data=df_long,
            x="method",
            y="score",
            hue="method",
            order=sorted_methods,
            palette=method_colors,
            legend=False,
            **BARPLOT_STYLE
        )
<<<<<<< HEAD

        # Set x-axis labels with 45-degree rotation (like plot_negative_rmse)
        labels_with_bold = [method for method in sorted_methods]
        ax.set_xticks(range(len(sorted_methods)))
        ax.set_xticklabels(labels_with_bold, rotation=45, ha='right')
        ax.set_xlabel("")

        # Set label colors to match bar colors and make TabImpute slightly larger
=======
        
        # Set x-axis labels with 45-degree rotation
        # Bold TabImpute using LaTeX \textbf{}
        labels_with_bold = [r"\textbf{" + method + "}" if method == "TabImpute" else method for method in sorted_methods]
        ax.set_xticks(range(len(sorted_methods)))
        ax.set_xticklabels(labels_with_bold, rotation=45, ha='right')
        ax.set_xlabel("")
        
        # Set label colors to match bar colors and make TabImpute even bolder
>>>>>>> a577979 (added test time training)
        for i, label in enumerate(ax.get_xticklabels()):
            method_name = sorted_methods[i]
            if method_name in method_colors:
                label.set_color(method_colors[method_name])
            if method_name == "TabImpute":
<<<<<<< HEAD
=======
                # Make TabImpute slightly larger for extra boldness
>>>>>>> a577979 (added test time training)
                label.set_fontsize(label.get_fontsize() * 1.1)

        # Use LaTeX-formatted label
        plt.ylabel(r"1 - Normalized RMSE", fontsize=15)
        # plt.title(f"Comparison of Imputation Algorithms | {pattern_name}")
        # plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(f"figures/uci_negative_rmse_{pattern_name}.pdf", **PLOT_PARAMS)
        plt.close()

    # Average across datasets and patterns
<<<<<<< HEAD
    fig = plt.figure(figsize=FIGURE_SIZES['standard'])
    # sort df_norm_all by the mean of the rows (best first)
    sorted_methods_all = df_norm_all.mean(axis=0).sort_values(ascending=False).index
    sorted_methods_all = [method for method in sorted_methods_all if method not in dont_plot_methods]
=======
    fig = plt.figure(figsize=(4.5,4.5))
    # sort df_norm_all by the mean of the rows (descending order - best first)
    sorted_methods_all = df_norm_all.mean(axis=0).sort_values(ascending=False).index
>>>>>>> a577979 (added test time training)

    # Melt into long format
    df_long = df_norm_all.melt(var_name="method", value_name="score")

    # Use method_colors from plot_options for consistent mapping
    ax = sns.barplot(
        data=df_long,
        x="method",
        y="score",
        hue="method",
        order=sorted_methods_all,
        palette=method_colors,
        legend=False,
        **BARPLOT_STYLE
    )

<<<<<<< HEAD
    # Set x-axis labels with 45-degree rotation (like plot_negative_rmse)
    labels_with_bold = [method for method in sorted_methods_all]
=======
    # Set x-axis labels with 45-degree rotation
    # Bold TabImpute using LaTeX \textbf{}
    labels_with_bold = [r"\textbf{" + method + "}" if method == "TabImpute" else method for method in sorted_methods_all]
>>>>>>> a577979 (added test time training)
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

<<<<<<< HEAD
    # Set label colors to match bar colors and make TabImpute slightly larger
    for i, label in enumerate(ax.get_xticklabels()):
        method_name = sorted_methods_all[i]
        if method_name in method_colors:
            label.set_color(method_colors[method_name])
        if method_name == "TabImpute":
            label.set_fontsize(label.get_fontsize() * 1.1)

    plt.ylabel("1 - Normalized RMSE (0–1)", fontsize=12)
=======
    # Use LaTeX-formatted label
    plt.ylabel(r"1 - Normalized RMSE", fontsize=18)
>>>>>>> a577979 (added test time training)
    # plt.title("Comparison of Imputation Algorithms | All Patterns")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    plt.savefig(f"figures/uci_negative_rmse_overall.pdf", **PLOT_PARAMS)
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
    # "TabImpute (MAR)",
    # "TabImpute (Self-Masking-MNAR)",
    # "TabPFN Fine-Tuned No Preprocessing",
    "EWF-TabPFN",
    "TabPFN",
    # "TabImpute (MCAR then MAR)",
    # "TabImpute (More Heads)",
    # "TabImpute (Nonlinear)",
    "HyperImpute",
    "MissForest",
    "OT",
    "Col Mean",
    "SoftImpute",
    "ICE",
    "MICE",
    "GAIN",
    "MIWAE",
    "K-Nearest Neighbors",
    "ForestDiffusion",
    # "DiffPuter",
    # "ReMasker",
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
df_all_norm = (df_all - df_all.min(axis=1).values[:, None]) / (df_all.max(axis=1) - df_all.min(axis=1)).values[:, None]
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
    
    # Generate LaTeX table with maximum 4 columns
    max_cols = 5
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
        
        # Calculate global maximum across all values in this table (higher is better for normalized negative RMSE)
        table_global_max = float('-inf')
        for pattern in all_patterns:
            for method in table_methods:
                if f"{method}_mean" in summary_pivot.columns:
                    mean_val = summary_pivot.loc[pattern, f"{method}_mean"]
                    if pd.notna(mean_val) and mean_val > table_global_max:
                        table_global_max = mean_val
        
        # Data rows (patterns as rows)
        for i, pattern in enumerate(all_patterns):
            row = patern_latex_names[pattern]  # Escape underscores in pattern names
            
            # Add midrule before Overall performance
            if pattern == 'Overall' and i > 0:
                latex_content += "\\midrule\n"
            
            for method in table_methods:
                if f"{method}_mean" in summary_pivot.columns:
                    mean_val = summary_pivot.loc[pattern, f"{method}_mean"]
                    std_val = summary_pivot.loc[pattern, f"{method}_std"]
                    if pd.notna(mean_val) and pd.notna(std_val):
                        # Bold if this is the global maximum value in this table
                        if abs(mean_val - table_global_max) < 1e-6:  # Use small epsilon for float comparison
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
        filename = f"figures/uci_normalized_negative_rmse_table_{table_idx + 1}.txt" if num_tables > 1 else "figures/uci_normalized_negative_rmse_table.txt"
        with open(filename, "w") as f:
            f.write(latex_content)
        
        print(f"LaTeX table {table_idx + 1} saved to {filename}")