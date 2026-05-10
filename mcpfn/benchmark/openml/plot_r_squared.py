import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings
from scipy import stats
from sklearn.metrics import r2_score
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plot_options import METHOD_NAMES, METHOD_COLORS

# --- Plotting ---
sns.set(style="whitegrid")

base_path = "../datasets/openml"

datasets = os.listdir(base_path)

methods = [
    "softimpute", 
    # "column_mean", 
    "hyperimpute",
    "ot_sinkhorn",
    "missforest",
    "ice",
    "mice",
    "gain",
    "miwae",
    "masters_mcar",
    # "masters_mar",
    # "masters_mnar",
    # "tabimpute_new_stable_round2_trial_2",
    # "masters_mcar_nonlinear",
    # "tabpfn",
    "tabpfn_impute",
    "knn",
    "forestdiffusion",
    # "diffputer",
    "remasker",
    "cacti",
    "notmiwae",
]

patterns = {
    "MCAR",
    "MAR",
    "MNAR",
    "MAR_Neural",
    "MAR_BlockNeural",
    "MNARPolarizationPattern",
    "MNARSoftPolarizationPattern",
    "MNARLatentFactorPattern",
    "MNARClusterLevelPattern",
    "MNARTwoPhaseSubsetPattern",
    "MNARCensoringPattern",
}

method_names = METHOD_NAMES
method_colors = METHOD_COLORS

r_squared = {}

def compute_r_squared(X_true, X_imputed, mask):
    # loop over columns and calculate the R^2 score
    r_squared = []
    for col in range(X_true.shape[1]):
        if mask[:, col].sum() < 2:
            continue
        if np.isnan(X_true[:, col][mask[:, col]]).any() or np.isnan(X_imputed[:, col][mask[:, col]]).any():
            continue
        predicted = X_imputed[:, col][mask[:, col]]
        true = X_true[:, col][mask[:, col]]
        if np.isnan(predicted).any() or np.isnan(true).any():
            continue
        if np.isclose(np.std(predicted), 0, atol=1e-3) or np.isclose(np.std(true), 0, atol=1e-3):
            r2 = 0.0
        else:
            r2 = np.corrcoef(
                predicted,
                true
                )[0, 1]
            r2 **= 2
            if np.isnan(r2):
                r2 = 0.0
        r_squared.append(r2)
    if len(r_squared) == 0:
        return 0.0
    return np.mean(r_squared)

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
        
        if np.isnan(X_true).any():
            print(f"NaN values found in {dataset}/{pattern_name}_{p}/true.npy or {dataset}/{pattern_name}_{p}/missing.npy")
            continue
        
        mask = np.isnan(X_missing)
        
        for method in methods:
            X_imputed = np.load(f"{base_path}/{dataset}/{pattern_name}_{p}/{method}.npy")
            # print(dataset, config, method, X_imputed.shape, X_true.shape, mask.shape)
            name = method_names[method]
            r_squared[(dataset, pattern_name, name)] = compute_r_squared(X_true, X_imputed, mask)
            
df = pd.Series(r_squared).unstack()

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
            hue="method",
            order=sorted_methods,
            palette=method_colors,   # <- consistent colors
            capsize=0.2,
            err_kws={"color": "#999999"},
            legend=False
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

        plt.ylabel("Column-wise R^2", fontsize=15)
        # plt.title(f"Comparison of Imputation Algorithms | {pattern_name}")
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(f"figures/r_squared_{pattern_name}.png", dpi=300)
        plt.close()

    # Average across datasets and patterns
    fig = plt.figure(figsize=(6.5,4.5))
    # sort df_norm_all by the mean of the rows
    sorted_methods_all = df_norm_all.mean(axis=0).sort_values(ascending=True).index

    # Melt into long format
    df_long = df_norm_all.melt(var_name="method", value_name="score")
    print(df_long)

    # Use your method_colors dictionary for consistent mapping
    ax = sns.barplot(
        data=df_long,
        x="method",
        y="score",
        hue="method",
        order=sorted_methods_all,
        palette=method_colors,   # <- consistent colors
        capsize=0.2,
        err_kws={"color": "#999999"},
        legend=False
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

    plt.ylabel("Column-wise R^2", fontsize=18)
    # plt.title("Comparison of Imputation Algorithms | All Patterns")
    plt.ylim(0, 1.0)
    # plt.tight_layout()
    
    fig.subplots_adjust(left=0.2, right=0.95, bottom=0.05, top=0.95)
    
    plt.savefig(f"figures/r_squared_overall.pdf", dpi=300, bbox_inches=None)
    plt.close()

# Count how many times TabImpute is the best method
# print(df_norm_all)
# tabimpute_best = df_norm_all[df_norm_all.index.get_level_values(2) == "TabImpute"].mean(axis=0).sort_values(ascending=True).index
# print(f"TabImpute is the best method {len(tabimpute_best)} times")

# Generate LaTeX table for column-wise R^2
print("Creating LaTeX table for column-wise R^2...")


def _ci95_halfwidth(series: pd.Series) -> float:
    n = int(series.count())
    if n < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, n - 1)) * float(series.sem())

# Define methods to include in the table (subset of all methods)
table_methods = [
    "TabImpute",
    # "TabImpute (New Model)",
    # "TabImpute (MAR)",
    # "TabImpute (Self-Masking-MNAR)",
    # "TabPFN Fine-Tuned No Preprocessing",
    # "EWF-TabPFN",
    "TabPFN",
    # "TabImpute (MCAR then MAR)",
    # "TabImpute (More Heads)",
    # "TabImpute (Nonlinear)",
    "HyperImpute",
    "MissForest",
    "OT",
    # "Col Mean",
    "SoftImpute",
    "ICE",
    "MICE-Forest",
    "GAIN",
    "MIWAE",
    "K-NN",
    "ForestDiffusion",
    # "DiffPuter",
    "ReMasker",
    "CACTI",
    "not-MIWAE",
]

# Calculate column-wise R^2 for each pattern
pattern_r_squared = {}

for pattern_name in patterns:
    # Get dataframe for this pattern
    df_pattern = df[df.index.get_level_values(1) == pattern_name]
    
    if df_pattern.empty:
        print(f"No data found for pattern: {pattern_name}")
        continue
    
    # Normalize column-wise R^2 for this pattern
    df_norm = df_pattern
    
    # Calculate mean and 95% CI half-width for each method across datasets
    pattern_means = df_norm.mean(axis=0)
    # pattern_stds = df_norm.std(axis=0)
    pattern_stds = df_norm.apply(_ci95_halfwidth, axis=0)
    
    # Create summary data for this pattern
    pattern_data = []
    for method in pattern_means.index:
        pattern_data.append({
            'Method': method,
            'mean': pattern_means[method],
            'std': pattern_stds[method],
            'Pattern': pattern_name
        })
    
    pattern_r_squared[pattern_name] = pd.DataFrame(pattern_data)

# Calculate overall column-wise R^2
overall_means = df_all.mean(axis=0)
# overall_stds = df_all.std(axis=0)
overall_stds = df_all.apply(_ci95_halfwidth, axis=0)

# Create summary data for overall
overall_data = []
for method in overall_means.index:
    overall_data.append({
        'Method': method,
        'mean': overall_means[method],
        'std': overall_stds[method],
        'Pattern': 'Overall'
    })

pattern_r_squared['Overall'] = pd.DataFrame(overall_data)

# Create summary data for LaTeX table
summary_data = []
all_patterns = ['MCAR',
                'MAR',
                'MNAR',
                'MAR_Neural',
                'MAR_BlockNeural',
                "MNARPolarizationPattern",
                "MNARSoftPolarizationPattern",
                "MNARLatentFactorPattern",
                "MNARClusterLevelPattern",
                "MNARTwoPhaseSubsetPattern",
                "MNARCensoringPattern",
                'Overall'
                ]

patern_latex_names = {
    "MCAR": "\\mcar",
    "MAR_Neural": "\\nnmar",
    "MNAR": "\\mnarself",
    "MAR": "\\colmar",
    "MAR_BlockNeural": "\\marblockneural",
    "MAR_Sequential": "\\seqmar",
    "MNARPanelPattern": "\\panelmnar",
    "MNARPolarizationPattern": "\\polarmnar",
    "MNARSoftPolarizationPattern": "\\softpolarmnar",
    "MNARLatentFactorPattern": "\\latentmnar",
    "MNARClusterLevelPattern": "\\clustermnar",
    "MNARTwoPhaseSubsetPattern": "\\twophasemnar",
    "MNARCensoringPattern": "\\censormnar",
    "Overall": "Overall"
}

for pattern in all_patterns:
    if pattern in pattern_r_squared:
        pattern_df = pattern_r_squared[pattern]
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
    
    # Sort methods by overall performance (higher column-wise R^2 is better)
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
    
    # Per-pattern grouping into MCAR / MAR / MNAR rows (Overall is ungrouped).
    # Note: MAR_Neural and MAR_BlockNeural are listed under the MNAR group to
    # match the paper's table layout — they generate self-conditional dropout
    # patterns even though their names start with "MAR_".
    _pattern_groups = {
        "MCAR": "MCAR",
        "MAR": "MAR",
        "MNAR": "MNAR",
        "MAR_Neural": "MNAR",
        "MAR_BlockNeural": "MNAR",
        "MNARPolarizationPattern": "MNAR",
        "MNARSoftPolarizationPattern": "MNAR",
        "MNARLatentFactorPattern": "MNAR",
        "MNARClusterLevelPattern": "MNAR",
        "MNARTwoPhaseSubsetPattern": "MNAR",
        "MNARCensoringPattern": "MNAR",
        "Overall": None,
    }
    _group_latex_macro = {"MCAR": "\\mcar", "MAR": "\\mar", "MNAR": "\\mnar"}
    _group_size = {}
    for _p in all_patterns:
        _g = _pattern_groups.get(_p)
        if _g is not None:
            _group_size[_g] = _group_size.get(_g, 0) + 1

    # Start single table environment
    latex_content = "\\begin{table*}[!ht]\n"
    latex_content += "\\centering\n"
    latex_content += "\\caption{Mean Column-wise R^2 $\\pm$ 95\\% Confidence Interval by Missingness Pattern}\n"
    latex_content += "\\label{tab:r_squared_by_pattern}\n"

    # Generate each tabular environment
    for tabular_idx in range(num_tabulars):
        start_col = tabular_idx * max_cols
        end_col = min(start_col + max_cols, len(all_methods))
        table_methods = all_methods[start_col:end_col]

        # Create column specification: leading "ll" = (group label, pattern name)
        num_methods_table = len(table_methods)
        col_spec = "ll" + "c" * num_methods_table

        latex_content += "\\resizebox{\\linewidth}{!}{\n"
        latex_content += f"\\begin{{tabular}}{{{col_spec}}}\n"
        latex_content += "\\toprule\n"

        # Header row: first column (group) is unlabeled, second is "Pattern".
        header = "& Pattern"
        for method in table_methods:
            method_name = method.replace('_', '\\_')  # Escape underscores
            header += f" & {method_name}"
        header += " \\\\\n"
        latex_content += header
        latex_content += "\\midrule\n"

        # Data rows (patterns as rows)
        prev_group = None
        for i, pattern in enumerate(all_patterns):
            group = _pattern_groups.get(pattern)

            # Soft separator between MCAR / MAR / MNAR groups
            # (Overall uses a hard \midrule below, not a dashed line.)
            if (
                i > 0
                and group is not None
                and prev_group is not None
                and group != prev_group
            ):
                latex_content += "\\noalign{\\vskip 1.5pt}\n"
                latex_content += "\\hdashline\n"
                latex_content += "\\noalign{\\vskip 1.5pt}\n"

            # Hard rule before Overall
            if pattern == 'Overall' and i > 0:
                latex_content += "\\midrule\n"

            # Group cell: bold macro on the first row of each group, multirow if
            # the group has multiple patterns; empty for subsequent rows in the
            # group and for the ungrouped Overall row.
            if group is None:
                group_cell = ""
            elif group != prev_group:
                n_rows = _group_size[group]
                macro = _group_latex_macro[group]
                if n_rows == 1:
                    group_cell = f"\\textbf{{{macro}}}"
                else:
                    group_cell = f"\\multirow{{{n_rows}}}{{*}}{{\\textbf{{{macro}}}}}"
            else:
                group_cell = ""

            row = f"{group_cell} & {patern_latex_names[pattern]}"

            # Use global row maximum for this pattern (across all methods)
            row_max = global_row_maxs[pattern]

            for method in table_methods:
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
            prev_group = group

        latex_content += "\\bottomrule\n"
        latex_content += "\\end{tabular}\n"
        latex_content += "}\n"

        # Add spacing between tabulars if not the last one
        if tabular_idx < num_tabulars - 1:
            latex_content += "\\quad\n"

    # End table environment
    latex_content += "\\end{table*}\n"
    
    # Save single table file
    filename = "figures/r_squared_table.txt"
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
        if pattern in pattern_r_squared:
            pattern_df = pattern_r_squared[pattern]
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
latex_content_transposed += "\\caption{Mean Column-wise R^2 $\\pm$ 95\\% Confidence Interval by Method (Transposed)}\n"
latex_content_transposed += "\\label{tab:r_squared_by_method_transposed}\n"

# Generate each tabular environment
for tabular_idx in range(num_tabulars_transposed):
    start_col = tabular_idx * max_cols
    end_col = min(start_col + max_cols, len(all_patterns))
    table_patterns = all_patterns[start_col:end_col]
    
    # Create column specification
    num_patterns_table = len(table_patterns)
    col_spec = "l" + "c" * num_patterns_table  # l for method names, c for each pattern
    
    latex_content_transposed += "\\resizebox{\\linewidth}{!}{\n"
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
    latex_content_transposed += "}\n"

    # Add spacing between tabulars if not the last one
    if tabular_idx < num_tabulars_transposed - 1:
        latex_content_transposed += "\\quad\n"

# End table environment
latex_content_transposed += "\\end{table}\n"

# Save single transposed table file
transposed_filename = "figures/r_squared_table_transposed.txt"
with open(transposed_filename, "w") as f:
    f.write(latex_content_transposed)

print(f"Transposed LaTeX table saved to {transposed_filename}")

# Generate LaTeX table for column-wise R^2 values for MCAR with datasets as rows
print("Creating LaTeX table for column-wise R^2 values (MCAR, datasets as rows)...")

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
                    # Get the column-wise R^2 value
                    # Use .iloc[0] to get the scalar value from the Series
                    try:
                        r_squared = mcar_data.loc[dataset, method].iloc[0]
                        if pd.notna(r_squared):
                            dataset_data[method] = r_squared
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
        latex_content_mcar += "\\caption{Column-wise R^2 Values for MCAR Pattern by Dataset}\n"
        latex_content_mcar += "\\label{tab:r_squared_mcar_by_dataset}\n"
        
        # Create column specification
        num_methods = len(available_methods)
        col_spec = "l" + "c" * num_methods  # l for dataset names, c for each method
        
        latex_content_mcar += "\\resizebox{\\linewidth}{!}{\n"
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
            
            # Calculate row maximum for this dataset (across methods)
            row_max = float('-inf')
            for method in available_methods:
                r_squared_val = row_data[method]
                if pd.notna(r_squared_val) and r_squared_val > row_max:
                    row_max = r_squared_val
            
            for method in available_methods:
                r_squared_val = row_data[method]
                if pd.notna(r_squared_val):
                    # Bold if this is the maximum value in this row
                    if abs(r_squared_val - row_max) < 1e-6:  # Use small epsilon for float comparison
                        row += f" & \\textbf{{{r_squared_val:.3f}}}"
                    else:
                        row += f" & {r_squared_val:.3f}"
                else:
                    row += " & --"
            
            row += " \\\\\n"
            latex_content_mcar += row
        
        latex_content_mcar += "\\bottomrule\n"
        latex_content_mcar += "\\end{tabular}\n"
        latex_content_mcar += "}\n"
        latex_content_mcar += "\\end{table}\n"
        
        # Save MCAR table to file
        mcar_filename = "figures/r_squared_mcar_by_dataset.txt"
        with open(mcar_filename, "w") as f:
            f.write(latex_content_mcar)
        
        print(f"MCAR R^2 table saved to {mcar_filename}")
    else:
        print("No methods from table_methods found in MCAR data")
else:
    print("No MCAR data found")