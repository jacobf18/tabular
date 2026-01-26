import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings
from scipy import stats

# --- Plotting ---
sns.set(style="whitegrid")

base_path = "datasets/openml"

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
    "masters_mcar_nonlinear",
    "tabpfn",
    "tabpfn_impute",
    "knn",
    "forestdiffusion",
    # "diffputer",
    "remasker",
    "cacti",
]

patterns = {
    "MCAR",
    # "MAR",
    "MNAR",
    "MAR_Neural",
    # "MAR_BlockNeural",
    "MAR_Sequential",
    # "MNARPanelPattern",
    "MNARPolarizationPattern",
    "MNARSoftPolarizationPattern",
    "MNARLatentFactorPattern",
    "MNARClusterLevelPattern",
    "MNARTwoPhaseSubsetPattern",
    # "MNARCensoringPattern",
}

method_names = {
    "mixed_nonlinear": "TabImpute (Nonlinear FM)",
    "mcpfn_ensemble": "TabImpute+",
    "mcpfn_mixed_fixed": "TabImpute (Fixed)",
    "masters_mcar": "TabImpute",
    "masters_mar": "TabImpute (MAR)",
    "masters_mnar": "TabImpute (Self-Masking-MNAR)",
    "masters_mcar_nonlinear": "TabImpute (Nonlinear)",
    "tabimpute_ensemble": "TabImpute Ensemble",
    "tabimpute_ensemble_router": "TabImpute Router",
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
    "diffputer": "DiffPuter",
    "remasker": "ReMasker",
    "cacti": "CACTI",
}

# Define consistent color mapping for methods (using display names as they appear in the DataFrame)
method_colors = {
    "TabImpute+": "#2f88a8",  # Blue
    "TabImpute": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute Ensemble": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (Self-Masking-MNAR)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (Fixed)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (MCAR)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (MAR)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (MNAR)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (Nonlinear)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute Router": "#2f88a8",  # Sea Green (distinct from GPU)
    "EWF-TabPFN": "#3e3b6e",  #
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
    "ForestDiffusion": "#52b980",      # Medium Green
    "MCPFN": "#ff9896",        # Light Red
    "MCPFN (Linear Permuted)": "#c5b0d5",  # Light Purple
    "MCPFN (Nonlinear Permuted)": "#c49c94",  # Light Brown
    "DiffPuter": "#d62728",  # Red
    "ReMasker": "#f58231",  # Dark Orange
    "CACTI": "#98df8a",  # Light Green
}

wasserstein = {}

def compute_wasserstein_distance(X_true, X_imputed, mask):
    # loop over columns and calculate the 1-Wasserstein distance
    wasserstein_distances = []
    for col in range(X_true.shape[1]):
        if mask[:, col].sum() < 2:
            continue
        if np.isnan(X_true[:, col][mask[:, col]]).any() or np.isnan(X_imputed[:, col][mask[:, col]]).any():
            continue
        predicted = X_imputed[:, col][mask[:, col]]
        true = X_true[:, col][mask[:, col]]
        if np.isnan(predicted).any() or np.isnan(true).any():
            continue
        w1 = stats.wasserstein_distance(true, predicted)
        if np.isnan(w1):
            w1 = 0.0
        wasserstein_distances.append(w1)
    if len(wasserstein_distances) == 0:
        return 0.0
    return np.mean(wasserstein_distances)

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
            wasserstein[(dataset, pattern_name, name)] = compute_wasserstein_distance(X_true, X_imputed, mask)

df = pd.Series(wasserstein).unstack()

plot_pattern = False

# Plot for all patterns combined
# Get dataframe for all patterns
df = 1.-df
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

        plt.ylabel("Column-wise 1-Wasserstein Distance", fontsize=15)
        # plt.title(f"Comparison of Imputation Algorithms | {pattern_name}")
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(f"figures/wasserstein_{pattern_name}.png", dpi=300)
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

    plt.ylabel("Column-wise 1-Wasserstein Distance", fontsize=18)
    # plt.title("Comparison of Imputation Algorithms | All Patterns")
    plt.ylim(0, 1.0)
    # plt.tight_layout()

    fig.subplots_adjust(left=0.2, right=0.95, bottom=0.05, top=0.95)

    plt.savefig(f"figures/wasserstein_overall.pdf", dpi=300, bbox_inches=None)
    plt.close()

# Generate LaTeX table for column-wise 1-Wasserstein distance
print("Creating LaTeX table for column-wise 1-Wasserstein distance...")

# Define methods to include in the table (subset of all methods)
table_methods = [
    "TabImpute",
    "EWF-TabPFN",
    "TabPFN",
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
    "ReMasker",
    "CACTI",
]

# Calculate column-wise 1-Wasserstein distance for each pattern
pattern_wasserstein = {}

for pattern_name in patterns:
    # Get dataframe for this pattern
    df_pattern = df[df.index.get_level_values(1) == pattern_name]

    if df_pattern.empty:
        print(f"No data found for pattern: {pattern_name}")
        continue

    # Normalize column-wise 1-Wasserstein distance for this pattern (per dataset-pattern combination)
    df_norm = (df_pattern - df_pattern.min(axis=1).values[:, None]) / (
        df_pattern.max(axis=1) - df_pattern.min(axis=1)
    ).values[:, None]

    # Calculate mean and standard error for each method across datasets
    pattern_means = df_norm.mean(axis=0)
    pattern_sems = df_norm.sem(axis=0)

    # Create summary data for this pattern
    pattern_data = []
    for method in pattern_means.index:
        pattern_data.append({
            'Method': method,
            'mean': pattern_means[method],
            'sem': pattern_sems[method],
            'Pattern': pattern_name
        })

    pattern_wasserstein[pattern_name] = pd.DataFrame(pattern_data)

# Calculate overall column-wise 1-Wasserstein distance (using normalized values)
overall_means = df_norm_all.mean(axis=0)
overall_sems = df_norm_all.sem(axis=0)

# Create summary data for overall
overall_data = []
for method in overall_means.index:
    overall_data.append({
        'Method': method,
        'mean': overall_means[method],
        'sem': overall_sems[method],
        'Pattern': 'Overall'
    })

pattern_wasserstein['Overall'] = pd.DataFrame(overall_data)

# Create summary data for LaTeX table
summary_data = []
all_patterns = ['MCAR',
                'MAR_Neural',
                'MNAR',
                # 'MAR',
                # 'MAR_BlockNeural',
                'MAR_Sequential',
                # "MNARPanelPattern",
                "MNARPolarizationPattern",
                "MNARSoftPolarizationPattern",
                "MNARLatentFactorPattern",
                "MNARClusterLevelPattern",
                "MNARTwoPhaseSubsetPattern",
                # "MNARCensoringPattern",
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
    if pattern in pattern_wasserstein:
        pattern_df = pattern_wasserstein[pattern]
        # Filter to only include methods in table_methods
        filtered_pattern_df = pattern_df[pattern_df['Method'].isin(table_methods)]
        summary_data.append(filtered_pattern_df)

# print(summary_data)

if summary_data:
    summary_df = pd.concat(summary_data, ignore_index=True)
    summary_pivot = summary_df.pivot(index='Pattern', columns='Method', values=['mean', 'sem'])

    # Flatten column names
    summary_pivot.columns = [f"{col[1]}_{col[0]}" for col in summary_pivot.columns]

    # Get all methods and sort by overall performance (higher normalized score is better)
    all_methods = []
    for method in table_methods:
        if f"{method}_mean" in summary_pivot.columns:
            all_methods.append(method)

    # Sort methods by overall performance (higher normalized score is better)
    if 'Overall' in summary_pivot.index:
        overall_means = summary_pivot.loc['Overall', [f"{method}_mean" for method in all_methods]]
        all_methods = [method for _, method in sorted(zip(overall_means, all_methods), reverse=True)]

    # Generate LaTeX table with maximum 5 columns per tabular
    max_cols = 5
    num_tables = (len(all_methods) + max_cols - 1) // max_cols

    # Calculate maximum per pattern (row) across all methods (for bold formatting)
    pattern_max = {}
    for pattern in all_patterns:
        pattern_max_val = float('-inf')
        for method in all_methods:
            if f"{method}_mean" in summary_pivot.columns:
                mean_val = summary_pivot.loc[pattern, f"{method}_mean"]
                if pd.notna(mean_val) and mean_val > pattern_max_val:
                    pattern_max_val = mean_val
        pattern_max[pattern] = pattern_max_val

    # Start single table environment
    latex_content = "\\begin{table}[h]\n"
    latex_content += "\\centering\n"
    latex_content += "\\caption{Mean Column-wise 1-Wasserstein Distance ± Standard Error by Missingness Pattern}\n"
    latex_content += "\\label{tab:wasserstein_by_pattern}\n"

    # Generate all tabular environments
    tabular_contents = []
    for table_idx in range(num_tables):
        start_col = table_idx * max_cols
        end_col = min(start_col + max_cols, len(all_methods))
        table_methods = all_methods[start_col:end_col]

        # Create column specification
        num_methods_table = len(table_methods)
        col_spec = "l" + "c" * num_methods_table  # l for pattern names, c for each method

        tabular_content = f"\\begin{{tabular}}{{{col_spec}}}\n"
        tabular_content += "\\toprule\n"

        # Header row
        header = "Pattern"
        for method in table_methods:
            method_name = method.replace('_', '\\_')  # Escape underscores
            header += f" & {method_name}"
        header += " \\\\\n"
        tabular_content += header
        tabular_content += "\\midrule\n"

        # Data rows (patterns as rows)
        for i, pattern in enumerate(all_patterns):
            row = patern_latex_names[pattern]  # Escape underscores in pattern names

            # Add midrule before Overall performance
            if pattern == 'Overall' and i > 0:
                tabular_content += "\\midrule\n"

            for method in table_methods:
                if f"{method}_mean" in summary_pivot.columns:
                    mean_val = summary_pivot.loc[pattern, f"{method}_mean"]
                    sem_val = summary_pivot.loc[pattern, f"{method}_sem"]
                    if pd.notna(mean_val) and pd.notna(sem_val):
                        # Bold if this is the maximum value for this pattern (row)
                        if abs(mean_val - pattern_max[pattern]) < 1e-6:
                            row += f" & \\textbf{{{mean_val:.3f} ± {sem_val:.3f}}}"
                        else:
                            row += f" & {mean_val:.3f} ± {sem_val:.3f}"
                    else:
                        row += " & --"
            row += " \\\\\n"
            tabular_content += row

        tabular_content += "\\bottomrule\n"
        tabular_content += "\\end{tabular}\n"
        
        tabular_contents.append(tabular_content)
        
        # Add spacing between tabulars (except for the last one)
        if table_idx < num_tables - 1:
            tabular_contents.append("\\quad\n")

    # Combine all tabulars
    latex_content += "".join(tabular_contents)
    latex_content += "\\end{table}\n"

    # Save single table to file
    filename = "figures/wasserstein_table.txt"
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
        if pattern in pattern_wasserstein:
            pattern_df = pattern_wasserstein[pattern]
            method_row = pattern_df[pattern_df['Method'] == method]
            if not method_row.empty:
                method_data[f"{pattern}_mean"] = method_row.iloc[0]['mean']
                method_data[f"{pattern}_sem"] = method_row.iloc[0]['sem']
            else:
                method_data[f"{pattern}_mean"] = None
                method_data[f"{pattern}_sem"] = None
        else:
            method_data[f"{pattern}_mean"] = None
            method_data[f"{pattern}_sem"] = None
    transposed_summary_data.append(method_data)

transposed_df = pd.DataFrame(transposed_summary_data)

# Generate transposed LaTeX table with maximum 4 columns per tabular
max_cols = 4
num_tables_transposed = (len(all_patterns) + max_cols - 1) // max_cols

# Calculate maximum per method (row) across all patterns (for bold formatting)
method_max = {}
for method in all_methods:
    method_max_val = float('-inf')
    for pattern in all_patterns:
        mean_col = f"{pattern}_mean"
        if mean_col in transposed_df.columns:
            method_row = transposed_df[transposed_df['Method'] == method]
            if not method_row.empty:
                mean_val = method_row[mean_col].iloc[0]
                if pd.notna(mean_val) and mean_val > method_max_val:
                    method_max_val = mean_val
    method_max[method] = method_max_val

# Start single table environment
latex_content_transposed = "\\begin{table}[h]\n"
latex_content_transposed += "\\centering\n"
latex_content_transposed += "\\caption{Mean Column-wise 1-Wasserstein Distance ± Standard Error by Method (Transposed)}\n"
latex_content_transposed += "\\label{tab:wasserstein_by_method_transposed}\n"

# Generate all tabular environments
tabular_contents_transposed = []
for table_idx in range(num_tables_transposed):
    start_col = table_idx * max_cols
    end_col = min(start_col + max_cols, len(all_patterns))
    table_patterns = all_patterns[start_col:end_col]

    # Create column specification
    num_patterns_table = len(table_patterns)
    col_spec = "l" + "c" * num_patterns_table  # l for method names, c for each pattern

    tabular_content = f"\\begin{{tabular}}{{{col_spec}}}\n"
    tabular_content += "\\toprule\n"

    # Header row
    header = "Method"
    for pattern in table_patterns:
        pattern_name = pattern.replace('_', '\\_')  # Escape underscores
        header += f" & {pattern_name}"
    header += " \\\\\n"
    tabular_content += header
    tabular_content += "\\midrule\n"

    # Data rows (methods as rows)
    for method in all_methods:
        row = method.replace('_', '\\_')  # Escape underscores in method names

        for pattern in table_patterns:
            mean_col = f"{pattern}_mean"
            sem_col = f"{pattern}_sem"

            if mean_col in transposed_df.columns and sem_col in transposed_df.columns:
                method_row = transposed_df[transposed_df['Method'] == method]
                if not method_row.empty:
                    mean_val = method_row[mean_col].iloc[0]
                    sem_val = method_row[sem_col].iloc[0]
                    if pd.notna(mean_val) and pd.notna(sem_val):
                        # Bold if this is the maximum value for this method (row)
                        if abs(mean_val - method_max[method]) < 1e-6:
                            row += f" & \\textbf{{{mean_val:.3f} ± {sem_val:.3f}}}"
                        else:
                            row += f" & {mean_val:.3f} ± {sem_val:.3f}"
                    else:
                        row += " & --"
                else:
                    row += " & --"
            else:
                row += " & --"

        row += " \\\\\n"
        tabular_content += row

    tabular_content += "\\bottomrule\n"
    tabular_content += "\\end{tabular}\n"
    
    tabular_contents_transposed.append(tabular_content)
    
    # Add spacing between tabulars (except for the last one)
    if table_idx < num_tables_transposed - 1:
        tabular_contents_transposed.append("\\quad\n")

# Combine all tabulars
latex_content_transposed += "".join(tabular_contents_transposed)
latex_content_transposed += "\\end{table}\n"

# Save single transposed table to file
transposed_filename = "figures/wasserstein_table_transposed.txt"
with open(transposed_filename, "w") as f:
    f.write(latex_content_transposed)

print(f"Transposed LaTeX table saved to {transposed_filename}")

# Generate LaTeX table for column-wise 1-Wasserstein distance values for MCAR with datasets as rows
print("Creating LaTeX table for column-wise 1-Wasserstein distance values (MCAR, datasets as rows)...")

# Get data for MCAR pattern only
mcar_data = df[df.index.get_level_values(1) == 'MCAR']

if not mcar_data.empty:
    # Normalize per dataset (row-wise normalization)
    mcar_data_norm = (mcar_data - mcar_data.min(axis=1).values[:, None]) / (
        mcar_data.max(axis=1) - mcar_data.min(axis=1)
    ).values[:, None]
    
    # Filter to only include methods in table_methods
    available_methods = [method for method in table_methods if method in mcar_data_norm.columns]

    if available_methods:
        # Create the table data
        mcar_table_data = []
        for dataset in mcar_data_norm.index.get_level_values(0).unique():
            dataset_data = {'Dataset': dataset}
            for method in available_methods:
                if method in mcar_data_norm.columns:
                    # Get the normalized column-wise 1-Wasserstein distance value
                    # Use .iloc[0] to get the scalar value from the Series
                    try:
                        wasserstein_val = mcar_data_norm.loc[dataset, method].iloc[0]
                        if pd.notna(wasserstein_val):
                            dataset_data[method] = wasserstein_val
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
        latex_content_mcar += "\\caption{Column-wise 1-Wasserstein Distance Values for MCAR Pattern by Dataset}\n"
        latex_content_mcar += "\\label{tab:wasserstein_mcar_by_dataset}\n"

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

        # Calculate global maximum across all values in this table (higher normalized score is better)
        table_global_max = float('-inf')
        for _, row_data in mcar_df.iterrows():
            for method in available_methods:
                wasserstein_val = row_data[method]
                if pd.notna(wasserstein_val) and wasserstein_val > table_global_max:
                    table_global_max = wasserstein_val

        # Data rows (datasets as rows)
        for _, row_data in mcar_df.iterrows():
            dataset_name = row_data['Dataset'].replace('_', '\\_')  # Escape underscores
            row = dataset_name

            for method in available_methods:
                wasserstein_val = row_data[method]
                if pd.notna(wasserstein_val):
                    # Bold if this is the global maximum value in this table
                    if abs(wasserstein_val - table_global_max) < 1e-6:
                        row += f" & \\textbf{{{wasserstein_val:.3f}}}"
                    else:
                        row += f" & {wasserstein_val:.3f}"
                else:
                    row += " & --"

            row += " \\\\\n"
            latex_content_mcar += row

        latex_content_mcar += "\\bottomrule\n"
        latex_content_mcar += "\\end{tabular}\n"
        latex_content_mcar += "\\end{table}\n"

        # Save MCAR table to file
        mcar_filename = "figures/wasserstein_mcar_by_dataset.txt"
        with open(mcar_filename, "w") as f:
            f.write(latex_content_mcar)

        print(f"MCAR 1-Wasserstein table saved to {mcar_filename}")
    else:
        print("No methods from table_methods found in MCAR data")
else:
    print("No MCAR data found")
