import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os
import json
import pandas as pd
import warnings
from scipy import stats
import itertools
from pathlib import Path
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
    NEUTRAL_COLOR,
)

# --- Plotting ---
# setup_latex_fonts()
matplotlib.rcParams['font.family'] = 'serif'

base_path = "../datasets/uci"
figures_path = "figures/"

DWIP_V3R1_MANIFEST_PATH = Path(
    "/root/tabular/mcpfn/src/tabimpute/workdir/"
    "dwip_v3r1_9run_downstream_eval_manifest_20260415.json"
)


def load_manifest_plot_methods(manifest_path: Path) -> tuple[list[str], dict[str, str]]:
    if not manifest_path.is_file():
        return [], {}

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    methods: list[str] = []
    labels: dict[str, str] = {}
    for run in manifest.get("runs", []):
        trial_run_name = str(run.get("trial_run_name", ""))
        sweep_name = str(run.get("sweep_name", ""))
        if not trial_run_name:
            continue
        methods.append(trial_run_name)
        labels[trial_run_name] = f"TabImpute ({sweep_name})"

    return methods, labels


MANIFEST_METHODS, MANIFEST_METHOD_NAMES = load_manifest_plot_methods(
    DWIP_V3R1_MANIFEST_PATH
)

DWIP_V2_METHODS = [f"dwip_v2-trial_{trial_id:03d}" for trial_id in range(1, 13)]
DWIP_V1_METHODS = [f"dwip_v1-trial_{trial_id:03d}" for trial_id in range(3, 9)]
ADDITIONAL_TABIMPUTE_METHODS = [
    # "tabimpute-new-stable_v2-rank_1_30-50x50",
    # "tabimpute-new-stable_v2-rank_10_10-50x50",
    # "tabimpute-new-stable_v2-spectrum-decay-topk_p_0.2-50x50",
    "dwip_v2-trial_002"
]

methods = [
    "softimpute",
    # "column_mean",
    "hyperimpute",
    # "ot_sinkhorn",
    "missforest",
    "ice",
    "mice",
    "gain",
    "miwae",
    "masters_mcar",
    # "xrfm",
    # "tabimpute_mcar_p0.4_num_cls_8_rank_1_11",
    # "tabimpsute_mcar_p0.4_num_cls_12_rank_1_11",
    # "tabimpute_rope_v1",
    # "tabimpute_rope_trial_6_chunking",
    # "tabimpute_rope_trial_14_no_permute",
    # "tabimpute_new_stable_v1_100k",
    # "tabimpute_new_stable_v1_100k_w_permute",
    # "tabimpute_rope_trial_12_no_permute",
    # "tabimpute_new_stable_v4_27500_no_permute",
    # "tabimpute_new_stable_v3_15000",
    # "tabimpute_new_stable_v6_25000",
    # "tabimpute_new_stable_v3_15000_300_rows",
    # "tabimpute_new_stable_v3_15000_250_rows",
    # "tabimpute_new_stable_v3_15000_100_rows",
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
    # "notmiwae"
] + ADDITIONAL_TABIMPUTE_METHODS # + MANIFEST_METHODS

# Use patterns from plot_options, but filter to only active ones
patterns = {
    "MCAR",
    "MAR",
    "MNAR",
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

num_repeats = 3

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
method_names["tabimpute_new_stable_round2_trial_2"] = "TabImpute (New)"
method_names["tabimpute_new_stable_round2_trial_6"] = "TabImpute (New Stable Round 2 Trial 6)"
method_names["tabimpute_new_stable_round2_trial_10"] = "TabImpute (New Stable Round 2 Trial 10)"
method_names["tabimpute_new_stable_shallow_trial_3"] = "TabImpute (New Stable Shallow Trial 3)"
method_names["tabimpute_new_stable_shallow_trial_6"] = "TabImpute (New Stable Shallow Trial 6)"
method_names["tabimpute_new_stable_shallow_trial_2"] = "TabImpute (New Stable Shallow Trial 2)"
method_names["tabimpute_new_stable_deep_trial_3"] = "TabImpute (New Stable Deep Trial 3)"
method_names["tabimpute-new-stable_v2-rank_10_10-50x50"] = "TabImpute (10-10 50x50)"
method_names["tabimpute-new-stable_v2-spectrum-decay-topk_p_0.2-50x50"] = "TabImpute (Spectrum Top-k p=0.2)"

method_names.update(
    {
        "dwip_v2-trial_001": "TabImpute (DWIP v2 Trial 001)",
        "dwip_v2-trial_002": "MatFormer",
        "dwip_v2-trial_003": "TabImpute (DWIP v2 Trial 003)",
        "dwip_v2-trial_004": "TabImpute (DWIP v2 Trial 004)",
        "dwip_v2-trial_005": "TabImpute (DWIP v2 Trial 005)",
        "dwip_v2-trial_006": "TabImpute (DWIP v2 Trial 006)",
        "dwip_v2-trial_007": "TabImpute (DWIP v2 Trial 007)",
        "dwip_v2-trial_008": "TabImpute (DWIP v2 Trial 008)",
        "dwip_v2-trial_009": "TabImpute (DWIP v2 Trial 009)",
        "dwip_v2-trial_010": "TabImpute (DWIP v2 Trial 010)",
        "dwip_v2-trial_011": "TabImpute (DWIP v2 Trial 011)",
        "dwip_v2-trial_012": "TabImpute (DWIP v2 Trial 012)",
        "dwip_v1-trial_003": "TabImpute (DWIP v1 Trial 003)",
        "dwip_v1-trial_004": "TabImpute (DWIP v1 Trial 004)",
        "dwip_v1-trial_005": "TabImpute (DWIP v1 Trial 005)",
        "dwip_v1-trial_006": "TabImpute (DWIP v1 Trial 006)",
        "dwip_v1-trial_007": "TabImpute (DWIP v1 Trial 007)",
        "dwip_v1-trial_008": "TabImpute (DWIP v1 Trial 008)",
        "tabimpute-new-stable_v2-rank_1_30-50x50": "TabImpute (1-30)",
        "tabimpute-new-stable_v2-rank_10_10-50x50": "TabImpute (10-10)",
        "tabimpute-new-stable_v2-spectrum-decay-topk_p_0.2-50x50": "TabImpute (Spectrum Top-k p=0.2)",
    }
)
method_names.update(MANIFEST_METHOD_NAMES)
# Use method colors from plot_options
method_colors = METHOD_COLORS.copy()

for method in method_names.values():
    if "TabImpute" in method or "MatFormer" in method:
        method_colors[method] = HIGHLIGHT_COLOR
    else:
        method_colors[method] = NEUTRAL_COLOR

negative_rmse = {}

# This is what should be 
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
        if sigma < 1e-4:
            # If the error is also 0, score is 0. Otherwise, it's effectively infinite error.
            if rmse < 1e-4:
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
# df_norm = 1.0 - (df - df.min(axis=1).values[:, None]) / (df.max(axis=1) - df.min(axis=1)).values[:, None]
df_norm = df

# print(df_norm)

plot_pattern = True
dont_plot_methods = []

# Plot for all patterns combined
# Get dataframe for all patterns
df_all = df.copy()
# df_norm_all = 1.0 - (df_all - df_all.min(axis=1).values[:, None]) / (df_all.max(axis=1) - df_all.min(axis=1)).values[:, None]
df_norm_all = df_all

if plot_pattern:
    for pattern_name in patterns:
        # Get dataframe for 1 pattern
        df2 = df[df.index.get_level_values(1) == pattern_name]
        # df_norm = 1.0 - (df2 - df2.min(axis=1).values[:, None]) / (
        #     df2.max(axis=1) - df2.min(axis=1)
        # ).values[:, None]
        df_norm = df2
        
        # df_norm = 1.0 - df_norm
        
        # Average across datasets
        # --- Barplot ---
        plt.figure(figsize=FIGURE_SIZES['standard'])

        # sort methods by mean performance (best first)
        sorted_methods = df_norm.mean(axis=0).sort_values(ascending=False).index
        sorted_methods = [method for method in sorted_methods if method not in dont_plot_methods]

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

        # Set x-axis labels with 45-degree rotation (like plot_negative_rmse)
        labels_with_bold = [method for method in sorted_methods]
        ax.set_xticks(range(len(sorted_methods)))
        ax.set_xticklabels(labels_with_bold, rotation=45, ha='right')
        ax.set_xlabel("")

        # Set label colors to match bar colors and make TabImpute slightly larger
        for i, label in enumerate(ax.get_xticklabels()):
            method_name = sorted_methods[i]
            # if method_name in method_colors:
            #     label.set_color(method_colors[method_name])
            if "TabImpute" in method_name or "MatFormer" in method_name:
                label.set_color(method_colors[method_name])
                label.set_fontsize(label.get_fontsize() * 1.1)

        plt.ylabel("Normalized RMSE", fontsize=15)
        # plt.title(f"Comparison of Imputation Algorithms | {pattern_name}")
        # plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(f"{figures_path}uci_negative_rmse_{pattern_name}.pdf", **PLOT_PARAMS)
        plt.close()

    # Average across datasets and patterns
    fig = plt.figure(figsize=FIGURE_SIZES['standard'])
    # sort df_norm_all by the mean of the rows (best first)
    sorted_methods_all = df_norm_all.mean(axis=0).sort_values(ascending=False).index
    sorted_methods_all = [method for method in sorted_methods_all if method not in dont_plot_methods]

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

    # Set x-axis labels with 45-degree rotation (like plot_negative_rmse)
    labels_with_bold = [method for method in sorted_methods_all]
    ax.set_xticks(range(len(sorted_methods_all)))
    ax.set_xticklabels(labels_with_bold, rotation=45, ha='right', fontsize=14)
    ax.set_xlabel("")

    # Set label colors to match bar colors and make TabImpute slightly larger
    for i, label in enumerate(ax.get_xticklabels()):
        method_name = sorted_methods_all[i]
        # if method_name in method_colors:
        #     label.set_color(method_colors[method_name])
        if "TabImpute" in method_name or "MatFormer" in method_name:
            label.set_color(method_colors[method_name])
            label.set_fontsize(label.get_fontsize() * 1.1)

    plt.ylabel("Normalized RMSE", fontsize=12)
    # plt.title("Comparison of Imputation Algorithms | All Patterns")
    # plt.ylim(0, 1.0)
    plt.tight_layout()
    
    plt.savefig(f"{figures_path}uci_negative_rmse_overall.pdf", **PLOT_PARAMS)
    plt.close()

    # --- Row of four barplots: Overall, MCAR, MAR, MNAR ---
    # Mirrors the panel plot in openml/plot_negative_rmse.py: figure-level
    # legend above the titles, RMSE on the left axis, sorted lowest-first.
    def _prepare_pattern_panel(pattern_name: str) -> pd.DataFrame:
        # df has an unnamed MultiIndex (dataset, pattern, missingness_level, repeat);
        # level 1 is the pattern. Match the index access used by the rest of the file.
        return df[df.index.get_level_values(1) == pattern_name]

    _panel_dfs = [
        ("Overall", df_norm_all),
        ("MCAR", _prepare_pattern_panel("MCAR")),
        ("MAR", _prepare_pattern_panel("MAR")),
        ("MNAR", _prepare_pattern_panel("MNAR")),
    ]

    _w_std, _h_std = FIGURE_SIZES["standard"]
    fig_row, axes_row = plt.subplots(1, 4, figsize=(4 * _w_std, _h_std))

    for ax_i, (_title, _df_panel) in enumerate(_panel_dfs):
        ax_panel = axes_row[ax_i]
        _sorted = _df_panel.mean(axis=0).sort_values(ascending=True).index.tolist()
        _sorted = [m for m in _sorted if m not in dont_plot_methods]
        _long = _df_panel.melt(var_name="method", value_name="score")
        sns.barplot(
            data=_long,
            x="method",
            y="score",
            hue="method",
            order=_sorted,
            palette=method_colors,
            ax=ax_panel,
            legend=False,
            **BARPLOT_STYLE,
        )
        if _title in ("MCAR", "MAR", "MNAR"):
            ax_panel.set_title(f"$P$ = {_title}", fontsize=26)
        else:
            ax_panel.set_title(_title, fontsize=26)
        ax_panel.set_xticks(range(len(_sorted)))
        ax_panel.set_xticklabels(_sorted, rotation=45, ha="right", fontsize=13)
        ax_panel.set_xlabel("")
        ax_panel.spines["top"].set_visible(False)
        ax_panel.spines["right"].set_visible(False)
        ax_panel.grid(False)
        ax_panel.yaxis.grid(True, color="#E5E5E5", linewidth=0.6, alpha=0.7)
        ax_panel.set_axisbelow(True)
        for j, label in enumerate(ax_panel.get_xticklabels()):
            mn = _sorted[j]
            if "TabImpute" in mn or "MatFormer" in mn:
                label.set_color(method_colors[mn])
                label.set_fontsize(label.get_fontsize() * 1.1)
                label.set_weight("bold")

        if ax_i == 0:
            ax_panel.set_ylabel(r"RMSE", fontsize=26)
        else:
            ax_panel.set_ylabel("")

    _legend_handles = [
        Patch(facecolor=HIGHLIGHT_COLOR, label="TabImpute"),
        Patch(facecolor=NEUTRAL_COLOR, label="Other Baselines"),
    ]
    fig_row.tight_layout(pad=0.4)
    # Reserve a strip at the top for the figure-level legend, then place it
    # above all subplot titles as a single horizontal row.
    fig_row.subplots_adjust(wspace=0.13)
    fig_row.legend(
        handles=_legend_handles,
        loc="upper center",
        # Anchor slightly above the figure top so there's a small gap between
        # the legend and the subplot titles. Save with bbox_inches="tight" so
        # the lifted legend isn't clipped.
        bbox_to_anchor=(0.5, 1.1),
        ncol=len(_legend_handles),
        fontsize=18,
        frameon=False,
    )
    plt.savefig(
        f"{figures_path}uci_negative_rmse_row_overall_mcar_mar_mnar.pdf",
        dpi=300,
        bbox_inches="tight",
    )
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
        filename = f"{figures_path}uci_normalized_negative_rmse_table_{table_idx + 1}.txt" if num_tables > 1 else f"{figures_path}uci_normalized_negative_rmse_table.txt"
        with open(filename, "w") as f:
            f.write(latex_content)
        
        print(f"LaTeX table {table_idx + 1} saved to {filename}")