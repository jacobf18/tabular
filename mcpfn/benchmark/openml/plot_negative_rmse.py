import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os
import pandas as pd
import sys
import warnings
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    NEUTRAL_COLOR_DARK,
    compute_normalized_rmse_columnwise,
)

# --- Plotting ---
# Configure LaTeX rendering for all text in plots
# setup_latex_fonts()

# Ensure LaTeX is enabled (redundant but explicit)
# matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'

base_path = "../datasets/openml"
figure_base_path = "./figures"

datasets = os.listdir(base_path)

methods = [
    "softimpute", 
    # "tabimpute_mcar_lin",
    # "tabimpute_dynamic_cls",
    # "tabimpute_large_cls_8",
    # "tabimpute_75_75_rank_1_11",
    # "tabimpute_50_50_rank_1_11_cls_12",
    # "tabimpute_rope_v1",
    # "tabimpute_rope_trial_1",
    # "tabimpute_rope_trial_7",
    # # "tabimpute_rope_trial_6",
    # "tabimpute_rope_trial_14",
    # # "tabimpute_rope_trial_14_no_permute",
    # # "tabimpute_rope_trial_10",
    # # "tabimpute_rope_trial_12",
    # "tabimpute_rope_trial_12_w_permute",
    # "tabimpute_new_stable_v1_100k",
    # "tabimpute_new_stable_v1_15k",
    # "tabimpute_new_stable_v4_7500",
    # "tabimpute_new_stable_v4_27500",
    # "tabimpute_new_stable_v3_15000",
    # "tabimpute_new_stable_v6_25000",
    # "tabimpute_new_stable_round2_trial_3",
    # "tabimpute_new_stable_round2_trial_2",
    # "tabimpute_new_stable_round2_trial_6",
    # "tabimpute_new_stable_round2_trial_10",
    # "tabimpute_new_stable_shallow_trial_3",
    # "tabimpute_new_stable_shallow_trial_6",
    # "tabimpute_new_stable_shallow_trial_2",
    # "tabimpute_new_stable_deep_trial_3",
    # "tabimpute_rope_trial_2",
    # "tabimpute_large_mcar_rank_1_11",
    # "column_mean", 
    "hyperimpute",
    "ot_sinkhorn",
    "missforest",
    "ice",
    "mice",
    "gain",
    "miwae",
    "masters_mcar",
    # "masters_mcar_nonlinear",
    "masters_mnar_polarization",
    "masters_mnar",
    "masters_mar",
    "masters_mar_neural_lr_3e-4_decay_1e-2",
    "masters_mar_block_neural_lr_4e-4_decay_1e-2",
    "masters_mnar_soft_polarization",
    "masters_mnar_latent_factor_lr_3e-4_decay_2e-2",
    # "masters_mnar_cluster_level",
    # "masters_mnar_cluster_level_lr_3e-4_decay_2e-2",
    # "masters_mnar_cluster_level_lr_3e-4_decay_1e-2",
    "masters_mnar_cluster_level_lr_2e-4_decay_1e-2",
    # "masters_mnar_two_phase_subset",
    "masters_mnar_two_phase_subset_lr_3e-4_decay_1e-2",
    "masters_mnar_censoring",
    # # "tabpfn",
    "tabpfn_impute",
    "knn",
    "forestdiffusion",
    "remasker",
    "cacti",
    # # "diffputer",
    "notmiwae",
    # "not-miwae-self-mnar",
]



# Use patterns from plot_options
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
    "tabimpute_50_50_rank_1_11_cls_12": "TabImpute (New)",
    "tabimpute_rope_v1": "TabImpute (Rope v1)",
    "tabimpute_rope_trial_1": "TabImpute (Rope Trial 1)",
    "tabimpute_rope_trial_7": "TabImpute (Rope Trial 7)",
    "tabimpute_rope_trial_6": "TabImpute (Rope Trial 6)",
    "tabimpute_rope_trial_14": "TabImpute (Rope Trial 14)",
    "tabimpute_rope_trial_2": "TabImpute (Rope Trial 2)",
    "tabimpute_rope_trial_14_no_permute": "TabImpute (Rope Trial 14 No Permute)",
    "tabimpute_rope_trial_10": "TabImpute (Rope Trial 10 no permute)",
    "tabimpute_rope_trial_12": "TabImpute (Rope Trial 12 no permute)",
    "tabimpute_rope_trial_12_w_permute": "TabImpute (Rope Trial 12 w/ Permute)",
    "tabimpute_new_stable_v1_100k": "TabImpute (New Stable v1 100k)",
    "tabimpute_new_stable_v1_15k": "TabImpute (New Stable v1 15k)",
    "tabimpute_new_stable_v4_7500": "TabImpute (New Stable v4 7500)",
    "tabimpute_new_stable_v4_27500": "TabImpute (New Stable v4 27500)",
    "tabimpute_new_stable_v3_15000": "TabImpute (New Stable v3 15000)",
    "tabimpute_new_stable_v6_25000": "TabImpute (New Stable v6 25000)",
    "tabimpute_new_stable_round2_trial_3": "TabImpute (New Stable Round 2 Trial 3)",
    "tabimpute_new_stable_round2_trial_2": "TabImpute (New)",
    "tabimpute_new_stable_round2_trial_6": "TabImpute (New Stable Round 2 Trial 6)",
    "tabimpute_new_stable_round2_trial_10": "TabImpute (New Stable Round 2 Trial 10)",
    "tabimpute_new_stable_shallow_trial_3": "TabImpute (New Stable Shallow Trial 3)",
    "tabimpute_new_stable_shallow_trial_6": "TabImpute (New Stable Shallow Trial 6)",
    "tabimpute_new_stable_shallow_trial_2": "TabImpute (New Stable Shallow Trial 2)",
    "tabimpute_new_stable_deep_trial_3": "TabImpute (New Stable Deep Trial 3)",
    # "diffputer": "DiffPuter",
    "notmiwae": "not-MIWAE",
    "not-miwae-self-mnar": "not-MIWAE (Self-MNAR)",
    "masters_mcar": "TabImpute",
    "masters_mnar_polarization": "Polar",
    "masters_mnar": "Self-MNAR",
    "masters_mar_block_neural": "Block-MNAR",
    "masters_mnar_soft_polarization": "Soft-Pol",
    "masters_mnar_latent_factor": "Latent-MNAR",
    "masters_mnar_cluster_level": "Clustering",
    "masters_mnar_cluster_level_lr_3e-4_decay_2e-2": "Clustering",
    "masters_mnar_cluster_level_lr_3e-4_decay_1e-2": "Clustering",
    "masters_mnar_two_phase_subset": "Two-Phase",
    "masters_mnar_two_phase_subset_soft_polarization": "Soft-Pol",
    "masters_mnar_censoring": "Censoring",
    "masters_mar": "Col-MAR",
    "masters_mnar_two_phase_subset_lr_1e-4_decay_1e-2": "Two-Phase",
    "masters_mnar_two_phase_subset_lr_2e-4_decay_1e-2": "Two-Phase",
    "masters_mnar_two_phase_subset_lr_2e-4_decay_2e-2": "Two-Phase",
    "masters_mnar_two_phase_subset_lr_3e-4_decay_2e-2": "Two-Phase",
    "masters_mnar_latent_factor_lr_3e-4_decay_2e-2": "Latent-MNAR",
    "masters_mnar_two_phase_subset_lr_4e-4_decay_1e-2": "Two-Phase",
    "masters_mnar_two_phase_subset_lr_3e-4_decay_1e-2": "Two-Phase",
    "masters_mar_neural_lr_3e-4_decay_1e-2": "NN-MNAR",
    "masters_mnar_cluster_level_lr_2e-4_decay_1e-2": "Clustering",
    "masters_mar_block_neural_lr_4e-4_decay_1e-2": "Block-MNAR",
})

pattern_to_masters = {
    "MCAR": "masters/mcar",
    # "MCAR": "masters/mnar_cluster_level_lr_2e-4_decay_1e-2",
    "MAR": "masters/mar",
    "MNAR": "masters/mnar",
    # "MAR_Neural": "masters/mar_neural",
    "MAR_Neural": "masters/mar_neural_lr_3e-4_decay_1e-2",
    # "MAR_BlockNeural": "masters/mar_block_neural",
    "MAR_BlockNeural": "masters/mar_block_neural_lr_4e-4_decay_1e-2",
    "MAR_Sequential": "masters/mar_sequential",
    "MNARPanelPattern": "masters/mnar_panel",
    "MNARPolarizationPattern": "masters/mnar_polarization",
    "MNARSoftPolarizationPattern": "masters/mnar_soft_polarization",
    # "MNARLatentFactorPattern": "masters/mnar_latent_factor",
    "MNARLatentFactorPattern": "masters/mnar_latent_factor_lr_3e-4_decay_2e-2",
    # "MNARClusterLevelPattern": "masters/mnar_cluster_level",
    "MNARClusterLevelPattern": "masters/mnar_cluster_level_lr_2e-4_decay_1e-2",
    # "MNARTwoPhaseSubsetPattern": "masters/mnar_two_phase_subset_lr_4e-4_decay_1e-2",
    "MNARTwoPhaseSubsetPattern": "masters/mnar_two_phase_subset_lr_3e-4_decay_1e-2",
    # "MNARTwoPhaseSubsetPattern": "masters/mnar_two_phase_subset",
    "MNARCensoringPattern": "masters/mnar_censoring",
}


def _pattern_to_masters_method_key(pattern_name: str) -> str | None:
    """e.g. ``masters/mnar_censoring`` -> ``masters_mnar_censoring`` for lookup in ``methods`` / ``method_names``."""
    if pattern_name not in pattern_to_masters:
        return None
    path = pattern_to_masters[pattern_name]
    return path.replace("masters/", "masters_").replace("/", "_")


# Display names for all MASTERS runs listed in ``methods`` (per-pattern plots filter to one + MCAR)
_masters_display_labels = {
    method_names[m] for m in methods if m.startswith("masters_") and m in method_names
}

for method in method_names.values():
    if "TabImpute" in method:
        method_colors[method] = highlight_color
    else:
        if "not-MIWAE" in method:
            method_colors[method] = "#FF8F2B"
        else:
            method_colors[method] = neutral_color
            
special_color = "#7A2CA0"
        
method_colors.update({
    "MCAR": special_color,
    "Col-MAR": special_color,
    "Censoring": special_color,
    "Self-MNAR": special_color,
    "NN-MNAR": special_color,
    "Block-MNAR": special_color,
    "Polar": special_color,
    "Soft-Pol": special_color,
    "Latent-MNAR": special_color,
    "Clustering": special_color,
    "Two-Phase": special_color,
})

# Add missing method colors that may appear in the data

negative_rmse = {}

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
        
        _swapped_files = {
            "masters_mar_block_neural_lr_4e-4_decay_1e-2": "masters_mar_neural_lr_3e-4_decay_1e-2",
            "masters_mar_neural_lr_3e-4_decay_1e-2": "masters_mar_block_neural_lr_4e-4_decay_1e-2",
        }
        for method in methods:
            _file_method = _swapped_files.get(method, method)
            X_imputed = np.load(f"{base_path}/{dataset}/{pattern_name}_{p}/{_file_method}.npy")
            # print(dataset, config, method, X_imputed.shape, X_true.shape, mask.shape)
            name = method_names[method]
            negative_rmse[(dataset, pattern_name, name)] = compute_normalized_rmse_columnwise(X_true, X_imputed, mask)
            
df = pd.Series(negative_rmse).unstack()

plot_pattern = True

dont_plot_methods = [
    # "EWF-TabPFN",
]

pattern_y_lims = {
    "Overall": (0, 4.0),
    "MCAR": (0, 1.75),
    "MAR": (0, 2.5),
    "MNAR": (0, 7),
    "MAR_Neural": (0, 5),
    "MAR_BlockNeural": (0, 3),
    "MNARPolarizationPattern": (0, 10),
    "MNARSoftPolarizationPattern": (0, 7),
    "MNARLatentFactorPattern": (0, 3.5),
    "MNARClusterLevelPattern": (0, 1.75),
    "MNARTwoPhaseSubsetPattern": (0, 2.75),
    "MNARCensoringPattern": (0, 8),
}

# Plot for all patterns combined
# Get dataframe for all patterns
df_all = df.copy()
# df_norm_all = 1.0 - (df_all - df_all.min(axis=1).values[:, None]) / (df_all.max(axis=1) - df_all.min(axis=1)).values[:, None]
df_norm_all = df_all
if plot_pattern:
    # Average across datasets and patterns: only one TabImpute bar (masters_mcar), not other TabImpute-* runs
    _mcar_tabimpute_name = method_names["masters_mcar"]
    _cols_overall = [
        c
        for c in df_norm_all.columns
        if not ("MCAR" in c or "Col-MAR" in c or "Self-MNAR" in c or "Censoring" in c or "NN-MNAR" in c or "Block-MNAR" in c or "Polar" in c or "Soft-Pol" in c or "Latent-MNAR" in c or "Clustering" in c or "Two-Phase" in c)
    ]
    df_overall = df_norm_all[_cols_overall]

    # 3x4 grid: Overall + 11 patterns
    _grid_pattern_order = [
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
    ]

    _grid_panels = [("Overall", df_overall, None)]
    for pattern_name in _grid_pattern_order:
        df2 = df[df.index.get_level_values(1) == pattern_name]
        df_norm = df2.copy()
        _pkey = _pattern_to_masters_method_key(pattern_name)
        _pname = method_names.get(_pkey) if _pkey else None
        _mcar_name = method_names["masters_mcar"]
        _kept = [
            c
            for c in df_norm.columns
            if c not in _masters_display_labels
            or c == _mcar_name
            or (_pname is not None and c == _pname)
        ]
        # MCAR's _pname is the default TabImpute, not a specialized variant.
        _spec_label = _pname if pattern_name != "MCAR" else None
        _grid_panels.append((pattern_name, df_norm[_kept], _spec_label))

    _w_std, _h_std = FIGURE_SIZES['standard']
    fig_grid, axes_grid = plt.subplots(3, 4, figsize=(4 * _w_std, 3 * _h_std))
    axes_flat = axes_grid.flatten()

    for grid_i, (_title, _df_panel, _spec_label) in enumerate(_grid_panels):
        ax_g = axes_flat[grid_i]
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
            ax=ax_g,
            **BARPLOT_STYLE,
            legend=False,
        )
        _grid_title_names = {
            "Overall": "Overall",
            "MCAR": "MCAR",
            "MAR": "Col-MAR",
            "MNAR": "Self-MNAR",
            "MAR_Neural": "NN-MNAR",
            "MAR_BlockNeural": "Block-MNAR",
            "MNARPolarizationPattern": "Pol-MNAR",
            "MNARSoftPolarizationPattern": "Soft-Pol-MNAR",
            "MNARLatentFactorPattern": "Latent-MNAR",
            "MNARClusterLevelPattern": "Clust-MNAR",
            "MNARTwoPhaseSubsetPattern": "Phase-MNAR",
            "MNARCensoringPattern": "Censor-MNAR",
        }
        _display_title = _grid_title_names.get(_title, _title)
        if _spec_label is not None or _title == "MCAR":
            if _title == "MCAR":
                ax_g.set_title("$P$ = MCAR", fontsize=22, family="monospace")
            else:
                ax_g.set_title(f"$P$ = {_display_title}", fontsize=22, family="monospace")
        else:
            ax_g.set_title(_display_title, fontsize=22, family="monospace")
        _xtick_labels = [
            ("TabImpute-P" if (_spec_label is not None and m == _spec_label) else m)
            for m in _sorted
        ]
        ax_g.set_xticks(range(len(_sorted)))
        ax_g.set_xticklabels(_xtick_labels, rotation=45, ha="right", fontsize=12)
        ax_g.set_xlabel("")
        ax_g.spines["top"].set_visible(False)
        ax_g.spines["right"].set_visible(False)
        ax_g.grid(False)
        ax_g.yaxis.grid(True, color="#E5E5E5", linewidth=0.6, alpha=0.7)
        ax_g.set_axisbelow(True)
        for j, label in enumerate(ax_g.get_xticklabels()):
            mn = _sorted[j]
            if mn in method_colors:
                label.set_color(NEUTRAL_COLOR_DARK)
            if "TabImpute" in mn or mn in ("MCAR", "Col-MAR", "Self-MNAR", "Censoring", "NN-MNAR", "Block-MNAR", "Polar", "Soft-Pol", "Latent-MNAR", "Clustering", "Two-Phase"):
                label.set_color(method_colors[mn])
                label.set_fontsize(label.get_fontsize() * 1.1)
                label.set_weight("bold")

        ax_g.set_ylim(pattern_y_lims[_title])

        if grid_i % 4 == 0:
            ax_g.set_ylabel(r"RMSE", fontsize=20)
        else:
            ax_g.set_ylabel("")

    _legend_handles = [
        Patch(facecolor=HIGHLIGHT_COLOR, label="TabImpute (Default; P = MCAR)"),
        Patch(facecolor=special_color, label="TabImpute-P (Pattern-Specific)"),
        Patch(facecolor="#FF8F2B", label="not-MIWAE (Tracked Baseline)"),
        Patch(facecolor=NEUTRAL_COLOR, label="Other Baselines"),
    ]
    fig_grid.tight_layout(pad=0.5)
    # Reserve a strip at the top for the figure-level legend, then place it
    # above all subplot titles as a single horizontal row.
    fig_grid.subplots_adjust(top=0.93)
    fig_grid.legend(
        handles=_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(_legend_handles),
        fontsize=16,
        frameon=False,
    )
    plt.savefig(
        f"{figure_base_path}/negative_rmse_all_patterns_grid.pdf",
        dpi=300,
        bbox_inches=None,
    )
    plt.close()

    fig = plt.figure(figsize=FIGURE_SIZES['standard'])
    # sort by the mean of the rows
    sorted_methods_all = df_overall.mean(axis=0).sort_values(ascending=True).index

    # Melt into long format
    df_long = df_overall.melt(var_name="method", value_name="score")
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
    # TabImpute shown with same styling as others (LaTeX off)
    labels_with_bold = [method for method in sorted_methods_all]
    ax.set_xticks(range(len(sorted_methods_all)))
    ax.set_xticklabels(labels_with_bold, rotation=45, ha='right', fontsize=22)
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.yaxis.grid(True, color="#E5E5E5", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    
    # Set label colors to match bar colors and make TabImpute even bolder
    for i, label in enumerate(ax.get_xticklabels()):
        method_name = sorted_methods_all[i]
        if "TabImpute" in method_name:
            label.set_color(method_colors[method_name])
            # Make TabImpute slightly larger for extra boldness
            label.set_fontsize(label.get_fontsize() * 1.1)

    # Use LaTeX-formatted label
    # plt.ylabel(r"1 - Normalized RMSE", fontsize=18)
    plt.ylabel(r"RMSE", fontsize=22)
    # plt.title("Comparison of Imputation Algorithms | All Patterns")
    # plt.ylim(0.3, 0.95)
    plt.tight_layout()
    
    fig.subplots_adjust(left=0.2, right=0.95, bottom=0.05, top=0.95)
    
    plt.savefig(f"{figure_base_path}/negative_rmse_overall.pdf", dpi=300, bbox_inches=None)
    plt.close()

    # Four barplots in one row: Overall, MCAR, MAR, MNAR (Censoring)
    def _prepare_pattern_panel(pattern_name: str) -> pd.DataFrame:
        d2 = df[df.index.get_level_values(1) == pattern_name]
        dnorm = d2.copy()
        _pkey = _pattern_to_masters_method_key(pattern_name)
        _pname = method_names.get(_pkey) if _pkey else None
        _mcar_nm = method_names["masters_mcar"]
        _kept_panel = [
            c
            for c in dnorm.columns
            if c not in _masters_display_labels
            or c == _mcar_nm
            or (_pname is not None and c == _pname)
        ]
        return dnorm[_kept_panel]

    _panel_dfs = [
        ("Overall", df_overall, None),
        ("MCAR", _prepare_pattern_panel("MCAR"), None),
        # ("Col-MAR", _prepare_pattern_panel("MAR"), method_names["masters_mar"]),
        ("Censor-MNAR", _prepare_pattern_panel("MNARCensoringPattern"), method_names["masters_mnar_censoring"]),
        ("Self-MNAR", _prepare_pattern_panel("MNAR"), method_names["masters_mnar"]),
    ]

    _w_std, _h_std = FIGURE_SIZES["standard"]
    fig_row, axes_row = plt.subplots(1, 4, figsize=(4 * _w_std, _h_std))

    _overall_order = df_overall.mean(axis=0).sort_values(ascending=True).index.tolist()
    _specialized_labels = {"MCAR", "Censor-MNAR", "Self-MNAR"} | {
        method_names[m]
        for m in methods
        if m.startswith("masters_") and m in method_names
    }

    for ax_i, (_title, _df_panel, _spec_label) in enumerate(_panel_dfs):
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
            **BARPLOT_STYLE,
            legend=False,
        )
        if _spec_label is not None or _title == "MCAR":
            if _title == "MCAR":
                ax_panel.set_title("$P$ = MCAR", fontsize=26)
            else:
                # Display label can differ from _spec_label (e.g. Censoring →
                # Censor-MNAR). Keep _spec_label intact so the xtick comparison
                # below still matches the actual method name in _sorted.
                _title_label = "Censor-MNAR" if _spec_label == "Censoring" else _spec_label
                ax_panel.set_title(f"$P$ = {_title_label}", fontsize=26)
        else:
            ax_panel.set_title(_title, fontsize=26)
        _xtick_labels = [
            ("TabImpute-P" if (_spec_label is not None and m == _spec_label) else m)
            for m in _sorted
        ]
        ax_panel.set_xticks(range(len(_sorted)))
        ax_panel.set_xticklabels(_xtick_labels, rotation=45, ha="right", fontsize=13)
        ax_panel.set_xlabel("")
        ax_panel.spines["top"].set_visible(False)
        ax_panel.spines["right"].set_visible(False)
        ax_panel.grid(False)
        ax_panel.yaxis.grid(True, color="#E5E5E5", linewidth=0.6, alpha=0.7)
        ax_panel.set_axisbelow(True)
        for j, label in enumerate(ax_panel.get_xticklabels()):
            mn = _sorted[j]
            if mn in method_colors:
                # label.set_color(method_colors[mn])
                label.set_color(NEUTRAL_COLOR_DARK)
            if "TabImpute" in mn or "Censor-MNAR" == mn or "Self-MNAR" == mn or "MCAR" == mn or "Censoring" == mn:
                label.set_color(method_colors[mn])
                label.set_fontsize(label.get_fontsize() * 1.1)
                # bold the label
                label.set_weight("bold")

        if _title == "Self-MNAR":
            ax_panel.set_ylim(0, 7)
        elif _title == "Censor-MNAR":
            ax_panel.set_ylim(0, 8)

        if ax_i == 0:
            ax_panel.set_ylabel(r"RMSE", fontsize=26)
        else:
            ax_panel.set_ylabel("")

    _legend_handles = [
        Patch(facecolor=HIGHLIGHT_COLOR, label="TabImpute (Default; P = MCAR)"),
        Patch(facecolor=special_color, label="TabImpute-P (Pattern-Specific)"),
        Patch(facecolor="#FF8F2B", label="not-MIWAE (Tracked Baseline)"),
        Patch(facecolor=NEUTRAL_COLOR, label="Other Baselines"),
    ]
    # Pull panels together; negative wspace lets rotated x-labels bleed into neighbors slightly
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
        f"{figure_base_path}/negative_rmse_row_overall_mcar_mar_self_mnar.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# exit()
# Generate LaTeX table for normalized negative RMSE
print("Creating LaTeX table for normalized negative RMSE...")


def _ci95_halfwidth(series: pd.Series) -> float:
    n = int(series.count())
    if n < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, n - 1)) * float(series.sem())

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
    "SoftImpute",
    "ICE",
    "MICE-Forest",
    "GAIN",
    "MIWAE",
    "K-NN",
    "ForestDiffusion",
    "ReMasker",
    "CACTI",
    # "DiffPuter"
    "not-MIWAE",
    "not-MIWAE (Self-MNAR)",
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
    # df_norm = 1.0 - (df_pattern - df_pattern.min(axis=1).values[:, None]) / (df_pattern.max(axis=1) - df_pattern.min(axis=1)).values[:, None]
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
    
    pattern_normalized_rmse[pattern_name] = pd.DataFrame(pattern_data)

# Calculate overall normalized negative RMSE
# df_all_norm = 1.0 - (df_all - df_all.min(axis=1).values[:, None]) / (df_all.max(axis=1) - df_all.min(axis=1)).values[:, None]
df_all_norm = df_all
overall_means = df_all_norm.mean(axis=0)
# overall_stds = df_all_norm.std(axis=0)
overall_stds = df_all_norm.apply(_ci95_halfwidth, axis=0)

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
        all_methods = [method for _, method in sorted(zip(overall_means, all_methods), reverse=False)]
    
    # Generate LaTeX table with maximum 5 columns per tabular
    max_cols = 5
    num_tabulars = (len(all_methods) + max_cols - 1) // max_cols
    
    # Calculate global row maximums across ALL methods (for bolding across all tabulars)
    global_row_min = {}
    for pattern in all_patterns:
        row_min = float('inf')
        for method in all_methods:
            if f"{method}_mean" in summary_pivot.columns:
                mean_val = summary_pivot.loc[pattern, f"{method}_mean"]
                if pd.notna(mean_val) and mean_val < row_min:
                    row_min = mean_val
        global_row_min[pattern] = row_min
    
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
    latex_content += "\\caption{RMSE $\\pm$ 95\\% Confidence Interval by Missingness Pattern}\n"
    latex_content += "\\label{tab:normalized_negative_rmse_by_pattern}\n"

    # Generate each tabular environment
    for tabular_idx in range(num_tabulars):
        start_col = tabular_idx * max_cols
        end_col = min(start_col + max_cols, len(all_methods))
        this_table_methods = all_methods[start_col:end_col]

        # Create column specification: leading "ll" = (group label, pattern name)
        num_methods_table = len(this_table_methods)
        col_spec = "ll" + "c" * num_methods_table

        latex_content += f"\\begin{{tabular}}{{{col_spec}}}\n"
        latex_content += "\\toprule\n"

        # Header row: first column (group) is unlabeled, second is "Pattern".
        header = "& Pattern"
        for method in this_table_methods:
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

            # Use global row minimum for this pattern (across all methods)
            row_min = global_row_min[pattern]

            for method in this_table_methods:
                if f"{method}_mean" in summary_pivot.columns:
                    mean_val = summary_pivot.loc[pattern, f"{method}_mean"]
                    std_val = summary_pivot.loc[pattern, f"{method}_std"]
                    if pd.notna(mean_val) and pd.notna(std_val):
                        # Bold if this is the global minimum value in this row (across all tabulars)
                        if abs(mean_val - row_min) < 1e-6:  # Use small epsilon for float comparison
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

        # Add spacing between tabulars if not the last one
        if tabular_idx < num_tabulars - 1:
            latex_content += "\\quad\n"

    # End table environment
    latex_content += "\\end{table*}\n"
    
    # Save single table file
    filename = f"{figure_base_path}/normalized_negative_rmse_table.txt"
    with open(filename, "w") as f:
        f.write(latex_content)
    
    print(f"LaTeX table saved to {filename}")
    
# exit()
    
# Square table: specialized MASTERS models as columns, patterns as rows, Overall as final row
print("Creating square LaTeX table (specialized models x patterns)...")

_square_patterns = [
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
]

_square_columns = []
for _pat in _square_patterns:
    _key = _pattern_to_masters_method_key(_pat)
    _col_name = method_names.get(_key) if _key else None
    _square_columns.append((_pat, _col_name))

# Build value matrix: row pattern x column specialized model (mean and 95% CI half-width across datasets)
_square_means = {}
_square_sems = {}
for _row_pat in _square_patterns:
    _df_row = df[df.index.get_level_values(1) == _row_pat]
    for _col_pat, _col_name in _square_columns:
        if _col_name is not None and _col_name in _df_row.columns and not _df_row.empty:
            _square_means[(_row_pat, _col_pat)] = _df_row[_col_name].mean()
            _square_sems[(_row_pat, _col_pat)] = _ci95_halfwidth(_df_row[_col_name])
        else:
            _square_means[(_row_pat, _col_pat)] = float("nan")
            _square_sems[(_row_pat, _col_pat)] = float("nan")

# Overall row: mean / 95% CI half-width across all (dataset, pattern) entries for each specialized model
for _col_pat, _col_name in _square_columns:
    if _col_name is not None and _col_name in df.columns:
        _square_means[("Overall", _col_pat)] = df[_col_name].mean()
        _square_sems[("Overall", _col_pat)] = _ci95_halfwidth(df[_col_name])
    else:
        _square_means[("Overall", _col_pat)] = float("nan")
        _square_sems[("Overall", _col_pat)] = float("nan")

# Build LaTeX
def _fmt_cell(mean_val, sem_val, bold_threshold):
    if pd.isna(mean_val):
        return " & --"
    sem_str = f"{sem_val:.3f}" if pd.notna(sem_val) else "--"
    cell = f"{mean_val:.3f} $\\pm$ {sem_str}"
    if bold_threshold is not None and mean_val <= bold_threshold + 1e-6:
        return f" & \\textbf{{{cell}}}"
    return f" & {cell}"


def _top2_threshold(values):
    finite = sorted(v for v in values if pd.notna(v))
    if not finite:
        return None
    return finite[1] if len(finite) >= 2 else finite[0]


# Per-row group mapping (rows + Overall) — mirrors the main per-pattern table.
# MAR_Neural and MAR_BlockNeural fall under the MNAR group despite their names,
# matching the paper's table layout.
_square_pattern_groups = {
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
}
_square_group_macro = {"MCAR": "\\mcar", "MAR": "\\mar", "MNAR": "\\mnar"}
# Short labels used in the second header row (one per data column).
_square_col_header_macro = {
    "MCAR": "\\shortstack{\\mcar}",
    "MAR": "\\shortstack{\\colmar}",
    "MNAR": "\\shortstack{\\texttt{Self}}",
    "MAR_Neural": "\\shortstack{\\texttt{NN}}",
    "MAR_BlockNeural": "\\shortstack{\\texttt{Block}}",
    "MNARPolarizationPattern": "\\shortstack{\\texttt{Pol}}",
    "MNARSoftPolarizationPattern": "\\shortstack{\\texttt{Soft-Pol}}",
    "MNARLatentFactorPattern": "\\shortstack{\\texttt{Latent}}",
    "MNARClusterLevelPattern": "\\shortstack{\\texttt{Clust}}",
    "MNARTwoPhaseSubsetPattern": "\\shortstack{\\texttt{Phase}}",
    "MNARCensoringPattern": "\\shortstack{\\texttt{Censor}}",
}

# Row counts per group, used for the leftmost \multirow span.
_square_group_size = {}
for _p in _square_patterns:
    _g = _square_pattern_groups.get(_p)
    if _g is not None:
        _square_group_size[_g] = _square_group_size.get(_g, 0) + 1

# Top-2 bolding threshold computed across ALL columns per row, so the bolded
# cell stays the global top-2 even when the table is split into multiple
# tabulars.
_row_thresholds = {
    _row_pat: _top2_threshold(
        [_square_means[(_row_pat, _cp)] for _cp, _ in _square_columns]
    )
    for _row_pat in _square_patterns
}
_overall_thresh = _top2_threshold(
    [_square_means[("Overall", _cp)] for _cp, _ in _square_columns]
)

# Split into tabulars with at most max_cols_sq data columns each, mirroring the
# main per-pattern table's multi-tabular layout.
max_cols_sq = 5
num_tabulars_sq = (len(_square_columns) + max_cols_sq - 1) // max_cols_sq

latex_square = "\\begin{table*}[!ht]\n"
latex_square += "\\centering\n"
latex_square += "\\caption{RMSE $\\pm$ 95\\% Confidence Interval for Specialized TabImpute Models \\\\ The top-2 are bolded for each missingness pattern.}\n"
latex_square += "\\label{tab:specialized_square}\n"

for _tab_idx in range(num_tabulars_sq):
    _start = _tab_idx * max_cols_sq
    _end = min(_start + max_cols_sq, len(_square_columns))
    _this_cols = _square_columns[_start:_end]
    _this_pats = [cp for cp, _ in _this_cols]

    # "ll" = (group-label column, pattern-label column); one centered "c" per
    # data column.
    _col_spec = "ll" + "c" * len(_this_cols)
    latex_square += f"\\begin{{tabular}}{{{_col_spec}}}\n"
    latex_square += "\\toprule\n"

    # Header row 1: contiguous-group spans across this tabular's columns.
    # Single-column groups are emitted as a plain bold cell; longer spans use
    # \multicolumn, with ":c" (dashed left edge) when this isn't the first
    # group in the tabular — matching the example layout.
    _header1 = "&"  # leading slot for the group-label column
    _i = 0
    _first_group_in_tab = True
    while _i < len(_this_pats):
        _grp = _square_pattern_groups[_this_pats[_i]]
        _j = _i
        while _j < len(_this_pats) and _square_pattern_groups[_this_pats[_j]] == _grp:
            _j += 1
        _span = _j - _i
        _macro = "\\textbf{" + _square_group_macro[_grp] + "}"
        if _span == 1:
            _header1 += f" & {_macro}"
        else:
            _align = "c" if _first_group_in_tab else ":c"
            _header1 += f" & \\multicolumn{{{_span}}}{{{_align}}}{{{_macro}}}"
        _first_group_in_tab = False
        _i = _j
    _header1 += " \\\\\n"
    latex_square += _header1
    latex_square += "\\midrule\n"

    # Header row 2: per-column \shortstack labels.
    _header2 = "& Trained on $\\Rightarrow$"
    for _cp, _ in _this_cols:
        _header2 += f" & {_square_col_header_macro[_cp]}"
    _header2 += " \\\\\n"
    latex_square += _header2
    latex_square += "\\midrule\n"

    # Data rows.
    _prev_group = None
    for _row_pat in _square_patterns:
        _grp = _square_pattern_groups.get(_row_pat)
        if _grp is None:
            _group_cell = ""
        elif _grp != _prev_group:
            _n = _square_group_size[_grp]
            _macro = _square_group_macro[_grp]
            if _n == 1:
                _group_cell = f"\\textbf{{{_macro}}}"
            else:
                _group_cell = f"\\multirow{{{_n}}}{{*}}{{\\textbf{{{_macro}}}}}"
        else:
            _group_cell = ""

        _row_label = patern_latex_names.get(_row_pat, _row_pat).replace("_", "\\_")
        _row = f"{_group_cell} & {_row_label}"
        _row_thresh = _row_thresholds[_row_pat]
        for _cp, _ in _this_cols:
            _row += _fmt_cell(
                _square_means[(_row_pat, _cp)],
                _square_sems[(_row_pat, _cp)],
                _row_thresh,
            )
        _row += " \\\\\n"
        latex_square += _row
        _prev_group = _grp

    # Overall row (ungrouped: empty group cell, "Overall" label).
    latex_square += "\\midrule\n"
    _overall_row = " & Overall"
    for _cp, _ in _this_cols:
        _overall_row += _fmt_cell(
            _square_means[("Overall", _cp)],
            _square_sems[("Overall", _cp)],
            _overall_thresh,
        )
    _overall_row += " \\\\\n"
    latex_square += _overall_row

    latex_square += "\\bottomrule\n"
    latex_square += "\\end{tabular}\n"

    if _tab_idx < num_tabulars_sq - 1:
        latex_square += "\\quad\n"

latex_square += "\\end{table*}\n"

square_filename = f"{figure_base_path}/specialized_square_table.txt"
with open(square_filename, "w") as f:
    f.write(latex_square)
print(f"Square LaTeX table saved to {square_filename}")


exit()
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
latex_content_transposed += "\\caption{RMSE ± Standard Error by Method (Transposed)}\n"
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
transposed_filename = f"{figure_base_path}/normalized_negative_rmse_table_transposed.txt"
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
        mcar_filename = f"{figure_base_path}/rmse_mcar_by_dataset.txt"
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

