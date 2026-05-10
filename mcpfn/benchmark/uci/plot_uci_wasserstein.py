#!/usr/bin/env python3
"""Plot UCI benchmark column-wise 1-Wasserstein results."""

import itertools
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy import stats
from plot_options import (
    BARPLOT_STYLE,
    FIGURE_SIZES,
    HIGHLIGHT_COLOR,
    METHOD_COLORS,
    METHOD_NAMES,
    NEUTRAL_COLOR,
    PLOT_PARAMS,
)

# --- Plotting ---
sns.set(style="whitegrid")
matplotlib.rcParams["font.family"] = "serif"

base_path = "datasets/uci"

DWIP_V3R1_MANIFEST_PATH = Path(
    "/root/tabular/mcpfn/src/tabimpute/workdir/"
    "dwip_v3r1_9run_downstream_eval_manifest_20260415.json"
)


def load_manifest_plot_methods(manifest_path: Path) -> tuple[list[str], dict[str, str]]:
    if not manifest_path.is_file():
        return [], {}

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    manifest_methods: list[str] = []
    labels: dict[str, str] = {}
    for run in manifest.get("runs", []):
        trial_run_name = str(run.get("trial_run_name", ""))
        sweep_name = str(run.get("sweep_name", ""))
        if not trial_run_name:
            continue
        manifest_methods.append(trial_run_name)
        labels[trial_run_name] = f"TabImpute ({sweep_name})"

    return manifest_methods, labels


MANIFEST_METHODS, MANIFEST_METHOD_NAMES = load_manifest_plot_methods(
    DWIP_V3R1_MANIFEST_PATH
)

# DWIP_V2_METHODS = [f"dwip_v2-trial_{trial_id:03d}" for trial_id in range(1, 13)]
# DWIP_V1_METHODS = [f"dwip_v1-trial_{trial_id:03d}" for trial_id in range(3, 9)]
ADDITIONAL_TABIMPUTE_METHODS = [
    # "tabimpute-new-stable_v2-rank_1_30-50x50",
    # "tabimpute-new-stable_v2-rank_10_10-50x50",
    "tabimpute-new-stable_v2-spectrum-decay-topk_p_0.2-50x50",
    "dwip_v2-trial_002"
]

methods = [
    "softimpute",
    "hyperimpute",
    "missforest",
    "ice",
    "gain",
    "knn",
    "xrfm",
] + ADDITIONAL_TABIMPUTE_METHODS + MANIFEST_METHODS

patterns = {
    "MCAR",
    # "MAR",
    # "MNAR",
}

missingness_levels = [0.4]
num_repeats = 10

method_names = METHOD_NAMES.copy()
method_names["tabimpute_mcar_p0.4_num_cls_8_rank_1_11"] = "TabImpute (CLS-8)"
method_names["tabimpute_mcar_p0.4_num_cls_12_rank_1_11"] = "TabImpute (New)"
method_names["tabimpute_rope_v1"] = "TabImpute (Rope no chunks)"
method_names["tabimpute_rope_trial_6_chunking"] = "TabImpute (Rope w/ Chunks)"
method_names["tabimpute_rope_trial_14_no_permute"] = "TabImpute (Rope Trial 14 No Permute)"
method_names["tabimpute_rope_trial_12_no_permute"] = "TabImpute (Rope Trial 12 No Permute)"
method_names["tabimpute_rope_trial_12"] = "TabImpute (Rope Trial 12 w/ Permute)"
method_names["tabimpute_new_stable_v1_100k"] = "TabImpute (New Stable v1 100k)"
method_names["tabimpute_new_stable_v1_100k_w_permute"] = (
    "TabImpute (New Stable v1 100k w/ Permute)"
)
method_names["tabimpute_new_stable_v4_27500_no_permute"] = (
    "TabImpute (New Stable v4 27500 No Permute)"
)
method_names["tabimpute_new_stable_v3_15000"] = "TabImpute (New Stable v3 15000)"
method_names["tabimpute_new_stable_v6_25000"] = "TabImpute (New Stable v6 25000)"
method_names["tabimpute_new_stable_v3_15000_300_rows"] = (
    "TabImpute (New Stable v3 15000 300 Rows)"
)
method_names["tabimpute_new_stable_v3_15000_250_rows"] = (
    "TabImpute (New Stable v3 15000 250 Rows)"
)
method_names["tabimpute_new_stable_v3_15000_100_rows"] = (
    "TabImpute (New Stable v3 15000 100 Rows)"
)
method_names["tabimpute_new_stable_round2_trial_3"] = (
    "TabImpute (New Stable Round 2 Trial 3)"
)
method_names["tabimpute_new_stable_round2_trial_2"] = "TabImpute (New (Dwip))"
method_names["tabimpute_new_stable_round2_trial_6"] = (
    "TabImpute (New Stable Round 2 Trial 6)"
)
method_names["tabimpute_new_stable_round2_trial_10"] = (
    "TabImpute (New Stable Round 2 Trial 10)"
)
method_names["tabimpute_new_stable_shallow_trial_3"] = (
    "TabImpute (New Stable Shallow Trial 3)"
)
method_names["tabimpute_new_stable_shallow_trial_6"] = (
    "TabImpute (New Stable Shallow Trial 6)"
)
method_names["tabimpute_new_stable_shallow_trial_2"] = (
    "TabImpute (New Stable Shallow Trial 2)"
)
method_names["tabimpute_new_stable_deep_trial_3"] = "TabImpute (New Stable Deep Trial 3)"
method_names["tabimpute-new-stable_v2-rank_10_10-50x50"] = "TabImpute (10-10 50x50)"
method_names["tabimpute-new-stable_v2-spectrum-decay-topk_p_0.2-50x50"] = (
    "TabImpute (Spectrum Top-k p=0.2)"
)
method_names["dwip_v2-trial_002"] = "TabImpute (Dwip v2 Trial 2)"
method_names.update(MANIFEST_METHOD_NAMES)

method_colors = METHOD_COLORS.copy()
for display_name in method_names.values():
    method_colors[display_name] = (
        HIGHLIGHT_COLOR if "TabImpute" in display_name else NEUTRAL_COLOR
    )

wasserstein = {}


def compute_wasserstein_distance(
    true_values: np.ndarray, imputed_values: np.ndarray, missing_mask: np.ndarray
) -> float:
    distances = []

    for col_idx in range(true_values.shape[1]):
        col_mask = missing_mask[:, col_idx]
        if col_mask.sum() < 2:
            continue

        true_vals = true_values[col_mask, col_idx]
        pred_vals = imputed_values[col_mask, col_idx]

        if np.isnan(true_vals).any() or np.isnan(pred_vals).any():
            continue

        distance = stats.wasserstein_distance(true_vals, pred_vals)
        if np.isnan(distance):
            continue
        distances.append(float(distance))

    if not distances:
        return np.nan

    return float(np.mean(distances))


datasets = os.listdir(base_path)
for dataset in datasets:
    for pattern, missingness_level, repeat in itertools.product(
        patterns, missingness_levels, range(num_repeats)
    ):
        cfg_dir = f"{base_path}/{dataset}/{pattern}/missingness-{missingness_level}/repeat-{repeat}"

        X_missing = np.load(f"{cfg_dir}/missing.npy")
        X_true = np.load(f"{cfg_dir}/true.npy")
        mask = np.isnan(X_missing)

        for method in methods:
            X_imputed = np.load(f"{cfg_dir}/{method}.npy")
            name = method_names[method]
            wasserstein[(dataset, pattern, missingness_level, repeat, name)] = (
                compute_wasserstein_distance(X_true, X_imputed, mask)
            )

s = pd.Series(wasserstein)
s.index = pd.MultiIndex.from_tuples(
    s.index, names=["dataset", "pattern", "missingness_level", "repeat", "method"]
)
df_summary = s.groupby(["missingness_level", "method"]).mean().unstack("method")
print("\nMean by missingness_level and method:")
print(df_summary.sort_values(by="missingness_level", ascending=True))

df = pd.Series(wasserstein).unstack()
# df_norm = df
df_norm = df.apply(lambda x: x / x.max(), axis=0)

plot_pattern = True
dont_plot_methods = []

df_all = df.copy()
# df_norm_all = df_all
df_norm_all = df_all.apply(lambda x: x / x.max(), axis=0)

if plot_pattern:
    for pattern_name in patterns:
        df2 = df[df.index.get_level_values(1) == pattern_name]
        # df_norm = df2
        df_norm = df2.apply(lambda x: x / x.max(), axis=0)

        plt.figure(figsize=FIGURE_SIZES["standard"])

        sorted_methods = df_norm.mean(axis=0).sort_values(ascending=True).index
        sorted_methods = [method for method in sorted_methods if method not in dont_plot_methods]

        df_long = df_norm.melt(var_name="method", value_name="score")

        ax = sns.barplot(
            data=df_long,
            x="method",
            y="score",
            hue="method",
            order=sorted_methods,
            palette=method_colors,
            legend=False,
            **BARPLOT_STYLE,
        )

        ax.set_xticks(range(len(sorted_methods)))
        ax.set_xticklabels(sorted_methods, rotation=45, ha="right")
        ax.set_xlabel("")

        for i, label in enumerate(ax.get_xticklabels()):
            method_name = sorted_methods[i]
            if "TabImpute" in method_name:
                label.set_color(method_colors[method_name])
                label.set_fontsize(label.get_fontsize() * 1.1)

        plt.ylabel("Column-wise Wasserstein", fontsize=15)
        plt.tight_layout()
        plt.savefig(f"figures/uci_wasserstein_{pattern_name}.pdf", **PLOT_PARAMS)
        plt.close()

    fig = plt.figure(figsize=FIGURE_SIZES["standard"])
    sorted_methods_all = df_norm_all.mean(axis=0).sort_values(ascending=True).index
    sorted_methods_all = [
        method for method in sorted_methods_all if method not in dont_plot_methods
    ]

    df_long = df_norm_all.melt(var_name="method", value_name="score")

    ax = sns.barplot(
        data=df_long,
        x="method",
        y="score",
        hue="method",
        order=sorted_methods_all,
        palette=method_colors,
        legend=False,
        **BARPLOT_STYLE,
    )

    ax.set_xticks(range(len(sorted_methods_all)))
    ax.set_xticklabels(sorted_methods_all, rotation=45, ha="right", fontsize=14)
    ax.set_xlabel("")

    for i, label in enumerate(ax.get_xticklabels()):
        method_name = sorted_methods_all[i]
        if "TabImpute" in method_name:
            label.set_color(method_colors[method_name])
            label.set_fontsize(label.get_fontsize() * 1.1)

    plt.ylabel("Column-wise 1-Wasserstein", fontsize=12)
    plt.tight_layout()

    fig.savefig("figures/uci_wasserstein_overall.pdf", **PLOT_PARAMS)
    plt.close(fig)
