#!/usr/bin/env python3
"""
Script to analyze UCI imputation times and plot them against dataset sizes.
"""

import glob
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_options import (
    setup_latex_fonts,
    METHOD_NAMES,
    METHOD_COLORS,
    HIGHLIGHT_COLOR,
    NEUTRAL_COLOR,
    FIGURE_SIZES,
    BARPLOT_STYLE,
)


def _darken_hex(hex_color: str, factor: float) -> str:
    """Darken a #RRGGBB color by multiplying RGB channels (0–1)."""
    hc = hex_color.lstrip("#")
    r, g, b = (int(hc[i : i + 2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, int(r * factor))),
        max(0, min(255, int(g * factor))),
        max(0, min(255, int(b * factor))),
    )


def format_seconds_human_readable(seconds: float, _pos=None) -> str:
    """Tick labels for values in seconds: ms, s, min, or days."""
    if seconds <= 0 or not np.isfinite(seconds):
        return ""
    s = float(seconds)
    if s < 1.0:
        return f"{s * 1000:.3g} ms"
    if s < 60.0:
        return f"{s:.3g} s"
    if s < 86400.0:
        return f"{s / 60.0:.3g} min"
    return f"{s / 86400.0:.3g} d"


def extract_dataset_name(path_str):
    """Extract the UCI dataset name from a benchmark path."""
    path_parts = Path(path_str).parts
    if "uci" in path_parts:
        idx = path_parts.index("uci")
        if idx + 1 < len(path_parts):
            return path_parts[idx + 1]
    return None


def parse_uci_pattern_missingness(file_path: str):
    """Return (pattern, missingness) from .../uci/<dataset>/<pattern>/missingness-<p>/..., else (None, None)."""
    parts = Path(file_path).parts
    try:
        uci_idx = parts.index("uci")
    except ValueError:
        return None, None
    if uci_idx + 3 >= len(parts):
        return None, None
    pattern = parts[uci_idx + 2]
    miss_part = parts[uci_idx + 3]
    if not miss_part.startswith("missingness-"):
        return None, None
    try:
        level = float(miss_part[len("missingness-") :])
    except ValueError:
        return pattern, None
    return pattern, level


# Runtime figures and tables in create_plots use only this configuration
RUNTIME_PATTERN = "MCAR"
RUNTIME_MISSINGNESS = 0.4


def parse_uci_dataset_sizes(base_path):
    """Infer dataset sizes from stored UCI benchmark arrays."""
    dataset_info = {}
    true_files = glob.glob(
        os.path.join(base_path, "*", "*", "missingness-*", "repeat-*", "true.npy")
    )

    if not true_files:
        true_files = glob.glob(
            os.path.join(base_path, "*", "*", "missingness-*", "repeat-*", "missing.npy")
        )

    for array_file in true_files:
        dataset_name = extract_dataset_name(array_file)
        if not dataset_name or dataset_name in dataset_info:
            continue

        try:
            dataset_array = np.load(array_file, mmap_mode="r")
            rows, cols = dataset_array.shape
            dataset_info[dataset_name] = {"rows": rows, "cols": cols, "size": rows * cols}
        except (ValueError, FileNotFoundError, OSError) as exc:
            print(f"Error reading {array_file}: {exc}")

    return dataset_info


def find_imputation_times(base_path):
    """Find all UCI imputation time files and extract the latest times."""
    imputation_data = []
    time_files = glob.glob(
        os.path.join(base_path, "*", "*", "missingness-*", "repeat-*", "*_imputation_time.txt")
    )

    print(f"Found {len(time_files)} imputation time files")

    for time_file in time_files:
        dataset_name = extract_dataset_name(time_file)
        if not dataset_name:
            print(f"Could not extract dataset name from {time_file}")
            continue

        try:
            with open(time_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
                if not lines:
                    continue

            imputation_time = float(lines[-1])
            filename = os.path.basename(time_file)
            method_name = filename.replace("_imputation_time.txt", "")

            imputation_data.append(
                {
                    "dataset": dataset_name,
                    "method": method_name,
                    "time": imputation_time,
                    "file_path": time_file,
                }
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"Error reading {time_file}: {exc}")

    return imputation_data


method_names = METHOD_NAMES.copy()
method_names.update(
    {
        "hyperimpute_hyperimpute": "HyperImpute (GPU)",
        "hyperimpute_ot_sinkhorn": "OT",
        "hyperimpute_hyperimpute_missforest": "MissForest",
        "hyperimpute_hyperimpute_ice": "ICE",
        "hyperimpute_hyperimpute_mice": "MICE",
        "hyperimpute_hyperimpute_gain": "GAIN (GPU)",
        "hyperimpute_hyperimpute_miwae": "MIWAE (GPU)",
        "tabimpute_new_stable_round2_trial_2": "MatFormer",
        "masters_mcar": "TabImpute",
    }
)

neutral_color = NEUTRAL_COLOR
# Slightly darker than plot_options NEUTRAL_COLOR for non–TabImpute methods
neutral_method_color = _darken_hex(NEUTRAL_COLOR, 0.82)
highlight_color = HIGHLIGHT_COLOR
darker_neutral_color = "#333333"
method_colors = METHOD_COLORS.copy()
method_colors.update(
    {
        "MatFormer": highlight_color,
        "HyperImpute (GPU)": neutral_method_color,
        "GAIN (GPU)": neutral_method_color,
        "MIWAE (GPU)": neutral_method_color,
        "MissForest": neutral_method_color,
        "ICE": neutral_method_color,
        "MICE": neutral_method_color,
        "OT": neutral_method_color,
        "SoftImpute": neutral_method_color,
        "KNN": neutral_method_color,
        "TabImpute": highlight_color,
    }
)

include_methods = [
    "masters_mcar"
    "tabimpute_new_stable_round2_trial_2",
    "hyperimpute_hyperimpute",
    "hyperimpute_hyperimpute_missforest",
    "hyperimpute_ot_sinkhorn",
    "hyperimpute_hyperimpute_ice",
    "hyperimpute_hyperimpute_mice",
    "hyperimpute_hyperimpute_gain",
    "hyperimpute_hyperimpute_miwae",
    "knn",
    "softimpute",
]


def create_runtime_tables(df_filtered):
    """Create LaTeX and Markdown tables with mean runtime per dataset for each method."""
    runtime_pivot = df_filtered.pivot_table(
        index="dataset",
        columns="Method",
        values="time",
        aggfunc="mean",
    )
    runtime_pivot = runtime_pivot.sort_index()

    ordered_columns = [
        method_names[method]
        for method in include_methods
        if method in method_names and method_names[method] in runtime_pivot.columns
    ]
    runtime_pivot = runtime_pivot[ordered_columns]

    overall_row = runtime_pivot.mean(axis=0)
    overall_row.name = "Overall"
    runtime_table = pd.concat([runtime_pivot, overall_row.to_frame().T])

    latex_formatted_table = runtime_table.copy().astype(object)
    markdown_formatted_table = runtime_table.copy().astype(object)
    for row_name, row in runtime_table.iterrows():
        valid_values = row.dropna()
        row_min = valid_values.min() if not valid_values.empty else None

        for col_name, value in row.items():
            if pd.isna(value):
                latex_formatted_table.at[row_name, col_name] = "--"
                markdown_formatted_table.at[row_name, col_name] = "--"
                continue

            formatted_value = f"{value:.3f}"
            if row_min is not None and abs(value - row_min) < 1e-12:
                latex_formatted_table.at[row_name, col_name] = f"\\textbf{{{formatted_value}}}"
                markdown_formatted_table.at[row_name, col_name] = f"**{formatted_value}**"
            else:
                latex_formatted_table.at[row_name, col_name] = formatted_value
                markdown_formatted_table.at[row_name, col_name] = formatted_value

    latex_formatted_table.index = [
        index_name.replace("_", "\\_") if isinstance(index_name, str) else index_name
        for index_name in latex_formatted_table.index
    ]

    latex_table = latex_formatted_table.to_latex(
        escape=False,
        column_format="l" + "c" * len(latex_formatted_table.columns),
    )

    latex_content = "\\begin{table}[h]\n"
    latex_content += "\\centering\n"
    latex_content += "\\caption{Mean imputation runtime in seconds for each UCI dataset and method. Bold indicates the fastest method for each dataset.}\n"
    latex_content += "\\label{tab:uci_imputation_runtime}\n"
    latex_content += latex_table
    latex_content += "\\end{table}\n"

    output_file = "figures/uci_imputation_runtime_table.tex"
    with open(output_file, "w") as f:
        f.write(latex_content)

    print(f"LaTeX runtime table saved to {output_file}")

    markdown_output_file = "figures/uci_imputation_runtime_table.md"
    markdown_table = markdown_formatted_table.to_markdown()
    with open(markdown_output_file, "w") as f:
        f.write(markdown_table + "\n")

    print(f"Markdown runtime table saved to {markdown_output_file}")


def create_plots(imputation_data, dataset_info):
    """Create plots of imputation times vs dataset sizes."""
    df = pd.DataFrame(imputation_data)
    if df.empty:
        raise ValueError("No imputation time data found.")

    df["dataset_size"] = df["dataset"].map(lambda x: dataset_info.get(x, {}).get("size", 0))
    df["rows"] = df["dataset"].map(lambda x: dataset_info.get(x, {}).get("rows", 0))
    df["cols"] = df["dataset"].map(lambda x: dataset_info.get(x, {}).get("cols", 0))
    df = df[df["dataset_size"] > 0]

    n_before_runtime_filter = len(df)
    pm = df["file_path"].map(parse_uci_pattern_missingness)
    df["_pattern"] = pm.map(lambda x: x[0])
    df["_missingness"] = pm.map(lambda x: x[1])
    df = df[
        (df["_pattern"] == RUNTIME_PATTERN)
        & df["_missingness"].notna()
        & np.isclose(df["_missingness"], RUNTIME_MISSINGNESS)
    ].drop(columns=["_pattern", "_missingness"])
    print(
        f"Filtered to {RUNTIME_PATTERN} / missingness {RUNTIME_MISSINGNESS}: "
        f"{len(df)} records (from {n_before_runtime_filter} with sizes)"
    )

    if df.empty:
        raise ValueError(
            f"No imputation times under {RUNTIME_PATTERN} missingness-{RUNTIME_MISSINGNESS}."
        )

    print(f"Found {len(df)} imputation time records with dataset sizes")
    print(f"Unique methods: {df['method'].unique()}")
    print(f"Unique datasets: {df['dataset'].unique()}")

    setup_latex_fonts()
    matplotlib.rcParams["text.usetex"] = False
    matplotlib.rcParams["font.family"] = "serif"

    plt.figure(figsize=FIGURE_SIZES["standard"])

    df["efficiency"] = df["time"] / df["dataset_size"]
    df_filtered = df[df["method"].isin(include_methods)].copy()
    if df_filtered.empty:
        raise ValueError("No matching UCI methods found for plotting.")

    baseline_method_key = "hyperimpute_hyperimpute"
    speed_up_method_key = "tabimpute_new_stable_round2_trial_2"
    baseline_data = df_filtered[df_filtered["method"] == baseline_method_key]
    speed_up_data = df_filtered[df_filtered["method"] == speed_up_method_key]

    if len(baseline_data) > 0 and len(speed_up_data) > 0:
        baseline_mean_time = baseline_data["time"].mean()
        speed_up_mean_time = speed_up_data["time"].mean()
        speedup = baseline_mean_time / speed_up_mean_time
        baseline_display_name = method_names[baseline_method_key]
        speed_up_display_name = method_names[speed_up_method_key]
        print(
            f"\nSpeedup of {speed_up_display_name} compared to {baseline_display_name}: {speedup:.2f}x"
        )
        print(f"{speed_up_display_name} mean time: {speed_up_mean_time:.3f} seconds")
        print(f"{baseline_display_name} mean time: {baseline_mean_time:.3f} seconds")
    else:
        print(
            "\nWarning: Could not calculate speedup - missing data for "
            "MatFormer or HyperImpute (GPU)"
        )

    df_filtered["Method"] = df_filtered["method"].map(method_names)
    efficiency_means = df_filtered.groupby("Method")["efficiency"].mean().sort_values(ascending=True)

    ax = sns.barplot(
        data=df_filtered,
        x="Method",
        y="efficiency",
        hue="Method",
        order=efficiency_means.index,
        palette=method_colors,
        **BARPLOT_STYLE,
        legend=False,
    )

    ax.set_xticks(range(len(efficiency_means.index)))
    ax.set_xticklabels(efficiency_means.index, rotation=45, ha="right", fontsize=14)
    ax.set_xlabel("")

    for i, label in enumerate(ax.get_xticklabels()):
        method_name = efficiency_means.index[i]
        if "TabImpute" in method_name:
            if method_name in method_colors:
                label.set_color(method_colors[method_name])
            label.set_fontsize(label.get_fontsize() * 1.1)
        else:
            label.set_color(darker_neutral_color)

    plt.ylabel("Time per entry", fontsize=18)
    plt.yscale("log")

    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_seconds_human_readable))
    ax.yaxis.grid(True, which="major", alpha=0.3, linestyle="-")
    ax.yaxis.grid(True, which="minor", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig("figures/uci_imputation_efficiency_barplot.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    # Mean imputation time vs dataset size (rows × columns), one line per method
    agg_time = (
        df_filtered.groupby(["dataset", "method", "Method"], as_index=False)
        .agg(time_mean=("time", "mean"), dataset_size=("dataset_size", "first"))
    )

    _, ax_line = plt.subplots(figsize=FIGURE_SIZES["wide"])
    for method_display in efficiency_means.index:
        sub = agg_time[agg_time["Method"] == method_display].sort_values("dataset_size")
        if sub.empty:
            continue
        color = method_colors.get(method_display, neutral_method_color)
        ax_line.plot(
            sub["dataset_size"],
            sub["time_mean"],
            marker="o",
            markersize=4,
            color=color,
            linewidth=1.6,
            alpha=0.9,
        )
        x_last = float(sub["dataset_size"].iloc[-1])
        y_last = float(sub["time_mean"].iloc[-1])
        ax_line.annotate(
            method_display,
            xy=(x_last, y_last),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color=color,
            clip_on=False,
        )

    ax_line.set_xlabel("Dataset size (rows × columns)", fontsize=16)
    ax_line.set_ylabel("Imputation time", fontsize=16)
    ax_line.set_xscale("log")
    ax_line.set_yscale("log")
    ax_line.yaxis.set_major_formatter(plt.FuncFormatter(format_seconds_human_readable))
    ax_line.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.3g}"))
    ax_line.yaxis.grid(True, which="major", alpha=0.3, linestyle="-")
    ax_line.yaxis.grid(True, which="minor", alpha=0.3, linestyle="--")
    ax_line.xaxis.grid(True, which="major", alpha=0.3, linestyle="-")
    x_left, x_right = ax_line.get_xlim()
    ax_line.set_xlim(x_left, x_right * 1.45)
    plt.tight_layout()
    plt.savefig(
        "figures/uci_imputation_time_vs_dataset_size.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("\n" + "=" * 60)
    print("UCI EFFICIENCY ANALYSIS (Runtime per Dataset Size)")
    print("=" * 60)
    print("Lower values indicate better efficiency:")

    efficiency_by_method = df_filtered.groupby("method")["efficiency"].mean().sort_values()
    for method, efficiency in efficiency_by_method.items():
        print(f"{method_names[method]:<25}: {efficiency:.2e} seconds per data point")

    summary_stats = df.groupby("method").agg(
        {
            "time": ["count", "mean", "std", "min", "max"],
            "dataset_size": ["mean", "std"],
        }
    ).round(3)

    print("\nSummary Statistics:")
    print(summary_stats)

    create_runtime_tables(df_filtered)

    return df


def main():
    """Main function to run the analysis."""
    base_path = "../datasets/uci"

    print("Parsing UCI dataset sizes...")
    dataset_info = parse_uci_dataset_sizes(base_path)
    print(f"Found {len(dataset_info)} datasets with size information")

    print("\nFinding UCI imputation times...")
    imputation_data = find_imputation_times(base_path)
    print(f"Found {len(imputation_data)} imputation time records")

    print("\nCreating plots...")
    create_plots(imputation_data, dataset_info)

    print("\nAnalysis complete! Results saved to:")
    print("- figures/uci_imputation_efficiency_barplot.pdf")
    print("- figures/uci_imputation_time_vs_dataset_size.pdf")
    print("- figures/uci_imputation_runtime_table.tex")
    print("- figures/uci_imputation_runtime_table.md")


if __name__ == "__main__":
    main()
