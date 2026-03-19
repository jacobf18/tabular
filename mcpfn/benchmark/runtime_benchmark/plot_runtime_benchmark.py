"""
Plot runtime benchmark results comparing old and new models.
Shows runtime vs matrix size with error bars (standard error) from multiple runs.
Includes standard imputation and TTT (test-time training) variants for both models.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_runtime_benchmark(
    csv_file: str = "runtime_benchmark_results.csv",
    output_file: str = "runtime_benchmark.png",
):
    """Plot runtime comparison between old and new models (standard and TTT)."""

    df = pd.read_csv(csv_file)

    # Support both old CSV (no mode) and new CSV (with mode)
    has_mode = "mode" in df.columns
    if not has_mode:
        df["mode"] = "standard"

    group_cols = ["model_type", "mode", "num_rows"] if has_mode else ["model_type", "num_rows"]
    stats = (
        df.groupby(group_cols)
        .agg({"runtime": ["mean", "std", "count", "min", "max"]})
        .reset_index()
    )
    # Flatten MultiIndex columns from agg: keys are (name, ''), agg cols are (key, name)
    if isinstance(stats.columns, pd.MultiIndex):
        stats.columns = [
            c[0] if isinstance(c, tuple) and c[1] == "" else (c[1] if isinstance(c, tuple) else c)
            for c in stats.columns
        ]
    stats["std_error"] = stats["std"] / np.sqrt(stats["count"])

    fig, ax = plt.subplots(figsize=(12, 7))

    markers = {"new": "o", "old": "s"}
    linestyles = {"standard": "-", "ttt": "--"}
    colors = {
        ("new", "standard"): "#2ecc71",
        ("new", "ttt"): "#27ae60",
        ("old", "standard"): "#e74c3c",
        ("old", "ttt"): "#c0392b",
    }

    for model_type in ["new", "old"]:
        for mode in ["standard", "ttt"]:
            subset = stats[
                (stats["model_type"] == model_type) & (stats["mode"] == mode)
            ].sort_values("num_rows")
            if subset.empty:
                continue
            label = f"{model_type.capitalize()} Model"
            if has_mode and mode == "ttt":
                label += " (TTT)"
            ax.errorbar(
                subset["num_rows"],
                subset["mean"],
                yerr=subset["std_error"],
                label=label,
                marker=markers[model_type],
                linestyle=linestyles.get(mode, "-"),
                capsize=5,
                capthick=2,
                linewidth=2,
                markersize=8,
                color=colors.get((model_type, mode), None),
            )

    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Number of Rows (log scale, base 10)", fontsize=12)
    ax.set_ylabel("Runtime (seconds, log scale, base 10)", fontsize=12)
    ax.set_title(
        "Runtime Comparison: Old vs New Model (Standard & TTT)\n(Fixed Columns: 10)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")

    # Print summary table
    print("\n" + "=" * 100)
    print("Runtime Summary (Mean ± Standard Error)")
    print("=" * 100)
    if has_mode:
        cols = ["Rows", "New (std)", "Old (std)", "New TTT", "Old TTT"]
        print(f"{'Rows':<8} {'New (std)':<22} {'Old (std)':<22} {'New TTT':<22} {'Old TTT':<22}")
        print("-" * 100)

        for num_rows in sorted(stats["num_rows"].unique()):
            def get_val(mt, m):
                row = stats[
                    (stats["model_type"] == mt)
                    & (stats["mode"] == m)
                    & (stats["num_rows"] == num_rows)
                ]
                if row.empty:
                    return "N/A"
                r = row.iloc[0]
                return f"{r['mean']:.4f} ± {r['std_error']:.4f}"

            print(
                f"{num_rows:<8} "
                f"{get_val('new', 'standard'):<22} "
                f"{get_val('old', 'standard'):<22} "
                f"{get_val('new', 'ttt'):<22} "
                f"{get_val('old', 'ttt'):<22}"
            )
    else:
        new_stats = stats[stats["model_type"] == "new"].sort_values("num_rows")
        old_stats = stats[stats["model_type"] == "old"].sort_values("num_rows")
        print(f"{'Rows':<8} {'New Model':<25} {'Old Model':<25} {'Speedup':<10}")
        print("-" * 80)
        for num_rows in sorted(new_stats["num_rows"].unique()):
            new_row = new_stats[new_stats["num_rows"] == num_rows].iloc[0]
            old_row = old_stats[old_stats["num_rows"] == num_rows].iloc[0]
            speedup = old_row["mean"] / new_row["mean"] if new_row["mean"] > 0 else 0
            print(
                f"{num_rows:<8} "
                f"{new_row['mean']:.4f} ± {new_row['std_error']:.4f}    "
                f"{old_row['mean']:.4f} ± {old_row['std_error']:.4f}    "
                f"{speedup:.2f}x"
            )
    plt.show()

if __name__ == "__main__":
    # Check if CSV file exists
    csv_file = 'runtime_benchmark_results.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        print("Please run get_runtime_models.py first to generate the CSV file.")
    else:
        plot_runtime_benchmark(csv_file, 'runtime_benchmark.png')
