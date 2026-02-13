#!/usr/bin/env python3
"""
Script to analyze imputation times from MCAR_0.4 folders and plot them against dataset sizes.
"""

import os
import re
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from scipy import stats
from plot_options import (
    setup_latex_fonts,
    METHOD_NAMES,
    METHOD_COLORS,
    HIGHLIGHT_COLOR,
    NEUTRAL_COLOR,
    FIGURE_SIZES,
    BARPLOT_STYLE,
)

def parse_dataset_sizes(file_path):
    """Parse the dataset_sizes.txt file to extract dataset names and dimensions."""
    dataset_info = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '|' in line:
                # Parse format: "dataset_name | rows \times cols"
                parts = line.split('|')
                if len(parts) == 2:
                    dataset_name = parts[0].strip()
                    dimensions = parts[1].strip()
                    
                    # Extract rows and columns from "rows \times cols" format
                    match = re.match(r'(\d+)\s*\\times\s*(\d+)', dimensions)
                    if match:
                        rows = int(match.group(1))
                        cols = int(match.group(2))
                        dataset_info[dataset_name] = {'rows': rows, 'cols': cols, 'size': rows * cols}
    
    return dataset_info

def find_imputation_times(base_path):
    """Find all imputation_time.txt files in MCAR_0.4 folders and extract times."""
    imputation_data = []
    
    # Find all MCAR_0.4 folders
    mcar_pattern = os.path.join(base_path, "**", "MCAR_0.4")
    mcar_folders = glob.glob(mcar_pattern, recursive=True)
    
    print(f"Found {len(mcar_folders)} MCAR_0.4 folders")
    
    for folder in mcar_folders:
        # Extract dataset name from path
        path_parts = Path(folder).parts
        dataset_name = None
        for part in path_parts:
            if part in ['openml']:
                # Get the next part as dataset name
                idx = path_parts.index(part)
                if idx + 1 < len(path_parts):
                    dataset_name = path_parts[idx + 1]
                    break
        
        if not dataset_name:
            print(f"Could not extract dataset name from {folder}")
            continue
            
        # Find all imputation_time.txt files in this folder
        time_files = glob.glob(os.path.join(folder, "*_imputation_time.txt"))
        
        for time_file in time_files:
            try:
                with open(time_file, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        imputation_time = float(first_line)
                        
                        # Extract method name from filename
                        filename = os.path.basename(time_file)
                        method_name = filename.replace('_imputation_time.txt', '')
                        
                        imputation_data.append({
                            'dataset': dataset_name,
                            'method': method_name,
                            'time': imputation_time,
                            'file_path': time_file
                        })
                        
            except (ValueError, FileNotFoundError) as e:
                print(f"Error reading {time_file}: {e}")
                continue
    
    return imputation_data

# Use method names from plot_options and add file-specific mappings
method_names = METHOD_NAMES.copy()
# Add file-specific method name mappings
method_names.update({
    "mcpfn": "TabImpute (GPU)",
    "mcpfn_cpu": "TabImpute (CPU)",
    "tabpfn_unsupervised": "Col-TabPFN (GPU)",
    "tabpfn": "EWF-TabPFN (GPU)",
    "hyperimpute_hyperimpute": "HyperImpute (GPU)",
    "hyperimpute_ot_sinkhorn": "OT",
    "hyperimpute_hyperimpute_missforest": "MissForest",
    "hyperimpute_hyperimpute_ice": "ICE",
    "hyperimpute_hyperimpute_mice": "MICE",
    "hyperimpute_hyperimpute_gain": "GAIN (GPU)",
    "hyperimpute_hyperimpute_miwae": "MIWAE (GPU)",
    "remasker": "ReMasker (GPU)",
    "cacti": "CACTI (GPU)",
    # "tabimpute_mcar_lin": "TabImpute (Lin. Emb.)",
    "tabimpute_dynamic_cls": "TabImpute (New)",
})

# Use colors from plot_options
neutral_color = NEUTRAL_COLOR
highlight_color = HIGHLIGHT_COLOR
# Use darker gray for x-axis labels (not bars) to match plot_negative_rmse.py
darker_neutral_color = "#333333"  # Very dark gray for x-axis label text
method_colors = METHOD_COLORS.copy()

# Add file-specific method colors (for bars)
method_colors.update({
    "TabImpute (GPU)": highlight_color,
    "TabImpute (CPU)": highlight_color,
    "EWF-TabPFN (GPU)": neutral_color,
    "HyperImpute (GPU)": neutral_color,
    "GAIN (GPU)": neutral_color,
    "MIWAE (GPU)": neutral_color,
    "Col-TabPFN (GPU)": neutral_color,
    "ReMasker (GPU)": neutral_color,
    "CACTI (GPU)": neutral_color,
    # "TabImpute (Lin. Emb.)": highlight_color,
    "TabImpute (New)": highlight_color,
})

include_methods = [
    "mcpfn",
    "mcpfn_cpu",
    # "tabimpute_large_mcar",
    # "mcpfn_ensemble",
    # "mcpfn_ensemble_cpu",
    "tabpfn_unsupervised",
    # "masters_mcar",
    "tabpfn",
    # "tabimpute_mcar_lin",
    "tabimpute_large_mcar_rank_1_11",
    # "tabpfn_impute",
    "hyperimpute_hyperimpute",
    "hyperimpute_hyperimpute_missforest",
    "hyperimpute_ot_sinkhorn",
    "hyperimpute_hyperimpute_ice",
    "hyperimpute_hyperimpute_mice",
    "hyperimpute_hyperimpute_gain",
    "hyperimpute_hyperimpute_miwae",
    # "column_mean",
    "knn",
    "softimpute",
    "forestdiffusion",
    "remasker",
    # "diffputer",
    "cacti",
]

def create_plots(imputation_data, dataset_info):
    """Create plots of imputation times vs dataset sizes."""
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(imputation_data)
    
    # Add dataset size information
    df['dataset_size'] = df['dataset'].map(lambda x: dataset_info.get(x, {}).get('size', 0))
    df['rows'] = df['dataset'].map(lambda x: dataset_info.get(x, {}).get('rows', 0))
    df['cols'] = df['dataset'].map(lambda x: dataset_info.get(x, {}).get('cols', 0))
    
    # Filter out datasets without size information
    df = df[df['dataset_size'] > 0]
    
    print(f"Found {len(df)} imputation time records")
    print(f"Unique methods: {df['method'].unique()}")
    print(f"Unique datasets: {df['dataset'].unique()}")
    
    
    # Configure LaTeX rendering for all text in plots
    setup_latex_fonts()
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'serif'
    
    # Create efficiency bar plot (runtime per dataset size) using seaborn
    plt.figure(figsize=FIGURE_SIZES['standard'])
    
    # Calculate efficiency metric: time per dataset size
    df['efficiency'] = df['time'] / df['dataset_size']
    
    # Filter to only include methods in include_methods list
    df_filtered = df[df['method'].isin(include_methods)].copy()
    
    # Calculate and print speedup (using method keys, not display names)
    baseline_method_key = 'tabpfn'  # Method key for EWF-TabPFN (GPU)
    speed_up_method_key = 'mcpfn'   # Method key for TabImpute (GPU)
    baseline_data = df_filtered[df_filtered['method'] == baseline_method_key]
    speed_up_data = df_filtered[df_filtered['method'] == speed_up_method_key]
    
    if len(baseline_data) > 0 and len(speed_up_data) > 0:
        baseline_mean_time = baseline_data['time'].mean()
        speed_up_mean_time = speed_up_data['time'].mean()
        speedup = baseline_mean_time / speed_up_mean_time
        baseline_display_name = method_names[baseline_method_key]
        speed_up_display_name = method_names[speed_up_method_key]
        print(f"\nSpeedup of {speed_up_display_name} compared to {baseline_display_name}: {speedup:.2f}x")
        print(f"{speed_up_display_name} mean time: {speed_up_mean_time:.3f} seconds")
        print(f"{baseline_display_name} mean time: {baseline_mean_time:.3f} seconds")
    else:
        print("\nWarning: Could not calculate speedup - missing data for TabPFN (GPU) or TabImpute (GPU)")
    
    # Add method names for plotting
    df_filtered['Method'] = df_filtered['method'].map(method_names)
    
    # Calculate mean efficiency to determine sort order (decreasing time = increasing efficiency values)
    efficiency_means = df_filtered.groupby('Method')['efficiency'].mean().sort_values(ascending=True)
    
    # Create seaborn bar plot with error bars, sorted by efficiency (decreasing time)
    ax = sns.barplot(data=df_filtered, x='Method', y='efficiency', hue='Method',
                order=efficiency_means.index,
                palette=method_colors,
                **BARPLOT_STYLE,
                legend=False)
    
    # Set x-axis labels with 45-degree rotation
    # Bold TabImpute methods using LaTeX \textbf{}
    labels_with_bold = [r"\textbf{" + method + "}" if "TabImpute" in method else method for method in efficiency_means.index]
    ax.set_xticks(range(len(efficiency_means.index)))
    ax.set_xticklabels(labels_with_bold, rotation=45, ha='right', fontsize=14)
    ax.set_xlabel('')
    
    # Set label colors - use darker color for non-TabImpute x-axis labels
    for i, label in enumerate(ax.get_xticklabels()):
        method_name = efficiency_means.index[i]
        if "TabImpute" in method_name:
            # TabImpute methods use highlight color and are larger
            if method_name in method_colors:
                label.set_color(method_colors[method_name])
            # Make TabImpute methods slightly larger for extra boldness
            label.set_fontsize(label.get_fontsize() * 1.1)
        else:
            # Non-TabImpute methods use darker gray for x-axis labels
            label.set_color(darker_neutral_color)
    
    # Use LaTeX-formatted label
    plt.ylabel(r'Milliseconds per entry', fontsize=18)
    # plt.title('Runtime per entry \n(seconds per number of entries (rows × columns))', fontsize=18.0)
    plt.yscale('log')  # Set y-axis to log scale
    
    # Convert y-axis to milliseconds and format ticks without scientific notation
    ax = plt.gca()
    
    # fig.subplots_adjust(left=0.2, right=0.95, bottom=0.05, top=0.95)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1000:.2f}'))
    
    # Configure grid for log scale - enable both major and minor grid lines
    ax.yaxis.grid(True, which='major', alpha=0.3, linestyle='-')
    ax.yaxis.grid(True, which='minor', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('/home/jacobf18/tabular/mcpfn/benchmark/imputation_efficiency_barplot.pdf', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print efficiency statistics
    print("\n" + "="*60)
    print("EFFICIENCY ANALYSIS (Runtime per Dataset Size)")
    print("="*60)
    print("Lower values indicate better efficiency:")
    
    # Calculate mean efficiency for each method for printing
    efficiency_by_method = df_filtered.groupby('method')['efficiency'].mean().sort_values()
    for method, efficiency in efficiency_by_method.items():
        print(f"{method_names[method]:<25}: {efficiency:.2e} seconds per data point")
    
    # Create a summary table
    summary_stats = df.groupby('method').agg({
        'time': ['count', 'mean', 'std', 'min', 'max'],
        'dataset_size': ['mean', 'std']
    }).round(3)
    
    print("\nSummary Statistics:")
    print(summary_stats)
    
    # Save summary to file
    # summary_stats.to_csv('/home/jacobf18/tabular/mcpfn/benchmark/imputation_times_summary.csv')
    
    return df

def main():
    """Main function to run the analysis."""
    # Paths
    base_path = "/home/jacobf18/tabular/mcpfn/benchmark/datasets"
    dataset_sizes_file = "/home/jacobf18/tabular/mcpfn/benchmark/dataset_sizes.txt"
    
    print("Parsing dataset sizes...")
    dataset_info = parse_dataset_sizes(dataset_sizes_file)
    print(f"Found {len(dataset_info)} datasets with size information")
    
    print("\nFinding imputation times...")
    imputation_data = find_imputation_times(base_path)
    print(f"Found {len(imputation_data)} imputation time records")
    
    print("\nCreating plots...")
    df = create_plots(imputation_data, dataset_info)
    
    print(f"\nAnalysis complete! Results saved to:")
    print("- /home/jacobf18/tabular/mcpfn/benchmark/imputation_efficiency_barplot.pdf")

if __name__ == "__main__":
    main()
