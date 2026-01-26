#!/usr/bin/env python3
"""
Script to analyze imputation times from MCAR_0.4 folders and plot them against dataset sizes.
"""

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from scipy import stats

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

method_names = {
    "mixed_nonlinear": "TabImpute (Nonlinear FM)",
    "tabimpute_large_mcar": "TabImpute (New Model)",
    "mcpfn_ensemble": "TabImpute+",
    "mcpfn_mnar": "TabImpute (MNAR)",
    "mcpfn_mixed_fixed": "TabImpute (Fixed)",
    "mcpfn": "TabImpute (GPU)",
    "masters_mar": "TabImpute (MAR)",
    "masters_mnar": "TabImpute (MNAR)",
    "mcpfn_cpu": "TabImpute (CPU)",
    "tabimpute_ensemble": "TabImpute Ensemble",
    "tabimpute_ensemble_router": "TabImpute Router",
    "mcpfn_mixed_adaptive": "TabImpute",
    "mcpfn_mar_linear": "TabImpute (MCAR then MAR)",
    "mixed_more_heads": "TabImpute (More Heads)",
    "tabpfn_no_proprocessing": "TabPFN Fine-Tuned No Preprocessing",
    # "mixed_perm_both_row_col": "TabImpute",
    "tabpfn_unsupervised": "Col-TabPFN (GPU)",
    "tabpfn": "EWF-TabPFN (GPU)",
    "column_mean": "Col Mean",
    "softimpute": "SoftImpute",
    "hyperimpute_hyperimpute": "HyperImpute (GPU)",
    "hyperimpute_ot_sinkhorn": "OT",
    "hyperimpute_hyperimpute_missforest": "MissForest",
    "hyperimpute_hyperimpute_ice": "ICE",
    "hyperimpute_hyperimpute_mice": "MICE",
    "hyperimpute_hyperimpute_gain": "GAIN (GPU)",
    "hyperimpute_hyperimpute_miwae": "MIWAE (GPU)",
    "forestdiffusion": "ForestDiffusion",
    "knn": "K-Nearest Neighbors",
    "remasker": "ReMasker (GPU)",
    "cacti": "CACTI (GPU)",
    # "diffputer": "DiffPuter (GPU)",
}

# Define consistent color mapping for methods (using display names as they appear in the DataFrame)
method_colors = {
    "TabImpute+": "#2f88a8",  # Blue
    "TabImpute (New Model)": "#2f88a8",  # Blue
    "TabImpute": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute Ensemble": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (MNAR)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (Fixed)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (GPU)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (CPU)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (MCAR)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (MAR)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (MNAR)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute Router": "#2f88a8",  # Sea Green (distinct from GPU)
    "EWF-TabPFN (GPU)": "#3e3b6e",  # 
    "HyperImpute (GPU)": "#ff7f0e",  # Orange
    "MissForest": "#2ca02c",   # Green
    "OT": "#591942",           # Red
    "Col Mean": "#9467bd",     # Purple
    "SoftImpute": "#8c564b",   # Brown
    "ICE": "#a14d88",          # Pink
    "MICE": "#7f7f7f",         # Gray
    "GAIN (GPU)": "#286b33",         # Dark Green
    "MIWAE (GPU)": "#17becf",        # Cyan
    "Col-TabPFN (GPU)": "#3e3b6e",       # Blue
    "K-Nearest Neighbors": "#a36424",  # Orange
    "ForestDiffusion": "#52b980",      # Medium Green
    "MCPFN": "#ff9896",        # Light Red
    "MCPFN (Linear Permuted)": "#c5b0d5",  # Light Purple
    "MCPFN (Nonlinear Permuted)": "#c49c94",  # Light Brown
    "DiffPuter (GPU)": "#d62728",  # Red
    "ReMasker (GPU)": "#d62728",  # Red
    "CACTI (GPU)": "#9467bd",  # Purple
}

include_methods = [
    "mcpfn",
    "mcpfn_cpu",
    # "tabimpute_large_mcar",
    # "mcpfn_ensemble",
    # "mcpfn_ensemble_cpu",
    "tabpfn_unsupervised",
    # "masters_mcar",
    "tabpfn",
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
    
    
    # Create efficiency bar plot (runtime per dataset size) using seaborn
    fig = plt.figure(figsize=(6.5,4.5))
    
    # Calculate efficiency metric: time per dataset size
    df['efficiency'] = df['time'] / df['dataset_size']
    
    # Filter to only include methods in include_methods list
    df_filtered = df[df['method'].isin(include_methods)].copy()
    
    # Calculate and print speedup
    baseline_method = 'mcpfn'
    speed_up_method = 'tabimpute_large_mcar'
    baseline_data = df_filtered[df_filtered['method'] == baseline_method]
    speed_up_data = df_filtered[df_filtered['method'] == speed_up_method]
    
    if len(baseline_data) > 0 and len(speed_up_data) > 0:
        baseline_mean_time = baseline_data['time'].mean()
        speed_up_mean_time = speed_up_data['time'].mean()
        speedup = baseline_mean_time / speed_up_mean_time
        print(f"\nSpeedup of {method_names[speed_up_method]} compared to {method_names[baseline_method]}: {speedup:.2f}x")
        print(f"{method_names[speed_up_method]} mean time: {speed_up_mean_time:.3f} seconds")
        print(f"{method_names[baseline_method]} mean time: {baseline_mean_time:.3f} seconds")
    else:
        print("\nWarning: Could not calculate speedup - missing data for TabPFN+ (GPU) or TabPFN (GPU)")
    
    # Add method names for plotting
    df_filtered['Method'] = df_filtered['method'].map(method_names)
    
    # Calculate mean efficiency to determine sort order (decreasing time = increasing efficiency values)
    efficiency_means = df_filtered.groupby('Method')['efficiency'].mean().sort_values(ascending=False)
    
    # --- Plotting ---
    sns.set(style="whitegrid")
    
    # Create seaborn bar plot with error bars, sorted by efficiency (decreasing time)
    ax = sns.barplot(data=df_filtered, x='Method', y='efficiency', hue='Method',
                order=efficiency_means.index,
                palette=method_colors,  # Use consistent colors from palette
                legend=False, capsize=0.2, err_kws={'color': '#999999'})
    
    # Remove x-axis labels since we'll put text inside/above bars
    ax.set_xticklabels([])
    ax.set_xlabel('')
    
    # Get bar heights and positions
    bars = ax.patches
    bar_heights = [bar.get_height() for bar in bars]
    bar_centers = [bar.get_x() + bar.get_width()/2 for bar in bars]
    # Add method names at bottom of bars, or above if they don't fit
    # Get the actual order from the seaborn plot by matching bar heights to efficiency values
    # Create a mapping from bar height to method name
    height_to_method = {}
    for method, efficiency in efficiency_means.items():
        height_to_method[efficiency] = method
    
    # Match each bar height to its corresponding method
    plot_order = []
    for height in bar_heights:
        # Find the closest efficiency value to this bar height
        closest_efficiency = min(height_to_method.keys(), key=lambda x: abs(x - height))
        plot_order.append(height_to_method[closest_efficiency])
    
    for i, (bar, height, center) in enumerate(zip(bars, bar_heights, bar_centers)):
        method_name = plot_order[i]
        
        # Calculate if bar is tall enough for text at bottom (use 15% of max height as threshold)
        max_height = max(bar_heights)
        threshold = max_height * 0.5
        
        if method_name == "K-Nearest Neighbors" or method_name == "Col Mean":
            ax.text(center, 0.000005, method_name, ha='center', va='bottom', 
                   color='black', fontweight='bold', fontsize=14, rotation=90)
            
        else:
            ax.text(center, 0.000002, method_name, ha='center', va='bottom', 
                   color='white', fontweight='bold', fontsize=14, rotation=90)
    
    plt.ylabel('Milliseconds per entry', fontsize=18)
    # plt.title('Runtime per entry \n(seconds per number of entries (rows × columns))', fontsize=18.0)
    plt.yscale('log')  # Set y-axis to log scale
    
    # Convert y-axis to milliseconds and format ticks without scientific notation
    ax = plt.gca()
    
    fig.subplots_adjust(left=0.2, right=0.95, bottom=0.05, top=0.95)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*1000:.2f}'))
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # plt.tight_layout()
    plt.savefig('/home/jacobf18/tabular/mcpfn/benchmark/imputation_efficiency_barplot.pdf', 
                dpi=300, bbox_inches=None)
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
