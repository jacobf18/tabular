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

neutral_color = "#B8B8B8"
highlight_color = "#2A6FBB"

# Define consistent color mapping for methods (using display names as they appear in the DataFrame)
method_colors = {
    "TabImpute+": highlight_color,  # Blue
    "TabImpute (New Model)": highlight_color,  # Blue
    "TabImpute": highlight_color,  # Sea Green (distinct from GPU)
    "TabImpute Ensemble": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (MNAR)": highlight_color,  # Sea Green (distinct from GPU)
    "TabImpute (Fixed)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (GPU)": highlight_color,  # Sea Green (distinct from GPU)
    "TabImpute (CPU)": highlight_color,  # Sea Green (distinct from GPU)
    "TabImpute (MCAR)": highlight_color,  # Sea Green (distinct from GPU)
    "TabImpute (MAR)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (MNAR)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute Router": "#2f88a8",  # Sea Green (distinct from GPU)
    "EWF-TabPFN (GPU)": neutral_color,  # 
    "HyperImpute (GPU)": neutral_color,  # Orange
    "MissForest": neutral_color,   # Green
    "OT": neutral_color,           # Red
    "Col Mean": neutral_color,     # Purple
    "SoftImpute": neutral_color,   # Brown
    "ICE": neutral_color,          # Pink
    "MICE": neutral_color,         # Gray
    "GAIN (GPU)": neutral_color,         # Dark Green
    "MIWAE (GPU)": neutral_color,        # Cyan
    "Col-TabPFN (GPU)": neutral_color,       # Blue
    "K-Nearest Neighbors": neutral_color,  # Orange
    "ForestDiffusion": neutral_color,      # Medium Green
    "MCPFN": neutral_color,        # Light Red
    "MCPFN (Linear Permuted)": neutral_color,  # Light Purple
    "MCPFN (Nonlinear Permuted)": neutral_color,  # Light Brown
    "DiffPuter (GPU)": neutral_color,  # Red
    "ReMasker (GPU)": neutral_color,  # Red
    "CACTI (GPU)": neutral_color,  # Purple
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
    plt.figure(figsize=(7.5,6.5))
    
    # Calculate efficiency metric: time per dataset size
    df['efficiency'] = df['time'] / df['dataset_size']
    
    # Filter to only include methods in include_methods list
    df_filtered = df[df['method'].isin(include_methods)].copy()
    
    # Calculate and print speedup
    baseline_method = 'EWF-TabPFN (GPU)'
    speed_up_method = 'TabImpute (GPU)'
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
    efficiency_means = df_filtered.groupby('Method')['efficiency'].mean().sort_values(ascending=True)
    
    # --- Plotting ---
    sns.set(style="whitegrid")
    
    # Create seaborn bar plot with error bars, sorted by efficiency (decreasing time)
    ax = sns.barplot(data=df_filtered, x='Method', y='efficiency', hue='Method',
                order=efficiency_means.index,
                palette=method_colors,  # Use consistent colors from palette
                legend=False, capsize=0.2, err_kws={'color': '#999999'}, edgecolor="#6E6E6E")
    
    # Set x-axis labels with 45-degree rotation
    ax.set_xticklabels(efficiency_means.index, rotation=45, ha='right', fontsize=14)
    ax.set_xlabel('')
    
    # Set label colors to match bar colors
    for label in ax.get_xticklabels():
        method_name = label.get_text()
        if method_name == "TabImpute (GPU)" or method_name == "TabImpute (CPU)":
            label.set_color(method_colors[method_name])
    
    plt.ylabel('Milliseconds per entry', fontsize=18)
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
