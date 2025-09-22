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
    "mcpfn_ensemble": "TabImpute++",
    "mcpfn": "TabImpute",
    "mixed_perm_both_row_col": "MCPFN (Linear Permuted)",
    "tabpfn": "TabPFN-Impute",
    "tabpfn_unsupervised": "TabPFN",
    "column_mean": "Col Mean",
    "hyperimpute_hyperimpute": "HyperImpute",
    "hyperimpute_hyperimpute_ot_sinkhorn": "OT",
    "hyperimpute_hyperimpute_missforest": "MissForest",
    "softimpute": "SoftImpute",
    "hyperimpute_ot_sinkhorn": "OT",
    "hyperimpute_hyperimpute_ice": "ICE",
    "hyperimpute_hyperimpute_mice": "MICE",
    "hyperimpute_hyperimpute_gain": "GAIN",
    "hyperimpute_hyperimpute_miwae": "MIWAE",
}

include_methods = [
    "mcpfn_ensemble",
    # "mcpfn",
    # "tabpfn",
    "tabpfn_unsupervised",
    "hyperimpute_hyperimpute",
    "hyperimpute_hyperimpute_missforest",
    "hyperimpute_ot_sinkhorn",
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
    
    # Create three plots: dataset size, rows, and columns
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Get unique methods and assign colors
    methods = [method for method in df['method'].unique() if method in include_methods]
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    method_colors = dict(zip(methods, colors))
    
    # Plot 1: Time vs Dataset Size
    ax1 = axes[0]
    for method in methods:
        method_data = df[df['method'] == method]
        
        # Create scatter plot with alpha=0.5
        ax1.scatter(method_data['dataset_size'], method_data['time'], 
                   label=method_names[method], alpha=0.5, s=50, color=method_colors[method])
        
        # Add line of best fit if we have enough data points
        if len(method_data) > 1:
            # Use log scale for fitting
            x_log = np.log10(method_data['dataset_size'])
            y_log = np.log10(method_data['time'])
            
            # Fit linear regression in log space
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
            
            # Generate points for the line
            x_line = np.logspace(np.log10(method_data['dataset_size'].min()), 
                               np.log10(method_data['dataset_size'].max()), 100)
            y_line = 10**(intercept + slope * np.log10(x_line))
            
            # Plot the line of best fit in bold
            ax1.plot(x_line, y_line, color=method_colors[method], 
                    linewidth=3, alpha=0.8)
    
    ax1.set_xlabel('Dataset Size (rows × columns)')
    ax1.set_ylabel('Imputation Time (seconds)')
    ax1.set_title('Imputation Time vs Dataset Size')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Time vs Number of Rows
    ax2 = axes[1]
    for method in methods:
        method_data = df[df['method'] == method]
        
        # Create scatter plot with alpha=0.5
        ax2.scatter(method_data['rows'], method_data['time'], 
                   label=method_names[method], alpha=0.5, s=50, color=method_colors[method])
        
        # Add line of best fit if we have enough data points
        if len(method_data) > 1:
            # Use log scale for fitting
            x_log = np.log10(method_data['rows'])
            y_log = np.log10(method_data['time'])
            
            # Fit linear regression in log space
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
            
            # Generate points for the line
            x_line = np.logspace(np.log10(method_data['rows'].min()), 
                               np.log10(method_data['rows'].max()), 100)
            y_line = 10**(intercept + slope * np.log10(x_line))
            
            # Plot the line of best fit in bold
            ax2.plot(x_line, y_line, color=method_colors[method], 
                    linewidth=3, alpha=0.8)
    
    ax2.set_xlabel('Number of Rows')
    ax2.set_ylabel('Imputation Time (seconds)')
    ax2.set_title('Imputation Time vs Number of Rows')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # Plot 3: Time vs Number of Columns
    ax3 = axes[2]
    for method in methods:
        method_data = df[df['method'] == method]
        
        # Create scatter plot with alpha=0.5
        ax3.scatter(method_data['cols'], method_data['time'], 
                   label=method_names[method], alpha=0.5, s=50, color=method_colors[method])
        
        # Add line of best fit if we have enough data points
        if len(method_data) > 1:
            # Use log scale for fitting
            x_log = np.log10(method_data['cols'])
            y_log = np.log10(method_data['time'])
            
            # Fit linear regression in log space
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
            
            # Generate points for the line
            x_line = np.logspace(np.log10(method_data['cols'].min()), 
                               np.log10(method_data['cols'].max()), 100)
            y_line = 10**(intercept + slope * np.log10(x_line))
            
            # Plot the line of best fit in bold
            ax3.plot(x_line, y_line, color=method_colors[method], 
                    linewidth=3, alpha=0.8)
    
    ax3.set_xlabel('Number of Columns')
    ax3.set_ylabel('Imputation Time (seconds)')
    ax3.set_title('Imputation Time vs Number of Columns')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    # Add legend to the right of all plots - only for methods that were actually plotted
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.05, 0.5), loc='center left')
    
    plt.tight_layout()
    plt.savefig('/root/tabular/mcpfn/benchmark/imputation_times_analysis.pdf', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create efficiency bar plot (runtime per dataset size) using seaborn
    plt.figure(figsize=(6,5))
    
    # Calculate efficiency metric: time per dataset size
    df['efficiency'] = df['time'] / df['dataset_size']
    
    # Filter to only include methods in include_methods list
    df_filtered = df[df['method'].isin(include_methods)].copy()
    
    # Add method names for plotting
    df_filtered['Method'] = df_filtered['method'].map(method_names)
    
    # Calculate mean efficiency to determine sort order (decreasing time = increasing efficiency values)
    efficiency_means = df_filtered.groupby('Method')['efficiency'].mean().sort_values(ascending=False)
    
    # --- Plotting ---
    sns.set(style="whitegrid")
    
    # Create seaborn bar plot with error bars, sorted by efficiency (decreasing time)
    ax = sns.barplot(data=df_filtered, x='Method', y='efficiency', hue='Method', 
                legend=False, order=efficiency_means.index,
                capsize=0.2, err_kws={'color': 'dimgray'})
    
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
        
        if height > threshold:
            # Place text at bottom of the bar (white text, vertical)
            # Position at 15% of bar height to ensure it's well within the bar
            ax.text(center, height/2, method_name, ha='center', va='center', 
                   color='white', fontweight='bold', fontsize=18, rotation=90)
        else:
            # Place text above the bar (black text, vertical)
            # Position slightly above the bar top
            ax.text(center, height*1.6, method_name, ha='center', va='bottom', 
                   color='black', fontweight='bold', fontsize=18, rotation=90)
    
    plt.ylabel('Seconds per entry', fontsize=18)
    # plt.title('Runtime per entry \n(seconds per number of entries (rows × columns))', fontsize=18.0)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/root/tabular/mcpfn/benchmark/imputation_efficiency_barplot.pdf', 
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
    # summary_stats.to_csv('/root/tabular/mcpfn/benchmark/imputation_times_summary.csv')
    
    return df

def main():
    """Main function to run the analysis."""
    # Paths
    base_path = "/root/tabular/mcpfn/benchmark/datasets"
    dataset_sizes_file = "/root/tabular/mcpfn/benchmark/dataset_sizes.txt"
    
    print("Parsing dataset sizes...")
    dataset_info = parse_dataset_sizes(dataset_sizes_file)
    print(f"Found {len(dataset_info)} datasets with size information")
    
    print("\nFinding imputation times...")
    imputation_data = find_imputation_times(base_path)
    print(f"Found {len(imputation_data)} imputation time records")
    
    print("\nCreating plots...")
    df = create_plots(imputation_data, dataset_info)
    
    print(f"\nAnalysis complete! Results saved to:")
    print("- /root/tabular/mcpfn/benchmark/imputation_times_analysis.png")
    print("- /root/tabular/mcpfn/benchmark/imputation_efficiency_barplot.png")
    print("- /root/tabular/mcpfn/benchmark/imputation_times_summary.csv")

if __name__ == "__main__":
    main()
