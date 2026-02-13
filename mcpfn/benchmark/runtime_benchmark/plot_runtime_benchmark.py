"""
Plot runtime benchmark results comparing old and new models.
Shows runtime vs matrix size with error bars (standard error) from multiple runs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_runtime_benchmark(csv_file='runtime_benchmark_results.csv', output_file='runtime_benchmark.png'):
    """Plot runtime comparison between old and new models."""
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Calculate statistics for each model and matrix size
    stats = df.groupby(['model_type', 'num_rows']).agg({
        'runtime': ['mean', 'std', 'count', 'min', 'max']
    }).reset_index()
    
    stats.columns = ['model_type', 'num_rows', 'mean', 'std', 'count', 'min', 'max']
    
    # Calculate standard error: std / sqrt(n)
    stats['std_error'] = stats['std'] / np.sqrt(stats['count'])
    
    # Separate data for old and new models
    new_stats = stats[stats['model_type'] == 'new'].sort_values('num_rows')
    old_stats = stats[stats['model_type'] == 'old'].sort_values('num_rows')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with error bars (standard error)
    ax.errorbar(new_stats['num_rows'], new_stats['mean'], 
                yerr=new_stats['std_error'], 
                label='New Model', 
                marker='o', 
                linestyle='-', 
                capsize=5,
                capthick=2,
                linewidth=2,
                markersize=8)
    
    ax.errorbar(old_stats['num_rows'], old_stats['mean'], 
                yerr=old_stats['std_error'], 
                label='Old Model', 
                marker='s', 
                linestyle='--', 
                capsize=5,
                capthick=2,
                linewidth=2,
                markersize=8)
    
    # Set log scale for x-axis (since rows grow exponentially)
    ax.set_xscale('log', base=2)
    # ax.set_yscale('log')
    
    # Labels and title
    ax.set_xlabel('Number of Rows (log scale, base 2)', fontsize=12)
    ax.set_ylabel('Runtime (seconds, log scale)', fontsize=12)
    ax.set_title('Runtime Comparison: Old vs New Model\n(Fixed Columns: 10)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add speedup annotations
    for num_rows in new_stats['num_rows'].unique():
        new_mean = new_stats[new_stats['num_rows'] == num_rows]['mean'].values[0]
        old_mean = old_stats[old_stats['num_rows'] == num_rows]['mean'].values[0]
        speedup = old_mean / new_mean if new_mean > 0 else 0
        
        # Position annotation above the old model point
        ax.annotate(f'{speedup:.1f}x', 
                   xy=(num_rows, old_mean),
                   xytext=(num_rows, old_mean * 1.5),
                   fontsize=9,
                   ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("Runtime Summary (Mean ± Standard Error)")
    print("="*80)
    print(f"{'Rows':<8} {'New Model':<25} {'Old Model':<25} {'Speedup':<10}")
    print("-"*80)
    
    for num_rows in sorted(new_stats['num_rows'].unique()):
        new_row = new_stats[new_stats['num_rows'] == num_rows].iloc[0]
        old_row = old_stats[old_stats['num_rows'] == num_rows].iloc[0]
        speedup = old_row['mean'] / new_row['mean'] if new_row['mean'] > 0 else 0
        
        print(f"{num_rows:<8} "
              f"{new_row['mean']:.4f} ± {new_row['std_error']:.4f}    "
              f"{old_row['mean']:.4f} ± {old_row['std_error']:.4f}    "
              f"{speedup:.2f}x")
    
    plt.show()

if __name__ == "__main__":
    # Check if CSV file exists
    csv_file = 'runtime_benchmark_results.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        print("Please run get_runtime_models.py first to generate the CSV file.")
    else:
        plot_runtime_benchmark(csv_file, 'runtime_benchmark.png')
