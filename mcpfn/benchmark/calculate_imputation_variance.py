import numpy as np
import os
import pandas as pd
from collections import defaultdict

base_path = "datasets/openml"

datasets = os.listdir(base_path)

methods = [
    "softimpute", 
    "column_mean", 
    "hyperimpute",
    "ot_sinkhorn",
    "missforest",
    "ice",
    "mice",
    "gain",
    "miwae",
    "masters_mcar",
    # "tabpfn",
    # "tabpfn_impute",
    # "knn",
    "forestdiffusion",
]

method_names = {
    "mixed_nonlinear": "TabImpute (Nonlinear FM)",
    "mcpfn_ensemble": "TabImpute+",
    "mcpfn_mnar": "TabImpute (MNAR)",
    "mcpfn_mixed_fixed": "TabImpute (Fixed)",
    "masters_mcar": "TabImpute",
    "masters_mar": "TabImpute (MAR)",
    "masters_mnar": "TabImpute (MNAR)",
    "tabimpute_ensemble": "TabImpute Ensemble",
    "tabimpute_ensemble_router": "TabImpute Router",
    "mcpfn_mixed_adaptive": "TabImpute",
    "mcpfn_mar_linear": "TabImpute (MCAR then MAR)",
    "mixed_more_heads": "TabImpute (More Heads)",
    "tabpfn_no_proprocessing": "TabPFN Fine-Tuned No Preprocessing",
    "tabpfn_impute": "TabPFN",
    "tabpfn": "EWF-TabPFN",
    "column_mean": "Col Mean",
    "hyperimpute": "HyperImpute",
    "ot_sinkhorn": "OT",
    "missforest": "MissForest",
    "softimpute": "SoftImpute",
    "ice": "ICE",
    "mice": "MICE",
    "gain": "GAIN",
    "miwae": "MIWAE",
    "forestdiffusion": "ForestDiffusion",
    "knn": "K-Nearest Neighbors",
}

# Dictionary to store MAD data: method -> list of MADs
method_mads = defaultdict(list)
# Dictionary to store RMSE data: method -> list of (min_rmse, avg_rmse) per entry
method_min_rmses = defaultdict(list)
method_avg_rmses = defaultdict(list)

pattern_name = "MCAR"
p = "0.4"
repeats_dir = f"{pattern_name}_{p}/repeats"

print("Calculating median absolute deviation per imputed entry across repeats...")

for dataset in datasets:
    dataset_path = f"{base_path}/{dataset}"
    repeats_path = f"{dataset_path}/{repeats_dir}"
    
    # Check if repeats directory exists
    if not os.path.exists(repeats_path):
        continue
    
    # Load the missing mask (same for all repeats)
    missing_path = f"{dataset_path}/{pattern_name}_{p}/missing.npy"
    if not os.path.exists(missing_path):
        continue
    
    X_true = np.load(f"{dataset_path}/{pattern_name}_{p}/true.npy")
    X_missing = np.load(missing_path)
    mask = np.isnan(X_missing)  # True for missing entries
    
    # Get list of repeat directories
    repeat_dirs = sorted([d for d in os.listdir(repeats_path) if os.path.isdir(f"{repeats_path}/{d}")])
    
    if len(repeat_dirs) == 0:
        continue
    
    print(f"Processing {dataset} with {len(repeat_dirs)} repeats...")
    
    # For each method, collect imputations across all repeats
    for method in methods:
        method_imputations = []
        
        # Load imputation from each repeat
        for repeat_dir in repeat_dirs:
            imputation_path = f"{repeats_path}/{repeat_dir}/{method}.npy"
            
            if os.path.exists(imputation_path):
                X_imputed = np.load(imputation_path)
                method_imputations.append(X_imputed)
        
        # If we have at least 2 repeats, calculate MAD
        if len(method_imputations) >= 2:
            # Stack all imputations: shape (num_repeats, n_samples, n_features)
            imputations_stack = np.stack(method_imputations, axis=0)
            
            # Calculate MAD across repeats for each entry
            # MAD = median(|x_i - median(x)|)
            # For each entry position, get values across repeats
            entry_mads = []
            entry_min_rmses = []
            entry_avg_rmses = []
            
            for i in range(imputations_stack.shape[1]):
                for j in range(imputations_stack.shape[2]):
                    if mask[i, j]:  # Only consider missing entries
                        # Get all imputed values for this entry across repeats
                        entry_values = imputations_stack[:, i, j]
                        true_value = X_true[i, j]
                        
                        # Calculate MAD
                        median_val = np.median(entry_values)
                        abs_deviations = np.abs(entry_values - median_val)
                        mad = np.median(abs_deviations)
                        entry_mads.append(mad)
                        
                        # Calculate RMSE for this entry across repeats
                        # RMSE = sqrt(mean((imputed - true)^2))
                        squared_errors = (entry_values - true_value) ** 2
                        rmse = np.sqrt(np.mean(squared_errors))
                        
                        # For minimum RMSE: take the minimum squared error, then sqrt
                        min_squared_error = np.min(squared_errors)
                        min_rmse = np.sqrt(min_squared_error)
                        
                        # For average RMSE: calculate RMSE for this entry
                        avg_rmse = rmse
                        
                        entry_min_rmses.append(min_rmse)
                        entry_avg_rmses.append(avg_rmse)
            
            # Store MADs and RMSEs for this method and dataset
            method_name = method_names.get(method, method)
            method_mads[method_name].extend(entry_mads)
            method_min_rmses[method_name].extend(entry_min_rmses)
            method_avg_rmses[method_name].extend(entry_avg_rmses)

# Calculate average MAD and RMSE per method
print("\n" + "="*60)
print("MAD and RMSE Metrics per Imputed Entry (across datasets and missing entries)")
print("="*60)

results = []
for method_name in method_mads.keys():
    mads = method_mads[method_name]
    min_rmses = method_min_rmses[method_name]
    avg_rmses = method_avg_rmses[method_name]
    
    if len(mads) > 0:
        avg_mad = np.mean(mads)
        std_mad = np.std(mads)
        
        # Average minimum RMSE across all samples
        avg_min_rmse = np.mean(min_rmses)
        std_min_rmse = np.std(min_rmses)
        
        # Average of average RMSE across all samples
        avg_avg_rmse = np.mean(avg_rmses)
        std_avg_rmse = np.std(avg_rmses)
        
        num_entries = len(mads)
        results.append({
            'Method': method_name,
            'Average MAD': avg_mad,
            'Std MAD': std_mad,
            'Average Min RMSE': avg_min_rmse,
            'Std Min RMSE': std_min_rmse,
            'Average Avg RMSE': avg_avg_rmse,
            'Std Avg RMSE': std_avg_rmse,
            'Number of Entries': num_entries
        })

# Sort by average MAD (lower is better - more consistent)
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Average MAD', ascending=True)

print("\nResults:")
print(results_df.to_string(index=False))

# Save to CSV
output_file = "figures/imputation_mad_rmse_mcar_0.4.csv"
os.makedirs("figures", exist_ok=True)
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")

# Generate LaTeX table
print("\nGenerating LaTeX table...")

latex_content = "\\begin{table}[h]\n"
latex_content += "\\centering\n"
latex_content += "\\caption{MAD and RMSE Metrics per Imputed Entry Across Repeats (MCAR, p=0.4)}\n"
latex_content += "\\label{tab:imputation_mad_rmse_mcar_0_4}\n"

# Create column specification: l for method names, c for each metric
col_spec = "lccc"

latex_content += f"\\begin{{tabular}}{{{col_spec}}}\n"
latex_content += "\\toprule\n"

# Header row
header = "Method & MAD & Min RMSE & Avg RMSE \\\\\n"
latex_content += header
latex_content += "\\midrule\n"

# Find maximum values for bolding (lower is better for all metrics, so we'll bold minimums)
max_mad = results_df['Average MAD'].max()
min_min_rmse = results_df['Average Min RMSE'].min()
min_avg_rmse = results_df['Average Avg RMSE'].min()

results_df.sort_values('Average MAD', ascending=False, inplace=True)

# Data rows (methods as rows)
for _, row_data in results_df.iterrows():
    method_name = row_data['Method'].replace('_', '\\_')  # Escape underscores
    avg_mad = row_data['Average MAD']
    std_mad = row_data['Std MAD']
    avg_min_rmse = row_data['Average Min RMSE']
    std_min_rmse = row_data['Std Min RMSE']
    avg_avg_rmse = row_data['Average Avg RMSE']
    std_avg_rmse = row_data['Std Avg RMSE']
    
    # Format MAD (bold if minimum)
    if abs(avg_mad - max_mad) < 1e-6:
        mad_str = f"\\textbf{{{avg_mad:.3f} $\\pm$ {std_mad:.3f}}}"
    else:
        mad_str = f"{avg_mad:.3f} $\\pm$ {std_mad:.3f}"
    
    # Format Min RMSE (bold if minimum)
    if abs(avg_min_rmse - min_min_rmse) < 1e-6:
        min_rmse_str = f"\\textbf{{{avg_min_rmse:.3f} $\\pm$ {std_min_rmse:.3f}}}"
    else:
        min_rmse_str = f"{avg_min_rmse:.3f} $\\pm$ {std_min_rmse:.3f}"
    
    # Format Avg RMSE (bold if minimum)
    if abs(avg_avg_rmse - min_avg_rmse) < 1e-6:
        avg_rmse_str = f"\\textbf{{{avg_avg_rmse:.3f} $\\pm$ {std_avg_rmse:.3f}}}"
    else:
        avg_rmse_str = f"{avg_avg_rmse:.3f} $\\pm$ {std_avg_rmse:.3f}"
    
    row = f"{method_name} & {mad_str} & {min_rmse_str} & {avg_rmse_str} \\\\\n"
    latex_content += row

latex_content += "\\bottomrule\n"
latex_content += "\\end{tabular}\n"
latex_content += "\\end{table}\n"

# Save LaTeX table to file
latex_filename = "figures/imputation_mad_rmse_mcar_0_4_table.txt"
with open(latex_filename, "w", encoding="utf-8") as f:
    f.write(latex_content)

print(f"LaTeX table saved to {latex_filename}")

# Also print summary statistics
print("\n" + "="*60)
print("Summary Statistics")
print("="*60)
print(f"Total methods analyzed: {len(results_df)}")
print(f"Total imputed entries across all datasets: {results_df['Number of Entries'].sum()}")
print(f"\nMethod with lowest MAD (most consistent): {results_df.iloc[0]['Method']}")
print(f"  Average MAD: {results_df.iloc[0]['Average MAD']:.6f}")
print(f"\nMethod with lowest Min RMSE: {results_df.loc[results_df['Average Min RMSE'].idxmin(), 'Method']}")
print(f"  Average Min RMSE: {results_df['Average Min RMSE'].min():.6f}")
print(f"\nMethod with lowest Avg RMSE: {results_df.loc[results_df['Average Avg RMSE'].idxmin(), 'Method']}")
print(f"  Average Avg RMSE: {results_df['Average Avg RMSE'].min():.6f}")

