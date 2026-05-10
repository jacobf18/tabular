import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
import warnings

warnings.filterwarnings("ignore")

# --- Plotting ---
sns.set(style="whitegrid")

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)

base_path = "datasets/openml"

datasets = os.listdir(base_path)

methods = [
    "softimpute", 
    # "column_mean", 
    "hyperimpute",
    "ot_sinkhorn",
    "missforest",
    "ice",
    # "mice",
    # "gain",
    # "miwae",
    "masters_mcar",
    # "tabimpute_large_mcar",
    # "tabimpute_large_mcar_rank_1_11",
    # "tabimpute_large_mcar_mnar",
    # "masters_mar",
    # "masters_mnar",
    # "masters_mcar_nonlinear",
    "tabpfn",
    # "tabpfn_impute",
    "knn",
    "forestdiffusion",
    # "diffputer",
    # "remasker",
]

patterns = {
    "MCAR",
    "MAR",
    "MNAR",
    "MAR_Neural",
    "MAR_BlockNeural",
    "MAR_Sequential",
    "MNARPanelPattern",
    "MNARPolarizationPattern",
    "MNARSoftPolarizationPattern",
    "MNARLatentFactorPattern",
    "MNARClusterLevelPattern",
    "MNARTwoPhaseSubsetPattern",
    "MNARCensoringPattern",
}

method_names = {
    "tabimpute_large_mcar": "TabImpute (New Model)",
    "tabimpute_large_mcar_mnar": "TabImpute (MNAR)",
    "tabimpute_large_mcar_rank_1_11": "TabImpute (New)",
    "mixed_nonlinear": "TabImpute (Nonlinear FM)",
    "mcpfn_ensemble": "TabImpute+",
    "mcpfn_mixed_fixed": "TabImpute (Fixed)",
    "masters_mcar": "TabImpute",
    "masters_mar": "TabImpute (MAR)",
    "masters_mnar": "TabImpute (Self-Masking-MNAR)",
    "masters_mcar_nonlinear": "TabImpute (Nonlinear)",
    "tabimpute_ensemble": "TabImpute Ensemble",
    "tabimpute_ensemble_router": "TabImpute Router",
    "mcpfn_mixed_adaptive": "TabImpute",
    "mcpfn_mar_linear": "TabImpute (MCAR then MAR)",
    "mixed_more_heads": "TabImpute (More Heads)",
    "tabpfn_no_proprocessing": "TabPFN Fine-Tuned No Preprocessing",
    # "mixed_perm_both_row_col": "TabImpute",
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
    "diffputer": "DiffPuter",
    "remasker": "ReMasker",
}

# Define consistent color mapping for methods (using display names as they appear in the DataFrame)
method_colors = {
    "TabImpute (New Model)": "#2f88a8",  # Blue
    "TabImpute (MNAR)": "#2f88a8",  # Blue
    "TabImpute (New)": "#2f88a8",  # Blue
    "TabImpute+": "#2f88a8",  # Blue
    "TabImpute": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute Ensemble": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (Self-Masking-MNAR)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (Fixed)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (MCAR)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (MAR)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (MNAR)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute (Nonlinear)": "#2f88a8",  # Sea Green (distinct from GPU)
    "TabImpute Router": "#2f88a8",  # Sea Green (distinct from GPU)
    "EWF-TabPFN": "#3e3b6e",  # 
    "HyperImpute": "#ff7f0e",  # Orange
    "MissForest": "#2ca02c",   # Green
    "OT": "#591942",           # Red
    "Col Mean": "#9467bd",     # Purple
    "SoftImpute": "#8c564b",   # Brown
    "ICE": "#a14d88",          # Pink
    "MICE": "#7f7f7f",         # Gray
    "GAIN": "#286b33",         # Dark Green
    "MIWAE": "#17becf",        # Cyan
    "TabPFN": "#3e3b6e",       # Blue
    "K-Nearest Neighbors": "#a36424",  # Orange
    "ForestDiffusion": "#52b980",      # Medium Green
    "MCPFN": "#ff9896",        # Light Red
    "MCPFN (Linear Permuted)": "#c5b0d5",  # Light Purple
    "MCPFN (Nonlinear Permuted)": "#c49c94",  # Light Brown
    "DiffPuter": "#d62728",  # Red
    "ReMasker": "#f58231",  # Dark Orange
}

def compute_normalized_rmse_columnwise(X_true, X_imputed, mask):
    """
    Computes the Average Column-wise NRMSE.
    
    The metric is calculated for each column j as:
        NRMSE_j = RMSE_j / sigma_j
    
    Where:
        RMSE_j  = RMSE of the imputed values (masked entries only) in column j
        sigma_j = Standard deviation of the TRUE values (masked entries only) in column j
        
    The final score is the mean of NRMSE_j across all columns.

    Args:
        X_true (np.ndarray): The ground truth matrix (complete).
        X_imputed (np.ndarray): The matrix with imputed values.
        mask (np.ndarray): Boolean mask where True indicates a missing value 
                           (the value was imputed) and False indicates observed.

    Returns:
        float: The average NRMSE across all columns.
    """
    # Ensure inputs are numpy arrays
    X_true = np.asarray(X_true)
    X_imputed = np.asarray(X_imputed)
    mask = np.asarray(mask, dtype=bool)

    n_cols = X_true.shape[1]
    nrmse_list = []

    for j in range(n_cols):
        # Extract the boolean mask for the current column
        col_mask = mask[:, j]
        
        # If there are no missing values in this column, skip it
        if not np.any(col_mask):
            continue
            
        # Get the true and imputed values corresponding to the mask
        true_vals = X_true[col_mask, j]
        imp_vals = X_imputed[col_mask, j]
        
        # Calculate MSE for this column
        mse = np.mean((true_vals - imp_vals) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate Standard Deviation of the TRUE values in the missing set
        # (Matches the logic of the MissForest R code provided)
        sigma = np.std(true_vals)
        
        # Edge case: If sigma is 0 (all true values are identical), 
        # avoid division by zero.
        if sigma < 1e-9:
            # If the error is also 0, score is 0. Otherwise, it's effectively infinite error.
            if rmse < 1e-9:
                nrmse = 0.0
            else:
                # Log warning or handle as appropriate; here we skip or set to high value
                continue 
        else:
            nrmse = rmse / sigma
            
        nrmse_list.append(nrmse)

    # Return the average across all valid columns
    if not nrmse_list:
        return 0.0
    
    return np.mean(nrmse_list)

# Load data (same as plot_negative_rmse.py)
nrmse_data = {}

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
        
        for method in methods:
            X_imputed = np.load(f"{base_path}/{dataset}/{pattern_name}_{p}/{method}.npy")
            name = method_names[method]
            nrmse_data[(dataset, pattern_name, name)] = compute_normalized_rmse_columnwise(X_true, X_imputed, mask)

# Convert to DataFrame
df = pd.Series(nrmse_data).unstack()

# Remove rows/columns with all NaN
df = df.dropna(how='all').dropna(axis=1, how='all')

# Compute rankings for each dataset-pattern combination
# Lower NRMSE is better, so rank 1 = best
rankings = df.rank(axis=1, method='average', ascending=True)

# Compute average ranks across all dataset-pattern combinations
avg_ranks = rankings.mean(axis=0).sort_values()

# Number of datasets/problems (N) and methods (k)
N = len(df)
k = len(df.columns)

print(f"Number of dataset-pattern combinations: {N}")
print(f"Number of methods: {k}")
print(f"\nAverage ranks (lower is better):")
for method, rank in avg_ranks.items():
    print(f"  {method}: {rank:.3f}")

# Perform Friedman test
# Remove rows with any NaN (Friedman test requires complete data)
df_complete = df.dropna()

if len(df_complete) > 0:
    # Perform Friedman test using scipy
    friedman_stat, friedman_p = stats.friedmanchisquare(*[df_complete[col] for col in df_complete.columns])
    print(f"\nFriedman test:")
    print(f"  Chi-square statistic: {friedman_stat:.4f}")
    print(f"  p-value: {friedman_p:.4e}")
    
    if friedman_p < 0.05:
        print("  Result: Significant differences detected (p < 0.05)")
    else:
        print("  Result: No significant differences detected (p >= 0.05)")
    
    # Perform Nemenyi post-hoc test using scikit-posthocs
    # posthoc_nemenyi_friedman expects raw data values as numpy array
    # Rows are datasets (blocks), columns are methods (groups)
    data_array = df_complete.values.astype(float)
    p_values = sp.posthoc_nemenyi_friedman(data_array, melted=False)
    
    print(f"\nNemenyi post-hoc test completed")
    print(f"  P-values matrix shape: {p_values.shape}")
    
    # Generate critical difference diagram using scikit-posthocs
    # The function expects avg_ranks as a Series and p_values as a DataFrame
    plt.figure(figsize=(12, 6))
    sp.critical_difference_diagram(avg_ranks, p_values)
    plt.title('Critical Difference Diagram (α = 0.05)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("figures/critical_difference.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nCritical difference diagram saved to figures/critical_difference.png")
else:
    print("\nWarning: No complete rows found for statistical tests")
    friedman_stat = None
    friedman_p = None
    p_values = None

# Also generate diagram for each pattern separately
print("\n" + "="*60)
print("Generating critical difference diagrams for each pattern...")
print("="*60)

for pattern_name in patterns:
    # Get data for this pattern only
    df_pattern = df[df.index.get_level_values(1) == pattern_name]
    
    if df_pattern.empty:
        continue
    
    # Remove columns with all NaN
    df_pattern = df_pattern.dropna(axis=1, how='all')
    
    if len(df_pattern.columns) < 2:
        continue
    
    # Remove rows with any NaN for statistical tests
    df_pattern_complete = df_pattern.dropna()
    
    if len(df_pattern_complete) < 2 or len(df_pattern_complete.columns) < 2:
        continue
    
    # Compute rankings
    rankings_pattern = df_pattern_complete.rank(axis=1, method='average', ascending=True)
    avg_ranks_pattern = rankings_pattern.mean(axis=0).sort_values()
    
    # Number of datasets for this pattern
    N_pattern = len(df_pattern_complete)
    k_pattern = len(df_pattern_complete.columns)
    
    print(f"\n{pattern_name}:")
    print(f"  N={N_pattern}, k={k_pattern}")
    
    # Perform Nemenyi post-hoc test using scikit-posthocs
    # posthoc_nemenyi_friedman expects raw data values as numpy array
    # Rows are datasets (blocks), columns are methods (groups)
    data_array_pattern = df_pattern_complete.values.astype(float)
    p_values_pattern = sp.posthoc_nemenyi_friedman(data_array_pattern, melted=False)
    
    # Generate critical difference diagram using scikit-posthocs
    plt.figure(figsize=(12, 6))
    sp.critical_difference_diagram(avg_ranks_pattern, p_values_pattern)
    plt.title(f'Critical Difference Diagram - {pattern_name} (α = 0.05)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    output_path_pattern = f"figures/critical_difference_{pattern_name}.png"
    plt.savefig(output_path_pattern, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Diagram saved to {output_path_pattern}")

print("\nDone!")
