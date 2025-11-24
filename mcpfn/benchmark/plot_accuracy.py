import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# --- Plotting ---
sns.set(style="whitegrid")

def calculate_auc_score(y_true, y_pred, y_probs=None):
    """
    Calculate AUC score for categorical imputation.
    
    For categorical data, computes macro-averaged AUC using one-vs-rest approach.
    If probability predictions are available, uses them. Otherwise, converts hard
    predictions to one-hot encoded probabilities.
    
    Args:
        y_true: True categorical values (array-like)
        y_pred: Predicted categorical values (array-like)
        y_probs: Optional probability predictions (array-like of shape (n_samples, n_classes))
                 If None, converts hard predictions to one-hot probabilities.
    
    Returns:
        float: Macro-averaged AUC score, or np.nan if calculation fails
    """
    # Convert to numpy arrays and ensure they are 1D
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Remove NaN values
    valid_mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    if not np.any(valid_mask):
        return np.nan
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    # Get unique classes from both true and predicted values for encoding
    all_unique_classes = np.unique(np.concatenate([y_true_valid, y_pred_valid]))
    
    # Encode labels to integers (sklearn's roc_auc_score accepts integer labels)
    le = LabelEncoder()
    le.fit(all_unique_classes)
    y_true_encoded = le.transform(y_true_valid)
    y_pred_encoded = le.transform(y_pred_valid)
    
    # Ensure encoded arrays are 1D
    y_true_encoded = np.asarray(y_true_encoded).ravel()
    y_pred_encoded = np.asarray(y_pred_encoded).ravel()
    
    # Get the actual number of unique classes in y_true_encoded
    # This is what sklearn will use for validation
    unique_true_classes = np.unique(y_true_encoded)
    n_classes = len(unique_true_classes)
    
    if n_classes < 2:
        # Need at least 2 classes for AUC calculation
        return np.nan
    
    # Remap y_true_encoded to be 0..n_classes-1 (sklearn requires this)
    # and remap y_pred_encoded accordingly
    class_mapping = {old_val: new_idx for new_idx, old_val in enumerate(unique_true_classes)}
    y_true_encoded = np.array([class_mapping[val] for val in y_true_encoded])
    # For predictions, map to closest class in true classes (or 0 if not found)
    y_pred_encoded_mapped = np.array([class_mapping.get(val, 0) for val in y_pred_encoded])
    
    # Create probability predictions
    # Note: n_classes now refers only to classes in y_true
    if y_probs is not None:
        # Use provided probabilities (filter to valid samples)
        y_probs_valid = np.array(y_probs)[valid_mask]
        if y_probs_valid.ndim == 1:
            y_probs_valid = y_probs_valid.reshape(-1, 1)
        if y_probs_valid.shape[1] != n_classes:
            # Mismatch in number of classes, fall back to one-hot from mapped predictions
            y_probs_matrix = np.zeros((len(y_pred_valid), n_classes))
            y_probs_matrix[np.arange(len(y_pred_valid)), y_pred_encoded_mapped] = 1.0
        else:
            y_probs_matrix = y_probs_valid
    else:
        # Convert hard predictions to one-hot probabilities using mapped indices
        y_probs_matrix = np.zeros((len(y_pred_valid), n_classes))
        y_probs_matrix[np.arange(len(y_pred_valid)), y_pred_encoded_mapped] = 1.0
    
    # Handle binary classification (2 classes) differently
    # For binary, sklearn expects y_score to be 1D (probabilities for positive class)
    if n_classes == 2:
        # For binary classification, use probabilities for the positive class (class 1)
        y_probs_binary = y_probs_matrix[:, 1]  # Probabilities for class 1
        return roc_auc_score(y_true_encoded, y_probs_binary)
    else:
        # For multi-class, use sklearn's built-in multi-class support with macro averaging
        # roc_auc_score handles multi-class automatically when given integer labels
        # and probability matrix, using one-vs-rest approach
        return roc_auc_score(
            y_true=y_true_encoded,
            y_score=y_probs_matrix,
            multi_class='ovr',
            average='macro'
        )

base_path = "datasets/openml_categorical"

datasets = os.listdir(base_path)

methods = [
    # "softimpute", 
    # "column_mean", 
    "hyperimpute",
    # "ot_sinkhorn",
    "missforest",
    # "ice",
    # "mice",
    # "gain",
    # "miwae",
    "mcpfn",
    "mode",
    # "tabpfn",
    # "tabpfn_impute",
    # "knn",
    # "forestdiffusion",
    # "diffputer",
    # "remasker",
]

method_names = {
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

accuracy_results = {}
auc_results = {}

rows = []

for dataset in datasets:
    configs = os.listdir(f"{base_path}/{dataset}")
    df_missing = pd.read_pickle(f"{base_path}/{dataset}/dataframe_missing.pkl")
    df = pd.read_pickle(f"{base_path}/{dataset}/dataframe.pkl")
    mask = pd.isnull(df_missing).values
    
    for method in methods:
        if not os.path.exists(f"{base_path}/{dataset}/dataframe_imputed_{method}.pkl"):
            continue
        df_imputed = pd.read_pickle(f"{base_path}/{dataset}/dataframe_imputed_{method}.pkl")
        
        # Calculate accuracy
        accuracy = np.mean(df.values[mask] == df_imputed.values[mask])
        accuracy_results[(dataset, method)] = accuracy
        
        # Calculate AUC for each column and average
        auc_scores_per_column = []
        for col_idx in range(df.shape[1]):
            col_mask = mask[:, col_idx]
            if np.sum(col_mask) > 0:  # Only calculate if there are missing values in this column
                y_true_col = df.iloc[:, col_idx].values[col_mask]
                y_pred_col = df_imputed.iloc[:, col_idx].values[col_mask]
                
                # Check if column is categorical (has object/category dtype or non-numeric)
                if df.dtypes.iloc[col_idx] in ['object', 'category'] or not pd.api.types.is_numeric_dtype(df.iloc[:, col_idx]):
                    auc = calculate_auc_score(y_true_col, y_pred_col)
                    if not np.isnan(auc):
                        auc_scores_per_column.append(auc)
        
        # Average AUC across columns
        avg_auc = np.mean(auc_scores_per_column) if auc_scores_per_column else np.nan
        auc_results[(dataset, method)] = avg_auc
        
        rows.append({
            "dataset": dataset,
            "method": method,
            "accuracy": accuracy,
            "auc": avg_auc
        })

# create a dataframe with the accuracy and AUC results (dataset, method, accuracy, auc)    
df = pd.DataFrame(rows)

# Replace underscores with spaces in dataset names
df['dataset'] = df['dataset'].str.replace('_', ' ', regex=False)

print("Columns:", df.columns)
print("\nAccuracy by method:")
print(df.groupby("method")['accuracy'].mean())
print("\nAccuracy std by method:")
print(df.groupby("method")['accuracy'].std())
print("\nAUC by method:")
print(df.groupby("method")['auc'].mean())
print("\nAUC std by method:")
print(df.groupby("method")['auc'].std())

# Get normalized AUC by method
# # Pivot the dataframe: rows=datasets, columns=methods, values=auc
auc_pivot = df.pivot(index='dataset', columns='method', values='auc')

auc_pivot.dropna(inplace=True)

# Create LaTeX table with AUC scores

# Calculate mean AUC for each method across all datasets
mean_row = auc_pivot.mean().round(3)
mean_row.name = 'Overall'

# Add mean row to the pivoted dataframe
auc_pivot_with_mean = pd.concat([auc_pivot, mean_row.to_frame().T])

# Format AUC values to 3 decimal places
auc_pivot_formatted = auc_pivot_with_mean.round(3)

auc_pivot_formatted = auc_pivot_formatted[['hyperimpute', 'missforest', 'mcpfn', 'mode']]

# Generate LaTeX table
latex_table = auc_pivot_formatted.to_latex(
    na_rep='--',
    float_format='{:.3f}'.format,
    escape=False
)

# Write LaTeX table to file
output_file = "auc_table.tex"
with open(output_file, 'w') as f:
    f.write(latex_table)

print(f"\nLaTeX table written to {output_file}")