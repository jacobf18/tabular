import numpy as np

def compute_rmse(X_true, X_imputed, mask):
    return np.sqrt(np.mean((X_true[mask] - X_imputed[mask]) ** 2))

# This is what should be used
def compute_negative_rmse(X_true, X_imputed, mask):
    return -compute_rmse(X_true, X_imputed, mask)

def compute_normalized_rmse(X_true, X_imputed, mask):
    return compute_rmse(X_true, X_imputed, mask) / np.std(X_true[mask])

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
        if sigma < 1e-4:
            # If the error is also 0, score is 0. Otherwise, it's effectively infinite error.
            if rmse < 1e-4:
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