import numpy as np

def create_train_test_sets(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Create train and test sets from a matrix with missing values.

    Args:
        X (np.ndarray): Matrix with missing values.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Train and test sets.
    """
    # Get missing indices in X
    missing_indices = np.where(np.isnan(X))
    
    non_missing_indices = np.where(~np.isnan(X))
    
    train_X = np.zeros((len(non_missing_indices[0]), X.shape[0] + X.shape[1] - 2))
    train_y = np.zeros(len(non_missing_indices[0]))
    test_X = np.zeros((len(missing_indices[0]), X.shape[0] + X.shape[1] - 2))
    test_y = np.zeros(len(missing_indices[0]))
    
    for k, (i,j) in enumerate(zip(non_missing_indices[0], non_missing_indices[1])):
        # Get row without j-th column
        row = np.delete(X[i,:], j)
        # Get column without i-th row
        col = np.delete(X[:,j], i)
        
        # Create train set
        train_X[k,:] = np.concatenate((row, col))
        train_y[k] = X[i,j]
        
    
    for k, (i,j) in enumerate(zip(missing_indices[0], missing_indices[1])):
        # Get row without j-th column
        row = np.delete(X[i,:], j)
        # Get column without i-th row
        col = np.delete(X[:,j], i)
        
        # Create train set
        test_X[k,:] = np.concatenate((row, col))
        test_y[k] = np.nan
       
    return train_X, train_y, test_X, test_y