#!/usr/bin/env python3
"""
Download UCI datasets listed in README.md (lines 17-28) and save them to datasets/uci folder.
Uses ucimlrepo package to fetch datasets and ensures data types match variables.type table.
"""

import os
import pandas as pd
from urllib import request
from sklearn.datasets import fetch_california_housing, load_diabetes, load_iris
from ucimlrepo import fetch_ucirepo

# Dataset name mappings - using UCI IDs or names where available
# For sklearn datasets, we'll use sklearn functions
DATASETS_CONFIG = {
    'airfoil_self_noise': {
        'type': 'ucimlrepo',
        'id': 291,  # Airfoil Self-Noise
    },
    'blood_transfusion': {
        'type': 'ucimlrepo',
        'id': 146,  # Blood Transfusion Service Center
    },
    'california_housing': {
        'type': 'sklearn',
        'function': fetch_california_housing
    },
    'concrete_compression': {
        'type': 'ucimlrepo',
        'id': 165,  # Concrete Compressive Strength
    },
    'diabetes': {
        'type': 'sklearn',
        'function': load_diabetes
    },
    'ionosphere': {
        'type': 'ucimlrepo',
        'id': 52,  # Ionosphere
    },
    'iris': {
        'type': 'sklearn',
        'function': load_iris
    },
    'letter_recognition': {
        'type': 'ucimlrepo',
        'id': 59,  # Letter Recognition
    },
    'libras_movement': {
        'type': 'uci_url',
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data',
        'filename': 'movement_libras.data',
        'delimiter': ',',
        'header': None
    },
    'spam_base': {
        'type': 'ucimlrepo',
        'id': 94,  # Spambase
    },
    'wine_quality_red': {
        'type': 'ucimlrepo',
        'id': 186,  # Wine Quality - contains both red and white, filter by 'color' column
    },
    'wine_quality_white': {
        'type': 'ucimlrepo',
        'id': 186,  # Wine Quality - contains both red and white, filter by 'color' column
    }
}


def convert_dtypes_from_variables(df, variables_df):
    """
    Convert DataFrame column dtypes based on variables.type table.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to convert
    variables_df : pd.DataFrame
        Variables dataframe with 'name' and 'type' columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with converted dtypes
    """
    # Create mapping from variable name to type
    var_type_map = dict(zip(variables_df['name'], variables_df['type']))
    
    for col in df.columns:
        if col in var_type_map:
            var_type = var_type_map[col].lower()
            
            if var_type == 'categorical':
                df[col] = df[col].astype('category')
            elif var_type == 'integer':
                # Convert to numeric first, then to integer
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
            elif var_type == 'continuous':
                # Convert to numeric (float)
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
            elif var_type == 'binary':
                # Convert to category or boolean
                df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
            # If type is not recognized, leave as is
    
    return df


def download_sklearn_dataset(dataset_name, dataset_config, save_dir):
    """Download dataset from sklearn."""
    print(f"\nDownloading {dataset_name} from sklearn...")
    func = dataset_config['function']
    
    if dataset_name == 'california_housing':
        data = func(as_frame=True)
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset_name == 'diabetes':
        data = func(as_frame=True)
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    elif dataset_name == 'iris':
        data = func(as_frame=True)
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
    
    # Create dataset-specific folder
    dataset_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    save_path = os.path.join(dataset_dir, "dataset.pkl")
    df.to_pickle(save_path)
    print(f"Saved {dataset_name} to {save_path}")
    return save_path


def download_file(url, filepath):
    """Download a file from URL to filepath."""
    print(f"Downloading {url}...")
    request.urlretrieve(url, filepath)
    print(f"Downloaded to {filepath}")


def download_uci_url_dataset(dataset_name, dataset_config, save_dir):
    """Download dataset from UCI URL."""
    print(f"\nDownloading {dataset_name} from UCI URL...")
    
    dataset_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    url = dataset_config['url']
    filename = dataset_config['filename']
    filepath = os.path.join(dataset_dir, filename)
    
    # Download if not exists
    if not os.path.exists(filepath):
        download_file(url, filepath)
    else:
        print(f"File already exists: {filepath}")
    
    # Read the data file
    delimiter = dataset_config['delimiter']
    if delimiter == r'\s+':
        df = pd.read_csv(filepath, delimiter=r'\s+', header=dataset_config['header'], engine='python')
    else:
        df = pd.read_csv(filepath, delimiter=delimiter, header=dataset_config['header'])
    
    # Extract only features (exclude target column, which is typically the last column)
    # For libras_movement, the last column is the target
    df_features = df.iloc[:, :-1].copy()
    
    # Create dataset-specific folder
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save as pickle
    pkl_path = os.path.join(dataset_dir, "dataset.pkl")
    df_features.to_pickle(pkl_path)
    print(f"Saved {dataset_name} to {pkl_path}")
    print(f"  Shape: {df_features.shape}")
    print(f"  Dtypes: {df_features.dtypes.value_counts().to_dict()}")
    return pkl_path


def download_ucimlrepo_dataset(dataset_name, dataset_config, save_dir):
    """Download dataset from ucimlrepo and convert dtypes based on variables.type."""
    print(f"\nDownloading {dataset_name} from ucimlrepo...")
    
    dataset = None
    try:
        # Try fetching by ID if available
        if 'id' in dataset_config:
            dataset = fetch_ucirepo(id=dataset_config['id'])
        # Try fetching by name if available
        elif 'name' in dataset_config:
            dataset = fetch_ucirepo(name=dataset_config['name'])
        else:
            raise ValueError(f"No 'id' or 'name' specified for {dataset_name}")
            
    except Exception as e:
        # If ID failed, try name as fallback
        if 'id' in dataset_config and 'name' not in dataset_config:
            try:
                print(f"  Failed with ID {dataset_config['id']}, trying by name...")
                dataset = fetch_ucirepo(name=dataset_name)
            except Exception as e2:
                raise RuntimeError(f"Failed to fetch {dataset_name} by ID or name: {e2}") from e
        else:
            raise RuntimeError(f"Failed to fetch {dataset_name}: {e}") from e
    
    # Handle wine quality datasets (ID 186) - filter by 'color' column, then extract features
    if dataset_config.get('id') == 186:
        # Get the original dataframe - it contains both red and white wine with a 'color' column
        if isinstance(dataset.data.original, pd.DataFrame):
            df_original = dataset.data.original.copy()
        elif isinstance(dataset.data, dict) and 'original' in dataset.data:
            df_original = dataset.data['original'].copy()
        else:
            raise ValueError(f"Could not access data.original for wine quality dataset. Type: {type(dataset.data.original)}")
        
        # Check if 'color' column exists
        if 'color' not in df_original.columns:
            raise ValueError(f"'color' column not found in wine quality dataset. Columns: {df_original.columns.tolist()}")
        
        # Filter by color based on dataset name
        if dataset_name == 'wine_quality_red':
            df_filtered = df_original[df_original['color'] == 'red'].copy()
        elif dataset_name == 'wine_quality_white':
            df_filtered = df_original[df_original['color'] == 'white'].copy()
        else:
            raise ValueError(f"Unexpected dataset name for wine quality: {dataset_name}")
        
        # Get only the feature columns (exclude color and target columns)
        # Use dataset.data.features to get the feature column names
        if isinstance(dataset.data.features, pd.DataFrame):
            feature_columns = dataset.data.features.columns.tolist()
        else:
            raise ValueError(f"Could not access data.features for wine quality dataset. Type: {type(dataset.data.features)}")
        
        # Extract only the feature columns from the filtered dataframe
        df = df_filtered[feature_columns].copy()
    else:
        # Get the features dataframe directly
        if isinstance(dataset.data.features, pd.DataFrame):
            df = dataset.data.features.copy()
        elif isinstance(dataset.data, dict) and 'features' in dataset.data:
            df = dataset.data['features'].copy()
        else:
            raise ValueError(f"Could not access data.features. Type: {type(dataset.data.features)}")
    
    # Get variables dataframe
    variables_df = dataset.variables
    
    # Convert data types based on variables.type
    df = convert_dtypes_from_variables(df, variables_df)
    
    # Create dataset-specific folder
    dataset_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save as pickle
    pkl_path = os.path.join(dataset_dir, "dataset.pkl")
    df.to_pickle(pkl_path)
    print(f"Saved {dataset_name} to {pkl_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Dtypes: {df.dtypes.value_counts().to_dict()}")
    return pkl_path


def main():
    """Download all UCI datasets."""
    # Create datasets/uci directory
    save_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'uci')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Downloading UCI datasets to {save_dir}\n")
    print("=" * 60)
    
    for dataset_name, dataset_config in DATASETS_CONFIG.items():
        try:
            if dataset_config['type'] == 'sklearn':
                download_sklearn_dataset(dataset_name, dataset_config, save_dir)
            elif dataset_config['type'] == 'ucimlrepo':
                download_ucimlrepo_dataset(dataset_name, dataset_config, save_dir)
            elif dataset_config['type'] == 'uci_url':
                download_uci_url_dataset(dataset_name, dataset_config, save_dir)
            else:
                print(f"Unknown dataset type for {dataset_name}")
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Download complete!")


if __name__ == '__main__':
    main()
