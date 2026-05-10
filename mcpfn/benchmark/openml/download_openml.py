import openml
import numpy as np
import torch
import pandas as pd
import os

def fetch_clean_openml_datasets(
    num_datasets: int = 5,
    min_rows: int = 50,
    max_rows: int = 250,
    min_cols: int = 5,
    max_cols: int = 60,
    seed: int = 42,
    verbose: bool = True
):
    """
    Fetch OpenML datasets with no missing values and numeric-only features.
    """
    datasets_df = openml.datasets.list_datasets(output_format='dataframe')
    datasets_df = datasets_df[
        (datasets_df['NumberOfMissingValues'] == 0) &
        (datasets_df['NumberOfFeatures'] >= min_cols) &
        (datasets_df['NumberOfInstances'] >= min_rows) &
        (datasets_df['NumberOfInstances'] <= max_rows) & 
        (datasets_df['NumberOfFeatures'] <= max_cols)
    ]

    np.random.seed(seed)
    dataset_ids = datasets_df.sample(frac=1).index.tolist()  # shuffle
    collected = []
    
    openml.config.timeout = 5 # 5 seconds

    for did in dataset_ids:
        if len(collected) >= num_datasets:
            break
        try:
            dataset = openml.datasets.get_dataset(did)
            if dataset.name[:3] == "fri":
                if verbose:
                    print(f"Skipping {dataset.name} because it's a synthetic dataset")
                continue
            df, _, _, _ = dataset.get_data(dataset_format="dataframe")

            if not all(np.issubdtype(dtype, np.number) for dtype in df.dtypes):
                continue

            X = torch.tensor(df.values, dtype=torch.float32)
            collected.append((X, dataset.name, did))
            if verbose:
                print(f"✅ Loaded: {dataset.name} (ID: {did}, shape: {X.shape})")

        except Exception as e:
            if verbose:
                print(f"⚠️ Skipped ID {did}: {e}")
            continue

    if not collected:
        raise RuntimeError("❌ No valid datasets found under the given constraints.")

    return collected

def fetch_clean_openml_datasets_categorical(
    num_datasets: int = 5,
    min_rows: int = 50,
    max_rows: int = 250,
    min_cols: int = 5,
    max_cols: int = 60,
    seed: int = 42,
    verbose: bool = True
):
    """
    Fetch OpenML datasets with no missing values and mixed features.
    """
    datasets_df = openml.datasets.list_datasets(output_format='dataframe')
    datasets_df = datasets_df[
        (datasets_df['NumberOfMissingValues'] == 0) &
        (datasets_df['NumberOfFeatures'] >= min_cols) &
        (datasets_df['NumberOfInstances'] >= min_rows) &
        (datasets_df['NumberOfInstances'] <= max_rows) &
        (datasets_df['NumberOfFeatures'] <= max_cols) &
        (datasets_df['NumberOfSymbolicFeatures'] > 1)
    ]

    np.random.seed(seed)
    dataset_ids = datasets_df.sample(frac=1).index.tolist()  # shuffle
    collected = []
    
    openml.config.timeout = 5 # 5 seconds

    for did in dataset_ids:
        if len(collected) >= num_datasets:
            break
        try:
            dataset = openml.datasets.get_dataset(did)
            if dataset.name[:3] == "fri":
                if verbose:
                    print(f"Skipping {dataset.name} because it's a synthetic dataset")
                continue
            df, _, _, _ = dataset.get_data(dataset_format="dataframe")
            
            allowed_dtypes = [np.number, pd.CategoricalDtype]
            
            for dtype in df.dtypes:
                if dtype not in allowed_dtypes:
                    continue
                
            if dataset.description is None:
                continue
            
            # remove object columns
            df = df.select_dtypes(exclude=['object'])
            
            # convert numeric columns to float32
            df[df.select_dtypes(include=['number']).columns] = df.select_dtypes(include=['number']).astype(np.float32)
            df[df.select_dtypes(include=['float']).columns] = df.select_dtypes(include=['float']).astype(np.float32)
            
            # X = torch.tensor(df.values, dtype=torch.float32 if all(np.issubdtype(dtype, np.number) for dtype in df.dtypes) else torch.object)
            collected.append((df, dataset.name, did))
            if verbose:
                print(f"✅ Loaded: {dataset.name} (ID: {did}, shape: {df.shape})")

        except Exception as e:
            if verbose:
                print(f"⚠️ Skipped ID {did}: {e}")
            continue

    if not collected:
        raise RuntimeError("❌ No valid datasets found under the given constraints.")

    return collected


if __name__ == "__main__":
    datasets = fetch_clean_openml_datasets_categorical(num_datasets=100, verbose=True)
    
    for df, name, did in datasets:
        # save the dataframe to a pickle
        # Create the directory if it doesn't exist
        os.makedirs(f"datasets/openml_categorical/{name}", exist_ok=True)
        df.to_pickle(f"datasets/openml_categorical/{name}/dataframe.pkl")
    
    # # calculate the percentage of missing values
    # for X, name, did in datasets:
    #     with open("dataset_sizes_missing.txt", "a") as f:
    #         f.write(f"{name} | {torch.mean(torch.isnan(X).float())}\n")