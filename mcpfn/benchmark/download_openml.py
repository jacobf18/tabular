import openml
import numpy as np
import torch

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
        (datasets_df['NumberOfInstances'] <= max_rows)
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
