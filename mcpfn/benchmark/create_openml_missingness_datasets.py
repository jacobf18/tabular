import torch
import numpy as np
from mcpfn.prior.training_set_generation import MCARPattern, MARPattern, MNARPattern
from download_openml import fetch_clean_openml_datasets
from mcpfn.model.encoders import normalize_data
import os

# --- Fetch datasets ---
datasets = fetch_clean_openml_datasets(num_datasets=4)

# --- Define missingness patterns ---
p_mcar = 0.4
p_mar = 0.4
p_mnar = 0.4

patterns = {
    "MCAR": MCARPattern(config={"p_missing": p_mcar}),
    "MAR": MARPattern(config={"p_missing": p_mar}),
    "MNAR": MNARPattern(config={"p_missing": p_mnar}),
}

base_path = "datasets/openml"

# --- Run benchmark ---
for X, name, did in datasets:
    for pattern_name, pattern in patterns.items():
        print(f"Running {name} | {pattern_name}")
        torch.cuda.empty_cache()
        X_missing = pattern._induce_missingness(X.clone())
        
        # Normalize the data (after inducing missingness)
        X_missing, (mean, std) = normalize_data(X_missing, return_scaling=True)
        # std is set to 1 if all values are the same
        X_normalized = (X - mean) / std
        
        p = p_mcar if pattern_name == "MCAR" else p_mar if pattern_name == "MAR" else p_mnar
        
        # Create the directory if it doesn't exist
        os.makedirs(f"{base_path}/{name}", exist_ok=True)
        
        # Save the missingness dataset
        np.save(f"{base_path}/{name}/{pattern_name}_{p}_missing.npy", X_missing.numpy())
        np.save(f"{base_path}/{name}/{pattern_name}_{p}_true.npy", X_normalized.numpy())
        
        # Save the mean and std
        np.save(f"{base_path}/{name}/{pattern_name}_{p}_mean.npy", mean)
        np.save(f"{base_path}/{name}/{pattern_name}_{p}_std.npy", std)