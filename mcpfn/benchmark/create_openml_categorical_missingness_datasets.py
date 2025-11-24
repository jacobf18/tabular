import torch
import numpy as np
from tabimpute.prior.training_set_generation import (
    MCARPattern,
)
from tabimpute.model.encoders import normalize_data
import os
import shutil
import pandas as pd

# --- Fetch datasets ---
# datasets = fetch_clean_openml_datasets(num_datasets=100, verbose=False)

# --- Define missingness patterns ---
p_mcar = 0.4

base_path = "datasets/openml_categorical"

# # outpu  out dataset sizes
# for X, name, did in datasets:
#     print(f"{name} | {X.shape[0]} \\times {X.shape[1]}")
#     with open("dataset_sizes.txt", "a") as f:
#         f.write(f"{name} | {X.shape[0]} \\times {X.shape[1]}\n")

max_attempts = 10
# print(datasets)
# --- Run benchmark ---
# for X, name, did in datasets:
for name in os.listdir(base_path):
    df = pd.read_pickle(f"{base_path}/{name}/dataframe.pkl").copy()
    
    # induce MCAR missingness with p_mcar
    # X[torch.rand(*X.shape) < self.config.get("p_missing", 0.6)] = torch.nan
    mask = pd.DataFrame(np.random.rand(*df.shape) < p_mcar, index=df.index, columns=df.columns)
    df = df.mask(mask, np.nan)
    
    # save the dataframe to a pickle 
    df.to_pickle(f"{base_path}/{name}/dataframe_missing.pkl")