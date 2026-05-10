import torch
import numpy as np
import pandas as pd
from tabimpute.prior.training_set_generation import (
    MCARPattern, 
    MARPattern, 
    MNARPattern, 
    MARNeuralNetwork,
    MARBlockNeuralNetwork,
    MARSequentialBandit,
    MNARPanelPattern,
    MNARSequentialPattern,
    MNARPolarizationPattern,
    MNARSoftPolarizationPattern,
    MNARLatentFactorPattern,
    MNARPositivityViolationPattern,
    MNARClusterLevelPattern,
    MNARSpatialBlockPattern,
    MNARCensoringPattern,
    MNARTwoPhaseSubsetPattern,
    MNARSkipLogicPattern
)
from download_openml import fetch_clean_openml_datasets
from tabimpute.model.encoders import normalize_data
import os
import shutil
from tqdm import tqdm

# --- Fetch datasets ---
# datasets = fetch_clean_openml_datasets(num_datasets=100, verbose=False)

# --- Define missingness patterns ---
p_missing = [0.1, 0.2, 0.3, 0.4, 0.5]

patterns = [
    # MCARPattern,
    MARPattern,
    MNARPattern,
]

base_path = "datasets/uci"
num_repeats = 10

for name in tqdm(os.listdir(base_path)):
    for p in tqdm(p_missing):
        for pattern in patterns:
            inducer = pattern(config={"p_missing": p})
        
            for repeat in range(num_repeats):
                path = f"{base_path}/{name}/{str(inducer)}/missingness-{p}/repeat-{repeat}"
                os.makedirs(path, exist_ok=True)
                
                df = pd.read_pickle(f"{base_path}/{name}/dataset.pkl")
                X_npy = df.to_numpy()
                X_npy = X_npy.astype(np.float32)
                
                X_tensor = torch.from_numpy(X_npy).clone()
                
                X_missing = inducer._induce_missingness(X_tensor)
                
                # Save the missingness dataset
                np.save(f"{path}/missing.npy", X_missing.numpy())
                np.save(f"{path}/true.npy", X_npy)