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

# --- Fetch datasets ---
# datasets = fetch_clean_openml_datasets(num_datasets=100, verbose=False)

# --- Define missingness patterns ---
p_mcar = 0.5

patterns = {
    "MCAR": MCARPattern(config={"p_missing": p_mcar}),
    # "MAR": MARPattern(config={"p_missing": p_mar}),
    # "MNAR": MNARPattern(config={"p_missing": p_mnar}),
    # "MAR_Neural": MARNeuralNetwork(config={
    #     "p_missing": p_mar, 
    #     'mar_config': {
    #     "num_layers_upper": 3,
    #     "hidden_lower": 1,
    #     "hidden_upper": 100,
    #     "activation": "relu",
    #     "seed": 42,
    #     "neighbor_type": "random"
    # }}),
    # "MAR_BlockNeural": MARBlockNeuralNetwork(config={
    #     'p_missing': p_mar,
    #     'mar_block_config': {
    #         "N": 100,
    #         "T": 50,
    #         "row_blocks": 10,
    #         "col_blocks": 10,
    #         "convolution_type": "mean"
    #     }
    # }),
    # "MAR_Sequential": MARSequentialBandit(config={
    #     'p_missing': p_mar,
    #     'mar_sequential_bandit_config': {
    #         'algorithm': 'epsilon_greedy',
    #         'pooling': False, 
    #         'epsilon': 0.4, 
    #         'epsilon_decay': 0.99, 
    #         'random_seed': 42
    #     }
    # }),
    # 'MNARPanelPattern': MNARPanelPattern(config={}),
    # 'MNARSequentialPattern': MNARSequentialPattern(config={'n_policies': 2}),
    # 'MNARPolarizationPattern': MNARPolarizationPattern(config={'threshold_quantile': 0.25}),
    # 'MNARSoftPolarizationPattern': MNARSoftPolarizationPattern(config={'soft_polarization_alpha': 2.5, 'soft_polarization_epsilon': 0.05}),
    # 'MNARLatentFactorPattern': MNARLatentFactorPattern(config={'latent_rank_low': 1, 'latent_rank_high': 5}),
    # 'MNARPositivityViolationPattern': MNARPositivityViolationPattern(config={}),
    # 'MNARClusterLevelPattern': MNARClusterLevelPattern(config={'cluster_level_n_row_clusters': 5, 'cluster_level_n_col_clusters': 4}),
    # 'MNARCensoringPattern': MNARCensoringPattern(config={'censor_quantile': 0.25}),
    # 'MNARTwoPhaseSubsetPattern': MNARTwoPhaseSubsetPattern(config={'two_phase_cheap_fraction': 0.4}),
    # "MAR_Diffusion": MARDiffusion(config={
    #     'missingness_type': 'bandit',
    #     'device': 'cuda',
    #     'target_shape': None,
    #     'num_samples': 1
    # }),
}

base_path = "datasets/uci"
num_repeats = 10

for name in os.listdir(base_path):
    for p in [0.1, 0.2, 0.3, 0.4, 0.5]:
        pattern = MCARPattern(config={"p_missing": p})
        for repeat in range(num_repeats):
            path = f"{base_path}/{name}/MCAR/missingness-{p}/repeat-{repeat}"
            os.makedirs(path, exist_ok=True)
            
            df = pd.read_pickle(f"{base_path}/{name}/dataset.pkl")
            X_npy = df.to_numpy()
            X_npy = X_npy.astype(np.float32)
            
            X_tensor = torch.from_numpy(X_npy).clone()
            
            X_missing = pattern._induce_missingness(X_tensor)
            
            # Save the missingness dataset
            np.save(f"{path}/missing.npy", X_missing.numpy())
            np.save(f"{path}/true.npy", X_npy)