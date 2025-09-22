import torch
import numpy as np
from mcpfn.prior.training_set_generation import (
    MCARPattern, 
    MARPattern, 
    MNARPattern, 
    MARNeuralNetwork,
    MARBlockNeuralNetwork,
    MARSequentialBandit,
    MARDiffusion,
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
from mcpfn.model.encoders import normalize_data
import os
import shutil

# --- Fetch datasets ---
datasets = fetch_clean_openml_datasets(num_datasets=100, verbose=False)

# --- Define missingness patterns ---
p_mcar = 0.4
p_mar = 0.4
p_mnar = 0.4

patterns = {
    # "MCAR": MCARPattern(config={"p_missing": p_mcar}),
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
    'MNARSequentialPattern': MNARSequentialPattern(config={'n_policies': 2}),
    # 'MNARPolarizationPattern': MNARPolarizationPattern(config={'threshold_quantile': 0.25}),
    # 'MNARSoftPolarizationPattern': MNARSoftPolarizationPattern(config={'soft_polarization_alpha': 2.5, 'soft_polarization_epsilon': 0.05}),
    # 'MNARLatentFactorPattern': MNARLatentFactorPattern(config={'latent_rank_low': 1, 'latent_rank_high': 5}),
    # 'MNARPositivityViolationPattern': MNARPositivityViolationPattern(config={}),
    # 'MNARClusterLevelPattern': MNARClusterLevelPattern(config={'cluster_level_n_row_clusters': 5, 'cluster_level_n_col_clusters': 4}),
    # 'MNARCensoringPattern': MNARCensoringPattern(config={'censor_quantile': 0.1}),
    # 'MNARTwoPhaseSubsetPattern': MNARTwoPhaseSubsetPattern(config={'two_phase_cheap_fraction': 0.4}),
    # "MAR_Diffusion": MARDiffusion(config={
    #     'missingness_type': 'bandit',
    #     'device': 'cuda',
    #     'target_shape': None,
    #     'num_samples': 1
    # }),
}

base_path = "datasets/openml"

# outpu  out dataset sizes
for X, name, did in datasets:
    print(f"{name} | {X.shape[0]} \\times {X.shape[1]}")
    with open("dataset_sizes.txt", "a") as f:
        f.write(f"{name} | {X.shape[0]} \\times {X.shape[1]}\n")

max_attempts = 10
# --- Run benchmark ---
for X, name, did in datasets:
    for pattern_name, pattern in patterns.items():
        num_attempts = 0
        while num_attempts < max_attempts:
            num_attempts += 1
            # print(f"Running {name} | {pattern_name}")
            torch.cuda.empty_cache()
            X_missing = pattern._induce_missingness(X.clone())
            
            # Normalize the data (after inducing missingness)
            X_missing, (mean, std) = normalize_data(X_missing, return_scaling=True)
            if X_missing.shape[0] == 0:
                continue
            
            # std is set to 1 if all values are the same
            X_normalized = (X - mean) / std
            
            # p = p_mcar if pattern_name == "MCAR" else p_mar if pattern_name == "MAR" else p_mnar
            # p = pattern.config['p_missing']
            
            # Create the directory if it doesn't exist
            print(f"{base_path}/{name}/{pattern_name}_{p}")
            os.makedirs(f"{base_path}/{name}/{pattern_name}_{p}", exist_ok=True)
            
            # Save the missingness dataset
            np.save(f"{base_path}/{name}/{pattern_name}_{p}/missing.npy", X_missing.numpy())
            np.save(f"{base_path}/{name}/{pattern_name}_{p}/true.npy", X_normalized.numpy())
            
            # Save the mean and std
            np.save(f"{base_path}/{name}/{pattern_name}_{p}/mean.npy", mean)
            np.save(f"{base_path}/{name}/{pattern_name}_{p}/std.npy", std)
            break