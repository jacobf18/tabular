import os
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tabimpute.interface import ImputePFN, TabPFNImputer, TabPFNUnsupervisedModel, MCTabPFNEnsemble, TabImputeEnsemble, TabImputeRouter, TabImputeCategorical
from tabimpute.prepreocess import (
    RandomRowColumnPermutation, 
    RandomRowPermutation, 
    RandomColumnPermutation, 
)
from hyperimpute.plugins.imputers import Imputers
from sklearn.impute import KNNImputer
import time

# Optional: ForestDiffusion (pip install ForestDiffusion)
try:
    from ForestDiffusion import ForestDiffusionModel
    HAS_FORESTDIFFUSION = True
except ImportError:
    HAS_FORESTDIFFUSION = False
import shutil
from diffputer_wrapper import DiffPuterImputer
from remasker_wrapper import ReMaskerImputer


repeats = 1

os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

warnings.filterwarnings("ignore")

force_rerun = True

# --- Choose which imputers to run ---
imputers = set([
    "mcpfn",
    # "mcpfn_ensemble",
    # "tabimpute_ensemble",
    # "knn",
    # "tabpfn",
    # "tabpfn_unsupervised",
    # "hyperimpute_mode",
    # "softimpute",
    # "hyperimpute_ot", # Sinkhorn / Optimal Transport
    "hyperimpute",
    "hyperimpute_missforest",
    # "hyperimpute_ice",
    # "hyperimpute_mice",
    # "hyperimpute_gain",
    # "hyperimpute_miwae",
    # "forestdiffusion",
    # "diffputer",
    # "remasker",
])

# --- Initialize classes once ---
if "mcpfn" in imputers:
    preprocessors = [
        RandomRowColumnPermutation(),
        RandomRowColumnPermutation(),
        RandomRowPermutation(),
        RandomColumnPermutation(),
        # StandardizeWhiten(whiten=True),
    ]
    
    mcpfn = TabImputeCategorical(
        device="cuda",
        # checkpoint_path="/home/jacobf18/mcpfn_data/checkpoints/mnar_fixed/step-10000.ckpt",
        # checkpoint_path="/home/jacobf18/mcpfn_data/checkpoints/mixed_mcar_mar_mnar/step-13500.ckpt",
        # checkpoint_path="/home/jacobf18/mcpfn_data/checkpoints/mixed_mcar_mar_mnar_reweighted_zscore/step-85000.ckpt",
        # checkpoint_path="/home/jacobf18/mcpfn_data/checkpoints/mixed_mcar_mar_mnar_gradnorm/step-41000.ckpt",
        checkpoint_path="/home/jacobf18/mcpfn_data/checkpoints/masters/mcar/step-117000.ckpt",
        # checkpoint_path="/home/jacobf18/mcpfn_data/checkpoints/masters/mcar_nonlinear/step-100000.ckpt",
        # checkpoint_path="/home/jacobf18/mcpfn_data/checkpoints/masters/mar/step-40000.ckpt",
        # checkpoint_path="/home/jacobf18/mcpfn_data/checkpoints/masters/mnar/step-40000.ckpt",
        # checkpoint_path = "/mnt/mcpfn_data/checkpoints/mixed_nonlinear/step-7000.ckpt",
        # checkpoint_path = "/mnt/mcpfn_data/checkpoints/mar_batch_size_64/step-49900.ckpt",
        # checkpoint_path = "/mnt/mcpfn_data/checkpoints/mixed_adaptive_more_heads/step-100000.ckpt",
        nhead=2,
        preprocessors=preprocessors
    )
    mcpfn_name = "masters_mcar"
    
if "tabpfn" in imputers:
    tabpfn = TabPFNImputer(device="cuda")
    
if "tabpfn_unsupervised" in imputers:
    tabpfn_unsupervised = TabPFNUnsupervisedModel(device="cuda")
    
if "diffputer" in imputers:
    diffputer = DiffPuterImputer(device="cuda")

# Map friendly names → HyperImpute plugin ids and output filenames
HYPERIMPUTE_MAP = {
    "hyperimpute": ("hyperimpute", "hyperimpute"),
    "hyperimpute_ot": ("sinkhorn", "ot_sinkhorn"),
    "hyperimpute_missforest": ("missforest", "missforest"),
    "hyperimpute_ice": ("ice", "ice"),
    "hyperimpute_mice": ("mice", "mice"),
    "hyperimpute_em": ("em", "em"),
    "hyperimpute_gain": ("gain", "gain"),
    "hyperimpute_miracle": ("miracle", "miracle"),
    "hyperimpute_miwae": ("miwae", "miwae"),
    "hyperimpute_mean": ("mean", "column_mean"),
    "hyperimpute_mode": ("mode", "mode"),
}

def fill_all_nan_columns(X_missing: np.ndarray) -> np.ndarray:
    """Fill columns that are entirely NaN with 0's, leaving other NaN values unchanged."""
    X_processed = X_missing.copy()
    for col_idx in range(X_processed.shape[1]):
        if np.all(np.isnan(X_processed[:, col_idx])):
            X_processed[:, col_idx] = 0.0
    return X_processed

def fill_all_nan_rows(X_missing: np.ndarray) -> np.ndarray:
    """Fill rows that are entirely NaN with column means"""
    X_processed = X_missing.copy()
    for row_idx in range(X_processed.shape[0]):
        if np.all(np.isnan(X_processed[row_idx, :])):
            X_processed[row_idx, :] = np.nanmean(X_processed, axis=0)
    return X_processed

def run_hyperimpute(plugin_name: str, X_missing: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Run a single HyperImpute plugin on X_missing (numpy with NaNs)."""
    # Fill all-NaN columns with 0's before running hyperimpute
    X_processed = fill_all_nan_rows(fill_all_nan_columns(X_missing))
    plugin = Imputers().get(plugin_name, random_state = random_state)
    # HyperImpute expects pandas DataFrame and returns a DataFrame
    out_df = plugin.fit_transform(pd.DataFrame(X_processed))
    return out_df.to_numpy()

def run_forest_diffusion(X_missing: np.ndarray, n_t: int = 50,
                         cat_indexes=None, int_indexes=None,
                         n_jobs: int = -1, repaint=False, num_repeats: int = 1) -> np.ndarray:
    """ForestDiffusion imputation wrapper (numeric defaults)."""
    if not HAS_FORESTDIFFUSION:
        raise RuntimeError("ForestDiffusion not installed. `pip install ForestDiffusion`.")
    if cat_indexes is None: cat_indexes = []
    if int_indexes is None: int_indexes = []
    # Per authors, vp diffusion is required for imputation.
    model = ForestDiffusionModel(
        fill_all_nan_rows(fill_all_nan_columns(X_missing)),
        n_t=n_t,
        diffusion_type="vp",
        cat_indexes=cat_indexes,
        int_indexes=int_indexes,
        n_jobs=n_jobs,
    )
    if repaint:
        return model.impute(repaint=True, r=10, j=max(1, n_t // 10), k=num_repeats)
    return model.impute(k=num_repeats)

# --- Run benchmark ---
base_path = "datasets/openml_categorical"
datasets = os.listdir(base_path)

force_rerun = False

pbar = tqdm(datasets)
for name in pbar:
    pbar.set_description(f"Running {name}")
    configs = os.listdir(f"{base_path}/{name}")
    
    df = pd.read_pickle(f"{base_path}/{name}/dataframe.pkl")
    df_missing = pd.read_pickle(f"{base_path}/{name}/dataframe_missing.pkl")
    
    # Get categorical column indices
    categorical_columns = df_missing.select_dtypes(include=['category']).columns
    categorical_indices = df.columns.get_indexer(categorical_columns)
    
    if "mcpfn" in imputers:
        print("Running MCPFN...")
        if os.path.exists(f"{base_path}/{name}/dataframe_imputed_mcpfn.pkl") and not force_rerun:
            continue
        try:
            df_imputed_mcpfn = mcpfn.impute(df_missing.to_numpy(), categorical_columns=categorical_indices)
            df_imputed_mcpfn = pd.DataFrame(df_imputed_mcpfn, columns=df.columns)
            df_imputed_mcpfn.to_pickle(f"{base_path}/{name}/dataframe_imputed_mcpfn.pkl")
        except Exception as e:
            print(f"Error running MCPFN: {e}")
            continue
    
    if "hyperimpute" in imputers:
        print("Running HyperImpute...")
        if os.path.exists(f"{base_path}/{name}/dataframe_imputed_hyperimpute.pkl") and not force_rerun:
            continue
        # Convert categorical columns to object type to avoid category errors during imputation
        df_missing_processed = df_missing.copy()
        for col in categorical_columns:
            df_missing_processed[col] = df_missing_processed[col].astype('object')
        
        plugin = Imputers().get('hyperimpute')
        imputed_hyperimpute = plugin.fit_transform(df_missing_processed)
        imputed_hyperimpute.to_pickle(f"{base_path}/{name}/dataframe_imputed_hyperimpute.pkl")
    
    if "hyperimpute_missforest" in imputers:
        print("Running MissForest...")
        if os.path.exists(f"{base_path}/{name}/dataframe_imputed_missforest.pkl") and not force_rerun:
            continue
        # Convert categorical columns to object type to avoid category errors during imputation
        df_missing_processed = df_missing.copy()
        for col in categorical_columns:
            df_missing_processed[col] = df_missing_processed[col].astype('object')
        
        plugin = Imputers().get('missforest')
        imputed_missforest = plugin.fit_transform(df_missing_processed)
        imputed_missforest.to_pickle(f"{base_path}/{name}/dataframe_imputed_missforest.pkl")
        
    if "hyperimpute_mice" in imputers:
        print("Running MICE...")
        if os.path.exists(f"{base_path}/{name}/dataframe_imputed_mice.pkl") and not force_rerun:
            continue
        # Convert categorical columns to object type to avoid category errors during imputation
        df_missing_processed = df_missing.copy()
        for col in categorical_columns:
            df_missing_processed[col] = df_missing_processed[col].astype('object')
        
        plugin = Imputers().get('mice')
        imputed_mice = plugin.fit_transform(df_missing_processed)
        imputed_mice.to_pickle(f"{base_path}/{name}/dataframe_imputed_mice.pkl")
        
    if "hyperimpute_gain" in imputers:
        print("Running GAIN...")
        if os.path.exists(f"{base_path}/{name}/dataframe_imputed_gain.pkl") and not force_rerun:
            continue
        # Convert categorical columns to object type to avoid category errors during imputation
        df_missing_processed = df_missing.copy()
        for col in categorical_columns:
            df_missing_processed[col] = df_missing_processed[col].astype('object')
        
        plugin = Imputers().get('gain')
        imputed_gain = plugin.fit_transform(df_missing_processed)
        imputed_gain.to_pickle(f"{base_path}/{name}/dataframe_imputed_gain.pkl")
        
    if "hyperimpute_ot" in imputers:
        print("Running OT...")
        if os.path.exists(f"{base_path}/{name}/dataframe_imputed_ot.pkl") and not force_rerun:
            continue
        # Convert categorical columns to object type to avoid category errors during imputation
        df_missing_processed = df_missing.copy()
        for col in categorical_columns:
            df_missing_processed[col] = df_missing_processed[col].astype('object')
        
        plugin = Imputers().get('sinkhorn')
        imputed_ot = plugin.fit_transform(df_missing_processed)
        imputed_ot.to_pickle(f"{base_path}/{name}/dataframe_imputed_ot.pkl")
        
    if "hyperimpute_mode" in imputers:
        print("Running Mode...")
        if os.path.exists(f"{base_path}/{name}/dataframe_imputed_mode.pkl") and not force_rerun:
            continue
        # Convert categorical columns to object type to avoid category errors during imputation
        df_missing_processed = df_missing.copy()
        for col in categorical_columns:
            df_missing_processed[col] = df_missing_processed[col].astype('object')
        
        plugin = Imputers().get('most_frequent')
        imputed_mode = plugin.fit_transform(df_missing_processed)
        imputed_mode.to_pickle(f"{base_path}/{name}/dataframe_imputed_mode.pkl")
    
    # if "mcpfn" in imputers:
    #     df_imputed_mcpfn = mcpfn.impute(df_missing.to_numpy(), categorical_columns=categorical_indices)
        
    #     break
    # if "tabpfn" in imputers:
    #     df_imputed_tabpfn = tabpfn.impute(df_missing.to_numpy())
    #     print(df_imputed_tabpfn)
    #     break
    # if "tabpfn_unsupervised" in imputers:
    #     df_imputed_tabpfn_unsupervised = tabpfn_unsupervised.impute(df_missing.to_numpy())
    #     print(df_imputed_tabpfn_unsupervised)
    #     break
    # if "diffputer" in imputers:
    #     df_imputed_diffputer = diffputer.impute(df_missing.to_numpy())
    #     print(df_imputed_diffputer)