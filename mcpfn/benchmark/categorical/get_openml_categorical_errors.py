import os
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tabimpute.categorical_adapter_interface import TabImputeCategoricalAdapter
from tabimpute.interface import (
    ImputePFN,
    TabImputeCategorical,
    TabPFNImputer,
    TabPFNUnsupervisedModel,
)
from tabimpute.tabimpute_v2 import TabImputeV2
from tabimpute.prepreocess import (
    RandomColumnPermutation,
    RandomRowColumnPermutation,
    RandomRowPermutation,
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
    # "mcpfn",
    "mcpfn_adapter",
    # "mcpfn_ensemble",
    # "tabimpute_ensemble",
    # "knn",
    # "tabpfn",
    # "tabpfn_unsupervised",
    # "hyperimpute_mode",
    # "softimpute",
    # "hyperimpute_ot", # Sinkhorn / Optimal Transport
    # "hyperimpute",
    # "hyperimpute_missforest",
    # "hyperimpute_ice",
    # "hyperimpute_mice",
    # "hyperimpute_gain",
    # "hyperimpute_miwae",
    # "forestdiffusion",
    # "diffputer",
    # "remasker",
])

# --- Initialize classes once ---
ROUND2_T2_CHECKPOINT = (
    "/home/jacobf18/tabular/mcpfn/src/tabimpute/workdir/"
    "tabimpute-round2-t2/checkpoint_10000.pth"
)


def build_round2_t2_preprocessors():
    """Fresh instances per imputer (fitted state must not be shared)."""
    return [
        RandomRowColumnPermutation(),
        RandomRowColumnPermutation(),
        RandomRowPermutation(),
        RandomColumnPermutation(),
    ]


# TabImputeCategoricalAdapter hyperparameters (test-time heads on a frozen backbone).
#
# The legacy one-hot path runs the full TabImpute forward on an expanded numeric matrix,
# so it can look much stronger if adapter training is under-powered. Very small learning
# rates (e.g. 2e-5) with only ~100 steps barely move the adapter weights away from the
# scalar-encoder bootstrap init — use AdamW-style rates (1e-3 … 1e-2) and enough steps.
#
# Tune knobs (rough guidance):
# - max_steps: 200–800 depending on n_rows and number of categorical columns
# - lr: 1e-3 to 1e-2; if loss oscillates, lower lr or add weight_decay
# - mask_prob: 0.25–0.5 (higher = harder masked-label recovery, often better if stable)
# - batch_rows: None = use all observed rows for masking each step; set e.g. 512–2048
#   on large tables for speed (noisier gradients)
# - weight_decay: 1e-5 … 1e-3 for light regularization on the small heads
ADAPTER_MAX_STEPS = 400
ADAPTER_LR = 3e-3
ADAPTER_MASK_PROB = 0.35
ADAPTER_WEIGHT_DECAY = 1e-4
ADAPTER_BATCH_ROWS = None  # e.g. 1024 for large datasets / faster epochs
ADAPTER_RANDOM_STATE = 0
ADAPTER_VERBOSE = True

if "mcpfn" in imputers:
    mcpfn_numeric = TabImputeV2(
        device="cuda",
        checkpoint_path=ROUND2_T2_CHECKPOINT,
        preprocessors=build_round2_t2_preprocessors(),
    )
    mcpfn = TabImputeCategorical(
        device="cuda",
        imputer=mcpfn_numeric,
    )
    mcpfn_name = "tabimpute_round2_t2"

if "mcpfn_adapter" in imputers:
    mcpfn_adapter = TabImputeCategoricalAdapter(
        device="cuda",
        checkpoint_path=ROUND2_T2_CHECKPOINT,
        preprocessors=build_round2_t2_preprocessors(),
    )
    mcpfn_adapter_name = "tabimpute_round2_t2_adapter"
    
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


def dataframe_to_adapter_array(df: pd.DataFrame) -> np.ndarray:
    # Preserve arbitrary categorical Python values while converting pandas
    # missing markers into a form the adapter recognizes.
    return df.astype(object).where(pd.notna(df), np.nan).to_numpy(dtype=object)


def restore_categorical_columns(
    df_imputed: pd.DataFrame,
    reference_df: pd.DataFrame,
    categorical_indices: np.ndarray,
) -> pd.DataFrame:
    for col_idx in categorical_indices:
        col_name = reference_df.columns[col_idx]
        categories = reference_df[col_name].cat.categories
        df_imputed[col_name] = pd.Categorical(
            df_imputed[col_name],
            categories=categories,
        )
    return df_imputed

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
        if os.path.exists(f"{base_path}/{name}/dataframe_imputed_{mcpfn_name}.pkl") and not force_rerun:
            continue
        try:
            df_imputed_mcpfn = mcpfn.impute(
                dataframe_to_adapter_array(df_missing),
                categorical_columns=categorical_indices.tolist(),
            )
            df_imputed_mcpfn = pd.DataFrame(df_imputed_mcpfn, columns=df.columns)
            df_imputed_mcpfn = restore_categorical_columns(
                df_imputed_mcpfn,
                reference_df=df,
                categorical_indices=categorical_indices,
            )
            df_imputed_mcpfn.to_pickle(f"{base_path}/{name}/dataframe_imputed_{mcpfn_name}.pkl")
        except Exception as e:
            print(f"Error running MCPFN: {e}")
            continue

    if "mcpfn_adapter" in imputers:
        print("Running MCPFN adapter...")
        if (
            os.path.exists(f"{base_path}/{name}/dataframe_imputed_{mcpfn_adapter_name}.pkl")
            and not force_rerun
        ):
            continue
        try:
            X_missing_adapter = dataframe_to_adapter_array(df_missing)
            X_imputed_mcpfn_adapter, _ = mcpfn_adapter.impute_categorical_columns(
                X_missing_adapter,
                target_cols=categorical_indices.tolist(),
                max_steps=ADAPTER_MAX_STEPS,
                mask_prob=ADAPTER_MASK_PROB,
                lr=ADAPTER_LR,
                weight_decay=ADAPTER_WEIGHT_DECAY,
                batch_rows=ADAPTER_BATCH_ROWS,
                random_state=ADAPTER_RANDOM_STATE,
                verbose=ADAPTER_VERBOSE,
            )
            df_imputed_mcpfn_adapter = pd.DataFrame(
                X_imputed_mcpfn_adapter, columns=df.columns
            )
            df_imputed_mcpfn_adapter = restore_categorical_columns(
                df_imputed_mcpfn_adapter,
                reference_df=df,
                categorical_indices=categorical_indices,
            )
            df_imputed_mcpfn_adapter.to_pickle(
                f"{base_path}/{name}/dataframe_imputed_{mcpfn_adapter_name}.pkl"
            )
        except Exception as e:
            print(f"Error running MCPFN adapter: {e}")
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