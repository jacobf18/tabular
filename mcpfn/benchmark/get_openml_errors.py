import os
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tabimpute.interface import ImputePFN, TabPFNImputer, TabPFNUnsupervisedModel, MCTabPFNEnsemble, TabImputeEnsemble, TabImputeRouter
from tabimpute.prepreocess import (
    RandomRowColumnPermutation, 
    RandomRowPermutation, 
    RandomColumnPermutation, 
)
from hyperimpute.plugins.imputers import Imputers
from sklearn.impute import KNNImputer
import time
import itertools

# Optional: ForestDiffusion (pip install ForestDiffusion)
try:
    from ForestDiffusion import ForestDiffusionModel
    HAS_FORESTDIFFUSION = True
except ImportError:
    HAS_FORESTDIFFUSION = False
import shutil
from diffputer_wrapper import DiffPuterImputer
from remasker_wrapper import ReMaskerImputer
from cacti_wrapper import CACTIImputer
from remasker.remasker_impute import ReMasker

os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

warnings.filterwarnings("ignore")

force_rerun = False

# --- Choose which imputers to run ---
imputers = set([
    # "mcpfn",
    # "mcpfn_ensemble",
    # "tabimpute_ensemble",
    # "knn",
    "tabpfn",
    # "tabpfn_unsupervised",
    # "hyperimpute_mean",
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
    # "cacti",
])

patterns = {
    "MCAR",
    # "MAR",
    # "MNAR",
    # "MAR_Neural",
    # "MAR_BlockNeural",
    # "MAR_Sequential",
    # "MNARPanelPattern",
    # "MNARPolarizationPattern",
    # "MNARSoftPolarizationPattern",
    # "MNARLatentFactorPattern",
    # "MNARClusterLevelPattern",
    # "MNARTwoPhaseSubsetPattern",
    # "MNARCensoringPattern",
}

missingness_levels = [
    # 0.1, 0.2, 
    # 0.3, 
    0.4, 
    # 0.5
]

num_repeats = 1

# --- Initialize classes once ---
if "mcpfn" in imputers:
    preprocessors = [
        RandomRowColumnPermutation(),
        RandomRowColumnPermutation(),
        RandomRowPermutation(),
        RandomColumnPermutation(),
        # StandardizeWhiten(whiten=True),
    ]
    
    mcpfn = ImputePFN(
        device="cuda",
        # checkpoint_path="/home/jacobf18/mcpfn_data/checkpoints/mnar_fixed/step-10000.ckpt",
        # checkpoint_path="/home/jacobf18/mcpfn_data/checkpoints/mixed_mcar_mar_mnar/step-13500.ckpt",
        # checkpoint_path="/home/jacobf18/mcpfn_data/checkpoints/mixed_mcar_mar_mnar_reweighted_zscore/step-85000.ckpt",
        # checkpoint_path="/home/jacobf18/mcpfn_data/checkpoints/mixed_mcar_mar_mnar_gradnorm/step-41000.ckpt",
        # checkpoint_path="/home/jacobf18/mcpfn_data/checkpoints/masters/mcar/step-117000.ckpt",
        # checkpoint_path="/home/jacobf18/mcpfn_data/checkpoints/masters/mcar_nonlinear/step-100000.ckpt",
        # checkpoint_path="/home/jacobf18/mcpfn_data/checkpoints/masters/mar/step-40000.ckpt",
        # checkpoint_path="/home/jacobf18/mcpfn_data/checkpoints/masters/mnar/step-40000.ckpt",
        # checkpoint_path = "/mnt/mcpfn_data/checkpoints/mixed_nonlinear/step-7000.ckpt",
        # checkpoint_path = "/mnt/mcpfn_data/checkpoints/mar_batch_size_64/step-49900.ckpt",
        # checkpoint_path = "/mnt/mcpfn_data/checkpoints/mixed_adaptive_more_heads/step-100000.ckpt",
        nhead=2,
        preprocessors=preprocessors,
        entry_wise_features=False,
        checkpoint_path='/home/jacobf18/tabular/mcpfn/src/tabimpute/workdir/tabimpute-large-pancake-model-mcar_mnar-p0.4-num-cls-8-rank-1-15/checkpoint_40000.pth'
        # max_num_rows=100,
        # max_num_chunks=2,
    )
    mcpfn_name = "tabimpute_large_mcar_mnar"
    
if "tabimpute_ensemble" in imputers:
    preprocessors = [
        RandomRowColumnPermutation(),
        RandomRowColumnPermutation(),
        RandomRowPermutation(),
        RandomColumnPermutation(),
        # StandardizeWhiten(whiten=True),
    ]
    tabimpute_ensemble = TabImputeRouter(device="cuda", preprocessors=preprocessors, checkpoint_paths=[
        "/home/jacobf18/mcpfn_data/checkpoints/masters/mcar/step-78500.ckpt",
        "/home/jacobf18/mcpfn_data/checkpoints/masters/mar/step-60000.ckpt",
        "/home/jacobf18/mcpfn_data/checkpoints/masters/mnar/step-60000.ckpt",
    ], nhead=2)
    mcpfn_name = "tabimpute_ensemble_router"
    
if "mcpfn_ensemble" in imputers:
    preprocessors = [
        RandomRowColumnPermutation(),
        RandomRowColumnPermutation(),
        RandomRowPermutation(),
        RandomColumnPermutation(),
        # StandardizeWhiten(whiten=True),
    ]
    mcpfn_ensemble = MCTabPFNEnsemble(device="cuda", 
                                    # checkpoint_path="/mnt/mcpfn_data/checkpoints/mixed_adaptive/step-125000.ckpt",
                                      nhead=2,
                                      preprocessors=preprocessors)

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
base_path = "datasets/openml"
datasets = os.listdir(base_path)
repeats = 5

pbar = tqdm(datasets)
for name in pbar:
    pbar.set_description(f"Running {name}")
    
    configs = itertools.product(patterns, missingness_levels, range(num_repeats))
    
    for pattern, missingness_level, r in configs:
        # cfg_dir = f"{base_path}/{name}/{pattern}/missingness-{missingness_level}/repeat-{r}"
        cfg_dir = f"{base_path}/{name}/{pattern}_{missingness_level}"

        X_missing = np.load(f"{cfg_dir}/missing.npy")
        X_true = np.load(f"{cfg_dir}/true.npy")  # normalized/ground truth

        # mask for reference if needed downstream
        mask = np.isnan(X_missing)
        
        original_cfg_dir = cfg_dir

        for repeat in range(repeats):
            if repeats > 1:
                cfg_dir = f"{original_cfg_dir}/repeats/{repeat}/"
                # set the random state for the repeat
                np.random.seed(repeat + 1)
                torch.manual_seed(repeat + 1)
            else:
                cfg_dir = original_cfg_dir
            # create the directory if it doesn't exist
            if not os.path.exists(cfg_dir):
                os.makedirs(cfg_dir)
            # --- MCPFN ---
            if "mcpfn" in imputers:
                out_path = f"{cfg_dir}/{mcpfn_name}.npy"
                if not os.path.exists(out_path) or force_rerun:
                    # Time the imputation
                    if repeats > 1:
                        start_time = time.time()
                        X_mcpfn_list = mcpfn.impute(X_missing.copy(), num_repeats=repeats)
                        end_time = time.time()
                        print(f"MCPFN imputation time: {end_time - start_time} seconds for {cfg_dir}")
                        # save the imputation time
                        for i in range(repeats):
                            cfg_dir = f"{original_cfg_dir}/repeats/{i}/"
                            out_path = f"{cfg_dir}/{mcpfn_name}.npy"
                            with open(f"{cfg_dir}/mcpfn_imputation_time.txt", "a") as f:
                                f.write(f"{(end_time - start_time) / repeats}\n")
                            np.save(out_path, X_mcpfn_list[i])
                        break
                    else:
                        start_time = time.time()
                        X_mcpfn = mcpfn.impute(X_missing.copy())
                        end_time = time.time()
                        np.save(out_path, X_mcpfn)
                        print(f"MCPFN imputation time: {end_time - start_time} seconds for {cfg_dir}")
                        # save the imputation time
                        with open(f"{cfg_dir}/{mcpfn_name}_imputation_time.txt", "a") as f:
                            f.write(f"{end_time - start_time}\n")
            
            # --- KNN ---
            if "knn" in imputers:
                out_path = f"{cfg_dir}/knn.npy"
                if not os.path.exists(out_path) or force_rerun:
                    start_time = time.time()
                    X_knn = KNNImputer(n_neighbors=5).fit_transform(fill_all_nan_columns(X_missing.copy()))
                    end_time = time.time()
                    print(f"KNN imputation time: {end_time - start_time} seconds")
                    # save the imputation time
                    with open(f"{cfg_dir}/knn_imputation_time.txt", "a") as f:
                        f.write(f"{end_time - start_time}\n")
                    np.save(out_path, X_knn)
                
            if "mcpfn_ensemble" in imputers:
                out_path = f"{cfg_dir}/mcpfn_ensemble.npy"
                if not os.path.exists(out_path) or force_rerun:
                    start_time = time.time()
                    X_mcpfn_ensemble = mcpfn_ensemble.impute(X_missing.copy())
                    end_time = time.time()
                    print(f"MCPFN Ensemble imputation time: {end_time - start_time} seconds")
                    np.save(out_path, X_mcpfn_ensemble)
                    # save the imputation time
                    with open(f"{cfg_dir}/mcpfn_ensemble_cpu_imputation_time.txt", "a") as f:
                        f.write(f"{end_time - start_time}\n")

            if "tabimpute_ensemble" in imputers:
                out_path = f"{cfg_dir}/{mcpfn_name}.npy"
                if not os.path.exists(out_path) or force_rerun:
                    start_time = time.time()
                    X_tabimpute_ensemble = tabimpute_ensemble.impute(X_missing.copy())
                    end_time = time.time()
                    print(f"TabImpute Ensemble imputation time: {end_time - start_time} seconds")
                    np.save(out_path, X_tabimpute_ensemble)
                    # save the imputation time
                    with open(f"{cfg_dir}/{mcpfn_name}_imputation_time.txt", "a") as f:
                        f.write(f"{end_time - start_time}\n")

            # --- TabPFN ---
            if "tabpfn" in imputers:
                out_path = f"{cfg_dir}/tabpfn.npy"
                if not os.path.exists(out_path) or force_rerun:
                    # Time the imputation
                    if repeats > 1:
                        start_time = time.time()
                        X_tabpfn_list = tabpfn.impute(X_missing.copy(), num_repeats=repeats)
                        end_time = time.time()
                        print(f"TabPFN imputation time: {end_time - start_time} seconds for {cfg_dir}")
                        # save the imputation time
                        for i in range(repeats):
                            cfg_dir = f"{original_cfg_dir}/repeats/{i}/"
                            out_path = f"{cfg_dir}/tabpfn.npy"
                            with open(f"{cfg_dir}/tabpfn_imputation_time.txt", "a") as f:
                                f.write(f"{(end_time - start_time) / repeats}\n")
                            np.save(out_path, X_tabpfn_list[i])
                        break
                    else:
                        start_time = time.time()
                        X_tabpfn = tabpfn.impute(X_missing.copy())
                        end_time = time.time()
                        np.save(out_path, X_tabpfn)
                        print(f"TabPFN imputation time: {end_time - start_time} seconds for {cfg_dir}")
                        # save the imputation time
                        with open(f"{cfg_dir}/tabpfn_imputation_time.txt", "a") as f:
                            f.write(f"{end_time - start_time}\n")

            # --- TabPFN Unsupervised ---
            if "tabpfn_unsupervised" in imputers:
                out_path = f"{cfg_dir}/tabpfn_impute.npy"
                if not os.path.exists(out_path) or force_rerun:
                    start_time = time.time()
                    X_tabpfn_unsupervised = tabpfn_unsupervised.impute(fill_all_nan_columns(X_missing.copy()))
                    end_time = time.time()
                    print(f"TabPFN Unsupervised imputation time: {end_time - start_time} seconds")
                    # save the imputation time
                    with open(f"{cfg_dir}/tabpfn_unsupervised_imputation_time.txt", "a") as f:
                        f.write(f"{end_time - start_time}\n")
                    np.save(out_path, X_tabpfn_unsupervised)

            # --- Column Mean (HyperImpute simple baseline) ---
            if "column_mean" in imputers:
                out_path = f"{cfg_dir}/column_mean.npy"
                
                if not os.path.exists(out_path) or force_rerun:
                    start_time = time.time()
                    plugin = Imputers().get("mean", random_state=repeat)
                    out = plugin.fit_transform(pd.DataFrame(fill_all_nan_columns(X_missing.copy()))).to_numpy()
                    
                    end_time = time.time()
                    print(f"Column Mean imputation time: {end_time - start_time} seconds")
                    # save the imputation time
                    with open(f"{cfg_dir}/column_mean_imputation_time.txt", "a") as f:
                        f.write(f"{end_time - start_time}\n")
                    np.save(out_path, out)

            # --- SoftImpute (HyperImpute) ---
            if "softimpute" in imputers:
                out_path = f"{cfg_dir}/softimpute.npy"
                if not os.path.exists(out_path) or force_rerun:
                    start_time = time.time()
                    plugin = Imputers().get("softimpute", random_state=repeat)
                    out = plugin.fit_transform(pd.DataFrame(fill_all_nan_columns(X_missing.copy()))).to_numpy()
                    end_time = time.time()
                    print(f"SoftImpute imputation time: {end_time - start_time} seconds")
                    # save the imputation time
                    with open(f"{cfg_dir}/softimpute_imputation_time.txt", "a") as f:
                        f.write(f"{end_time - start_time}\n")
                    np.save(out_path, out)

            # --- HyperImpute family (OT/Sinkhorn, MissForest, ICE, MICE, EM, GAIN, MIRACLE, MIWAE) ---
            for key, (plugin_id, fname) in HYPERIMPUTE_MAP.items():
                # print(key, fname)
                if fname == "ot_sinkhorn":
                    if key in imputers:
                        out_path = f"{cfg_dir}/{fname}.npy"
                        if not os.path.exists(out_path) or force_rerun:
                            start_time = time.time()
                            out = run_hyperimpute(plugin_id, X_missing.astype(np.float64).copy(), random_state=repeat)
                            end_time = time.time()
                            print(f"HyperImpute (OT/Sinkhorn) imputation time: {end_time - start_time} seconds")
                            # save the imputation time
                            with open(f"{cfg_dir}/hyperimpute_ot_sinkhorn_imputation_time.txt", "a") as f:
                                f.write(f"{end_time - start_time}\n")
                            np.save(out_path, out)
                else:
                    if key in imputers:
                        print(key)
                        out_path = f"{cfg_dir}/{fname}.npy"
                        if not os.path.exists(out_path) or force_rerun:
                            try:
                                start_time = time.time()
                                out = run_hyperimpute(plugin_id, X_missing.copy(), random_state=repeat)
                                end_time = time.time()
                                print(f"HyperImpute ({key}) imputation time: {end_time - start_time} seconds")
                                # save the imputation time
                                with open(f"{cfg_dir}/hyperimpute_{key}_imputation_time.txt", "a") as f:
                                    f.write(f"{end_time - start_time}\n")
                            except Exception as e:
                                print(f"Error running {plugin_id}: {e}")
                                out = np.zeros_like(X_missing)
                            np.save(out_path, out)

            
            # --- ForestDiffusion (separate package) ---
            if "forestdiffusion" in imputers:
                out_path = f"{cfg_dir}/forestdiffusion.npy"
                
                if not os.path.exists(out_path) or force_rerun:
                    # If you have metadata, pass cat/int indexes here for better handling of categorical/ordinal cols.
                    start_time = time.time()
                    out = run_forest_diffusion(X_missing.copy(), n_t=50, cat_indexes=[], int_indexes=[], n_jobs=-1, repaint=False, num_repeats=repeats)
                    end_time = time.time()
                    if repeats > 1:
                        for i in range(repeats):
                            cfg_dir = f"{original_cfg_dir}/repeats/{i}/"
                            out_path = f"{cfg_dir}/forestdiffusion.npy"
                            np.save(out_path, out[i])
                            with open(f"{cfg_dir}/forestdiffusion_imputation_time.txt", "a") as f:
                                f.write(f"{(end_time - start_time) / repeats}\n")
                        break
                    else:
                        np.save(out_path, out)
                        print(f"ForestDiffusion imputation time: {end_time - start_time} seconds")
                        # save the imputation time
                        with open(f"{cfg_dir}/forestdiffusion_imputation_time.txt", "a") as f:
                            f.write(f"{end_time - start_time}\n")
                            
            if "diffputer" in imputers:
                out_path = f"{cfg_dir}/diffputer.npy"
                if not os.path.exists(out_path) or force_rerun:
                    start_time = time.time()
                    X_diffputer = diffputer.fit_transform(X_missing.copy())
                    end_time = time.time()
                    print(f"DiffPuter imputation time: {end_time - start_time} seconds")
                    np.save(out_path, X_diffputer)
                    # save the imputation time
                    with open(f"{cfg_dir}/diffputer_imputation_time.txt", "a") as f:
                        f.write(f"{end_time - start_time}\n")
            
            # --- ReMasker ---
            if "remasker" in imputers:
                remasker = ReMasker()
                out_path = f"{cfg_dir}/remasker.npy"
                if not os.path.exists(out_path) or force_rerun:
                    start_time = time.time()
                    X_remasker = remasker.fit_transform(fill_all_nan_columns(X_missing.copy()))
                    end_time = time.time()
                    print(f"ReMasker imputation time: {end_time - start_time} seconds")
                    # save the imputation time
                    with open(f"{cfg_dir}/remasker_imputation_time.txt", "w") as f:
                        f.write(f"{end_time - start_time}\n")
                    np.save(out_path, X_remasker)

            # --- CACTI ---
            if "cacti" in imputers:
                cacti = CACTIImputer(device="cuda", model="CMAE", mask_ratio=0.9, epochs=300)
                out_path = f"{cfg_dir}/cacti.npy"
                print(out_path)
                if not os.path.exists(out_path) or force_rerun:
                    try:
                        start_time = time.time()
                        X_cacti = cacti.fit_transform(fill_all_nan_columns(X_missing.copy()))
                        end_time = time.time()
                        print(f"CACTI imputation time: {end_time - start_time} seconds")
                        np.save(out_path, X_cacti)
                        # save the imputation time
                        with open(f"{cfg_dir}/cacti_imputation_time.txt", "w") as f:
                            f.write(f"{end_time - start_time}\n")
                    except Exception as e:
                        print(out_path)
                        print(f"Error running CACTI: {e}")
            # GPU housekeeping (for MCPFN/TabPFN)
            torch.cuda.empty_cache()
