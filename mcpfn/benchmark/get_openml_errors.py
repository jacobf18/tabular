import os
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mcpfn.interface import ImputePFN, TabPFNImputer, TabPFNUnsupervisedModel
from hyperimpute.plugins.imputers import Imputers

# Optional: ForestDiffusion (pip install ForestDiffusion)
try:
    from ForestDiffusion import ForestDiffusionModel
    HAS_FORESTDIFFUSION = True
except Exception:
    HAS_FORESTDIFFUSION = False

from mcpfn.prepreocess import (
    RandomRowColumnPermutation, 
    PowerTransform, 
    SequentialPreprocess, 
    RandomRowPermutation, 
    RandomColumnPermutation, 
    StandardizeWhiten
)

warnings.filterwarnings("ignore")

# --- Choose which imputers to run ---
imputers = set([
    "mcpfn",
    # "tabpfn",
    # "tabpfn_unsupervised",
    # "column_mean",
    # "softimpute",
    # "hyperimpute_ot", # Sinkhorn / Optimal Transport
    # "hyperimpute",        
    # "hyperimpute_missforest",
    # "hyperimpute_ice",
    # "hyperimpute_mice",
    # # "hyperimpute_em", # doesn't work well
    # "hyperimpute_gain",
    # # "hyperimpute_miracle",
    # "hyperimpute_miwae",
    # # "forestdiffusion",
])

patterns = {
    "MCAR",
    "MAR",
    "MAR_Neural",
    "MAR_BlockNeural",
    "MAR_Sequential",
    "MNAR",
}

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
        encoder_path="/root/tabular/mcpfn/src/mcpfn/model/encoder.pth",
        borders_path="/root/tabular/mcpfn/borders.pt",
        checkpoint_path="/mnt/mcpfn_data/checkpoints/mixed_adaptive/step-125000.ckpt",
        nhead=2,
        preprocessors=preprocessors
    )
    mcpfn_name = "mixed_adaptive"

if "tabpfn" in imputers:
    tabpfn = TabPFNImputer(device="cuda")
    
if "tabpfn_unsupervised" in imputers:
    tabpfn_unsupervised = TabPFNUnsupervisedModel(device="cuda")

# Map friendly names → HyperImpute plugin ids and output filenames
HYPERIMPUTE_MAP = {
    "hyperimpute_ot": ("sinkhorn", "ot_sinkhorn"),
    "hyperimpute_missforest": ("missforest", "missforest"),
    "hyperimpute_ice": ("ice", "ice"),
    "hyperimpute_mice": ("mice", "mice"),
    "hyperimpute_em": ("em", "em"),
    "hyperimpute_gain": ("gain", "gain"),
    "hyperimpute_miracle": ("miracle", "miracle"),
    "hyperimpute_miwae": ("miwae", "miwae"),
}

def run_hyperimpute(plugin_name: str, X_missing: np.ndarray) -> np.ndarray:
    """Run a single HyperImpute plugin on X_missing (numpy with NaNs)."""
    plugin = Imputers().get(plugin_name)
    # HyperImpute expects pandas DataFrame and returns a DataFrame
    out_df = plugin.fit_transform(pd.DataFrame(X_missing))
    return out_df.to_numpy()

def run_forest_diffusion(X_missing: np.ndarray, n_t: int = 50,
                         cat_indexes=None, int_indexes=None,
                         n_jobs: int = -1, repaint=False) -> np.ndarray:
    """ForestDiffusion imputation wrapper (numeric defaults)."""
    if not HAS_FORESTDIFFUSION:
        raise RuntimeError("ForestDiffusion not installed. `pip install ForestDiffusion`.")
    if cat_indexes is None: cat_indexes = []
    if int_indexes is None: int_indexes = []
    # Per authors, vp diffusion is required for imputation.
    model = ForestDiffusionModel(
        X_missing,
        n_t=n_t,
        diffusion_type="vp",
        cat_indexes=cat_indexes,
        int_indexes=int_indexes,
        n_jobs=n_jobs,
    )
    model.y_label = np.zeros_like(X_missing[:, 0])
    if repaint:
        return model.impute(repaint=True, r=10, j=max(1, n_t // 10), k=1)
    return model.impute(k=1)

# --- Run benchmark ---
base_path = "datasets/openml"
datasets = os.listdir(base_path)

pbar = tqdm(datasets)
for name in pbar:
    pbar.set_description(f"Running {name}")
    configs = os.listdir(f"{base_path}/{name}")
    for config in configs:
        cfg_dir = f"{base_path}/{name}/{config}"
        pattern_name = config.split("_")[0]
        p = config.split("_")[1]
        if p != "0.4":
            continue
        if pattern_name not in patterns:
            continue

        X_missing = np.load(f"{cfg_dir}/missing.npy")
        X_true = np.load(f"{cfg_dir}/true.npy")  # normalized/ground truth

        # mask for reference if needed downstream
        mask = np.isnan(X_missing)

        # --- MCPFN ---
        if "mcpfn" in imputers:
            out_path = f"{cfg_dir}/{mcpfn_name}.npy"
            # if not os.path.exists(out_path):
            X_mcpfn = mcpfn.impute(X_missing.copy(), calibrate=False)
            np.save(out_path, X_mcpfn)

        # --- TabPFN ---
        if "tabpfn" in imputers:
            out_path = f"{cfg_dir}/tabpfn.npy"
            if not os.path.exists(out_path):
                X_tabpfn = tabpfn.impute(X_missing.copy())
                np.save(out_path, X_tabpfn)

        # --- TabPFN Unsupervised ---
        if "tabpfn_unsupervised" in imputers:
            out_path = f"{cfg_dir}/tabpfn_impute.npy"
            if not os.path.exists(out_path):
                X_tabpfn_unsupervised = tabpfn_unsupervised.impute(X_missing.copy())
                np.save(out_path, X_tabpfn_unsupervised)

        # --- Column Mean (HyperImpute simple baseline) ---
        if "column_mean" in imputers:
            out_path = f"{cfg_dir}/column_mean.npy"
            
            # if not os.path.exists(out_path):
            # Column means (ignoring NaN). This will be NaN if the column is all NaNs.
            col_means = np.nanmean(X_missing, axis=0)

            # Replace NaN means with 0 for all-NaN columns
            col_means = np.where(np.isnan(col_means), 0, col_means)

            # Impute: broadcast col_means into missing entries
            X_missing[mask] = np.take(col_means, np.where(mask)[1])
            
            np.save(out_path, X_missing)

        # --- SoftImpute (HyperImpute) ---
        if "softimpute" in imputers:
            out_path = f"{cfg_dir}/softimpute.npy"
            if not os.path.exists(out_path):
                plugin = Imputers().get("softimpute")
                out = plugin.fit_transform(pd.DataFrame(X_missing.copy())).to_numpy()
                np.save(out_path, out)

        # --- HyperImpute family (OT/Sinkhorn, MissForest, ICE, MICE, EM, GAIN, MIRACLE, MIWAE) ---
        for key, (plugin_id, fname) in HYPERIMPUTE_MAP.items():
            # print(key, fname)
            if fname == "ot_sinkhorn":
                if key in imputers:
                    out_path = f"{cfg_dir}/{fname}.npy"
                    if not os.path.exists(out_path):
                        out = run_hyperimpute(plugin_id, X_missing.astype(np.float64).copy())
                        np.save(out_path, out)
            else:
                if key in imputers:
                    out_path = f"{cfg_dir}/{fname}.npy"
                    if not os.path.exists(out_path):
                        try:
                            out = run_hyperimpute(plugin_id, X_missing.copy())
                        except Exception as e:
                            print(f"Error running {plugin_id}: {e}")
                            out = np.zeros_like(X_missing)
                        np.save(out_path, out)

            
        # --- ForestDiffusion (separate package) ---
        if "forestdiffusion" in imputers:
            out_path = f"{cfg_dir}/forestdiffusion.npy"
            if not os.path.exists(out_path):
                # If you have metadata, pass cat/int indexes here for better handling of categorical/ordinal cols.
                out = run_forest_diffusion(X_missing.copy(), n_t=50, cat_indexes=[], int_indexes=[], n_jobs=-1, repaint=False)
                np.save(out_path, out)

        # GPU housekeeping (for MCPFN/TabPFN)
        torch.cuda.empty_cache()


# import torch
# import numpy as np
# from mcpfn.prior.training_set_generation import (
#     MCARPattern, MARPattern, MNARPattern
# )
# from mcpfn.interface import ImputePFN, TabPFNImputer
# from hyperimpute.plugins.imputers import Imputers
# import warnings
# from mcpfn.model.encoders import normalize_data
# import pandas as pd
# import os
# from tqdm import tqdm

# # --- Suppress warnings ---
# warnings.filterwarnings("ignore")

# imputers = set([
#     # "mcpfn", 
#     # "hyperimpute", 
#     # "softimpute", 
#     # "column_mean", 
#     "tabpfn"
# ])

# # --- Load imputer classes ---
# if "mcpfn" in imputers:
#     mcpfn = ImputePFN(
#         device='cuda',
#         encoder_path='/root/tabular/mcpfn/src/mcpfn/model/encoder.pth',
#         borders_path='/root/tabular/mcpfn/borders.pt',
#         # checkpoint_path='/mnt/mcpfn_data/checkpoints/mixed/step-445000.ckpt'
#         # checkpoint_path='/root/checkpoints/mar_mixed/step-57000.ckpt'
#         # checkpoint_path='/mnt/mcpfn_data/checkpoints/mixed_random_2/step-121000.ckpt'
#         # checkpoint_path='/mnt/mcpfn_data/checkpoints/mcar_linear/step-121000.ckpt'
#         # checkpoint_path='/mnt/mcpfn_data/checkpoints/mixed_linear_fixed/step-50000.ckpt'
#         checkpoint_path='/mnt/mcpfn_data/checkpoints/mixed_adaptive/step-49000.ckpt',
#         nhead=6
#     )
#     mcpfn.model.model.load_state_dict(torch.load('/root/tabular/mcpfn/src/mcpfn/model/tabpfn_model.pt', weights_only=True))
#     mcpfn_name = "mcpfn_tabpfn"

# if "tabpfn" in imputers:
#     tabpfn = TabPFNImputer(device='cuda', encoder_path='/root/tabular/mcpfn/src/mcpfn/model/encoder.pth')

# # --- Store all results ---
# base_path = "/root/tabular/mcpfn/benchmark/datasets/openml"

# # Get dataset names from base path
# datasets = os.listdir(base_path)

# # --- Run benchmark ---
# pbar = tqdm(datasets)
# for name in pbar:
#     pbar.set_description(f"Running {name}")
#     # Get filenames
#     configs = os.listdir(f"{base_path}/{name}")
#     for config in configs:
#         # print(f"Running {name} | {config}")
#         pattern_name = config.split("_")[0]
#         p = config.split("_")[1]
#         X_missing = np.load(f"{base_path}/{name}/{config}/missing.npy")
#         X_normalized = np.load(f"{base_path}/{name}/{config}/true.npy")
    
#         mask = np.isnan(X_missing)

#         # MCPFN
#         if "mcpfn" in imputers:
#             # check if file exists. if not, run imputation
#             # if not os.path.exists(f"{base_path}/{name}/{config}/{mcpfn_name}.npy"):
#             X_mcpfn = mcpfn.impute(X_missing.copy())
#             np.save(f"{base_path}/{name}/{config}/{mcpfn_name}.npy", X_mcpfn)
        
#         # TabPFN
#         if "tabpfn" in imputers:
#             # if not os.path.exists(f"{base_path}/{name}/{config}/tabpfn.npy"):
#             X_tabpfn = tabpfn.impute(X_missing.copy())
#             np.save(f"{base_path}/{name}/{config}/mcpfn_tabpfn_with_preprocessing.npy", X_tabpfn)
        
#         # SoftImpute
#         if "softimpute" in imputers:
#             if not os.path.exists(f"{base_path}/{name}/{config}/softimpute.npy"):
#                 plugin = Imputers().get("softimpute")
#                 out = plugin.fit_transform(X_missing.copy()).to_numpy()
#                 np.save(f"{base_path}/{name}/{config}/softimpute.npy", out)
        
#         # Column Mean
#         if "column_mean" in imputers:
#             if not os.path.exists(f"{base_path}/{name}/{config}/column_mean.npy"):
#                 plugin = Imputers().get("mean")
#                 out = plugin.fit_transform(X_missing.copy()).to_numpy()
#                 np.save(f"{base_path}/{name}/{config}/column_mean.npy", out)
        
#         # HyperImpute
#         if "hyperimpute" in imputers:
#             if not os.path.exists(f"{base_path}/{name}/{config}/hyperimpute.npy"):
#                 plugin = Imputers().get("hyperimpute")
#                 out = plugin.fit_transform(X_missing.copy()).to_numpy()
#                 np.save(f"{base_path}/{name}/{config}/hyperimpute.npy", out)
        
#         # Empty cache
#         torch.cuda.empty_cache()