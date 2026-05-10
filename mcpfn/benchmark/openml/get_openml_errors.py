from __future__ import annotations

import math
import os
import warnings
import numpy as np
import pandas as pd
import torch
import miceforest as mf
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
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from notmiwae_wrapper import NotMIWAEImputer

os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

warnings.filterwarnings("ignore")

force_rerun = True

# --- Checkpoint sweep mode ---
# Point SWEEP_CHECKPOINT_DIR at a single training run directory to iterate over
# every step-*.ckpt inside it, impute on the configured (pattern,
# missingness_level) datasets, score with compute_normalized_rmse_columnwise
# (lower = better), and write the winning checkpoint to
# <SWEEP_CHECKPOINT_DIR>/best_rmse_checkpoint.txt. Only the winner's imputations
# are persisted as <SWEEP_DISPLAY_NAME>.npy in the per-dataset directories.
# Set SWEEP_CHECKPOINT_DIR = None to disable.
# SWEEP_CHECKPOINT_DIR: str | None = "/root/checkpoints/masters/mnar_latent_factor_lr_3e-4_decay_2e-2"
SWEEP_CHECKPOINT_DIR = None
SWEEP_DISPLAY_NAME: str | None = None  # default: f"masters_{Path(SWEEP_CHECKPOINT_DIR).name}"

# --- Choose which imputers to run ---
imputers = set([
    # "mcpfn",
    # "mcpfn_ensemble",
    # "tabimpute_ensemble",
    # "knn",
    # "tabpfn",
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
    # "notmiwae",
    "not-miwae-self-mnar",
])

patterns = {
    # "MCAR",
    # "MAR",
    "MNAR",
    # "MAR_Neural",
    # "MAR_BlockNeural",
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

# Root for train scripts' CHECKPOINT_DIR (e.g. train_mar.sh: BASE_DIR/masters/mar).
MCPFN_CHECKPOINT_BASE = Path(os.environ.get("MCPFN_CHECKPOINT_BASE", "/root/checkpoints"))

# CHECKPOINT_NAME values from tabular/mcpfn/scripts/train*.sh (entry-wise / MCPFN architecture).
MCPFN_MASTERS_CHECKPOINT_NAMES = [
    # "masters/mcar",
    # "masters/mcar_nonlinear",
    # "masters/mar",
    # "masters/mar_block_neural",
    # "masters/mar_block_neural_lr_3e-4_decay_2e-2",
    # "masters/mar_block_neural_lr_3e-4_decay_1e-2",
    # "masters/mar_block_neural_lr_4e-4_decay_1e-2",
    # "masters/mar_neural",
    # "masters/mar_neural_lr_3e-4_decay_1e-2",
    # "masters/mnar",
    # "masters/mnar_censoring",
    # "masters/mnar_cluster_level",
    # "masters/mnar_cluster_level_lr_3e-4_decay_2e-2",
    # "masters/mnar_cluster_level_lr_3e-4_decay_1e-2",
    # "masters/mnar_cluster_level_lr_2e-4_decay_1e-2",
    # "masters/mnar_latent_factor",
    "masters/mnar_latent_factor_lr_3e-4_decay_2e-2",
    # "masters/mnar_panel",
    # "masters/mnar_polarization",
    # "masters/mnar_soft_polarization",
    # "masters/mnar_two_phase_subset",
    # "masters/mnar_two_phase_subset_lr_3e-4_decay_1e-2",
]

# If True, also load the legacy direct TabImpute (non-entry-wise) checkpoint below.
INCLUDE_LEGACY_DIRECT_TABIMPUTE = False

LEGACY_DIRECT_TABIMPUTE = {
    "name": "tabimpute_new_stable_deep_trial_3",
    "checkpoint_path": (
        "/root/tabular/mcpfn/src/tabimpute/workdir/"
        "tabimpute-stable-deep-refine-25000-v1-20260309-trial_003/checkpoint_15000.pth"
    ),
    "json_config": {
        "config": {
            "embedding_size": 768,
            "num_attention_heads": 16,
            "mlp_hidden_size": 2048,
            "num_layers": 12,
            "num_outputs": 5000,
            "num_cls": 12,
            "use_rope": True,
            "rope_base": 10000.0,
            "rope_fraction": 0.5,
            "use_absolute_positional_embeddings": False,
            "positional_damping_factor": 0.1,
            "attention_dropout": 0.01,
            "ffn_dropout": 0.06,
            "drop_path_rate": 0.05,
            "residual_scale_init": 0.25,
            "embedding_dropout": 0.0,
            "rms_norm_eps": 1e-6,
        }
    },
}

def _latest_step_ckpt(ckpt_dir: Path) -> str | None:
    """Newest step-NNNNN.ckpt under ckpt_dir, or None if missing."""
    if not ckpt_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for p in ckpt_dir.glob("step-*.ckpt"):
        try:
            n = int(p.stem.split("-")[1])
        except (IndexError, ValueError):
            continue
        if best is None or n > best[0]:
            best = (n, p)
    return str(best[1]) if best else None


def _all_ckpts(ckpt_dir: Path) -> list[tuple[str, Path]]:
    """Return [(label, path), ...] for every step-*.ckpt under ckpt_dir, sorted by step."""
    if not ckpt_dir.is_dir():
        return []
    out: list[tuple[int, str, Path]] = []
    for p in ckpt_dir.glob("step-*.ckpt"):
        try:
            n = int(p.stem.split("-")[1])
        except (IndexError, ValueError):
            continue
        out.append((n, p.stem, p))
    out.sort(key=lambda x: x[0])
    return [(label, path) for _, label, path in out]


def _compute_normalized_rmse_columnwise(X_true, X_imputed, mask):
    """Average column-wise NRMSE; matches plot_negative_rmse.py.

    Lower = better. NaN-safe handling for zero-variance columns. Returns a
    finite float; if no valid columns, returns +inf so it loses any sweep.
    """
    X_true = np.asarray(X_true)
    X_imputed = np.asarray(X_imputed)
    mask = np.asarray(mask, dtype=bool)
    nrmses: list[float] = []
    for j in range(X_true.shape[1]):
        col_mask = mask[:, j]
        if not np.any(col_mask):
            continue
        true_vals = X_true[col_mask, j]
        imp_vals = X_imputed[col_mask, j]
        rmse = float(np.sqrt(np.mean((true_vals - imp_vals) ** 2)))
        sigma = float(np.std(true_vals))
        if sigma < 1e-3:
            if rmse < 1e-3:
                nrmses.append(0.0)
            continue
        nrmses.append(rmse / sigma)
    if not nrmses:
        return float("inf")
    return float(np.mean(nrmses))


def _iter_eval_dataset_paths(
    base_path: str,
    datasets_list: list[str],
    eval_patterns: set,
    eval_missingness_levels: list[float],
    num_repeats_inner: int,
):
    """Yield (cfg_dir, X_missing, X_true, mask) for every (dataset, pattern, level, repeat).

    Used by the checkpoint sweep so each candidate sees identical data.
    """
    for ds_name in datasets_list:
        for pat, lvl, _r in itertools.product(
            eval_patterns, eval_missingness_levels, range(num_repeats_inner)
        ):
            cfg = f"{base_path}/{ds_name}/{pat}_{lvl}"
            miss_path = f"{cfg}/missing.npy"
            true_path = f"{cfg}/true.npy"
            if not (os.path.exists(miss_path) and os.path.exists(true_path)):
                continue
            X_miss = np.load(miss_path)
            X_tru = np.load(true_path)
            yield cfg, X_miss, X_tru, np.isnan(X_miss)


def sweep_checkpoints_for_run(
    run_dir: Path,
    display_name: str,
    mcpfn_shared_kwargs: dict,
    base_path: str,
    datasets_list: list[str],
    eval_patterns: set,
    eval_missingness_levels: list[float],
) -> str | None:
    """Score every checkpoint in `run_dir` by avg column-wise NRMSE and
    persist the winner.

    Returns the path of the winning checkpoint, or None if no checkpoints
    were found / scored. Side effects:
      - writes <run_dir>/best_rmse_checkpoint.txt with metadata
      - writes per-dataset <cfg_dir>/<display_name>.npy using the winner
    """
    candidates = _all_ckpts(run_dir)
    if not candidates:
        print(f"[sweep] No checkpoints found in {run_dir}")
        return None

    print(f"[sweep] {display_name}: evaluating {len(candidates)} checkpoints in {run_dir}")
    scores: list[tuple[float, str, Path, dict]] = []
    # Cache imputations per checkpoint so we can re-emit the winner's .npy files
    # without re-running inference at the end.
    imputations_cache: dict[str, dict[str, np.ndarray]] = {}

    for label, ckpt_path in candidates:
        try:
            model = ImputePFN(checkpoint_path=str(ckpt_path), use_ema=False, **mcpfn_shared_kwargs)
        except Exception as e:
            print(f"[sweep] Failed to load {ckpt_path}: {e}")
            continue

        per_cfg_imputed: dict[str, np.ndarray] = {}
        per_dataset_scores: list[float] = []
        for cfg_dir, X_missing, X_true, mask in _iter_eval_dataset_paths(
            base_path, datasets_list, eval_patterns, eval_missingness_levels, 1
        ):
            try:
                X_imp = model.impute(X_missing.copy())
            except Exception as e:
                print(f"[sweep] {label}: impute failed on {cfg_dir}: {e}")
                continue
            per_cfg_imputed[cfg_dir] = X_imp
            score = _compute_normalized_rmse_columnwise(X_true, X_imp, mask)
            if math.isfinite(score):
                per_dataset_scores.append(score)

        # Free this candidate's GPU memory before loading the next one.
        del model
        torch.cuda.empty_cache()

        if not per_dataset_scores:
            print(f"[sweep] {label}: no scorable datasets, skipping")
            continue

        mean_score = float(np.mean(per_dataset_scores))
        print(
            f"[sweep] {label}: avg NRMSE = {mean_score:.6f} "
            f"(over {len(per_dataset_scores)} dataset/pattern combos)"
        )
        imputations_cache[label] = per_cfg_imputed
        scores.append((mean_score, label, ckpt_path, per_cfg_imputed))

    if not scores:
        print(f"[sweep] {display_name}: no checkpoint produced a valid score")
        return None

    scores.sort(key=lambda x: x[0])
    best_score, best_label, best_path, best_imps = scores[0]
    print(
        f"[sweep] {display_name}: WINNER = {best_label} (path={best_path}) "
        f"with avg NRMSE = {best_score:.6f}"
    )

    # Persist winner metadata to the run directory.
    try:
        with open(run_dir / "best_rmse_checkpoint.txt", "w") as f:
            f.write(f"checkpoint: {best_label}\n")
            f.write(f"path: {best_path}\n")
            f.write(f"avg_nrmse: {best_score:.8f}\n")
            f.write(f"num_eval_combos: {len(best_imps)}\n")
            f.write("ranking:\n")
            for s, lbl, pth, _ in scores:
                f.write(f"  {lbl}\t{s:.8f}\t{pth}\n")
    except Exception as e:
        print(f"[sweep] Failed to write best_rmse_checkpoint.txt: {e}")

    # Emit the winner's imputations as <display_name>.npy files.
    for cfg_dir, X_imp in best_imps.items():
        os.makedirs(cfg_dir, exist_ok=True)
        np.save(f"{cfg_dir}/{display_name}.npy", X_imp)

    return str(best_path)


# --- Initialize classes once ---
if "mcpfn" in imputers:
    preprocessors = [
        RandomRowColumnPermutation(),
        RandomRowColumnPermutation(),
        RandomRowPermutation(),
        RandomColumnPermutation(),
    ]
    # preprocessors = None

    mcpfn_masters_shared = {
        "device": "cuda",
        "nhead": 2,
        "preprocessors": preprocessors,
        "entry_wise_features": True,
        "max_num_rows": None,
        "max_num_cols": None,
    }

    mcpfn_imputers: list[dict] = []
    for ck_name in MCPFN_MASTERS_CHECKPOINT_NAMES:
        rel = Path(ck_name)
        run_dir = MCPFN_CHECKPOINT_BASE / rel
        display_name = f"masters_{rel.name}"
        
        # Check if best exists in the run directory
        best_path = run_dir / "best.ckpt"
        if not best_path.exists():
            ckpt_path = _latest_step_ckpt(run_dir)
        else:
            ckpt_path = best_path
        if ck_name == "masters/mnar_cluster_level_lr_2e-4_decay_1e-2":
            ckpt_path = "/root/checkpoints/masters/mnar_cluster_level_lr_2e-4_decay_1e-2/step-35000.ckpt"
        if ck_name == "masters/mar_neural_lr_3e-4_decay_1e-2":
            ckpt_path = "/root/checkpoints/masters/mar_neural_lr_3e-4_decay_1e-2/step-35000.ckpt"
        if ck_name == "masters/mnar_latent_factor_lr_3e-4_decay_2e-2":
            ckpt_path = "/root/checkpoints/masters/mnar_latent_factor_lr_3e-4_decay_2e-2/step-49500.ckpt"
        print(ck_name)
        print(f"Latest checkpoint: {ckpt_path}")
        mcpfn_imputers.append(
            {
                "name": display_name,
                "model": ImputePFN(
                    checkpoint_path=str(ckpt_path),
                    use_ema=False,
                    **mcpfn_masters_shared,
                ),
            }
        )

    if INCLUDE_LEGACY_DIRECT_TABIMPUTE:
        mcpfn_imputers.append(
            {
                "name": LEGACY_DIRECT_TABIMPUTE["name"],
                "model": ImputePFN(
                    device="cuda",
                    nhead=2,
                    preprocessors=preprocessors,
                    entry_wise_features=False,
                    checkpoint_path=LEGACY_DIRECT_TABIMPUTE["checkpoint_path"],
                    json_config=LEGACY_DIRECT_TABIMPUTE["json_config"],
                    model_family="new_stable",
                ),
            }
        )

    if not mcpfn_imputers and SWEEP_CHECKPOINT_DIR is None:
        raise RuntimeError(
            "imputers includes 'mcpfn' but no checkpoints were found. "
            "Train models (see tabular/mcpfn/scripts) or set MCPFN_CHECKPOINT_BASE."
        )

    if mcpfn_imputers:
        mcpfn_name = mcpfn_imputers[0]["name"]
    elif SWEEP_CHECKPOINT_DIR is not None:
        mcpfn_name = SWEEP_DISPLAY_NAME or f"masters_{Path(SWEEP_CHECKPOINT_DIR).name}"
    
if "tabpfn" in imputers:
    tabpfn = TabPFNImputer(device="cuda")
    
if "tabpfn_unsupervised" in imputers:
    tabpfn_unsupervised = TabPFNUnsupervisedModel(device="cuda")
    
if "diffputer" in imputers:
    diffputer = DiffPuterImputer(device="cuda")

if "notmiwae" in imputers:
    # Pin the original SelfMaskingProcess so cached notmiwae.npy artifacts stay
    # reproducible even though the wrapper's default flipped to 'self_mnar'.
    notmiwae = NotMIWAEImputer(
        device="cuda" if torch.cuda.is_available() else "cpu",
        missing_process="selfmasking",
    )

if "not-miwae-self-mnar" in imputers:
    # Self-MNAR variant: uses SelfMNARMissingProcess initialized to mirror the
    # MNARPattern propensity in tabimpute.prior.training_set_generation.
    notmiwae_self_mnar = NotMIWAEImputer(
        device="cuda" if torch.cuda.is_available() else "cpu",
        missing_process="self_mnar",
    )

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
    if plugin_name == "mice":
        miceforest_df = pd.DataFrame(
            X_processed,
            columns=[str(col_idx) for col_idx in range(X_processed.shape[1])],
        )
        num_imputations = 5
        kernel_kwargs = {
            "num_datasets": num_imputations,
            # Avoid mean-matching failures on small benchmark slices.
            "mean_match_candidates": 0,
            "random_state": random_state,
        }
        print(f"Running miceforest MICE with 5 iterations and {num_imputations} datasets on CPU")
        kernel = mf.ImputationKernel(miceforest_df, **kernel_kwargs)
        kernel.mice(iterations=5, verbose=False)
        # Collect each completed dataset as a numpy array
        imputed_arrays = [
            kernel.complete_data(dataset=i).to_numpy()
            for i in range(num_imputations)
        ]
        # Average across imputations
        return np.mean(imputed_arrays, axis=0)
    else:
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
base_path = "../datasets/openml"
datasets = os.listdir(base_path)
repeats = 1

# --- Optional MCPFN checkpoint sweep over a single run directory ---
# Iterate over every step-*.ckpt in SWEEP_CHECKPOINT_DIR, score each by avg
# column-wise NRMSE, write the winner manifest to that directory, and persist
# the winner's imputations as the per-dataset .npy files.
if "mcpfn" in imputers and SWEEP_CHECKPOINT_DIR is not None:
    sweep_dir = Path(SWEEP_CHECKPOINT_DIR)
    sweep_name = SWEEP_DISPLAY_NAME or f"masters_{sweep_dir.name}"
    sweep_checkpoints_for_run(
        run_dir=sweep_dir,
        display_name=sweep_name,
        mcpfn_shared_kwargs=mcpfn_masters_shared,
        base_path=base_path,
        datasets_list=datasets,
        eval_patterns=patterns,
        eval_missingness_levels=missingness_levels,
    )

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
            # --- MCPFN (one output .npy per model in mcpfn_imputers) ---
            if "mcpfn" in imputers:
                mcpfn_did_multirepeat = False
                for mcpfn_spec in mcpfn_imputers:
                    mcpfn_name = mcpfn_spec["name"]
                    mcpfn = mcpfn_spec["model"]
                    out_path = f"{cfg_dir}/{mcpfn_name}.npy"
                    if not os.path.exists(out_path) or force_rerun:
                        if repeats > 1:
                            start_time = time.time()
                            X_mcpfn_list = mcpfn.impute(
                                X_missing.copy(), num_repeats=repeats
                            )
                            end_time = time.time()
                            print(
                                f"MCPFN ({mcpfn_name}) imputation time: "
                                f"{end_time - start_time} seconds for {cfg_dir}"
                            )
                            for i in range(repeats):
                                r_cfg = f"{original_cfg_dir}/repeats/{i}/"
                                if not os.path.exists(r_cfg):
                                    os.makedirs(r_cfg)
                                out_p = f"{r_cfg}/{mcpfn_name}.npy"
                                with open(
                                    f"{r_cfg}/{mcpfn_name}_imputation_time.txt", "a"
                                ) as f:
                                    f.write(f"{(end_time - start_time) / repeats}\n")
                                np.save(out_p, X_mcpfn_list[i])
                            mcpfn_did_multirepeat = True
                        else:
                            start_time = time.time()
                            X_mcpfn = mcpfn.impute(X_missing.copy())
                            end_time = time.time()
                            np.save(out_path, X_mcpfn)
                            print(
                                f"MCPFN ({mcpfn_name}) imputation time: "
                                f"{end_time - start_time} seconds for {cfg_dir}"
                            )
                            with open(
                                f"{cfg_dir}/{mcpfn_name}_imputation_time.txt", "w"
                            ) as f:
                                f.write(f"{end_time - start_time}\n")
                if mcpfn_did_multirepeat and repeats > 1:
                    break
            
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

            if "notmiwae" in imputers:
                out_path = f"{cfg_dir}/notmiwae.npy"
                if not os.path.exists(out_path) or force_rerun:
                    try:
                        start_time = time.time()
                        X_notmiwae = notmiwae.fit_transform(
                            fill_all_nan_rows(fill_all_nan_columns(X_missing.copy()))
                        )
                        end_time = time.time()
                        print(f"NotMIWAE imputation time: {end_time - start_time} seconds")
                        np.save(out_path, X_notmiwae)
                        with open(f"{cfg_dir}/notmiwae_imputation_time.txt", "a") as f:
                            f.write(f"{end_time - start_time}\n")
                    except Exception as e:
                        print(f"Error running NotMIWAE on {cfg_dir}: {e}")

            if "not-miwae-self-mnar" in imputers:
                out_path = f"{cfg_dir}/not-miwae-self-mnar.npy"
                if not os.path.exists(out_path) or force_rerun:
                    try:
                        start_time = time.time()
                        X_notmiwae_self_mnar = notmiwae_self_mnar.fit_transform(
                            fill_all_nan_rows(fill_all_nan_columns(X_missing.copy()))
                        )
                        end_time = time.time()
                        print(f"NotMIWAE (self-MNAR) imputation time: {end_time - start_time} seconds")
                        np.save(out_path, X_notmiwae_self_mnar)
                        with open(f"{cfg_dir}/not-miwae-self-mnar_imputation_time.txt", "a") as f:
                            f.write(f"{end_time - start_time}\n")
                    except Exception as e:
                        print(f"Error running NotMIWAE (self-MNAR) on {cfg_dir}: {e}")
            
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
