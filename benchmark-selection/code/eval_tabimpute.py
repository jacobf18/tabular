"""Evaluate benchmark selection with TabImpute V2 as the imputation backend.

Same CV protocol as eval_entropy_vs_mi.py (10-fold, 10% holdout), but replaces
Gaussian conditional mean with TabImpute V2 for imputing unselected benchmarks.

Runs entropy, MI, and random selection on all 4 datasets, reports R².
"""

import pathlib
import sys
import time
import os
import json
import numpy as np
from collections import defaultdict

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from path_config import data_dir, figures_dir, logs_dir

DATA = data_dir()
OUT = figures_dir()
LOGS = logs_dir()
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

from greedy_select import greedy_entropy, greedy_mi, random_select
from em_cov import em_cov
from cv_split import cv_folds, cv_train_val

import torch
from huggingface_hub import hf_hub_download
from tabimpute.interface import CompressedCrossImputePFN, ImputePFN
from tabimpute.interface_rfm import RFMImputePFN
from tabimpute.model.rfm_head_agop import RFMHeadAGOP
from tabimpute.tabimpute_v2 import TabImputeV2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_MAX = int(os.environ.get("BENCHSELECT_K_MAX", "15"))
N_FOLDS = int(os.environ.get("BENCHSELECT_N_FOLDS", "10"))
HOLDOUT_FRAC = float(os.environ.get("BENCHSELECT_HOLDOUT_FRAC", "0.1"))
SEED = 42
MAX_ROWS = int(os.environ.get("BENCHSELECT_MAX_ROWS", "4000"))
TEST_BATCH = int(os.environ.get("BENCHSELECT_TEST_BATCH", "200"))
TABIMPUTE_MODEL = os.environ.get("BENCHSELECT_TABIMPUTE_MODEL", "v2").lower()
TABIMPUTE_LABEL = os.environ.get("BENCHSELECT_TABIMPUTE_LABEL", TABIMPUTE_MODEL)
ROW_LIMIT = os.environ.get("BENCHSELECT_ROW_LIMIT")
ROW_LIMIT = int(ROW_LIMIT) if ROW_LIMIT else None
METHODS = os.environ.get("BENCHSELECT_METHODS")
METHODS = METHODS.split(",") if METHODS else ["entropy", "mi", "random"]

DEFAULT_DATASETS = ["mmlu", "mteb", "merged", "benchpress"]
DATASETS = os.environ.get("BENCHSELECT_DATASETS")
DATASETS = DATASETS.split(",") if DATASETS else DEFAULT_DATASETS

DISPLAY_NAME = {
    "mmlu": "MMLU", "mteb": "MTEB",
    "merged": "Merged", "benchpress": "BenchPress",
}

# ── Covariance estimation (same as other scripts) ─────────────────

def estimate_corr_pairwise(B_train, O_train):
    col_count = np.maximum(O_train.sum(axis=0), 1.0)
    col_mean = (B_train * O_train).sum(axis=0) / col_count
    B_c = O_train * (B_train - col_mean[None, :])
    col_var = (B_c ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    col_std = np.sqrt(np.maximum(col_var, 1e-12))
    B_std = O_train * (B_train - col_mean[None, :]) / col_std[None, :]
    gram = B_std.T @ B_std
    pair_counts = O_train.T @ O_train
    denom = np.maximum(pair_counts - 1.0, 1.0)
    R = gram / denom
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-6)
    R = (eigvecs * eigvals[None, :]) @ eigvecs.T
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(R), 1e-12))
    R = R * np.outer(d_inv, d_inv)
    return R, col_mean, col_std


def estimate_corr_em(B_train, O_train):
    M_tr, N_tr = B_train.shape
    obs_frac = O_train.sum() / (M_tr * N_tr)
    col_count = np.maximum(O_train.sum(axis=0), 1.0)
    col_mean = (B_train * O_train).sum(axis=0) / col_count
    B_c = O_train * (B_train - col_mean[None, :])
    col_var = (B_c ** 2).sum(axis=0) / np.maximum(col_count - 1.0, 1.0)
    col_std_floor = 0.01 * (np.abs(col_mean) + 1.0)
    col_std = np.sqrt(np.maximum(col_var, col_std_floor ** 2))
    B_std = O_train * (B_train - col_mean[None, :]) / col_std[None, :]
    regularized = M_tr < N_tr or obs_frac < 0.5
    use_shrink = "auto" if M_tr < N_tr else 0.0
    em_eps = 1e-3 if regularized else 1e-6
    em_iter = 1000 if regularized else 500
    result = em_cov(B_std, O_train, max_iter=em_iter, tol=5e-4,
                    shrinkage=use_shrink, eps_psd=em_eps, verbose=False)
    Sigma = result["Sigma"]
    d_inv = 1.0 / np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    R = Sigma * np.outer(d_inv, d_inv)
    return R, col_mean, col_std


def estimate_corr(B_train, O_train):
    M_tr, N_tr = B_train.shape
    obs_frac = O_train.sum() / (M_tr * N_tr)
    if obs_frac < 0.9 or M_tr < N_tr:
        return estimate_corr_em(B_train, O_train)
    else:
        return estimate_corr_pairwise(B_train, O_train)


# ── TabImpute-based imputation ────────────────────────────────────

def impute_tabimpute(imputer, B_train, O_train, B_test, O_test,
                     selected, col_mean, col_std, N):
    """Impute test models' unselected benchmarks using TabImpute V2.

    Constructs a matrix with training rows (context) and test rows
    (only selected benchmarks observed), runs TabImpute, evaluates R²
    in standardized (z-scored) space for comparability with the Gaussian
    conditional mean baseline.

    All training rows are used as context (subsampled only if needed
    to fit GPU memory). Test rows are batched so that each batch sees
    the maximum number of training rows.
    """
    A = set(selected)
    M_train = B_train.shape[0]
    M_test = B_test.shape[0]

    # Context rows: all training data with observed entries
    X_ctx = np.where(O_train == 1, B_train, np.nan)

    # Subsample training rows if needed to leave room for test batches
    max_ctx = MAX_ROWS - TEST_BATCH
    if M_train > max_ctx:
        rng = np.random.default_rng(42)
        ctx_idx = rng.choice(M_train, max_ctx, replace=False)
        X_ctx = X_ctx[ctx_idx]
        n_ctx = max_ctx
    else:
        n_ctx = M_train

    # Test rows: only reveal selected benchmarks
    X_test = np.full((M_test, N), np.nan)
    for j in selected:
        for i in range(M_test):
            if O_test[i, j] == 1:
                X_test[i, j] = B_test[i, j]

    # Process test rows in batches, each seeing all context rows
    max_test_per_batch = max(1, min(TEST_BATCH, MAX_ROWS - n_ctx))
    X_test_imputed = np.empty((M_test, N), dtype=np.float64)
    for start in range(0, M_test, max_test_per_batch):
        end = min(start + max_test_per_batch, M_test)
        X_batch = np.vstack([X_ctx, X_test[start:end]])
        X_batch_imp = imputer.impute(X_batch)
        X_test_imputed[start:end] = X_batch_imp[n_ctx:]

    # Evaluate in standardized space (same as Gaussian method):
    # z-score using training col_mean/col_std, then compute R²
    ss_res = 0.0
    ss_tot = 0.0
    n_eval = 0

    for i in range(M_test):
        eval_idx = np.array([j for j in range(N)
                             if j not in A and O_test[i, j] == 1])
        if len(eval_idx) == 0:
            continue

        true_std = (B_test[i, eval_idx] - col_mean[eval_idx]) / col_std[eval_idx]
        pred_std = (X_test_imputed[i, eval_idx] - col_mean[eval_idx]) / col_std[eval_idx]
        true_std = np.clip(true_std, -10, 10)
        pred_std = np.clip(pred_std, -10, 10)

        ss_res += np.sum((true_std - pred_std) ** 2)
        ss_tot += np.sum(true_std ** 2)  # mean is 0 in standardized space
        n_eval += len(eval_idx)

    if n_eval == 0 or ss_tot < 1e-12:
        return {"r2": np.nan, "rmse": np.nan, "n_eval": 0}

    r2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / n_eval)
    return {"r2": r2, "rmse": rmse, "n_eval": n_eval}


# ── Run one dataset ───────────────────────────────────────────────

def run_dataset(ds_name, imputer):
    print(f"\n{'#'*70}")
    print(f"# Dataset: {ds_name.upper()}")
    print(f"{'#'*70}")

    d = np.load(DATA / f"{ds_name}.matrix.npz", allow_pickle=True)
    B = d["B"].astype(np.float64)
    O = d["O"].astype(np.float64)
    benchmark_names = d["benchmark_names"]
    if ROW_LIMIT is not None:
        B = B[:ROW_LIMIT]
        O = O[:ROW_LIMIT]
    M, N = B.shape
    obs_frac = O.sum() / (M * N)
    print(f"{M} models × {N} tasks, {obs_frac*100:.1f}% observed", flush=True)

    folds = cv_folds(M, N_FOLDS, SEED)
    p = HOLDOUT_FRAC

    metrics = {method: defaultdict(list) for method in METHODS}

    for fold_idx in range(N_FOLDS):
        t0 = time.time()
        train_idx, val_idx = cv_train_val(folds, fold_idx, p, M, SEED)

        B_train, O_train = B[train_idx], O[train_idx]
        B_test, O_test = B[val_idx], O[val_idx]
        M_train = len(train_idx)
        M_test = len(val_idx)

        print(f"  Fold {fold_idx}: {M_train} train / {M_test} test",
              end="", flush=True)

        # Estimate correlation (still needed for selection)
        try:
            R, col_mean, col_std = estimate_corr(B_train, O_train)
        except Exception as e:
            print(f" — estimation failed: {e}", flush=True)
            for method in METHODS:
                metrics[method]["r2"].append([np.nan] * K_MAX)
                metrics[method]["rmse"].append([np.nan] * K_MAX)
            continue

        # Run all three selection methods
        res_ent = greedy_entropy(R, K_MAX, benchmark_names=benchmark_names)
        res_mi = greedy_mi(R, K_MAX, benchmark_names=benchmark_names)
        rand_seed = (SEED + 7, fold_idx, int(p * 10000))
        res_rand = random_select(R, K_MAX, seed=rand_seed,
                                 benchmark_names=benchmark_names)

        results = {"entropy": res_ent, "mi": res_mi, "random": res_rand}
        for method in METHODS:
            result = results[method]
            fold_r2, fold_rmse = [], []
            for k in range(1, K_MAX + 1):
                selected = result["selected"][:k]
                imp = impute_tabimpute(imputer, B_train, O_train,
                                       B_test, O_test, selected,
                                       col_mean, col_std, N)
                fold_r2.append(imp["r2"])
                fold_rmse.append(imp["rmse"])

            metrics[method]["r2"].append(fold_r2)
            metrics[method]["rmse"].append(fold_rmse)

        elapsed = time.time() - t0
        report_k = min(5, K_MAX)
        report_parts = []
        for method in METHODS:
            value = metrics[method]["r2"][-1][report_k - 1]
            report_parts.append(f"{method} R²@{report_k}={value:.3f}")
        print("  " + "  ".join(report_parts) + f"  ({elapsed:.1f}s)", flush=True)

    # Convert to arrays
    for method in METHODS:
        for key in metrics[method]:
            metrics[method][key] = np.array(metrics[method][key])

    return metrics, benchmark_names


# ── Plotting ──────────────────────────────────────────────────────

def plot_comparison(ds_name, metrics):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ks = np.arange(1, K_MAX + 1)

    colors = {"entropy": "#4477AA", "mi": "#CC3311", "random": "#888888"}
    labels = {"entropy": "Entropy (Alg. 1)", "mi": "Mutual Info (Alg. 2)",
              "random": "Random"}
    styles = {"entropy": "o-", "mi": "o-", "random": "x--"}

    for method in METHODS:
        c = colors[method]
        ls = styles[method]
        lw = 1.5 if method != "random" else 1.0
        ms = 4 if method != "random" else 3
        valid = ~np.all(np.isnan(metrics[method]["r2"]), axis=1)
        if not valid.any():
            continue

        mean_r2 = np.nanmean(metrics[method]["r2"][valid], axis=0)
        std_r2 = np.nanstd(metrics[method]["r2"][valid], axis=0)

        ax.plot(ks, mean_r2, ls, color=c, label=labels[method],
                markersize=ms, linewidth=lw)
        ax.fill_between(ks, mean_r2 - std_r2, mean_r2 + std_r2,
                        color=c, alpha=0.15)

    ax.set_ylabel(f"Imputation $R^2$ (TabImpute {TABIMPUTE_LABEL})")
    ax.set_xlabel("Selected benchmarks $k$")
    ax.set_xlim(0.5, K_MAX + 0.5)
    ax.set_xticks([k for k in [1, 3, 5, 7, 10, 13, 15] if k <= K_MAX])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"{DISPLAY_NAME[ds_name]}")

    plt.tight_layout()
    out_path = OUT / f"tabimpute_{TABIMPUTE_LABEL}_r2_{ds_name}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\nPlot saved to {out_path}", flush=True)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    total_t0 = time.time()

    print(f"Loading TabImpute model: {TABIMPUTE_MODEL}")
    device = os.environ.get("BENCHSELECT_TABIMPUTE_DEVICE")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.environ.get("BENCHSELECT_TABIMPUTE_CHECKPOINT")
    imputer_max_rows_env = os.environ.get("BENCHSELECT_IMPUTER_MAX_ROWS")
    imputer_max_rows = (
        int(imputer_max_rows_env) if imputer_max_rows_env is not None else None
    )
    imputer_max_cols_env = os.environ.get("BENCHSELECT_IMPUTER_MAX_COLS")
    imputer_max_cols = (
        int(imputer_max_cols_env) if imputer_max_cols_env is not None else None
    )
    prediction_stat = os.environ.get("BENCHSELECT_TABIMPUTE_PREDICTION_STAT", "mean")
    if TABIMPUTE_MODEL in {"v2", "tabimputev2", "tabimpute-v2"}:
        imputer = TabImputeV2(
            device=device,
            checkpoint_path=checkpoint_path,
            max_num_rows=imputer_max_rows,
        )
    elif TABIMPUTE_MODEL in {"current", "latest", "tabimpute_001", "001"}:
        if checkpoint_path is None:
            checkpoint_path = hf_hub_download(
                repo_id="Tabimpute/TabImpute", filename="tabimpute_001.ckpt"
            )
        imputer = ImputePFN(
            device=device,
            checkpoint_path=checkpoint_path,
            entry_wise_features=True,
            max_num_rows=imputer_max_rows,
            max_num_cols=imputer_max_cols,
        )
    elif TABIMPUTE_MODEL in {"direct_checkpoint", "checkpoint"}:
        if checkpoint_path is None:
            raise ValueError(
                "BENCHSELECT_TABIMPUTE_CHECKPOINT is required for "
                "BENCHSELECT_TABIMPUTE_MODEL=direct_checkpoint"
            )
        cfg_path = pathlib.Path(checkpoint_path).parent / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing checkpoint config: {cfg_path}")
        training_cfg = json.loads(cfg_path.read_text())
        model_cfg = training_cfg.get("model") or training_cfg.get("config")
        if model_cfg is None:
            raise ValueError(f"Could not find model config in {cfg_path}")
        imputer = ImputePFN(
            device=device,
            checkpoint_path=checkpoint_path,
            entry_wise_features=False,
            json_config={"config": model_cfg},
            model_family=os.environ.get(
                "BENCHSELECT_TABIMPUTE_MODEL_FAMILY",
                training_cfg.get("model_family", "new_stable"),
            ),
            precision_mode=os.environ.get("BENCHSELECT_TABIMPUTE_PRECISION_MODE", "full_bfloat16"),
            max_num_rows=imputer_max_rows,
            max_num_cols=imputer_max_cols,
            prediction_stat=prediction_stat,
        )
    elif TABIMPUTE_MODEL in {"rfm_agop_direct", "agop_direct"}:
        if checkpoint_path is None:
            raise ValueError(
                "BENCHSELECT_TABIMPUTE_CHECKPOINT is required for "
                "BENCHSELECT_TABIMPUTE_MODEL=rfm_agop_direct"
            )
        cfg_path = pathlib.Path(checkpoint_path).parent / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing checkpoint config: {cfg_path}")
        training_cfg = json.loads(cfg_path.read_text())
        model_cfg = training_cfg.get("model") or training_cfg.get("config")
        if model_cfg is None:
            raise ValueError(f"Could not find model config in {cfg_path}")

        agop_iters = int(os.environ.get("BENCHSELECT_RFM_AGOP_ITERS", "3"))
        num_anchors = int(os.environ.get("BENCHSELECT_RFM_NUM_ANCHORS", "1024"))
        lambda_init = float(os.environ.get("BENCHSELECT_RFM_LAMBDA_INIT", "0.01"))
        gamma_scale_init = float(
            os.environ.get("BENCHSELECT_RFM_GAMMA_SCALE_INIT", "1.0")
        )
        head = RFMHeadAGOP(
            lambda_init=lambda_init,
            gamma_scale_init=gamma_scale_init,
            iters=agop_iters,
            detach_M=True,
        )
        head_checkpoint = os.environ.get("BENCHSELECT_RFM_HEAD_CHECKPOINT")
        if head_checkpoint:
            ck = torch.load(head_checkpoint, map_location="cpu", weights_only=False)
            head.load_state_dict(ck.get("head", ck))

        imputer = RFMImputePFN(
            device=device,
            checkpoint_path=checkpoint_path,
            entry_wise_features=False,
            json_config={"config": model_cfg},
            model_family=os.environ.get(
                "BENCHSELECT_TABIMPUTE_MODEL_FAMILY",
                training_cfg.get("model_family", "new_stable"),
            ),
            max_num_rows=imputer_max_rows,
            max_num_cols=imputer_max_cols,
            head=head,
            num_anchors=num_anchors,
            per_matrix_steps=int(os.environ.get("BENCHSELECT_RFM_TUNE_STEPS", "0")),
            per_matrix_lr=float(os.environ.get("BENCHSELECT_RFM_TUNE_LR", "0.01")),
            per_matrix_val_frac=float(
                os.environ.get("BENCHSELECT_RFM_TUNE_VAL_FRAC", "0.1")
            ),
            per_matrix_patch_size=int(
                os.environ.get("BENCHSELECT_RFM_TUNE_PATCH_SIZE", "50")
            ),
            per_matrix_seed=int(os.environ.get("BENCHSELECT_RFM_TUNE_SEED", "0")),
        )
    elif TABIMPUTE_MODEL in {"manifest", "direct"}:
        manifest_path = pathlib.Path(
            os.environ.get(
                "BENCHSELECT_TABIMPUTE_MANIFEST",
                "/root/tabular/mcpfn/benchmark/movielens/configs/tabimpute_models.json",
            )
        )
        manifest_name = os.environ["BENCHSELECT_TABIMPUTE_MANIFEST_NAME"]
        manifest = json.loads(manifest_path.read_text())
        model_entry = None
        for candidate in manifest["models"]:
            aliases = {candidate["name"], *candidate.get("aliases", [])}
            if manifest_name in aliases:
                model_entry = candidate
                break
        if model_entry is None:
            raise ValueError(f"No manifest TabImpute model named {manifest_name!r}")
        if checkpoint_path is None:
            checkpoint_path = model_entry["checkpoint_path"]
        imputer = ImputePFN(
            device=device,
            checkpoint_path=checkpoint_path,
            entry_wise_features=False,
            json_config={"config": model_entry["model_config"]},
            model_family=model_entry.get("model_family", "new_stable"),
            max_num_rows=imputer_max_rows,
            max_num_cols=imputer_max_cols,
            prediction_stat=prediction_stat,
        )
    elif TABIMPUTE_MODEL in {"compressed", "compressor"}:
        if checkpoint_path is None:
            checkpoint_path = os.environ["BENCHSELECT_COMPRESSED_CHECKPOINT"]
        chunk_rows_env = os.environ.get("BENCHSELECT_COMPRESSED_CHUNK_ROWS")
        chunk_cols_env = os.environ.get("BENCHSELECT_COMPRESSED_CHUNK_COLS")
        imputer = CompressedCrossImputePFN(
            compressed_checkpoint_path=checkpoint_path,
            chunk_rows=int(chunk_rows_env) if chunk_rows_env is not None else None,
            chunk_cols=int(chunk_cols_env) if chunk_cols_env is not None else None,
            device=device,
        )
    else:
        raise ValueError(
            "BENCHSELECT_TABIMPUTE_MODEL must be one of: "
            "v2, current, manifest, direct_checkpoint, rfm_agop_direct, compressed"
        )
    print(f"Model loaded on {device}.\n")

    all_metrics = {}
    for ds_name in DATASETS:
        ds_t0 = time.time()
        metrics, bench_names = run_dataset(ds_name, imputer)
        all_metrics[ds_name] = metrics
        plot_comparison(ds_name, metrics)

        # Save results
        save_dict = {"benchmark_names": bench_names}
        for method in METHODS:
            for key, arr in metrics[method].items():
                save_dict[f"{method}_{key}"] = arr
        np.savez(LOGS / f"tabimpute_{TABIMPUTE_LABEL}_{ds_name}.npz", **save_dict)

        ds_elapsed = time.time() - ds_t0
        print(f"\n{ds_name.upper()} completed in {ds_elapsed:.0f}s", flush=True)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: Mean R² (TabImpute {TABIMPUTE_LABEL} imputation)")
    print(f"{'='*70}")
    report_ks = [k for k in [5, 10, 15] if k <= K_MAX]
    if not report_ks:
        report_ks = [K_MAX]
    metric_header = " ".join([f"R²@{k:>2}" for k in report_ks])
    print(f"{'Dataset':<12} {'Method':<10} {metric_header:>26}")
    print("-" * 50)
    for ds_name in DATASETS:
        for method in METHODS:
            valid = ~np.all(np.isnan(all_metrics[ds_name][method]["r2"]), axis=1)
            if valid.any():
                mr2 = np.nanmean(all_metrics[ds_name][method]["r2"][valid], axis=0)
                values = " ".join([f"{mr2[k - 1]:>8.4f}" for k in report_ks])
                print(f"{DISPLAY_NAME[ds_name]:<12} {method:<10} {values}")
        print("-" * 50)

    total_elapsed = time.time() - total_t0
    print(f"\nAll datasets completed in {total_elapsed:.0f}s")
