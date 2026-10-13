"""
Imputation methods compared in the benchmark.

Each method is registered under a short key (used on the CLI and as the file name
of its saved imputations) with a display name used in figures and tables. The
settings mirror the ones used to produce the paper results.

`legacy_files` / `legacy_time_files` list the file names the original benchmark
scripts used, so result folders produced before this package existed can still be
evaluated and plotted.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import pandas as pd

from .patterns import PATTERNS

BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Imputer = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Method:
    key: str
    display: str
    # build(device, seed) -> function mapping X_missing to X_imputed
    build: Callable[[str, int], Imputer]
    uses_gpu: bool = False
    # Run on CPU even if a GPU is available
    cpu_only: bool = False
    legacy_files: tuple = ()
    legacy_time_files: tuple = ()
    # pip packages / git submodules this method needs (shown when the import fails)
    requires: str = ""
    # Modules to import before PyTorch. On macOS, xgboost and lightgbm deadlock or crash
    # if their OpenMP runtime is loaded after PyTorch's.
    preload: tuple = ()


def fill_all_nan_columns(X: np.ndarray) -> np.ndarray:
    """Fill columns that are entirely NaN with 0's, leaving other NaN values unchanged."""
    X = X.copy()
    X[:, np.all(np.isnan(X), axis=0)] = 0.0
    return X


def fill_all_nan_rows(X: np.ndarray) -> np.ndarray:
    """Fill rows that are entirely NaN with column means."""
    X = X.copy()
    all_nan_rows = np.all(np.isnan(X), axis=1)
    if all_nan_rows.any():
        X[all_nan_rows, :] = np.nanmean(X, axis=0)
    return X


# --- TabImpute ---------------------------------------------------------------

def _tabimpute(checkpoint_path: str | None = None) -> Callable[[str, int], Imputer]:
    def build(device: str, seed: int) -> Imputer:
        from tabimpute.interface import ImputePFN
        from tabimpute.prepreocess import (
            RandomColumnPermutation,
            RandomRowColumnPermutation,
            RandomRowPermutation,
        )

        model = ImputePFN(
            device=device,
            nhead=2,
            preprocessors=[
                RandomRowColumnPermutation(),
                RandomRowColumnPermutation(),
                RandomRowPermutation(),
                RandomColumnPermutation(),
            ],
            checkpoint_path=checkpoint_path,
        )
        return lambda X: model.impute(X.copy())

    return build


def _tabimpute_v2(device: str, seed: int) -> Imputer:
    from tabimpute.tabimpute_v2 import TabImputeV2

    model = TabImputeV2(device=device)
    return lambda X: model.impute(X.copy())


# --- HyperImpute plugins -----------------------------------------------------

def _hyperimpute(plugin_name: str, fill_rows: bool = True, zeros_on_error: bool = True):
    def build(device: str, seed: int) -> Imputer:
        from hyperimpute.plugins.imputers import Imputers

        def impute(X: np.ndarray) -> np.ndarray:
            X = fill_all_nan_columns(X.astype(np.float64))
            if fill_rows:
                X = fill_all_nan_rows(X)
            try:
                plugin = Imputers().get(plugin_name, random_state=seed)
                return plugin.fit_transform(pd.DataFrame(X)).to_numpy()
            except Exception as e:
                if not zeros_on_error:
                    raise
                # The paper results score a failed run as an all-zero imputation
                print(f"    {plugin_name} failed ({e}); saving zeros")
                return np.zeros_like(X)

        return impute

    return build


def _mice(device: str, seed: int) -> Imputer:
    import miceforest as mf

    def impute(X: np.ndarray) -> np.ndarray:
        X = fill_all_nan_rows(fill_all_nan_columns(X))
        df = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
        num_imputations = 5
        kernel = mf.ImputationKernel(
            df,
            num_datasets=num_imputations,
            # Avoid mean-matching failures on small benchmark tables
            mean_match_candidates=0,
            random_state=seed,
        )
        kernel.mice(iterations=5, verbose=False)
        # Average across the imputed datasets
        return np.mean([kernel.complete_data(dataset=i).to_numpy() for i in range(num_imputations)], axis=0)

    return impute


# --- Other baselines ---------------------------------------------------------

def _knn(device: str, seed: int) -> Imputer:
    from sklearn.impute import KNNImputer

    return lambda X: KNNImputer(n_neighbors=5).fit_transform(fill_all_nan_columns(X))


def _tabpfn(device: str, seed: int) -> Imputer:
    from tabimpute.interface import TabPFNUnsupervisedModel

    model = TabPFNUnsupervisedModel(device=device)
    return lambda X: model.impute(fill_all_nan_columns(X))


def _ewf_tabpfn(device: str, seed: int) -> Imputer:
    from tabimpute.interface import TabPFNImputer

    model = TabPFNImputer(device=device)
    return lambda X: model.impute(X.copy())


def _forestdiffusion(device: str, seed: int) -> Imputer:
    from ForestDiffusion import ForestDiffusionModel

    def impute(X: np.ndarray) -> np.ndarray:
        # Per the authors, vp diffusion is required for imputation
        model = ForestDiffusionModel(
            fill_all_nan_rows(fill_all_nan_columns(X)),
            n_t=50,
            diffusion_type="vp",
            cat_indexes=[],
            int_indexes=[],
            n_jobs=-1,
        )
        return model.impute(k=1)

    return impute


def _remasker(device: str, seed: int) -> Imputer:
    for path in (BENCHMARK_DIR, os.path.join(BENCHMARK_DIR, "remasker")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from remasker.remasker_impute import ReMasker

    if "float" not in np.__dict__:
        np.float = float  # ReMasker uses this alias, which NumPy 1.24 removed

    def impute(X: np.ndarray) -> np.ndarray:
        # ReMasker reads its hyperparameters from the command line; give it the defaults
        argv, sys.argv = sys.argv, sys.argv[:1]
        try:
            model = ReMasker()
        finally:
            sys.argv = argv
        return np.asarray(model.fit_transform(pd.DataFrame(fill_all_nan_columns(X))))

    return impute


def _cacti(device: str, seed: int) -> Imputer:
    from .baselines.cacti import CACTIImputer

    def impute(X: np.ndarray) -> np.ndarray:
        model = CACTIImputer(device=device, model="CMAE", mask_ratio=0.9, epochs=300)
        return model.fit_transform(fill_all_nan_columns(X))

    return impute


def _diffputer(device: str, seed: int) -> Imputer:
    from .baselines.diffputer import DiffPuterImputer

    model = DiffPuterImputer(device=device)
    return lambda X: model.fit_transform(X.copy())


def _notmiwae(missing_process: str):
    def build(device: str, seed: int) -> Imputer:
        from .baselines.notmiwae import NotMIWAEImputer

        model = NotMIWAEImputer(device=device, missing_process=missing_process)
        return lambda X: model.fit_transform(fill_all_nan_rows(fill_all_nan_columns(X)))

    return build


_HYPERIMPUTE = "pip install hyperimpute"

METHODS: dict[str, Method] = {
    m.key: m
    for m in [
        Method("tabimpute", "TabImpute", _tabimpute(), uses_gpu=True,
               legacy_files=("masters_mcar",), legacy_time_files=("mcpfn", "masters_mcar")),
        Method("tabimpute_cpu", "TabImpute (CPU)", _tabimpute(), cpu_only=True,
               legacy_time_files=("mcpfn_cpu",)),
        Method("tabimpute_v2", "TabImpute V2", _tabimpute_v2, uses_gpu=True),
        Method("tabpfn", "TabPFN", _tabpfn, uses_gpu=True,
               legacy_files=("tabpfn_impute",), legacy_time_files=("tabpfn_unsupervised",),
               requires="pip install tabpfn_extensions"),
        Method("ewf_tabpfn", "EWF-TabPFN", _ewf_tabpfn, uses_gpu=True,
               legacy_files=("tabpfn",), requires="pip install tabpfn_extensions"),
        Method("hyperimpute", "HyperImpute", _hyperimpute("hyperimpute"), uses_gpu=True,
               legacy_time_files=("hyperimpute_hyperimpute",), requires=_HYPERIMPUTE),
        Method("missforest", "MissForest", _hyperimpute("missforest"),
               legacy_time_files=("hyperimpute_hyperimpute_missforest",), requires=_HYPERIMPUTE),
        Method("ot", "OT", _hyperimpute("sinkhorn", zeros_on_error=False),
               legacy_files=("ot_sinkhorn",), legacy_time_files=("hyperimpute_ot_sinkhorn",),
               requires=_HYPERIMPUTE),
        Method("ice", "ICE", _hyperimpute("ice"),
               legacy_time_files=("hyperimpute_hyperimpute_ice",), requires=_HYPERIMPUTE),
        Method("mice", "MICE-Forest", _mice, legacy_time_files=("hyperimpute_hyperimpute_mice",),
               requires="pip install miceforest", preload=("lightgbm",)),
        Method("gain", "GAIN", _hyperimpute("gain"), uses_gpu=True,
               legacy_time_files=("hyperimpute_hyperimpute_gain",), requires=_HYPERIMPUTE),
        Method("miwae", "MIWAE", _hyperimpute("miwae"), uses_gpu=True,
               legacy_time_files=("hyperimpute_hyperimpute_miwae",), requires=_HYPERIMPUTE),
        Method("softimpute", "SoftImpute", _hyperimpute("softimpute", fill_rows=False, zeros_on_error=False),
               requires=_HYPERIMPUTE),
        Method("mean", "Col Mean", _hyperimpute("mean", fill_rows=False, zeros_on_error=False),
               legacy_files=("column_mean",), requires=_HYPERIMPUTE),
        Method("knn", "K-NN", _knn),
        Method("forestdiffusion", "ForestDiffusion", _forestdiffusion, requires="pip install ForestDiffusion",
               preload=("xgboost",)),
        Method("remasker", "ReMasker", _remasker, uses_gpu=True,
               requires="git submodule update --init mcpfn/benchmark/remasker && pip install timm"),
        Method("cacti", "CACTI", _cacti, uses_gpu=True,
               requires="git submodule update --init mcpfn/benchmark/CACTI"),
        Method("diffputer", "DiffPuter", _diffputer, uses_gpu=True,
               requires="git submodule update --init mcpfn/benchmark/DiffPuter"),
        Method("notmiwae", "not-MIWAE", _notmiwae("selfmasking"), uses_gpu=True,
               requires="pip install notmiwae-pytorch"),
        Method("notmiwae_self_mnar", "not-MIWAE (Self-MNAR)", _notmiwae("self_mnar"), uses_gpu=True,
               legacy_files=("not-miwae-self-mnar",), requires="pip install notmiwae-pytorch"),
    ]
}

# Methods in the paper's main comparison (Tables 1/3/4, Figs. 1/5)
PAPER_METHODS = [
    "tabimpute", "missforest", "knn", "ot", "hyperimpute", "mice", "ice", "softimpute",
    "forestdiffusion", "miwae", "cacti", "notmiwae", "tabpfn", "remasker", "gain",
]
# Methods that need no GPU, git submodules or extra pip packages beyond `tabimpute[benchmark]`
QUICK_METHODS = ["tabimpute", "knn", "mean", "softimpute", "missforest", "ice", "mice", "ot", "hyperimpute"]
METHOD_GROUPS = {
    "paper": PAPER_METHODS,
    "quick": QUICK_METHODS,
    "all": [k for k in METHODS],
}

# Pattern-specific TabImpute models ("TabImpute-P"): method key per training pattern.
# These checkpoints are not on HuggingFace; pass them with --checkpoint PATTERN=PATH.
# `legacy` is the file name the original scripts used for each specialist. The NN-MNAR
# and Block-MNAR names are intentionally crossed: those two runs were saved under each
# other's names, and the paper's plotting code swapped them back when loading.
_SPECIALIST_LEGACY = {
    "mcar": "masters_mcar",
    "col_mar": "masters_mar",
    "self_mnar": "masters_mnar",
    "nn_mnar": "masters_mar_block_neural_lr_4e-4_decay_1e-2",
    "block_mnar": "masters_mar_neural_lr_3e-4_decay_1e-2",
    "pol_mnar": "masters_mnar_polarization",
    "soft_pol_mnar": "masters_mnar_soft_polarization",
    "latent_mnar": "masters_mnar_latent_factor_lr_3e-4_decay_2e-2",
    "clust_mnar": "masters_mnar_cluster_level_lr_2e-4_decay_1e-2",
    "phase_mnar": "masters_mnar_two_phase_subset_lr_3e-4_decay_1e-2",
    "censor_mnar": "masters_mnar_censoring",
}


def specialist_key(pattern_key: str) -> str:
    """Method key of the TabImpute model trained on `pattern_key` (MCAR is the default model)."""
    return "tabimpute" if pattern_key == "mcar" else f"tabimpute_{pattern_key}"


def _needs_checkpoint(key: str) -> Callable[[str, int], Imputer]:
    def build(device: str, seed: int) -> Imputer:
        raise SystemExit(f"{key} needs a model checkpoint: pass --checkpoint {key}=PATH")

    return build


def specialist(pattern_key: str) -> Method:
    key = specialist_key(pattern_key)
    return Method(
        key,
        f"TabImpute-{PATTERNS[pattern_key].display}",
        _needs_checkpoint(key),
        uses_gpu=True,
        legacy_files=(_SPECIALIST_LEGACY[pattern_key],),
    )


for _pattern_key in PATTERNS:
    if _pattern_key != "mcar":
        METHODS[specialist_key(_pattern_key)] = specialist(_pattern_key)

# TabImpute trained with a nonlinear factor model (paper Table 7)
METHODS["tabimpute_nonlinear"] = Method(
    "tabimpute_nonlinear", "TabImpute (Nonlinear)", _needs_checkpoint("tabimpute_nonlinear"), uses_gpu=True,
    legacy_files=("masters_mcar_nonlinear",),
)

CHECKPOINT_METHODS = [specialist_key(k) for k in PATTERNS if k != "mcar"] + ["tabimpute_nonlinear"]


def get_method(key: str, checkpoint_path: str | None = None) -> Method:
    """The method registered as `key`, loading `checkpoint_path` if given (TabImpute models only)."""
    if checkpoint_path is None:
        return METHODS[key]
    if key != "tabimpute" and key not in CHECKPOINT_METHODS:
        raise SystemExit(f"--checkpoint only applies to TabImpute models: tabimpute, {', '.join(CHECKPOINT_METHODS)}")
    return replace(METHODS[key], build=_tabimpute(checkpoint_path))


def resolve_methods(keys: list[str]) -> list[str]:
    out: list[str] = []
    for k in keys:
        for key in METHOD_GROUPS.get(k, [k]):
            if key not in METHODS:
                raise SystemExit(f"Unknown method {key!r}. Choose from: {', '.join(METHODS)} "
                                 f"or a group: {', '.join(METHOD_GROUPS)}")
            if key not in out:
                out.append(key)
    return out


def display_name(key: str) -> str:
    return METHODS[key].display if key in METHODS else key
