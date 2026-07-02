import os
import warnings
import json
import numpy as np
import pandas as pd
import torch
import miceforest as mf
from tqdm import tqdm
from pathlib import Path
import sys
from tabimpute.interface import ImputePFN, TabPFNImputer, TabPFNUnsupervisedModel
from tabimpute.interface_rfm import RFMImputePFN, BoostMatFormerImputer
from tabimpute.model.rfm_head_agop import RFMHeadAGOP
from tabimpute.prepreocess import (
    RandomRowColumnPermutation, 
    RandomRowPermutation, 
    RandomColumnPermutation, 
)
from hyperimpute.plugins.imputers import Imputers
from sklearn.impute import KNNImputer
from fancyimpute import SoftImpute as FancySoftImpute
import time
import itertools

# Optional: ForestDiffusion (pip install ForestDiffusion)
try:
    from ForestDiffusion import ForestDiffusionModel
    HAS_FORESTDIFFUSION = True
except ImportError:
    HAS_FORESTDIFFUSION = False
import shutil

UCI_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = UCI_DIR.parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from diffputer_wrapper import DiffPuterImputer
from remasker_wrapper import ReMaskerImputer
from cacti_wrapper import CACTIImputer
from notmiwae_wrapper import NotMIWAEImputer

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
])

MCPFN_SHARED_KWARGS = {
    "device": "cuda",
    "nhead": 2,
    "preprocessors": None,
    "entry_wise_features": False,
    "max_num_rows": 200,
    "max_num_chunks": 4,
}

ROUND2_T2_CONFIG = {"config": {
            "embedding_size": 768,
            "num_attention_heads": 16,
            "mlp_hidden_size": 1536,
            "num_layers": 8,
            "num_outputs": 5000,
            "num_cls": 12,
            "use_rope": True,
            "rope_base": 10000.0,
            "rope_fraction": 0.5,
            "use_absolute_positional_embeddings": False,
            "positional_damping_factor": 0.1,
            "attention_dropout": 0.0,
            "ffn_dropout": 0.05,
            "drop_path_rate": 0.01,
            "residual_scale_init": 0.5,
            "embedding_dropout": 0.0,
            "rms_norm_eps": 1e-6,
        }}

DWIP_V3R1_MANIFEST_PATH = Path(
    "/root/tabular/mcpfn/src/tabimpute/workdir/"
    "dwip_v3r1_9run_downstream_eval_manifest_20260415.json"
)


def load_manifest_mcpfn_specs(manifest_path: Path) -> list[dict]:
    if not manifest_path.is_file():
        return []

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    specs = []
    for run in manifest.get("runs", []):
        model_cfg = dict(run.get("model_cfg", {}))
        checkpoint_path = run.get("selection", {}).get("selected_checkpoint_path")
        trial_run_name = run.get("trial_run_name")
        source_model_file = str(run.get("source_model_file", ""))
        if not model_cfg or not checkpoint_path or not trial_run_name:
            continue

        model_family = "new_stable"
        if source_model_file.endswith("model_new.py"):
            model_family = "new"
        elif source_model_file.endswith("model_new_stable.py"):
            model_family = "new_stable"

        specs.append(
            {
                "name": trial_run_name,
                "model_family": model_family,
                "checkpoint_path": checkpoint_path,
                "json_config": {"config": model_cfg},
                "legacy_name_aliases": list(run.get("legacy_name_aliases", [])),
            }
        )

    return specs

MCPFN_IMPUTER_SPECS = [
    # {
    #     "name": "dwip_v2-trial_001",
    #     "model_family": "new_stable",
    #     "checkpoint_path": "/mnt/checkpoints_sweep/workdir/dwip_v2-trial_001/checkpoint_23000.pth",
    #     "json_config": {"config": {
    #         "embedding_size": 768,
    #         "num_attention_heads": 16,
    #         "mlp_hidden_size": 1536,
    #         "num_layers": 8,
    #         "num_outputs": 5000,
    #         "num_cls": 12,
    #         "use_rope": True,
    #         "rope_base": 10000.0,
    #         "rope_fraction": 0.5,
    #         "use_absolute_positional_embeddings": False,
    #         "positional_damping_factor": 0.1,
    #         "attention_dropout": 0.0,
    #         "ffn_dropout": 0.03,
    #         "drop_path_rate": 0.0,
    #         "residual_scale_init": 0.7,
    #         "embedding_dropout": 0.0,
    #         "rms_norm_eps": 1e-6,
    #     }},
    # },
    # {
    #     "name": "dwip_v2-trial_002",
    #     "model_family": "new_stable",
    #     "checkpoint_path": "/mnt/checkpoints_sweep/workdir/dwip_v2-trial_002/checkpoint_17000.pth",
    #     "json_config": {"config": {
    #         "embedding_size": 768,
    #         "num_attention_heads": 16,
    #         "mlp_hidden_size": 1536,
    #         "num_layers": 8,
    #         "num_outputs": 5000,
    #         "num_cls": 12,
    #         "use_rope": True,
    #         "rope_base": 10000.0,
    #         "rope_fraction": 0.5,
    #         "use_absolute_positional_embeddings": False,
    #         "positional_damping_factor": 0.1,
    #         "attention_dropout": 0.0,
    #         "ffn_dropout": 0.03,
    #         "drop_path_rate": 0.0,
    #         "residual_scale_init": 0.7,
    #         "embedding_dropout": 0.0,
    #         "rms_norm_eps": 1e-6,
    #     }},
    # },
    # {
    #     "name": "dwip_v2-trial_003",
    #     "model_family": "new_stable",
    #     "checkpoint_path": "/mnt/checkpoints_sweep/workdir/dwip_v2-trial_003/checkpoint_19000.pth",
    #     "json_config": {"config": {
    #         "embedding_size": 768,
    #         "num_attention_heads": 16,
    #         "mlp_hidden_size": 1536,
    #         "num_layers": 8,
    #         "num_outputs": 5000,
    #         "num_cls": 12,
    #         "use_rope": True,
    #         "rope_base": 10000.0,
    #         "rope_fraction": 0.5,
    #         "use_absolute_positional_embeddings": False,
    #         "positional_damping_factor": 0.1,
    #         "attention_dropout": 0.0,
    #         "ffn_dropout": 0.03,
    #         "drop_path_rate": 0.0,
    #         "residual_scale_init": 0.7,
    #         "embedding_dropout": 0.0,
    #         "rms_norm_eps": 1e-6,
    #     }},
    # },
    # {
    #     "name": "dwip_v2-trial_004",
    #     "model_family": "new_stable",
    #     "checkpoint_path": "/mnt/checkpoints_sweep/workdir/dwip_v2-trial_004/checkpoint_17000.pth",
    #     "json_config": {"config": {
    #         "embedding_size": 768,
    #         "num_attention_heads": 16,
    #         "mlp_hidden_size": 1536,
    #         "num_layers": 8,
    #         "num_outputs": 5000,
    #         "num_cls": 12,
    #         "use_rope": True,
    #         "rope_base": 10000.0,
    #         "rope_fraction": 0.5,
    #         "use_absolute_positional_embeddings": False,
    #         "positional_damping_factor": 0.1,
    #         "attention_dropout": 0.0,
    #         "ffn_dropout": 0.05,
    #         "drop_path_rate": 0.01,
    #         "residual_scale_init": 0.5,
    #         "embedding_dropout": 0.0,
    #         "rms_norm_eps": 1e-6,
    #     }},
    # },
    # {
    #     "name": "dwip_v2-trial_005",
    #     "model_family": "new_stable",
    #     "checkpoint_path": "/mnt/checkpoints_sweep/workdir/dwip_v2-trial_005/checkpoint_23000.pth",
    #     "json_config": {"config": {
    #         "embedding_size": 768,
    #         "num_attention_heads": 16,
    #         "mlp_hidden_size": 1536,
    #         "num_layers": 8,
    #         "num_outputs": 5000,
    #         "num_cls": 12,
    #         "use_rope": True,
    #         "rope_base": 10000.0,
    #         "rope_fraction": 0.5,
    #         "use_absolute_positional_embeddings": False,
    #         "positional_damping_factor": 0.1,
    #         "attention_dropout": 0.0,
    #         "ffn_dropout": 0.05,
    #         "drop_path_rate": 0.01,
    #         "residual_scale_init": 0.5,
    #         "embedding_dropout": 0.0,
    #         "rms_norm_eps": 1e-6,
    #     }},
    # },
    # {
    #     "name": "dwip_v2-trial_006",
    #     "model_family": "new_stable",
    #     "checkpoint_path": "/mnt/checkpoints_sweep/workdir/dwip_v2-trial_006/checkpoint_18000.pth",
    #     "json_config": {"config": {
    #         "embedding_size": 768,
    #         "num_attention_heads": 16,
    #         "mlp_hidden_size": 1536,
    #         "num_layers": 8,
    #         "num_outputs": 5000,
    #         "num_cls": 12,
    #         "use_rope": True,
    #         "rope_base": 10000.0,
    #         "rope_fraction": 0.5,
    #         "use_absolute_positional_embeddings": False,
    #         "positional_damping_factor": 0.1,
    #         "attention_dropout": 0.0,
    #         "ffn_dropout": 0.05,
    #         "drop_path_rate": 0.01,
    #         "residual_scale_init": 0.5,
    #         "embedding_dropout": 0.0,
    #         "rms_norm_eps": 1e-6,
    #     }},
    # },
    # {
    #     "name": "dwip_v2-trial_007",
    #     "model_family": "new_stable",
    #     "checkpoint_path": "/mnt/checkpoints_sweep/workdir/dwip_v2-trial_007/checkpoint_8000.pth",
    #     "json_config": {"config": {
    #         "embedding_size": 768,
    #         "num_attention_heads": 16,
    #         "mlp_hidden_size": 1536,
    #         "num_layers": 8,
    #         "num_outputs": 5000,
    #         "num_cls": 12,
    #         "use_rope": True,
    #         "rope_base": 10000.0,
    #         "rope_fraction": 0.75,
    #         "use_absolute_positional_embeddings": False,
    #         "positional_damping_factor": 0.1,
    #         "attention_dropout": 0.005,
    #         "ffn_dropout": 0.03,
    #         "drop_path_rate": 0.005,
    #         "residual_scale_init": 0.5,
    #         "embedding_dropout": 0.0,
    #         "rms_norm_eps": 1e-6,
    #     }},
    # },
    # {
    #     "name": "dwip_v2-trial_008",
    #     "model_family": "new_stable",
    #     "checkpoint_path": "/mnt/checkpoints_sweep/workdir/dwip_v2-trial_008/checkpoint_6000.pth",
    #     "json_config": {"config": {
    #         "embedding_size": 768,
    #         "num_attention_heads": 16,
    #         "mlp_hidden_size": 1536,
    #         "num_layers": 8,
    #         "num_outputs": 5000,
    #         "num_cls": 12,
    #         "use_rope": True,
    #         "rope_base": 10000.0,
    #         "rope_fraction": 0.75,
    #         "use_absolute_positional_embeddings": False,
    #         "positional_damping_factor": 0.1,
    #         "attention_dropout": 0.005,
    #         "ffn_dropout": 0.03,
    #         "drop_path_rate": 0.005,
    #         "residual_scale_init": 0.5,
    #         "embedding_dropout": 0.0,
    #         "rms_norm_eps": 1e-6,
    #     }},
    # },
    # {
    #     "name": "dwip_v2-trial_009",
    #     "model_family": "new_stable",
    #     "checkpoint_path": "/mnt/checkpoints_sweep/workdir/dwip_v2-trial_009/checkpoint_15000.pth",
    #     "json_config": {"config": {
    #         "embedding_size": 768,
    #         "num_attention_heads": 16,
    #         "mlp_hidden_size": 1536,
    #         "num_layers": 8,
    #         "num_outputs": 5000,
    #         "num_cls": 12,
    #         "use_rope": True,
    #         "rope_base": 10000.0,
    #         "rope_fraction": 0.75,
    #         "use_absolute_positional_embeddings": False,
    #         "positional_damping_factor": 0.1,
    #         "attention_dropout": 0.005,
    #         "ffn_dropout": 0.03,
    #         "drop_path_rate": 0.005,
    #         "residual_scale_init": 0.5,
    #         "embedding_dropout": 0.0,
    #         "rms_norm_eps": 1e-6,
    #     }},
    # },
    # {
    #     "name": "dwip_v2-trial_010",
    #     "model_family": "new_stable",
    #     "checkpoint_path": "/mnt/checkpoints_sweep/workdir/dwip_v2-trial_010/checkpoint_25000.pth",
    #     "json_config": {"config": {
    #         "embedding_size": 768,
    #         "num_attention_heads": 16,
    #         "mlp_hidden_size": 2048,
    #         "num_layers": 12,
    #         "num_outputs": 5000,
    #         "num_cls": 12,
    #         "use_rope": True,
    #         "rope_base": 10000.0,
    #         "rope_fraction": 0.75,
    #         "use_absolute_positional_embeddings": False,
    #         "positional_damping_factor": 0.1,
    #         "attention_dropout": 0.02,
    #         "ffn_dropout": 0.08,
    #         "drop_path_rate": 0.05,
    #         "residual_scale_init": 0.2,
    #         "embedding_dropout": 0.0,
    #         "rms_norm_eps": 1e-6,
    #     }},
    # },
    # {
    #     "name": "dwip_v2-trial_011",
    #     "model_family": "new_stable",
    #     "checkpoint_path": "/mnt/checkpoints_sweep/workdir/dwip_v2-trial_011/checkpoint_16000.pth",
    #     "json_config": {"config": {
    #         "embedding_size": 768,
    #         "num_attention_heads": 16,
    #         "mlp_hidden_size": 2048,
    #         "num_layers": 12,
    #         "num_outputs": 5000,
    #         "num_cls": 12,
    #         "use_rope": True,
    #         "rope_base": 10000.0,
    #         "rope_fraction": 0.75,
    #         "use_absolute_positional_embeddings": False,
    #         "positional_damping_factor": 0.1,
    #         "attention_dropout": 0.02,
    #         "ffn_dropout": 0.08,
    #         "drop_path_rate": 0.05,
    #         "residual_scale_init": 0.2,
    #         "embedding_dropout": 0.0,
    #         "rms_norm_eps": 1e-6,
    #     }},
    # },
    # {
    #     "name": "dwip_v2-trial_012",
    #     "model_family": "new_stable",
    #     "checkpoint_path": "/mnt/checkpoints_sweep/workdir/dwip_v2-trial_012/checkpoint_24000.pth",
    #     "json_config": {"config": {
    #         "embedding_size": 768,
    #         "num_attention_heads": 16,
    #         "mlp_hidden_size": 2048,
    #         "num_layers": 12,
    #         "num_outputs": 5000,
    #         "num_cls": 12,
    #         "use_rope": True,
    #         "rope_base": 10000.0,
    #         "rope_fraction": 0.75,
    #         "use_absolute_positional_embeddings": False,
    #         "positional_damping_factor": 0.1,
    #         "attention_dropout": 0.02,
    #         "ffn_dropout": 0.08,
    #         "drop_path_rate": 0.05,
    #         "residual_scale_init": 0.2,
    #         "embedding_dropout": 0.0,
    #         "rms_norm_eps": 1e-6,
    #     }},
    # },
    # {
    #     "name": "tabimpute-new-stable_v2-rank_1_30-50x50",
    #     "model_family": "new_stable",
    #     "checkpoint_path": "/root/tabular/mcpfn/src/tabimpute/workdir/tabimpute-new-stable_v2-rank_1_30-50x50/checkpoint_30000.pth",
    #     "json_config": ROUND2_T2_CONFIG,
    # },
    # {
    #     "name": "tabimpute-new-stable_v2-rank_10_10-50x50",
    #     "model_family": "new_stable",
    #     "checkpoint_path": "/root/tabular/mcpfn/src/tabimpute/workdir/tabimpute-new-stable_v2-rank_10_10-50x50/checkpoint_30000.pth",
    #     "json_config": ROUND2_T2_CONFIG,
    # },
    # {
    #     "name": "tabimpute-new-stable_v2-rank_15_15_from_10_10-50x50",
    #     "model_family": "new_stable",
    #     "checkpoint_path": "/root/tabular/mcpfn/src/tabimpute/workdir/tabimpute-new-stable_v2-rank_15_15-50x50_from_10_10/checkpoint_60000.pth",
    #     "json_config": ROUND2_T2_CONFIG,
    # },
    # {
    #     "name": "tabimpute-new-stable_v2-spectrum-decay-topk_p_0.2-50x50",
    #     "model_family": "new_stable",
    #     "checkpoint_path": "/root/tabular/mcpfn/src/tabimpute/workdir/tabimpute-new-stable_v2-spectrum-decay-topk_p_0.2-50x50/checkpoint_50000.pth",
    #     "json_config": ROUND2_T2_CONFIG,
    # },
    # {
    #     "name": "dwip_v1-trial_003",
    #     "model_family": "new",
    #     "checkpoint_path": "/mnt/checkpoints_sweep/workdir/dwip_v1-trial_003/checkpoint_10000.pth",
    #     "json_config": {"config": {
    #         "embedding_size": 768,
    #         "num_attention_heads": 16,
    #         "mlp_hidden_size": 1536,
    #         "num_layers": 8,
    #         "num_outputs": 5000,
    #         "num_cls": 12,
    #         "use_rope": True,
    #         "rope_base": 10000.0,
    #         "rope_fraction": 0.75,
    #         "use_absolute_positional_embeddings": False,
    #         "positional_damping_factor": 0.1,
    #     }},
    # },
    # {
    #     "name": "dwip_v1-trial_004",
    #     "model_family": "new",
    #     "checkpoint_path": "/mnt/checkpoints_sweep/workdir/dwip_v1-trial_004/checkpoint_7000.pth",
    #     "json_config": {"config": {
    #         "embedding_size": 768,
    #         "num_attention_heads": 16,
    #         "mlp_hidden_size": 1536,
    #         "num_layers": 8,
    #         "num_outputs": 5000,
    #         "num_cls": 12,
    #         "use_rope": True,
    #         "rope_base": 10000.0,
    #         "rope_fraction": 0.5,
    #         "use_absolute_positional_embeddings": False,
    #         "positional_damping_factor": 0.1,
    #     }},
    # },
    # {
    #     "name": "dwip_v1-trial_005",
    #     "model_family": "new",
    #     "checkpoint_path": "/mnt/checkpoints_sweep/workdir/dwip_v1-trial_005/checkpoint_17000.pth",
    #     "json_config": {"config": {
    #         "embedding_size": 768,
    #         "num_attention_heads": 16,
    #         "mlp_hidden_size": 1536,
    #         "num_layers": 8,
    #         "num_outputs": 5000,
    #         "num_cls": 12,
    #         "use_rope": True,
    #         "rope_base": 10000.0,
    #         "rope_fraction": 0.5,
    #         "use_absolute_positional_embeddings": False,
    #         "positional_damping_factor": 0.1,
    #     }},
    # },
    # {
    #     "name": "dwip_v1-trial_006",
    #     "model_family": "new",
    #     "checkpoint_path": "/mnt/checkpoints_sweep/workdir/dwip_v1-trial_006/checkpoint_19000.pth",
    #     "json_config": {"config": {
    #         "embedding_size": 768,
    #         "num_attention_heads": 16,
    #         "mlp_hidden_size": 1536,
    #         "num_layers": 8,
    #         "num_outputs": 5000,
    #         "num_cls": 12,
    #         "use_rope": True,
    #         "rope_base": 10000.0,
    #         "rope_fraction": 0.5,
    #         "use_absolute_positional_embeddings": False,
    #         "positional_damping_factor": 0.1,
    #     }},
    # },
    # {
    #     "name": "dwip_v1-trial_007",
    #     "model_family": "new",
    #     "checkpoint_path": "/mnt/checkpoints_sweep/workdir/dwip_v1-trial_007/checkpoint_15000.pth",
    #     "json_config": {"config": {
    #         "embedding_size": 1024,
    #         "num_attention_heads": 16,
    #         "mlp_hidden_size": 1536,
    #         "num_layers": 8,
    #         "num_outputs": 5000,
    #         "num_cls": 16,
    #         "use_rope": True,
    #         "rope_base": 10000.0,
    #         "rope_fraction": 0.5,
    #         "use_absolute_positional_embeddings": False,
    #         "positional_damping_factor": 0.1,
    #     }},
    # },
    # {
    #     "name": "dwip_v1-trial_008",
    #     "model_family": "new",
    #     "checkpoint_path": "/mnt/checkpoints_sweep/workdir/dwip_v1-trial_008/checkpoint_17000.pth",
    #     "json_config": {"config": {
    #         "embedding_size": 1024,
    #         "num_attention_heads": 16,
    #         "mlp_hidden_size": 1536,
    #         "num_layers": 8,
    #         "num_outputs": 5000,
    #         "num_cls": 16,
    #         "use_rope": True,
    #         "rope_base": 10000.0,
    #         "rope_fraction": 0.5,
    #         "use_absolute_positional_embeddings": False,
    #         "positional_damping_factor": 0.1,
    #     }},
    # },
]

# MCPFN_IMPUTER_SPECS.extend(load_manifest_mcpfn_specs(DWIP_V3R1_MANIFEST_PATH))

patterns = {
    "MCAR",
    "MAR",
    "MNAR",
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

num_repeats = 3

# --- Initialize classes once ---
if "mcpfn" in imputers:
    default_preprocessors = [
        RandomRowColumnPermutation(),
        RandomRowColumnPermutation(),
        RandomRowPermutation(),
        RandomColumnPermutation(),
        # StandardizeWhiten(whiten=True),
    ]

    # def build_mcpfn_imputers():
    #     built_imputers = []
    #     for spec in MCPFN_IMPUTER_SPECS:
    #         init_kwargs = dict(MCPFN_SHARED_KWARGS)
    #         init_kwargs.update(spec.get("init_kwargs", {}))

    #         if init_kwargs.get("preprocessors") == "default":
    #             init_kwargs["preprocessors"] = default_preprocessors

    #         built_imputers.append(
    #             {
    #                 "name": spec["name"],
    #                 "model": ImputePFN(
    #                     checkpoint_path=spec["checkpoint_path"],
    #                     json_config=spec["json_config"],
    #                     model_family=spec.get("model_family", "new_stable"),
    #                     **init_kwargs,
    #                 ),
    #             }
    #         )
    #     return built_imputers

    # mcpfn_imputers = build_mcpfn_imputers()
    
    # Step 1 trained head (log_lambda, log_gamma_scale fitted on synthetic data).
    _trained_head_ckpt = torch.load(
        '/root/tabular/mcpfn/src/tabimpute/workdir/agop_head_dwip_v2_t002_step1/head_final.pt',
        map_location='cpu', weights_only=False,
    )
    _trained_head_for_tuned = RFMHeadAGOP(iters=3)
    _trained_head_for_tuned.load_state_dict(_trained_head_ckpt['head'])
    _trained_head_for_per_matrix = RFMHeadAGOP(iters=3)
    _trained_head_for_per_matrix.load_state_dict(_trained_head_ckpt['head'])

    mcpfn_imputers = [
        # Boost: MatFormer's MLP prediction + xRFM column-wise residual correction.
        {"name": "boost_matformer_dwip_v2_t002_a0p1",
         "model": BoostMatFormerImputer(
             device='cuda',
             entry_wise_features=False,
             json_config=ROUND2_T2_CONFIG,
             checkpoint_path='/mnt/checkpoints_sweep/workdir/dwip_v2-trial_002/checkpoint_17000.pth',
             max_num_rows=50,
             max_num_cols=50,
             blend_alpha=0.1,
         )},
        # # Step 1: warm-started head (synthetic-trained), no per-matrix tuning.
        # {"name": "rfm_agop_dwip_v2_t002_trained",
        #  "model": RFMImputePFN(
        #      device='cuda',
        #      entry_wise_features=False,
        #      json_config=ROUND2_T2_CONFIG,
        #      checkpoint_path='/mnt/checkpoints_sweep/workdir/dwip_v2-trial_002/checkpoint_17000.pth',
        #      max_num_rows=50,
        #      max_num_cols=50,
        #      head=_trained_head_for_tuned,
        #  )},
        # # Step 1.5: warm-start + per-matrix Adam tuning of (log_lambda, log_gamma_scale).
        # {"name": "rfm_agop_dwip_v2_t002_per_matrix",
        #  "model": RFMImputePFN(
        #      device='cuda',
        #      entry_wise_features=False,
        #      json_config=ROUND2_T2_CONFIG,
        #      checkpoint_path='/mnt/checkpoints_sweep/workdir/dwip_v2-trial_002/checkpoint_17000.pth',
        #      max_num_rows=50,
        #      max_num_cols=50,
        #      head=_trained_head_for_per_matrix,
        #      per_matrix_steps=50,
        #      per_matrix_lr=1e-2,
        #      per_matrix_val_frac=0.1,
        #  )},
        # {"name": "masters_mcar",
        #  "model": ImputePFN(device='cuda', max_num_rows=300, entry_wise_features=True, max_num_cols=10,preprocessors=default_preprocessors)},
        # {"name": "rfm_agop_dwip_v2_t002",
        #  "model": RFMImputePFN(
        #      device='cuda',
        #      entry_wise_features=False,
        #      json_config=ROUND2_T2_CONFIG,
        #      checkpoint_path='/mnt/checkpoints_sweep/workdir/dwip_v2-trial_002/checkpoint_17000.pth',
        #      max_num_rows=50,
        #      max_num_cols=50,
        #      head=RFMHeadAGOP(iters=3),
        #  )},
        # {"name": "tabimpute_curriculum_100x100_rank10_best_chunked_100x100",
        #  "model": ImputePFN(
        #      device='cuda',
        #      entry_wise_features=False,
        #      json_config=ROUND2_T2_CONFIG,
        #      checkpoint_path='/root/tabular/mcpfn/src/tabimpute/workdir/tabimpute-new-stable_v2-rank_10_10-100x100-from_50x50_step30000/checkpoint_42500.pth',
        #      model_family='new_stable',
        #      max_num_rows=100,
        #      max_num_cols=100,
        #  )},
        # {"name": "tabimpute_memory_efficient_200x200_rank10_step60000_chunked_200x200",
        #  "model": ImputePFN(
        #      device='cuda',
        #      entry_wise_features=False,
        #      json_config=ROUND2_T2_CONFIG,
        #      checkpoint_path='/root/tabular/mcpfn/src/tabimpute/workdir/tabimpute-new-stable-memory-efficient_v2-rank_10_10-200x200-from_100x100_step45000/checkpoint_60000.pth',
        #      model_family='new_stable_memory_efficient',
        #      max_num_rows=200,
        #      max_num_cols=200,
        #  )},
        {"name": "dwip_v2-trial_002",
         "model": ImputePFN(
             device='cuda',
             entry_wise_features=False,
             json_config={"config": {
                 "embedding_size": 768,
                 "num_attention_heads": 16,
                 "mlp_hidden_size": 1536,
                 "num_layers": 8,
                 "num_outputs": 5000,
                 "num_cls": 12,
                 "use_rope": True,
                 "rope_base": 10000.0,
                 "rope_fraction": 0.5,
                 "use_absolute_positional_embeddings": False,
                 "positional_damping_factor": 0.1,
                 "attention_dropout": 0.0,
                 "ffn_dropout": 0.03,
                 "drop_path_rate": 0.0,
                 "residual_scale_init": 0.7,
                 "embedding_dropout": 0.0,
                 "rms_norm_eps": 1e-6,
             }},
             checkpoint_path='/mnt/checkpoints_sweep/workdir/dwip_v2-trial_002/checkpoint_17000.pth',
             model_family='new_stable',
             max_num_rows=50,
             max_num_cols=50,
         )},
    ]


if "tabpfn" in imputers:
    tabpfn = TabPFNImputer(device="cuda")
    
if "tabpfn_unsupervised" in imputers:
    tabpfn_unsupervised = TabPFNUnsupervisedModel(device="cuda")
    
if "diffputer" in imputers:
    diffputer = DiffPuterImputer(device="cuda")

if "notmiwae" in imputers:
    notmiwae = NotMIWAEImputer(device="cuda")

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
        kernel_kwargs = {
            "num_datasets": 1,
            # Avoid mean-matching failures on small benchmark slices.
            "mean_match_candidates": 0,
            "random_state": random_state,
        }
        print("Running miceforest MICE with 5 iterations on CPU")
        kernel = mf.ImputationKernel(miceforest_df, **kernel_kwargs)
        kernel.mice(iterations=5, verbose=False)
        return kernel.complete_data(dataset=0).to_numpy()
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
base_path = "../datasets/uci"
datasets = os.listdir(base_path)
repeats = 1

pbar = tqdm(datasets)
for name in pbar:
    pbar.set_description(f"Running {name}")
    
    configs = itertools.product(patterns, missingness_levels, range(num_repeats))
    
    for pattern, missingness_level, r in configs:
        cfg_dir = f"{base_path}/{name}/{pattern}/missingness-{missingness_level}/repeat-{r}"

        X_missing = np.load(f"{cfg_dir}/missing.npy")
        X_true = np.load(f"{cfg_dir}/true.npy")
        
        X_missing_mean = np.nanmean(X_missing, axis=0)
        X_missing_std = np.nanstd(X_missing, axis=0)
        
        X_missing = (X_missing - X_missing_mean) / X_missing_std

        # mask for reference if needed downstream
        mask = np.isnan(X_missing)
        
        original_cfg_dir = cfg_dir

        for repeat in range(repeats):
            if repeats > 1:
                cfg_dir = f"{original_cfg_dir}/repeats/{repeat}/"
            else:
                cfg_dir = original_cfg_dir
            # create the directory if it doesn't exist
            if not os.path.exists(cfg_dir):
                os.makedirs(cfg_dir)
            # --- MCPFN ---
            if "mcpfn" in imputers and (repeats == 1 or repeat == 0):
                for mcpfn_spec in mcpfn_imputers:
                    mcpfn_name = mcpfn_spec["name"]
                    mcpfn = mcpfn_spec["model"]
                    out_path = f"{cfg_dir}/{mcpfn_name}.npy"
                    if not os.path.exists(out_path) or force_rerun:
                        # Time the imputation
                        if repeats > 1:
                            start_time = time.time()
                            X_mcpfn_list = mcpfn.impute(X_missing.copy(), num_repeats=repeats)
                            end_time = time.time()
                            print(f"MCPFN ({mcpfn_name}) imputation time: {end_time - start_time} seconds for {cfg_dir}")
                            # save the imputation time
                            for i in range(repeats):
                                repeat_cfg_dir = f"{original_cfg_dir}/repeats/{i}/"
                                if not os.path.exists(repeat_cfg_dir):
                                    os.makedirs(repeat_cfg_dir)
                                out_path = f"{repeat_cfg_dir}/{mcpfn_name}.npy"
                                with open(f"{repeat_cfg_dir}/{mcpfn_name}_imputation_time.txt", "a") as f:
                                    f.write(f"{(end_time - start_time) / repeats}\n")
                                np.save(out_path, (X_mcpfn_list[i] * X_missing_std) + X_missing_mean)
                        else:
                            start_time = time.time()
                            X_mcpfn = mcpfn.impute(X_missing.copy())
                            end_time = time.time()
                            np.save(out_path, (X_mcpfn * X_missing_std) + X_missing_mean)
                            print(f"MCPFN ({mcpfn_name}) imputation time: {end_time - start_time} seconds for {cfg_dir}")
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
                    np.save(out_path, (X_knn * X_missing_std) + X_missing_mean)

            # --- TabPFN ---
            if "tabpfn" in imputers:
                out_path = f"{cfg_dir}/tabpfn.npy"
                if not os.path.exists(out_path) or force_rerun:
                    start_time = time.time()
                    X_tabpfn = tabpfn.impute(X_missing.copy())
                    end_time = time.time()
                    print(f"TabPFN imputation time: {end_time - start_time} seconds")
                    # save the imputation time
                    with open(f"{cfg_dir}/tabpfn_imputation_time.txt", "a") as f:
                        f.write(f"{end_time - start_time}\n")
                    np.save(out_path, (X_tabpfn * X_missing_std) + X_missing_mean)

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
                    np.save(out_path, (X_tabpfn_unsupervised * X_missing_std) + X_missing_mean)

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
                    np.save(out_path, (out * X_missing_std) + X_missing_mean)

            # --- SoftImpute (fancyimpute) ---
            if "softimpute" in imputers:
                out_path = f"{cfg_dir}/softimpute.npy"
                if not os.path.exists(out_path) or force_rerun:
                    start_time = time.time()
                    X_processed = fill_all_nan_columns(X_missing.copy())
                    out = FancySoftImpute(verbose=False).fit_transform(X_processed)
                    end_time = time.time()
                    print(f"SoftImpute imputation time: {end_time - start_time} seconds")
                    # save the imputation time
                    with open(f"{cfg_dir}/softimpute_imputation_time.txt", "a") as f:
                        f.write(f"{end_time - start_time}\n")
                    np.save(out_path, (out * X_missing_std) + X_missing_mean)

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
                            np.save(out_path, (out * X_missing_std) + X_missing_mean)
                else:
                    if key in imputers:
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
                            np.save(out_path, (out * X_missing_std) + X_missing_mean)

            
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
                            np.save(out_path, (out[i] * X_missing_std) + X_missing_mean)
                            with open(f"{cfg_dir}/forestdiffusion_imputation_time.txt", "a") as f:
                                f.write(f"{(end_time - start_time) / repeats}\n")
                        break
                    else:
                        np.save(out_path, (out * X_missing_std) + X_missing_mean)
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
                    np.save(out_path, (X_diffputer * X_missing_std) + X_missing_mean)
                    # save the imputation time
                    with open(f"{cfg_dir}/diffputer_imputation_time.txt", "a") as f:
                        f.write(f"{end_time - start_time}\n")
            
            # --- ReMasker ---
            if "remasker" in imputers:
                remasker = ReMaskerImputer(device="cuda")
                out_path = f"{cfg_dir}/remasker.npy"
                if not os.path.exists(out_path) or force_rerun:
                    start_time = time.time()
                    X_remasker = remasker.fit_transform(fill_all_nan_columns(X_missing.copy()))
                    end_time = time.time()
                    print(f"ReMasker imputation time: {end_time - start_time} seconds")
                    # save the imputation time
                    with open(f"{cfg_dir}/remasker_imputation_time.txt", "a") as f:
                        f.write(f"{end_time - start_time}\n")
                    np.save(out_path, (X_remasker * X_missing_std) + X_missing_mean)

            # --- CACTI ---
            if "cacti" in imputers:
                cacti = CACTIImputer(device="cuda", model="CMAE", mask_ratio=0.9, epochs=300)
                out_path = f"{cfg_dir}/cacti.npy"
                if not os.path.exists(out_path) or force_rerun:
                    try:
                        start_time = time.time()
                        X_cacti = cacti.fit_transform(fill_all_nan_columns(X_missing.copy()))
                        end_time = time.time()
                        print(f"CACTI imputation time: {end_time - start_time} seconds")
                        np.save(out_path, (X_cacti * X_missing_std) + X_missing_mean)
                        # save the imputation time
                        with open(f"{cfg_dir}/cacti_imputation_time.txt", "a") as f:
                            f.write(f"{end_time - start_time}\n")
                    except Exception as e:
                        print(f"Error running CACTI: {e}")

            # --- NotMIWAE ---
            if "notmiwae" in imputers:
                out_path = f"{cfg_dir}/notmiwae.npy"
                if not os.path.exists(out_path) or force_rerun:
                    try:
                        start_time = time.time()
                        X_notmiwae = notmiwae.fit_transform(X_missing.copy())
                        end_time = time.time()
                        print(f"NotMIWAE imputation time: {end_time - start_time} seconds")
                        np.save(out_path, (X_notmiwae * X_missing_std) + X_missing_mean)
                        with open(f"{cfg_dir}/notmiwae_imputation_time.txt", "a") as f:
                            f.write(f"{end_time - start_time}\n")
                    except Exception as e:
                        print(f"Error running NotMIWAE: {e}")
            # GPU housekeeping (for MCPFN/TabPFN)
            torch.cuda.empty_cache()
