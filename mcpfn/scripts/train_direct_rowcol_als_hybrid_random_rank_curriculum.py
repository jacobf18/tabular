#!/usr/bin/env python
from __future__ import annotations

"""Train the RowCol latent-attention + unrolled ALS hybrid architecture."""

from copy import deepcopy

import train_direct_pure_rabel_curriculum as base
from tabimpute.model.model_rowcol_als_hybrid import (
    TabImputeModelRowColumnALSHybrid,
)


base.TabImputeModelPureRaBEL = TabImputeModelRowColumnALSHybrid
base.MODEL_FAMILY = "rowcol_als_hybrid"
base.MODEL_CONFIG = deepcopy(base.MODEL_CONFIG)
base.MODEL_CONFIG.update(
    {
        "embedding_size": 768,
        "num_attention_heads": 12,
        "mlp_hidden_size": 2048,
        "num_layers": 10,
        "num_outputs": 5000,
        "num_cls": 0,
        "use_rope": False,
        "rope_fraction": 0.0,
        "attention_dropout": 0.0,
        "latent_attention_dropout": 0.0,
        "ffn_dropout": 0.03,
        "drop_path_rate": 0.02,
        "residual_scale_init": 0.12,
        "embedding_dropout": 0.0,
        "use_activation_checkpointing": True,
        "checkpoint_use_reentrant": False,
        "rabel_num_kernels": 96,
        "rabel_sigma": 1.0,
        "rabel_exponent_digits": 1,
        "rabel_exponent_embedding_size": 48,
        "count_hidden_size": 192,
        "decoder_interaction_scale_init": 0.20,
        "als_rank": 10,
        "als_steps": 4,
        "als_ridge": 1e-3,
        "als_update_damping": 0.85,
        "als_init_scale": 0.035,
        "logit_sigma_init": 0.025,
        "logit_sigma_min": 0.003,
        "use_neural_residual": False,
    }
)


if __name__ == "__main__":
    base.main()
