#!/usr/bin/env python
from __future__ import annotations

"""Train the explicit row/column latent matrix-completion model."""

from copy import deepcopy

import train_direct_pure_rabel_joint_matrix_causal as base
from tabimpute.model.model_rowcol_latent import TabImputeModelRowColumnLatent


base.TabImputeModelPureRaBEL = TabImputeModelRowColumnLatent
base.MODEL_FAMILY = "rowcol_latent"
base.BASE_SPECTRUM_CHECKPOINT = None
base.MODEL_CONFIG = deepcopy(base.MODEL_CONFIG)
base.MODEL_CONFIG.update(
    {
        "embedding_size": 512,
        "num_attention_heads": 8,
        "mlp_hidden_size": 1024,
        "num_layers": 8,
        "num_outputs": 5000,
        "num_cls": 0,
        "use_rope": False,
        "rope_fraction": 0.0,
        "attention_dropout": 0.0,
        "ffn_dropout": 0.05,
        "drop_path_rate": 0.02,
        "residual_scale_init": 0.20,
        "embedding_dropout": 0.0,
        "use_activation_checkpointing": True,
        "checkpoint_use_reentrant": False,
        "rabel_num_kernels": 64,
        "rabel_sigma": 1.0,
        "rabel_exponent_digits": 1,
        "rabel_exponent_embedding_size": 32,
        "count_hidden_size": 128,
        "decoder_interaction_scale_init": 0.10,
    }
)


if __name__ == "__main__":
    base.main()
