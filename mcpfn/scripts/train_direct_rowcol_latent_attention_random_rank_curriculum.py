#!/usr/bin/env python
from __future__ import annotations

"""Train the row/column latent-attention architecture on low-rank priors."""

from copy import deepcopy

import train_direct_pure_rabel_curriculum as base
from tabimpute.model.model_rowcol_latent_attention import (
    TabImputeModelRowColumnLatentAttention,
)


base.TabImputeModelPureRaBEL = TabImputeModelRowColumnLatentAttention
base.MODEL_FAMILY = "rowcol_latent_attention"
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
        "ffn_dropout": 0.05,
        "drop_path_rate": 0.03,
        "residual_scale_init": 0.15,
        "embedding_dropout": 0.0,
        "use_activation_checkpointing": True,
        "checkpoint_use_reentrant": False,
        "rabel_num_kernels": 96,
        "rabel_sigma": 1.0,
        "rabel_exponent_digits": 1,
        "rabel_exponent_embedding_size": 48,
        "count_hidden_size": 192,
        "decoder_interaction_scale_init": 0.20,
    }
)


if __name__ == "__main__":
    base.main()
