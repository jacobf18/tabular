#!/usr/bin/env python
from __future__ import annotations

"""Train the direct RaBEL model with learnable attention logit scales.

This reuses the existing mixed spectrum/CATE/skinny prior training loop but
swaps the model class for the scaled-attention variant. The model class keeps
the same parameter names as the baseline where possible, so ordinary RaBEL
checkpoints can warm-start it; only the new logit-scale parameters are freshly
initialized.
"""

from copy import deepcopy

import train_direct_pure_rabel_joint_matrix_causal as base
from tabimpute.model.model_new_stable_rabel_scaled_attention import (
    TabImputeModelPureRaBELScaledAttention,
)


base.TabImputeModelPureRaBEL = TabImputeModelPureRaBELScaledAttention
base.MODEL_CONFIG = deepcopy(base.MODEL_CONFIG)
base.MODEL_CONFIG.update(
    {
        "feature_logit_scale_init": 1.5,
        "datapoint_logit_scale_init": 4.0,
        "min_logit_scale": 0.125,
        "max_logit_scale": 32.0,
    }
)


if __name__ == "__main__":
    base.main()
