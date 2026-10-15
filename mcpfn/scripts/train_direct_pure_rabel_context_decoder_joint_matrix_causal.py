#!/usr/bin/env python
from __future__ import annotations

"""Train the direct RaBEL model with explicit row/column decoder context."""

from copy import deepcopy

import train_direct_pure_rabel_joint_matrix_causal as base
from tabimpute.model.model_new_stable_rabel_context_decoder import (
    TabImputeModelPureRaBELContextDecoder,
)


base.TabImputeModelPureRaBEL = TabImputeModelPureRaBELContextDecoder
base.MODEL_FAMILY = "pure_rabel_context_decoder"
base.MODEL_CONFIG = deepcopy(base.MODEL_CONFIG)
base.MODEL_CONFIG.update(
    {
        "context_scale_init": 0.10,
        "context_include_interaction": True,
    }
)


if __name__ == "__main__":
    base.main()
