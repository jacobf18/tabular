import numpy as np
import torch
from torch import nn

import tabimpute.tabimpute_v2 as legacy


def test_legacy_model_inherits_current_direct_interface(monkeypatch):
    class SmallModel(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(0.0))

        def forward(self, source):
            return torch.zeros((*source.shape, 5000), dtype=source.dtype, device=source.device) + self.anchor

    original_load = torch.load

    def load(source, *args, **kwargs):
        if source == "mock-v2.ckpt":
            return {"model": SmallModel().state_dict()}
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(legacy, "_LegacyTabImputeModel", SmallModel)
    monkeypatch.setattr(torch, "load", load)
    model = legacy.TabImputeV2(device="cpu", checkpoint_path="mock-v2.ckpt")
    assert model.direct_dtype == torch.bfloat16
    assert model.precision_mode == "full_bfloat16"
    assert model.direct_prior_index is None
    data = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0]], dtype=np.float32)
    result = model.impute(data.copy())
    assert result.shape == data.shape
    assert np.isfinite(result).all()
    assert np.allclose(result[~np.isnan(data)], data[~np.isnan(data)])
