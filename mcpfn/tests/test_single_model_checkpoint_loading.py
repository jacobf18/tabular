from copy import deepcopy

import pytest
import torch

from audit_single_model_checkpoint_loading import verify_loaded


def test_exact_cpu_inference_state_is_required():
    model = torch.nn.Linear(2, 3).eval()
    result = verify_loaded(model, deepcopy(model.state_dict()))
    assert result["state_tensors"] == 2
    assert result["state_elements"] == result["parameter_elements"] == 9
    assert result["all_tensors_bitwise_equal"] and result["all_tensors_on_cpu"]


@pytest.mark.parametrize("kind", ["value", "dtype", "shape", "missing", "nonfinite", "training"])
def test_loader_mismatch_is_not_silently_accepted(kind):
    model = torch.nn.Linear(2, 3).eval()
    expected = deepcopy(model.state_dict())
    if kind == "value":
        expected["weight"][0, 0] += .001
    elif kind == "dtype":
        expected["weight"] = expected["weight"].double()
    elif kind == "shape":
        expected["weight"] = expected["weight"][:1]
    elif kind == "missing":
        del expected["bias"]
    elif kind == "nonfinite":
        with torch.no_grad():
            model.weight[0, 0] = torch.nan
        expected = deepcopy(model.state_dict())
    else:
        model.train()
    with pytest.raises(ValueError):
        verify_loaded(model, expected)
