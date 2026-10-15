import numpy as np
import pytest
import torch

from audit_single_model_missingness_coverage import audit_normalization
from diagnose_single_model_missingness_coverage import prepare_input


def arrays():
    source = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    target = source + 2.
    query = torch.zeros_like(source, dtype=torch.bool)
    query[:3] = True
    query[:, 2] = True
    normalized, truth, mean, scale = prepare_input(source, target, query)
    stored = {"query": query.numpy(), "model_source": normalized.numpy(), "model_target": truth.numpy(),
              "observed_mean": mean.numpy(), "observed_scale": scale.numpy()}
    return source.numpy(), target.numpy(), stored


def test_independent_numpy_audit_handles_empty_observed_column():
    audit_normalization(*arrays())


@pytest.mark.parametrize("field", ["observed_mean", "observed_scale", "model_source", "model_target"])
def test_independent_audit_rejects_corrupted_normalization(field):
    source, target, stored = arrays()
    stored[field] = stored[field] + .1
    with pytest.raises(AssertionError):
        audit_normalization(source, target, stored)


def test_audit_rejects_noisy_input_substituted_for_clean_truth():
    source, target, stored = arrays()
    with pytest.raises(AssertionError):
        audit_normalization(source, source, stored)


def test_audit_rejects_non_boolean_query_mask():
    source, target, stored = arrays()
    stored["query"] = stored["query"].astype(np.int32)
    with pytest.raises(ValueError, match="mask"):
        audit_normalization(source, target, stored)
