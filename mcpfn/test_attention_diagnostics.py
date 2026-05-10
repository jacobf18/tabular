import torch

from scripts.synthetic_attention_washout_tests import (
    GENERATORS,
    aggregate_rows,
    flatten_diagnostics,
)
from tabimpute.model.model_new import TabImputeModel as TabImputeModelNew
from tabimpute.model.model_new_stable import TabImputeModel as TabImputeModelStable


def _build_kwargs():
    return {
        "embedding_size": 16,
        "num_attention_heads": 4,
        "mlp_hidden_size": 32,
        "num_layers": 2,
        "num_outputs": 8,
        "num_cls": 2,
        "use_rope": True,
    }


def _assert_attention_diagnostics(model_cls):
    model = model_cls(**_build_kwargs())
    model.eval()
    model.set_attention_capture(True, store_weights=True)

    x = torch.randn(1, 5, 4)
    x[0, 1, 2] = torch.nan
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 5, 4, 8)
    diagnostics = model.get_attention_diagnostics()
    assert len(diagnostics) == 2

    first_layer = diagnostics[0]
    feature_weights = first_layer["features"]["weights"]
    datapoint_weights = first_layer["datapoints"]["weights"]
    feature_stats = first_layer["features"]["stats"]
    datapoint_stats = first_layer["datapoints"]["stats"]

    assert feature_weights.shape == (7, 4, 6, 6)
    assert datapoint_weights.shape == (6, 4, 7, 7)
    assert torch.allclose(
        feature_weights.sum(dim=-1), torch.ones_like(feature_weights[..., 0])
    )
    assert torch.allclose(
        datapoint_weights.sum(dim=-1), torch.ones_like(datapoint_weights[..., 0])
    )
    assert 0.0 <= feature_stats["normalized_entropy_mean"] <= 1.0
    assert 0.0 <= datapoint_stats["normalized_entropy_mean"] <= 1.0
    assert feature_stats["seq_len"] == 6.0
    assert datapoint_stats["seq_len"] == 7.0


def test_model_new_attention_diagnostics():
    _assert_attention_diagnostics(TabImputeModelNew)


def test_model_new_stable_attention_diagnostics():
    _assert_attention_diagnostics(TabImputeModelStable)


def test_synthetic_generators_and_aggregation():
    import numpy as np

    rng = np.random.default_rng(0)
    for generator in GENERATORS.values():
        x = generator(10, 8, rng, missing_rate=0.25, rank=3)
        assert x.shape == (10, 8)
        assert x.dtype == np.float32
        assert np.isnan(x).any()

    diagnostics = [
        {
            "layer": 0,
            "features": {
                "stats": {
                    "seq_len": 10.0,
                    "normalized_entropy_mean": 0.9,
                    "effective_key_fraction_mean": 0.8,
                    "max_prob_mean": 0.2,
                    "top5_mass_mean": 0.6,
                    "kl_to_uniform_mean": 0.1,
                    "logit_std_mean": 0.3,
                    "logit_range_mean": 1.2,
                }
            },
            "datapoints": {
                "stats": {
                    "seq_len": 10.0,
                    "normalized_entropy_mean": 0.95,
                    "effective_key_fraction_mean": 0.9,
                    "max_prob_mean": 0.15,
                    "top5_mass_mean": 0.55,
                    "kl_to_uniform_mean": 0.05,
                    "logit_std_mean": 0.2,
                    "logit_range_mean": 1.0,
                }
            },
        }
    ]
    raw_rows = flatten_diagnostics("10x10", "low_rank", 0, diagnostics)
    summary = aggregate_rows(raw_rows)
    assert len(raw_rows) == 2
    assert len(summary) == 2
    assert summary[0]["n_seeds"] == 1
