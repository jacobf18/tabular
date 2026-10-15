import hashlib

import pytest
import torch

import single_model_compact_rating_replay as compact
from single_model_composite_gradient_storage import byte_equal


def saved_state():
    return {"means": torch.tensor([3., 3.5]), "stds": torch.tensor([1., 1.]),
            "row_offsets": torch.tensor([0., .1, -.1]), "row_scales": torch.tensor([1., 1.1, .9])}


def test_replays_only_compact_state_once_and_restores_sampler():
    saved = saved_state()
    with compact.runtime(saved) as receipt:
        assert compact.distillation.sample_rating_episode is compact.sample_with_saved_scaling
        actual = compact.restore_scaling(*(torch.zeros_like(saved[name]) for name in compact.NAMES))
        assert all(byte_equal(value, saved[name]) for name, value in zip(compact.NAMES, actual))
        assert receipt["vector_bytes"] == 40
        with pytest.raises(ValueError, match="exactly once"):
            compact.restore_scaling(*actual)
    assert receipt["calls"] == 1
    assert compact.ACTIVE.get() is None
    assert compact.distillation.sample_rating_episode is compact.ORIGINAL_SAMPLE


def test_no_state_keeps_imputation_sampler_unchanged():
    with compact.runtime(None) as receipt:
        assert compact.distillation.sample_rating_episode is compact.ORIGINAL_SAMPLE
    assert receipt == {"calls": 0, "vector_bytes": 0}


@pytest.mark.parametrize("mutation", ["missing", "extra", "nan", "zero_scale", "negative_scale", "matrix", "wrong_width", "dtype"])
def test_rejects_invalid_or_excess_state_before_patching(mutation):
    saved = saved_state()
    if mutation == "missing":
        saved.pop("means")
    elif mutation == "extra":
        saved["source"] = torch.ones(3, 2)
    elif mutation == "nan":
        saved["means"][0] = torch.nan
    elif mutation == "zero_scale":
        saved["stds"][0] = 0.0
    elif mutation == "negative_scale":
        saved["row_scales"][0] = -1.0
    elif mutation == "matrix":
        saved["means"] = saved["means"].reshape(1, 2)
    elif mutation == "wrong_width":
        saved["means"] = torch.ones(4)
    else:
        saved["row_offsets"] = saved["row_offsets"].double()
    with pytest.raises(ValueError):
        with compact.runtime(saved):
            pytest.fail("Invalid state must fail before sampler substitution")
    assert compact.distillation.sample_rating_episode is compact.ORIGINAL_SAMPLE


def test_errors_missing_application_and_nesting_restore_original_state():
    with pytest.raises(RuntimeError, match="stop"):
        with compact.runtime(saved_state()):
            raise RuntimeError("stop")
    assert compact.ACTIVE.get() is None
    with pytest.raises(ValueError, match="not applied"):
        with compact.runtime(saved_state()):
            pass
    assert compact.distillation.sample_rating_episode is compact.ORIGINAL_SAMPLE
    with pytest.raises(ValueError, match="nest"):
        with compact.runtime(saved_state()):
            with compact.runtime(None):
                pass
    assert compact.ACTIVE.get() is None


def test_shape_mismatch_cannot_silently_use_other_context_statistics():
    with pytest.raises(ValueError, match="cropped"):
        with compact.runtime(saved_state()):
            compact.restore_scaling(torch.ones(4), torch.ones(4), torch.ones(3), torch.ones(3))
    assert compact.distillation.sample_rating_episode is compact.ORIGINAL_SAMPLE


def test_actual_rating_prior_recovers_original_grid_despite_changed_fresh_fit(monkeypatch):
    arguments = dict(seed=715001, rows=8, cols=8, parent_rows=32, parent_cols=32, device="cpu",
                     normalization="two_way_user_scale", density_range=(.2, .3), query_limit=16,
                     context_ordering="first_observation", query_sampling="user_balanced", preference_family="gaussian",
                     exclude_unknown_queries=True, point_target="expected_rating")
    original = compact.ORIGINAL_SAMPLE(**arguments)
    saved = {name: getattr(original, name).clone() for name in compact.NAMES}
    original_fit = compact._namespace["fit_bias_normalization"]
    calls = []

    def altered_fit(ratings, observed, **options):
        means, stds, offsets = original_fit(ratings, observed, **options)
        calls.append(True)
        return means + .125, stds * 1.125, offsets

    monkeypatch.setitem(compact._namespace, "fit_bias_normalization", altered_fit)
    with compact.runtime(saved) as receipt:
        regenerated = compact.distillation.sample_rating_episode(**arguments)
    assert calls == [True] and receipt["calls"] == 1
    for name in (*compact.NAMES, "source", "target", "rating_query", "point_target"):
        assert byte_equal(getattr(regenerated, name), getattr(original, name)), name
    assert regenerated.metadata == original.metadata
    assert compact.distillation.sample_rating_episode is compact.ORIGINAL_SAMPLE


def test_source_keeps_original_cropping_and_normalization_arithmetic():
    source = compact.ADAPTED_SOURCE
    assert source.index("context_crop_indices(") < source.index("= restore_scaling(")
    assert source.index("= restore_scaling(") < source.index("    location =")
    assert "    normalized = (local_ratings - location) / stds" in source
    assert "        normalized = normalized / row_scales[:, None]" in source
    assert compact.ADAPTED_SOURCE_SHA256 == hashlib.sha256(source.encode()).hexdigest()
