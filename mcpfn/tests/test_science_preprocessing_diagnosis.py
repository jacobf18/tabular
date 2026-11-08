import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from diagnose_science_preprocessing import prepare


def test_cleanup_uses_only_current_observations():
    values = np.column_stack((np.arange(12, dtype=float), np.ones(12), np.full(12, np.nan)))
    normalized, restore, details = prepare(values, 'example', ['signal', 'constant', 'empty'], 'clean_column')
    assert normalized.shape == (12, 1)
    assert details['removed_columns'] == ['constant', 'empty']
    result = restore(normalized)
    np.testing.assert_array_equal(result[np.isfinite(values)], values[np.isfinite(values)])


def test_physical_transforms_roundtrip_and_preserve_mask():
    values = np.column_stack((np.linspace(30., 36., 12), np.linspace(2100., 2400., 12)))
    values[3, 1] = np.nan
    for variant in ('physical_salinity', 'physical_residual'):
        normalized, restore, _ = prepare(values, 'ocean_alkalinity', ['salinity', 'talk'], variant)
        assert np.isnan(normalized[3, 1])
        restored = restore(np.nan_to_num(normalized))
        np.testing.assert_array_equal(restored[np.isfinite(values)], values[np.isfinite(values)])
        assert np.isfinite(restored).all()


def test_observed_salinity_trend_removes_exact_linear_component():
    salinity = np.linspace(30., 36., 20)
    matrix = np.column_stack((salinity, 60. * salinity + 100.))
    matrix[5, 1] = np.nan
    normalized, restore, details = prepare(matrix, 'ocean_alkalinity', ['salinity', 'talk'], 'physical_residual')
    assert abs(details['salinity_slope'] - 60.) < 1e-8
    result = restore(np.nan_to_num(normalized))
    assert abs(result[5, 1] - (60. * salinity[5] + 100.)) < 1e-8


def test_validation_covers_each_target_without_exposing_missing_truth():
    from science_context_preprocessing import cover_validation_targets

    context = np.arange(180, dtype=float).reshape(60, 3)
    context[5:12, 0] = np.nan
    existing = np.zeros_like(context, dtype=bool)
    existing[20:24, 1] = True
    mask = cover_validation_targets(context, existing, [0, 1])
    assert not (mask & ~np.isfinite(context)).any()
    assert mask[:, 0].sum() == 12
    assert mask[:, 1].sum() == 12
    assert not mask[:, 2].any()
    assert mask[existing].all()
    np.testing.assert_array_equal(mask, cover_validation_targets(context, existing, [0, 1]))


def test_local_context_delays_never_reintroduce_masked_values():
    from science_context_preprocessing import temporal_context_predict

    matrix = np.column_stack((np.linspace(0., 1., 48), np.linspace(2., 3., 48)))
    matrix[20, 0] = np.nan
    seen = []

    def fake_forward(values, model, distribution):
        seen.append(values.copy())
        return np.nan_to_num(values), {}

    result = temporal_context_predict(matrix, [0], None, None, fake_forward, 32, [1, 3])
    assert seen
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[np.isfinite(matrix)], matrix[np.isfinite(matrix)])


def test_unsupported_local_column_falls_back_to_observed_global_level():
    from science_context_preprocessing import temporal_context_predict

    matrix = np.column_stack((np.full(96, 100.), np.linspace(0., 1., 96)))
    matrix[24:72, 0] = np.nan

    def fake_forward(values, model, distribution):
        return np.nan_to_num(values), {}

    result = temporal_context_predict(matrix, [0], None, None, fake_forward, 32, [])
    assert result[48, 0] == 100.
