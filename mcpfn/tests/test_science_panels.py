import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import evaluate_science_panels as panels
from prepare_science_panels import WATER_COLUMNS, approved_values


def example():
    frame = pd.DataFrame(np.arange(192 * 3, dtype=float).reshape(192, 3),
                         index=pd.date_range('2000-01-01', periods=192, freq='h'))
    frame.iloc[3:8, 0] = np.nan
    frame.iloc[80:86, 0] = np.nan
    frame.iloc[135:138, 0] = np.nan
    return frame, dict(window=64, targets=[0, 1])


def test_replay_no_missing_truth_or_overlap():
    frame, metadata = example()
    arrays, details = panels.make_case(frame, metadata, 128, 0, 64)
    assert not (arrays['missing'] & arrays['validation']).any()
    assert np.isfinite(arrays['truth'][arrays['missing']]).all()
    assert np.isfinite(arrays['truth'][arrays['validation']]).all()
    assert not arrays['missing'][:, 3:].any()
    assert details['target_cells'] == 4
    assert details['validation_target_cells'] == 6


def test_reject_overlapping_sources():
    frame, metadata = example()
    with pytest.raises(ValueError, match='overlap'):
        panels.make_case(frame, metadata, 0, 0, 64)


def test_actual_time_interpolation_and_observation_preservation():
    matrix = np.array([[0., 0.], [np.nan, .1], [10., 1.]])
    result, _ = panels.predict(matrix, 'Temporal interpolation', dict(transform='column'), [0], [[0]], {}, None)
    np.testing.assert_allclose(result[:, 0], [0., 1., 10.])
    np.testing.assert_array_equal(result[np.isfinite(matrix)], matrix[np.isfinite(matrix)])


def test_usgs_estimates_excluded_revised_approved_kept():
    records = [dict(value=value, qualifiers=qualifiers, dateTime=f'2019-01-01T0{position}:00:00Z')
               for position, (value, qualifiers) in enumerate([('1', ['A']), ('2', ['A', 'e']),
                                                              ('3', ['P']), ('-999999', ['A']), ('4', ['A', 'R'])])]
    np.testing.assert_array_equal(approved_values(records, -999999).to_numpy(), [1., 4.])


def test_wastewater_has_no_derived_efficiency_leakage():
    assert len(WATER_COLUMNS) == 29
    assert not any(name.startswith('RD-') for name in WATER_COLUMNS)


def test_time_delay_embedding_masks_every_copy_of_hidden_targets():
    from evaluate_science_delays import delayed_matrix

    matrix = np.column_stack((np.arange(10, dtype=float), np.arange(10, dtype=float) + 100.))
    matrix[4, 0] = np.nan
    augmented = delayed_matrix(matrix, [0], [1, 3])
    assert np.isnan(augmented[4, 0])
    assert np.isnan(augmented[5, 2]) and np.isnan(augmented[3, 3])
    assert np.isnan(augmented[7, 4]) and np.isnan(augmented[1, 5])
    assert not np.any(augmented == 4.)
    np.testing.assert_allclose(augmented[:, :2], matrix)


def test_ocean_preparation_excludes_interpolated_chemistry():
    from evaluate_ocean_panels import COLUMNS, clean_data

    source = {'G2' + name: np.array([1., 2., 3.]) for name in COLUMNS + ['cruise', 'month', 'day']}
    for name in ('salinity', 'oxygen', 'nitrate', 'silicate', 'phosphate', 'talk'):
        source['G2' + name + 'f'] = np.array([2., 0., 9.])
    frame = clean_data(source)
    assert frame.talk.iloc[0] == 1.
    assert frame.talk.iloc[1:].isna().all()
    assert not {'tco2', 'phts25p0', 'aou', 'sigma0'} & set(COLUMNS)


def test_vertical_interpolation_does_not_cross_casts():
    from evaluate_ocean_panels import COLUMNS, TARGET, profile_interpolate

    matrix = np.ones((6, len(COLUMNS)))
    matrix[:, 2] = [0., 5., 10., 0., 5., 10.]
    matrix[:, 10] = [1., 1., 1., 2., 2., 2.]
    matrix[:, TARGET] = [100., np.nan, 200., 1000., np.nan, 1200.]
    prediction = profile_interpolate(matrix)
    np.testing.assert_allclose(prediction[:, TARGET], [100., 150., 200., 1000., 1100., 1200.])


def test_panel_comparisons_require_complete_case_coverage():
    from report_science_panels import compare

    frame = pd.DataFrame([dict(dataset='wastewater', split='confirmation', identifier='case', method=method, nrmse=value)
                          for method, value in [('MatrixFormer', .2), ('MatrixFormer-Impute', .1), ('KNN', .3)]])
    audits = [dict(dataset='wastewater', split='confirmation', identifier='case', complete=False)]
    assert compare(frame, audits).empty


def test_panel_intervals_use_cases_not_cell_counts():
    from report_science_panels import summary
    from scipy.stats import t as student_t

    frame = pd.DataFrame([dict(method='example', nrmse=value, target_cells=cells)
                          for value, cells in [(1., 10), (2., 100), (3., 1000)]])
    result = summary(frame, ['method'], 'nrmse').iloc[0]
    assert result['mean'] == 2.
    assert result.windows == 3
    assert result.ci95_half_width == pytest.approx(student_t.ppf(.975, 2) / np.sqrt(3))
