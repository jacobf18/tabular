import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

import evaluate_realistic_science as evaluation


def test_transform_uses_only_observed_and_roundtrips():
    matrix = np.array([[1., -4.], [3., np.nan], [5., 2.]])
    for name in ('global', 'column', 'asinh_column'):
        normalized, inverse = evaluation.transform(matrix, name)
        np.testing.assert_allclose(inverse(normalized), matrix, equal_nan=True)
        assert np.isnan(normalized[1, 1])


def test_mask_replay_preserves_real_gaps_and_excludes_unmeasured_targets():
    frame = pd.DataFrame(np.ones((evaluation.WINDOW * 3, 2)),
                         index=pd.date_range('2000-01-01', periods=evaluation.WINDOW * 3, freq='h'))
    frame.iloc[5:10, 0] = np.nan
    frame.iloc[evaluation.WINDOW + 8:evaluation.WINDOW + 14, 0] = np.nan
    frame.iloc[2 * evaluation.WINDOW + 20:2 * evaluation.WINDOW + 23, 1] = np.nan
    frame.iloc[:, 1] = np.arange(len(frame))
    frame.iloc[2 * evaluation.WINDOW + 20:2 * evaluation.WINDOW + 23, 1] = np.nan
    frame.iloc[:, 0] += np.arange(len(frame)) / len(frame)
    arrays, details = evaluation.make_case(frame, [0, 1], (0, evaluation.WINDOW, 2 * evaluation.WINDOW))
    assert not arrays['missing'][8:10, 0].any()
    assert arrays['missing'][10:14, 0].all()
    assert not (arrays['missing'] & arrays['validation']).any()
    assert np.isfinite(arrays['truth'][arrays['missing']]).all()
    assert np.isfinite(arrays['truth'][arrays['validation']]).all()
    assert details['target_cells'] == 4
    assert not arrays['missing'][:, 2:].any()


def test_metrics_macro_average_columns_and_exclude_unscored_auxiliaries():
    truth = np.array([[2., 20., 10.], [4., 40., 11.], [6., 60., 12.]])
    missing = np.array([[True, True, False], [False, False, False], [False, False, False]])
    context = np.where(missing, np.nan, truth)
    prediction = truth.copy()
    prediction[0, :2] += [1., 10.]
    prediction[:, 2] = 1e9
    result = evaluation.metrics(truth, prediction, missing, context, [0, 1])
    assert result['nrmse'] == 1.
    assert result['target_cells'] == 2


def test_temporal_and_spatial_predictors_keep_observations():
    matrix = np.array([[1., 2.], [np.nan, 3.], [3., 4.], [4., np.nan]])
    for method in ('Temporal interpolation', 'Spatial residual interpolation', 'Temporal regression'):
        setting = evaluation.settings(method)[0]
        result, _ = evaluation.predict(matrix, method, setting, [0, 1], [[0, 1]], {}, None)
        observed = np.isfinite(matrix)
        np.testing.assert_allclose(result[observed], matrix[observed])
        assert np.isfinite(result).all()


def test_frozen_target_windows_do_not_overlap_within_dataset():
    for plans in evaluation.PLANS.values():
        starts = [start for cases in plans.values() for start, _, _ in cases]
        for position, start in enumerate(starts):
            for other in starts[position + 1:]:
                assert abs(other - start) >= evaluation.WINDOW


def test_calibration_controls_preserve_observed_measurements():
    from control_realistic_calibration import predict, settings

    positions = np.arange(24, dtype=float)
    matrix = np.column_stack((positions, positions ** 2, np.sin(positions)))
    matrix[5:9, 1] = np.nan
    observed = np.isfinite(matrix)
    for method in ('RBF-SVR', 'RBF-SVR + temporal residual', 'Gradient boosting'):
        result = predict(matrix, [1], method, settings(method)[0])
        assert np.isfinite(result).all()
        np.testing.assert_allclose(result[observed], matrix[observed])


def test_comparisons_withheld_until_all_baselines_complete():
    from report_realistic_science import compare

    frame = pd.DataFrame([dict(dataset='air_quality', split='confirmation', identifier='case',
                               method=method, nrmse=value)
                          for method, value in [('MatrixFormer', .1), ('MatrixFormer-Impute', .2), ('KNN', .3)]])
    audits = pd.DataFrame([dict(dataset='air_quality', split='confirmation', complete=False)])
    assert compare(frame, audits).empty


def test_linear_calibration_controls_preserve_observed_measurements():
    from control_realistic_linear import predict, settings

    positions = np.arange(24, dtype=float)
    matrix = np.column_stack((positions, positions ** 2, np.sin(positions)))
    matrix[5:9, 1] = np.nan
    observed = np.isfinite(matrix)
    for method in ('Ridge calibration', 'PLS calibration', 'Iterative random forest'):
        result = predict(matrix, [1], method, settings(method)[0])
        assert np.isfinite(result).all()
        np.testing.assert_allclose(result[observed], matrix[observed])


def test_replication_references_do_not_overlap_initial_target_windows():
    from register_science_replication import plans

    for dataset, splits in plans().items():
        original = [start for cases in evaluation.PLANS[dataset].values() for start, _, _ in cases]
        current = [start for start, _, _ in splits['confirmation']]
        for start in current:
            assert all(abs(start - previous) >= evaluation.WINDOW for previous in original)
        for position, start in enumerate(current):
            assert all(abs(start - following) >= evaluation.WINDOW for following in current[position + 1:])


def test_loader_respects_documented_sentinel_and_hourly_grid(tmp_path):
    from zipfile import ZipFile

    content = ('Date;Time;CO(GT);NOx(GT);NO2(GT);NMHC(GT);T;\n'
               '10/03/2004;18.00.00;2,6;166;113;150;13,6;\n'
               '10/03/2004;20.00.00;-200;103;92;-200;13,3;\n')
    with ZipFile(tmp_path / 'air_quality.zip', 'w') as archive:
        archive.writestr('AirQualityUCI.csv', content)
    frame, targets, _ = evaluation.load_data('air_quality', tmp_path)
    assert len(frame) == 3
    assert frame.iloc[1].isna().all()
    assert np.isnan(frame.iloc[2, 0])
    assert 'NMHC(GT)' not in frame.columns
    assert list(frame.columns[targets]) == ['CO(GT)', 'NOx(GT)', 'NO2(GT)']


def test_reporting_does_not_pool_different_physical_units():
    from report_realistic_science import summarize

    frame = pd.DataFrame([dict(dataset='air_quality', split='confirmation', method='MatrixFormer',
                               nrmse=.5, pooled_rmse=50.)])
    assert summarize(frame, 'pooled_rmse').empty
    assert len(summarize(frame, 'nrmse')) == 1
