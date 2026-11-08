from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import evaluate_physics_impute as evaluation
import report_physics_impute as reporting


def test_parent_artifacts_keep_original_layout():
    record = Path('/original/pilot/cases/example.json')
    assert evaluation.parent_artifact('cylinder', record) == Path('/original/pilot/predictions/example.npz')
    for benchmark in ('quantum', 'synthetic_mri', 'real_mri'):
        assert evaluation.parent_artifact(benchmark, record) == record.with_suffix('.npz')


def test_uncertainty_summary_uses_original_interval_order():
    arrays = dict(crps=np.array([1., 3.]), nll=np.array([2., 4.]),
                  quantiles=np.array([[-3., -2., -1., 1., 2., 3.], [-3., -2., -1., 1., 2., 3.]]))
    scores = evaluation.uncertainty_scores(arrays, np.array([0., 2.5]), 2.)
    assert scores['crps'] == 2. and scores['normalized_crps'] == 1.
    assert scores['coverage_50'] == .5 and scores['coverage_90'] == .5
    assert scores['coverage_95'] == 1. and scores['width_95'] == 6.


def test_quantum_specialist_uses_original_physical_projection():
    _, operators = evaluation.quantum.pauli_basis(2)
    density = np.diag([1., 0., 0., 0.]).astype(complex)
    truth = evaluation.quantum.pauli_coefficients(density, operators)
    matrix = truth.copy()
    matrix.flat[1::2] = np.nan
    scores, arrays = evaluation.quantum_scores(density, truth, matrix, truth, operators)
    np.testing.assert_allclose(arrays['density'], density, atol=1e-12)
    np.testing.assert_array_equal(arrays['completed'][np.isfinite(matrix)], matrix[np.isfinite(matrix)])
    assert scores['fidelity'] == pytest.approx(1.)
    assert scores['hidden_pauli_rmse'] == pytest.approx(0.)


def test_replay_checks_original_values_not_only_summary_scores(tmp_path):
    path = tmp_path / 'parent.npz'
    prediction = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.savez(path, prediction_4=prediction)
    case = dict(benchmark='cylinder', case_id='fixed', parent_artifact=str(path))
    result = evaluation.check_replay(case, dict(prediction=prediction.astype(float)))
    assert result['passed'] and result['comparisons'][0]['bit_identical']
    with pytest.raises(AssertionError):
        evaluation.check_replay(case, dict(prediction=prediction + 1.))


def test_paired_comparison_requires_matching_cases_and_preserves_direction():
    rows = []
    for index in range(18):
        for method, value in (('MatrixFormer', .5), ('MatrixFormer-Impute', .6)):
            rows.append(dict(case_id=f'case-{index}', method=method, fidelity=value))
    comparison = pd.DataFrame(rows)
    result = reporting.paired_comparison(comparison, 'quantum')[0]
    assert result['cases'] == 18 and result['impute_wins'] == 18
    assert result['impute_minus_combined'] == pytest.approx(.1)
    with pytest.raises(ValueError, match='Incomplete'):
        reporting.paired_comparison(comparison.iloc[1:], 'quantum')


def test_reported_spread_is_sample_sd_not_standard_error():
    frame = pd.DataFrame(dict(rate=[.5] * 3, method=['method'] * 3, score=[1., 2., 3.]))
    table = reporting.mean_sd_table(frame, 'score', ['rate'], ['method'])
    assert '2.0000 ± 1.0000' in table


def test_report_scope_excludes_mri():
    assert reporting.REPORT_BENCHMARKS == ('cylinder', 'quantum')
    assert sum(evaluation.COUNTS[benchmark] for benchmark in reporting.REPORT_BENCHMARKS) == 36


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_native_distributions_match_original_helper():
    model = torch.nn.Module()
    model.decoder = torch.nn.Linear(5, 3).cuda().eval()
    distribution = evaluation.shared.FullSupportBarDistribution(torch.tensor([-2., -1., 1., 2.], device='cuda'))
    matrix = np.arange(20, dtype=float).reshape(4, 5)
    truth = matrix.copy()
    matrix.flat[::3] = np.nan
    indices = np.flatnonzero(~np.isfinite(matrix))
    features = torch.arange(100, device='cuda', dtype=torch.float32).reshape(20, 5) / 100
    current_scores, current_arrays = evaluation.native_uncertainty(model, features, distribution, matrix, truth, indices)
    names = ('single', 'general', 'specialist')
    original_scores, original_arrays = evaluation.shared.neural_uncertainty(
        {name: model for name in names}, {name: features for name in names}, distribution, matrix, truth, indices, .3)
    assert current_scores == original_scores['MatrixFormer']
    for name, values in current_arrays.items():
        np.testing.assert_array_equal(values, original_arrays['MatrixFormer'][name])
