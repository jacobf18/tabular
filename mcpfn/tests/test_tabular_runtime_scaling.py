import json
from contextlib import nullcontext
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import benchmark_tabular_runtime_scaling as benchmark


def test_full_call_is_timed_without_dividing_by_cells(monkeypatch):
    matrix = np.array([[1., np.nan], [np.nan, 2.]])
    events = []
    clock = iter([10., 17.])
    monkeypatch.setattr(benchmark.time, 'perf_counter', lambda: next(clock))

    def impute(method, sample, seed):
        events.append((method, sample.shape, seed))
        sample[np.isnan(sample)] = 0.
        return sample

    elapsed, prediction = benchmark.timed_call(impute, matrix, 'mice', 42)
    assert elapsed == 7.
    assert events == [('mice', (2, 2), 42)]
    assert np.isnan(matrix).sum() == 2
    assert np.isfinite(prediction).all()


@pytest.mark.parametrize('prediction', [np.array([[1., np.nan]]), np.ones((1, 1)), np.array([[3., 2.]])])
def test_full_matrix_output_is_validated(prediction):
    with pytest.raises(ValueError):
        benchmark.validate_prediction(np.array([[1., np.nan]]), prediction)


def test_collection_requires_all_five_full_calls(tmp_path):
    (tmp_path / 'cases').mkdir()
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(rows=[16], columns=10, repeats=5)))
    for method, label in benchmark.METHODS.items():
        benchmark.write(tmp_path / 'cases' / f'{method}_16.json', dict(status='ok', seconds=[1., 2., 3., 4., 5.]))
    frame = benchmark.collect(tmp_path)
    assert len(frame) == 25
    assert frame.groupby('model').seconds.sum().eq(15.).all()
    benchmark.write(tmp_path / 'cases/mice_16.json', dict(status='running', seconds=[1., 2.]))
    with pytest.raises(ValueError, match='Incomplete'):
        benchmark.collect(tmp_path)


def test_partial_resume_rewarms_new_process_without_repeating_timed_calls(tmp_path, monkeypatch):
    for folder in ('inputs', 'cases', 'predictions'):
        (tmp_path / folder).mkdir()
    matrix = np.array([[1., np.nan], [2., 3.]])
    path = tmp_path / 'inputs/rows_2.npy'
    np.save(path, matrix)
    benchmark.write(tmp_path / 'manifest.json', dict(
        seed=42, warmup=2, repeats=5, columns=2, input_sha256={'2': benchmark.digest(path)}))
    benchmark.write(tmp_path / 'cases/knn_2.json', dict(
        status='running', seconds=[9., 8.], warmup_seconds=[1., 1.]))
    calls = []

    def impute(method, sample, seed):
        calls.append(method)
        return np.nan_to_num(sample)

    monkeypatch.setattr(benchmark, 'verify_sources', lambda manifest: None)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(set_num_threads=lambda threads: None))
    monkeypatch.setitem(sys.modules, 'threadpoolctl', SimpleNamespace(
        threadpool_limits=lambda limits: nullcontext(), threadpool_info=lambda: []))
    monkeypatch.setitem(sys.modules, 'repeated_imputation_methods', SimpleNamespace(impute=impute))
    benchmark.worker(tmp_path, 'knn', 2)
    result = benchmark.read(tmp_path / 'cases/knn_2.json')
    assert result['status'] == 'ok'
    assert len(calls) == 5
    assert result['seconds'][:2] == [9., 8.]
    assert len(result['seconds']) == 5
    assert len(result['warmup_seconds']) == 2
    assert result['previous_warmup_seconds'] == [[1., 1.]]
