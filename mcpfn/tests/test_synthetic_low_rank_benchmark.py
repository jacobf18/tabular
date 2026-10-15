import argparse
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import evaluate_matrixformer_plus_synthetic as runner
import summarize_matrixformer_plus_synthetic as report
import synthetic_low_rank_benchmark as benchmark
from recsys_evaluation_common import sha256, write_json


def make_case(**overrides):
    return {'rows': 32, 'cols': 24, 'rank': 2, 'noise_std': 0., 'missing_rate': .5,
            'seed': 50000, **overrides}


def test_registered_sizes_and_invalid_profiles():
    config = runner.read(runner.DEFAULT_CONFIG)
    assert {name: len(benchmark.cases_for_profile(profile)) for name, profile in
            config['profiles'].items()} == {'smoke': 8, 'main': 840, 'stress': 800}
    profile = copy.deepcopy(config['profiles']['smoke'])
    profile['ranks'] = [25]
    with pytest.raises(ValueError, match='rank'):
        benchmark.cases_for_profile(profile)
    profile['ranks'] = [2, 2]
    with pytest.raises(ValueError, match='unique'):
        benchmark.cases_for_profile(profile)


def test_generation_reproduces_rank_nested_masks_and_independent_noise():
    signal, truth, observed, missing = benchmark.generate_case(make_case(), 123)
    assert benchmark.array_digest(signal, truth, observed, missing) == benchmark.array_digest(
        *benchmark.generate_case(make_case(), 123))
    assert np.linalg.matrix_rank(signal) == 2
    np.testing.assert_array_equal(truth, signal)
    assert missing.sum() == 384
    np.testing.assert_array_equal(observed[~missing], truth[~missing])
    larger = benchmark.generate_case(make_case(missing_rate=.9), 123)
    np.testing.assert_array_equal(signal, larger[0])
    assert np.all(larger[3][missing])
    noisy = benchmark.generate_case(make_case(noise_std=.1), 123)
    np.testing.assert_array_equal(signal, noisy[0])
    np.testing.assert_array_equal(missing, noisy[3])
    assert np.std(noisy[1] - truth) == pytest.approx(.1, rel=.1)
    assert not np.array_equal(truth, benchmark.generate_case(make_case(seed=50001), 123)[1])


@pytest.mark.parametrize('rank', [2, 16, 64])
def test_signal_scale_does_not_grow_with_rank(rank):
    signal, _, _, _ = benchmark.generate_case(make_case(rows=256, cols=256, rank=rank), 123)
    assert np.var(signal) == pytest.approx(1., abs=.15)


def test_no_support_repair_and_cold_cells_are_scored():
    signal, truth, observed, missing = benchmark.generate_case(
        make_case(rows=4, cols=4, missing_rate=.94), 123)
    diagnostics = benchmark.support_diagnostics(observed, 2)
    assert diagnostics['observed_cells'] == 1
    assert diagnostics['empty_rows'] == diagnostics['empty_columns'] == 3
    prediction = np.where(missing, 0., observed)
    scores, values = benchmark.score_prediction(truth, signal, observed, prediction)
    assert scores['test_cells'] == scores['cold_axis_cells'] == 15
    assert scores['supported_cells'] == 0 and scores['supported_rmse'] is None
    assert scores['rmse'] == pytest.approx(np.sqrt(np.mean(truth[missing].astype(np.float64) ** 2)))
    assert values.shape == (15,) and values.dtype == np.float32


@pytest.mark.parametrize('corruption', ['observed', 'nonfinite', 'overflow', 'shape'])
def test_scoring_rejects_bad_predictions(corruption):
    signal, truth, observed, missing = benchmark.generate_case(make_case(), 123)
    prediction = truth.astype(np.float64)
    if corruption == 'observed':
        prediction[~missing] += 1
    elif corruption == 'nonfinite':
        prediction[missing] = np.nan
    elif corruption == 'overflow':
        prediction[missing] = 1e300
    else:
        prediction = prediction[:, :-1]
    with np.errstate(over='ignore'), pytest.raises(ValueError):
        benchmark.score_prediction(truth, signal, observed, prediction)


@pytest.mark.parametrize('method', benchmark.BASELINES)
def test_baselines_preserve_observations_handle_empty_axes_and_repeat(method):
    matrix = np.array([[1., 2., np.nan], [3., np.nan, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32)
    settings = runner.read(runner.DEFAULT_CONFIG)['baseline_settings']
    settings['max_iters'] = 3
    settings['iterative_svd_ranks'] = [2]
    first, details = benchmark.evaluate_baseline(matrix, method, settings, 42)
    second, _ = benchmark.evaluate_baseline(matrix, method, settings, 42)
    assert first.shape == matrix.shape and np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[np.isfinite(matrix)], matrix[np.isfinite(matrix)])
    np.testing.assert_array_equal(first[2], [2., 2., 2.])
    np.testing.assert_array_equal(first[:, 2], [2., 2., 2.])
    assert details['validation_cells'] == 0
    if method == 'iterative_svd':
        assert details['effective_svd_rank'] == 1


def test_baseline_selection_sees_only_observations_and_refits(monkeypatch):
    matrix = np.arange(100, dtype=np.float32).reshape(10, 10)
    matrix[0, 0] = np.nan
    settings = runner.read(runner.DEFAULT_CONFIG)['baseline_settings']
    contexts = []

    def predictor(context, method, parameters, seed, max_iters, tolerance):
        contexts.append(context.copy())
        return np.where(np.isfinite(context), context, parameters['neighbors']).astype(np.float32)

    monkeypatch.setattr(benchmark, 'baseline_prediction', predictor)
    _, details = benchmark.evaluate_baseline(matrix, 'knn', settings, 42)
    heldout = benchmark.observed_holdout(matrix, .1, 42)
    assert details['validation_cells'] == heldout.sum() > 0
    assert len(contexts) == len(settings['knn_neighbors']) + 1
    for context in contexts[:-1]:
        assert np.isnan(context[heldout]).all() and np.isnan(context[0, 0])
        np.testing.assert_array_equal(context[~heldout], matrix[~heldout])
    np.testing.assert_array_equal(contexts[-1], matrix)
    reduced = matrix.copy()
    reduced[heldout] = np.nan
    _, mean, scale = benchmark.observed_normalization(reduced)
    np.testing.assert_allclose(mean, np.nanmean(reduced.astype(np.float64), axis=0))
    np.testing.assert_allclose(scale, np.nanstd(reduced.astype(np.float64), axis=0))


class FakeExpert:
    def __init__(self, value):
        self.value = value
        self.inputs = []

    def impute(self, matrix, return_full=False):
        assert return_full
        self.inputs.append(matrix.copy())
        raw = np.full_like(matrix, self.value)
        return np.where(np.isfinite(matrix), matrix, raw), raw


def test_neural_path_uses_existing_raw_observed_blend_once():
    matrix = np.array([[2., np.nan], [2., 2.]], dtype=np.float32)
    general, specialist = FakeExpert(0.), FakeExpert(4.)
    predictions, details = benchmark.predict_experts(matrix, general, specialist)
    assert details['decision']['specialist_weight'] == .5
    assert details['decision']['evidence_cells'] == 3
    assert predictions['matrixformer_plus'][0, 1] == 2.
    assert len(general.inputs) == len(specialist.inputs) == 1
    for context in general.inputs + specialist.inputs:
        np.testing.assert_array_equal(context, matrix)


def test_frozen_guard_and_inference_context(monkeypatch):
    experts = runner.FrozenExperts.__new__(runner.FrozenExperts)
    experts.models = {name: SimpleNamespace(model=torch.nn.Linear(2, 2).eval().requires_grad_(False))
                      for name in ('general', 'specialist')}
    experts.versions = experts.state_versions()

    def predict(observed, general, specialist):
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        return observed

    monkeypatch.setattr(runner, 'predict_experts', predict)
    experts.predict(np.ones((2, 2)))
    with torch.no_grad():
        experts.models['general'].model.weight.add_(1.)
    with pytest.raises(RuntimeError, match='mutated'):
        experts.check_frozen()


@pytest.fixture
def planned(tmp_path, monkeypatch):
    config = runner.read(runner.DEFAULT_CONFIG)
    config['profiles']['smoke'] = {'shapes': [[6, 5]], 'ranks': [2], 'noise_stds': [0.],
                                  'missing_rates': [.5], 'seeds': [1, 2]}
    config['baseline_settings']['max_iters'] = 2
    config['baseline_settings']['iterative_svd_ranks'] = [2]
    config_path = tmp_path / 'protocol_input.json'
    write_json(config_path, config)
    checkpoints = {}
    for name in ('general', 'specialist'):
        checkpoint = tmp_path / name / 'checkpoint.pth'
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(name.encode())
        write_json(checkpoint.parent / 'config.json', {'model': {}})
        checkpoints[name] = checkpoint
    source = tmp_path / 'source.py'
    source.write_text('evaluation_only = True\n')
    resource = tmp_path / 'borders.pt'
    resource.write_bytes(b'fixed resource')
    monkeypatch.setattr(runner, 'REPO', tmp_path)
    monkeypatch.setattr(runner, 'CHECKPOINTS', checkpoints)
    monkeypatch.setattr(runner, 'CHECKPOINT_SHA256', {name: sha256(path) for name, path in checkpoints.items()})
    monkeypatch.setattr(runner, 'source_paths', lambda: [source, resource])
    args = argparse.Namespace(output=tmp_path / 'output', config=config_path, profile='smoke',
        device='cpu', precision='float32', backend='dense', threads=1, reserve_gib=0.,
        methods=['baselines'], case_limit=1, retry_failed=False)
    runner.plan(args)
    return args


@pytest.mark.parametrize('artifact', ['manifest', 'config', 'source', 'resource', 'checkpoint'])
def test_manifest_rejects_tampering(planned, artifact):
    manifest = runner.verify_manifest(planned.output)
    path = {'manifest': planned.output / 'manifest.json', 'config': planned.config,
            'source': runner.REPO / 'source.py', 'resource': runner.REPO / 'borders.pt',
            'checkpoint': runner.CHECKPOINTS['general']}[artifact]
    assert manifest['training_permitted'] is False
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='changed'):
        runner.verify_manifest(planned.output)


def test_resume_rescore_reporting_and_provisional_export(planned, monkeypatch, tmp_path):
    runner.run(planned)
    manifest = runner.verify_manifest(planned.output)
    directory = planned.output / 'cases' / manifest['cases'][0]['case_id']
    timestamp = (directory / 'column_mean.npz').stat().st_mtime_ns
    assert runner.read(planned.output / 'status.json')['pending'] == 10
    planned.case_limit = None
    runner.run(planned)
    assert (directory / 'column_mean.npz').stat().st_mtime_ns == timestamp

    class FakeBundle:
        def __init__(self, manifest):
            pass

        def predict(self, observed):
            return benchmark.predict_experts(observed, FakeExpert(0.), FakeExpert(1.))

        def check_frozen(self):
            pass

    monkeypatch.setattr(runner, 'FrozenExperts', FakeBundle)
    planned.methods = ['neural']
    runner.run(planned)
    records, counts, _ = report.audit_results(planned.output, manifest)
    assert counts == {'ok': 14, 'failed': 0, 'pending': 0}
    summaries, paired = report.aggregate(records, manifest['cases'])
    assert all(row['rmse_n'] == 2 and row['pending_n'] == 0 for row in summaries)
    assert all(row['n'] == 2 for row in paired)
    report.summarize(planned.output)
    text = (planned.output / 'paper/synthetic_results.tex').read_text()
    assert r'UV^\top/\sqrt{r}' in text and '\t' not in text
    assert list((planned.output / 'figures').glob('*.pdf'))
    with pytest.raises(ValueError, match='Do not publish'):
        report.summarize(planned.output, tmp_path / 'paper')
    record_path = directory / 'column_mean.json'
    record = runner.read(record_path)
    record['rmse'] += .1
    write_json(record_path, record)
    with pytest.raises(ValueError, match='independently reproduce'):
        report.audit_results(planned.output, manifest)


def test_interrupted_runs_are_not_left_marked_running(planned):
    with pytest.raises(RuntimeError, match='busy'), runner.run_lock(planned.output):
        write_json(planned.output / 'status.json', {'state': 'running', 'neural_updates': 0})
        raise RuntimeError('GPU busy')
    assert runner.read(planned.output / 'status.json')['state'] == 'interrupted'


def test_busy_gpu_is_refused_without_loading_models(monkeypatch):
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *args, **kwargs: '123, python\n')
    with pytest.raises(RuntimeError, match='GPU is in use'):
        runner.require_idle_gpu()


def test_failed_solver_is_counted_and_retry_is_explicit(planned, monkeypatch):
    original = runner.evaluate_baseline

    def fail(*args, **kwargs):
        raise RuntimeError('Deliberate solver failure')

    monkeypatch.setattr(runner, 'evaluate_baseline', fail)
    planned.methods = ['column_mean']
    runner.run(planned)
    manifest = runner.verify_manifest(planned.output)
    directory = planned.output / 'cases' / manifest['cases'][0]['case_id']
    record_path = directory / 'column_mean.json'
    assert runner.read(record_path)['status'] == 'failed'
    assert runner.read(planned.output / 'status.json')['failed'] == 1
    monkeypatch.setattr(runner, 'evaluate_baseline', original)
    runner.run(planned)
    assert runner.read(record_path)['status'] == 'failed'
    planned.retry_failed = True
    runner.run(planned)
    assert runner.read(record_path)['status'] == 'ok'
    assert len(list(directory.glob('column_mean.failed_attempt_*.json'))) == 1
    (directory / 'column_mean.npz').write_bytes(b'changed prediction')
    with pytest.raises(ValueError, match='Archived prediction changed'):
        runner.run(planned)


def test_uncertainty_missing_and_failed_counts():
    assert report.statistics([])['mean'] is None
    assert report.statistics([2.])['ci95_low'] is None
    values = report.statistics([1., 3.])
    assert values['mean'] == 2. and values['std'] == pytest.approx(np.sqrt(2.))
    cases = [{**make_case(seed=seed), 'case_id': str(seed)} for seed in (1, 2)]
    records = [{**cases[0], 'method': 'column_mean', 'status': 'ok', 'rmse': 2., 'signal_rmse': 2.},
               {**cases[1], 'method': 'column_mean', 'status': 'failed'},
               {**cases[0], 'method': 'matrixformer_plus', 'status': 'ok', 'rmse': 1., 'signal_rmse': 1.}]
    summaries, paired = report.aggregate(records, cases)
    column = next(row for row in summaries if row['method'] == 'column_mean')
    plus = next(row for row in summaries if row['method'] == 'matrixformer_plus')
    assert column['rmse_n'] == column['failed_n'] == 1 and column['expected_n'] == 2
    assert plus['pending_n'] == 1
    assert paired[0]['n'] == 1 and paired[0]['mean'] == 1.
