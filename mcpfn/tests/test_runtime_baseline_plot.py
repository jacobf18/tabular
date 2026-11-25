import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def plotting(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / '69d7e55fd826e0325c3115d0/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    specification = importlib.util.spec_from_file_location('runtime_baseline_figure', scripts / 'plot_runtime_baselines.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def examples(plotting):
    manifest = dict(rows=[16, 32], columns=10, repeats=5)
    records = [dict(model=method, rows=rows, columns=10, repeat=repeat,
                    seconds=(index + 1) * repeat * rows / 1000)
               for index, method in enumerate(plotting.METHODS)
               for rows in manifest['rows'] for repeat in range(1, 6)]
    return pd.DataFrame(records), manifest


def test_summary_preserves_full_time_and_median_ratios(plotting):
    timings, manifest = examples(plotting)
    summary = plotting.summarize(timings, manifest)
    assert len(summary) == 14
    for index, method in enumerate(plotting.METHODS):
        selected = summary[summary.model == method]
        np.testing.assert_allclose(selected.relative_runtime, index + 1)
        np.testing.assert_allclose(selected.median_seconds, (index + 1) * 3 * np.array([16, 32]) / 1000)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'zero', 'nan'])
def test_summary_rejects_incomplete_or_invalid_timings(plotting, change):
    timings, manifest = examples(plotting)
    if change == 'missing':
        timings = timings.iloc[:-1]
    elif change == 'duplicate':
        timings = pd.concat([timings, timings.iloc[:1]])
    else:
        timings.loc[0, 'seconds'] = 0 if change == 'zero' else np.nan
    with pytest.raises(ValueError):
        plotting.summarize(timings, manifest)


def test_ratios_below_one_are_not_clipped(plotting, tmp_path):
    timings, manifest = examples(plotting)
    timings.loc[timings.model == 'K-NN', 'seconds'] /= 100
    summary = plotting.summarize(timings, manifest)
    assert (summary.loc[summary.model == 'K-NN', 'relative_runtime'] < 1).all()
    plotting.draw(timings, summary, manifest, tmp_path / 'runtime')
    assert (tmp_path / 'runtime.pdf').stat().st_size > 1000
    assert (tmp_path / 'runtime.png').stat().st_size > 1000


def test_portable_export_preserves_archive_and_exact_repeated_times(plotting, tmp_path):
    gpu = tmp_path / 'gpu'
    cpu = tmp_path / 'cpu'
    output = tmp_path / 'portable'
    gpu.mkdir()
    (cpu / 'cases').mkdir(parents=True)
    protocol = dict(rows=[16], columns=10, seed=42, missing_fraction=.4, repeats=5, warmup=2)
    (gpu / 'manifest.json').write_text(json.dumps(protocol))
    records = [dict(model=method, model_key=key, rows=16, columns=10, repeat=repeat, seconds=.12345678901234567)
               for method, key in [('TabImpute (legacy)', 'legacy'), ('MatrixFormer', 'matrixformer')]
               for repeat in range(1, 6)]
    pd.DataFrame(records).to_csv(gpu / 'timings.csv', index=False)
    hashes = {str(gpu / filename): plotting.digest(gpu / filename) for filename in ['manifest.json', 'timings.csv']}
    records = []
    methods = {method.lower(): method for method in plotting.METHODS[2:]}
    for method, label in methods.items():
        seconds = [.9876543210123456] * 5
        (cpu / 'cases' / f'{method}_16.json').write_text(json.dumps(dict(
            status='ok', input_sha256='same-input', seconds=seconds)))
        records.extend(dict(model=label, model_key=method, rows=16, columns=10, repeat=repeat,
                            seconds=seconds[repeat - 1]) for repeat in range(1, 6))
    pd.DataFrame(records).to_csv(cpu / 'timings.csv', index=False)
    (cpu / 'manifest.json').write_text(json.dumps(dict(
        **protocol, methods=methods, source_sha256=hashes, input_sha256={'16': 'same-input'},
        timing='whole matrix', exclusions='warmup', gpu_note='archived')))
    (cpu / 'status.json').write_text(json.dumps(dict(state='complete')))
    plotting.import_measurements(cpu, gpu, output)
    assert all(plotting.digest(filename) == expected for filename, expected in hashes.items())
    timings = pd.read_csv(output / 'timings.csv')
    assert len(timings) == 35
    assert set(timings.model) == set(plotting.METHODS)
    (cpu / 'status.json').write_text(json.dumps(dict(state='running')))
    with pytest.raises(ValueError, match='incomplete'):
        plotting.import_measurements(cpu, gpu, output)


def merged_examples(plotting, tmp_path):
    timings, protocol = examples(plotting)
    protocol.update(seed=42, missing_fraction=.4, warmup=2, methods=plotting.METHODS,
                    timing_scope='whole matrix', exclusions='warmup')
    cpu = dict(methods=plotting.METHODS[2:], versions={'numpy': 'same'}, thread_environment={},
               torch_threads=1, threadpool_limit=1, estimator_parallelism='same',
               mice='five', softimpute='cross-validation', hyperimpute='adaptive')
    gpu = dict(models={key: dict(checkpoint_sha256=key, precision='bf16', prediction_stat='mean')
                       for key in ('matrixformer', 'legacy')})
    paths = []
    for label, rows, fraction in [('base', 16, .25), ('extension', 32, .95)]:
        path = tmp_path / label
        path.mkdir()
        (path / 'manifest.json').write_text(json.dumps(dict(protocol, rows=[rows])))
        timings[timings.rows == rows].to_csv(path / 'timings.csv', index=False)
        (path / 'cpu_manifest.json').write_text(json.dumps(cpu))
        (path / 'gpu_manifest.json').write_text(json.dumps(dict(gpu, worker_memory_fraction=fraction)))
        paths.append(path)
    return paths


def test_extension_preserves_every_original_call_and_provenance(plotting, tmp_path):
    base, extension = merged_examples(plotting, tmp_path)
    output = tmp_path / 'merged'
    original_hash = plotting.digest(base / 'timings.csv')
    plotting.merge_measurements(base, extension, output)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['rows'] == [16, 32]
    assert manifest['gpu_memory_fractions'] == {'base': .25, 'extension': .95}
    combined = pd.read_csv(output / 'timings.csv', float_precision='round_trip')
    original = pd.read_csv(base / 'timings.csv', float_precision='round_trip')
    pd.testing.assert_frame_equal(combined[combined.rows == 16], original)
    assert plotting.digest(base / 'timings.csv') == original_hash
    assert (output / 'extension_gpu_manifest.json').exists()
    assert len(combined) == 70


@pytest.mark.parametrize('change', ['overlap', 'model', 'cpu', 'incomplete', 'destination'])
def test_extension_rejects_incompatible_or_partial_measurements(plotting, tmp_path, change):
    base, extension = merged_examples(plotting, tmp_path)
    output = tmp_path / 'merged'
    if change == 'destination':
        output = base
    elif change == 'incomplete':
        frame = pd.read_csv(extension / 'timings.csv')
        frame.iloc[:-1].to_csv(extension / 'timings.csv', index=False)
    else:
        filename = {'overlap': 'manifest.json', 'model': 'gpu_manifest.json', 'cpu': 'cpu_manifest.json'}[change]
        path = extension / filename
        manifest = json.loads(path.read_text())
        if change == 'overlap':
            manifest['rows'] = [16]
        elif change == 'model':
            manifest['models']['legacy']['checkpoint_sha256'] = 'different'
        else:
            manifest['threadpool_limit'] = 12
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        plotting.merge_measurements(base, extension, output)
