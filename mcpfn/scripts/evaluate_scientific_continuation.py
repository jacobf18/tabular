"""Evaluate a synthetic-selected continuation on the existing scientific benchmarks."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.impute import KNNImputer
import torch

import evaluate_cylinder_completion as shared
import evaluate_realistic_science as original
from tabimpute.prior.scientific_panels import TRANSFER_FAMILIES, sample_scientific_episode
from tabimpute.streamed_inference import streamed_point_predictions
from train_scientific_matrixformer import DEFAULT_OUTPUT, OPTIONS, science_episode, score_episode


SCIENCE_ROOT = Path('/mnt/matrixformer_scratch/science_panels_20261106')
REALISTIC_ROOT = Path('/mnt/matrixformer_scratch/realistic_science_20261106')
SUITES = {
    'science': [SCIENCE_ROOT / 'replay_v1'],
    'ocean': [SCIENCE_ROOT / 'ocean_v2'],
    'initial_sensors': [REALISTIC_ROOT / name for name in ('outage_replay_v2', 'linear_controls_v1', 'calibration_controls_v1')],
    'additional_sensors': [REALISTIC_ROOT / name for name in ('outage_replication_v2', 'replication_linear_controls_v1', 'replication_calibration_controls_v1')],
}


def register(run, output):
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    control_hashes = {}
    for suite, parents in SUITES.items():
        parent = parents[0]
        protocol = json.loads((parent / 'protocol.json').read_text())
        assert protocol['checkpoints']['MatrixFormer']['sha256'] == shared.SINGLE_SHA256
        for directory in parents:
            for path in (directory / 'records').iterdir():
                if path.suffix in ('.json', '.npz'):
                    control_hashes[str(path)] = shared.sha256(path)
        for case in protocol['cases']:
            path = parent / f'{case["identifier"]}.npz'
            assert shared.sha256(path) == case['input_sha256']
            targets = case['targets'] if 'targets' in case else protocol['datasets'][case['dataset']]['targets']
            cases.append(dict(case, suite=suite, parent=str(parent), targets=targets,
                              controls=[str(directory) for directory in parents], input_path=str(path),
                              original_protocol_sha256=shared.sha256(parent / 'protocol.json')))
    shared.write_json(output / 'protocol.json', dict(
        created=datetime.now(timezone.utc).isoformat(), training_run=str(run), cases=cases,
        control_sha256=control_hashes,
        source_sha256={str(Path(__file__).resolve()): shared.sha256(__file__),
                       str(Path(original.__file__).resolve()): shared.sha256(original.__file__)},
        checkpoint_selection='Use only the training run synthetic-selected checkpoint; real results never choose a checkpoint in this run',
        interpretation='All these real datasets have previously been inspected and are development benchmarks; original split labels are retained for comparison, not fresh transfer claims',
        inference='Original full-window standard-orientation evaluator, original masks, original observed-validation transform grid; no local-context adapter',
        transfer_specs=[dict(seed=99310600 + index * 100 + draw, family=family, rows=rows, columns=columns,
                             pattern=pattern, allow_transfer=True)
                        for index, family in enumerate(TRANSFER_FAMILIES)
                        for draw, (rows, columns, pattern) in enumerate(((96, 16, 'isolated'), (192, 32, 'bursts'), (128, 24, 'network_outage')))],
    ))


def wait_for_training(run, output):
    while True:
        status = json.loads((run / 'status.json').read_text())
        if status['state'] == 'complete':
            return status
        if status['state'] in ('failed', 'interrupted'):
            raise RuntimeError(f'Training did not complete: {status}')
        worker = int((run / 'worker.pid').read_text())
        os.kill(worker, 0)
        shared.write_json(output / 'status.json', dict(state='waiting_for_training', training=status, updated=time.time()))
        time.sleep(20)


def cached_controls(case, arrays):
    rows = []
    seen = set()
    current = np.where(arrays['missing'], np.nan, arrays['truth'])
    for directory in case['controls']:
        parent = Path(directory)
        protocol = json.loads((parent / 'protocol.json').read_text())
        matches = [record for record in protocol['cases'] if record['identifier'] == case['identifier']]
        if len(matches) != 1 or matches[0]['input_sha256'] != case['input_sha256']:
            raise ValueError('Control inputs do not match the registered case')
        for path in sorted((parent / 'records').glob(f'{case["identifier"]}_*.json')):
            record = json.loads(path.read_text())
            method = record['method']
            if method in seen:
                continue
            prediction_path = path.with_suffix('.npz')
            if 'prediction_sha256' in record and shared.sha256(prediction_path) != record['prediction_sha256']:
                raise ValueError('Control prediction fingerprint mismatch')
            prediction = np.load(prediction_path)['prediction']
            np.testing.assert_array_equal(prediction[np.isfinite(current)], current[np.isfinite(current)].astype(prediction.dtype))
            score = original.metrics(arrays['truth'], prediction, arrays['missing'], current, case['targets'])
            rounding_bounds = []
            for column in score['per_column']:
                selected = arrays['missing'][:, column['column']]
                spacing = np.spacing(prediction[selected, column['column']]).astype(float)
                rounding_bounds.append(np.sqrt(np.mean(spacing ** 2)) / column['observed_sd'])
            if abs(score['nrmse'] - record['scores']['nrmse']) > max(1e-10, float(np.mean(rounding_bounds))):
                raise ValueError(f'Cached score differs beyond saved-precision rounding: {path}')
            rows.append(dict(method=method, nrmse=score['nrmse'], source=str(path)))
            seen.add(method)
    if 'MatrixFormer' not in seen:
        raise ValueError('Missing unchanged-parent control')
    return rows


def evaluate(run, output, status):
    protocol = json.loads((output / 'protocol.json').read_text())
    for path, expected in protocol['source_sha256'].items():
        assert shared.sha256(path) == expected
    for path, expected in protocol['control_sha256'].items():
        assert shared.sha256(path) == expected
    frozen = output / 'checkpoint'
    frozen.mkdir()
    checkpoint = frozen / 'best_validation_checkpoint.pth'
    os.link(run / checkpoint.name, checkpoint)
    shutil.copyfile(run / 'config.json', frozen / 'config.json')
    shared.write_json(output / 'checkpoint.json', dict(path=str(checkpoint), sha256=shared.sha256(checkpoint),
                      selected_step=status['best_step'], synthetic_score=status['best_score']))
    models, distribution = shared.load_models({'MatrixFormer-science': checkpoint})
    rows = []
    (output / 'records').mkdir()
    for case in protocol['cases']:
        assert shared.sha256(case['input_path']) == case['input_sha256']
        assert shared.sha256(Path(case['parent']) / 'protocol.json') == case['original_protocol_sha256']
        arrays = np.load(case['input_path'])
        current = np.where(arrays['missing'], np.nan, arrays['truth'])
        reduced = np.where(arrays['validation'], np.nan, current)
        common = dict(identifier=case['identifier'], dataset=case['dataset'], split=case['split'], suite=case['suite'])
        for record in cached_controls(case, arrays):
            rows.append(dict(common, **record))
        trials = []
        for setting in original.settings('MatrixFormer'):
            prediction, _ = original.predict(reduced, 'MatrixFormer-science', setting, case['targets'], [], models, distribution)
            score = original.metrics(arrays['truth'], prediction, arrays['validation'], reduced, case['targets'])
            trials.append(dict(setting=setting, validation_nrmse=score['nrmse']))
        selected = min(trials, key=lambda record: record['validation_nrmse'])
        prediction, _ = original.predict(current, 'MatrixFormer-science', selected['setting'], case['targets'], [], models, distribution)
        np.testing.assert_array_equal(prediction[np.isfinite(current)], current[np.isfinite(current)])
        scores = original.metrics(arrays['truth'], prediction, arrays['missing'], current, case['targets'])
        destination = output / 'records' / f'{case["suite"]}_{case["identifier"]}.json'
        np.savez_compressed(destination.with_suffix('.npz'), prediction=prediction)
        shared.write_json(destination, dict(common, scores=scores, tuning=dict(selected=selected, trials=trials),
                          prediction_sha256=shared.sha256(destination.with_suffix('.npz'))))
        rows.append(dict(common, method='MatrixFormer-science', nrmse=scores['nrmse'], source=str(destination)))
        pd.DataFrame(rows).to_csv(output / 'per_case.csv', index=False)
        print(case['suite'], case['identifier'], scores['nrmse'], flush=True)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(['dataset', 'split', 'method']).nrmse.agg(['mean', 'std', 'count']).reset_index()
    summary['ci95_half_width'] = summary.apply(lambda row: student_t.ppf(.975, row['count'] - 1) * row['std'] / np.sqrt(row['count']) if row['count'] > 1 else np.nan, axis=1)
    expected = frame[frame.method == 'MatrixFormer-science'].groupby(['dataset', 'split']).size().to_dict()
    summary['expected_count'] = [expected[(row.dataset, row.split)] for row in summary.itertuples()]
    summary.to_csv(output / 'summary.csv', index=False)
    candidate = frame[frame.method == 'MatrixFormer-science']
    parent = frame[frame.method == 'MatrixFormer']
    paired = candidate.merge(parent, on=['dataset', 'suite', 'identifier', 'split'], suffixes=('_new', '_parent'), validate='one_to_one')
    paired['improvement'] = paired.nrmse_parent - paired.nrmse_new
    paired.to_csv(output / 'paired_vs_parent.csv', index=False)
    assert len(candidate) == len(paired) == len(protocol['cases'])

    parent_models, _ = shared.load_models({'MatrixFormer': shared.SINGLE})
    models.update(parent_models)
    transfer_rows = []
    for index, spec in enumerate(protocol['transfer_specs']):
        episode = science_episode(spec)
        source = episode.source[0].cpu().numpy()
        predictions = {
            'Temporal interpolation': original.temporal_fill(source),
            'KNN': KNNImputer(n_neighbors=5, weights='distance', keep_empty_features=True).fit_transform(source),
        }
        for method, model in models.items():
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                predictions[method] = streamed_point_predictions(model, episode.source, distribution, **OPTIONS)[0].cpu().numpy()
        for method, prediction in predictions.items():
            scores = score_episode(torch.from_numpy(prediction)[None].cuda(), episode)
            transfer_rows.append(dict(index=index, family=spec['family'], method=method, **scores))
    pd.DataFrame(transfer_rows).to_csv(output / 'synthetic_transfer.csv', index=False)
    shared.write_json(output / 'status.json', dict(state='complete', real_cases=len(candidate), transfer_records=len(transfer_rows),
                      selected_step=status['best_step'], updated=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--wait-for-completion', action='store_true')
    arguments = parser.parse_args()
    output = arguments.output or arguments.run / 'real_benchmarks'
    register(arguments.run, output)
    try:
        status = wait_for_training(arguments.run, output) if arguments.wait_for_completion else json.loads((arguments.run / 'status.json').read_text())
        if status['state'] != 'complete':
            raise ValueError('Training must finish before this registered final evaluation')
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(.16)
        evaluate(arguments.run, output, status)
    except Exception as error:
        shared.write_json(output / 'status.json', dict(state='failed', error=repr(error), updated=time.time()))
        raise


if __name__ == '__main__':
    main()
