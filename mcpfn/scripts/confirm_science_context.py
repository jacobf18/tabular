"""Previously unused Intel days to confirm local-context preprocessing effects."""

import argparse
from datetime import datetime, timezone
from itertools import permutations
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

import evaluate_cylinder_completion as shared
import evaluate_realistic_science as original
import evaluate_science_panels as panels
from fetch_science_panels import ROOT
import science_context_preprocessing as preprocessing
import test_science_context_fix as context


OUTPUT = ROOT / 'context_confirmation_v1'


def register(output):
    original_protocol = json.loads((ROOT / 'replay_v1/protocol.json').read_text())
    parent_protocol = json.loads((context.OUTPUT / 'protocol.json').read_text())
    metadata = original_protocol['datasets']['intel_temperature']
    source = ROOT / 'prepared_v2/intel_temperature.csv'
    assert shared.sha256(source) == metadata['prepared_sha256']
    frame = pd.read_csv(source, index_col=0, parse_dates=True)
    width = metadata['window']
    used = {case['target_start'] // width for case in original_protocol['cases'] if case['dataset'] == 'intel_temperature'}
    output.mkdir(parents=True, exist_ok=False)
    cases, exclusions = [], []
    for block in range(3, len(frame) // width):
        if block in used:
            continue
        reasons = []
        for donor, validation_donor in permutations(range(3), 2):
            try:
                arrays, details = panels.make_case(frame, metadata, block * width, donor * width, validation_donor * width)
                break
            except ValueError as error:
                reasons.append(str(error))
        else:
            exclusions.append(dict(block=block, reasons=reasons))
            continue
        current = np.where(arrays['missing'], np.nan, arrays['truth'])
        arrays['validation'] = preprocessing.cover_validation_targets(current, arrays['validation'], details['targets'])
        identifier = f'intel_temperature_newday_{block:02}'
        destination = output / f'{identifier}.npz'
        np.savez_compressed(destination, **arrays)
        cases.append(dict(identifier=identifier, dataset='intel_temperature', split='confirmation', **details,
                          input_sha256=shared.sha256(destination), modified_input_sha256=shared.sha256(destination),
                          columns=metadata['columns'] + ['hour_sin', 'hour_cos', 'relative_time']))
    sources = dict(parent_protocol['source_sha256'])
    sources[str(Path(__file__))] = shared.sha256(__file__)
    protocol = dict(parent_protocol, created=datetime.now(timezone.utc).isoformat(), cases=cases, exclusions=exclusions,
                    source_sha256=sources, datasets={'intel_temperature': metadata},
                    selection='All remaining eligible nonoverlapping target days; original raw data, target rules and donor pool unchanged',
                    status='New target days, same sensor deployment; freeze before any predictions on these days',
                    local_knn_grid=[dict(**setting, neighbors=neighbors) for setting in context.SETTINGS for neighbors in (3, 10, 30)])
    shared.write_json(output / 'protocol.json', protocol)
    baseline_output = output / 'baselines'
    baseline_output.mkdir()
    for case in cases:
        shutil.copyfile(output / f'{case["identifier"]}.npz', baseline_output / f'{case["identifier"]}.npz')
    methods = ['Temporal interpolation', 'Temporal regression', 'KNN', 'MatrixFormer', 'MatrixFormer-Impute']
    shared.write_json(baseline_output / 'protocol.json', dict(protocol, methods=methods, screen_methods=methods,
                                                           grids={method: original.settings(method) for method in methods}))
    print(pd.DataFrame(cases)[['identifier', 'target_date', 'target_cells']].to_string(index=False))


def local_knn(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    for path, expected in protocol['source_sha256'].items():
        assert shared.sha256(path) == expected
    for case in protocol['cases']:
        arrays = np.load(output / f'{case["identifier"]}.npz')
        truth, missing, validation = [arrays[name] for name in ('truth', 'missing', 'validation')]
        current = np.where(missing, np.nan, truth)
        reduced = np.where(validation, np.nan, current)
        destination = output / 'records' / f'{case["identifier"]}_Local_KNN.json'
        if destination.exists():
            continue

        def predict(matrix, setting):
            def forward(values, model, distribution):
                return KNNImputer(n_neighbors=setting['neighbors'], weights='distance', keep_empty_features=True).fit_transform(values), {}
            return preprocessing.temporal_context_predict(matrix, case['targets'], None, None, forward,
                                                          setting['rows'], setting['delays'])

        trials = []
        for setting in protocol['local_knn_grid']:
            prediction = predict(reduced, setting)
            score = original.metrics(truth, prediction, validation, reduced, case['targets'])
            trials.append(dict(setting=setting, validation_nrmse=score['nrmse']))
        selected = min(trials, key=lambda trial: trial['validation_nrmse'])
        prediction = predict(current, selected['setting'])
        scores = original.metrics(truth, prediction, missing, current, case['targets'])
        destination.parent.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(destination.with_suffix('.npz'), prediction=prediction)
        shared.write_json(destination, dict(identifier=case['identifier'], model='Local KNN', scores=scores,
                          tuning=dict(selected=selected, trials=trials), prediction_sha256=shared.sha256(destination.with_suffix('.npz'))))
        print(case['identifier'], 'Local KNN', scores['nrmse'], flush=True)


def report(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    rows = []
    for case in protocol['cases']:
        arrays = np.load(output / f'{case["identifier"]}.npz')
        current = np.where(arrays['missing'], np.nan, arrays['truth'])
        paths = list((output / 'records').glob(f'{case["identifier"]}_*.json'))
        paths += list((output / 'baselines/records').glob(f'{case["identifier"]}_*.json'))
        for path in paths:
            record = json.loads(path.read_text())
            assert record['prediction_sha256'] == shared.sha256(path.with_suffix('.npz'))
            assert record['tuning']['selected'] == min(record['tuning']['trials'], key=lambda trial: trial['validation_nrmse'])
            prediction = np.load(path.with_suffix('.npz'))['prediction']
            np.testing.assert_array_equal(prediction[np.isfinite(current)], current[np.isfinite(current)])
            score = original.metrics(arrays['truth'], prediction, arrays['missing'], current, case['targets'])
            np.testing.assert_allclose(score['nrmse'], record['scores']['nrmse'])
            method = record.get('model', record.get('method'))
            if 'baselines' in path.parts and method.startswith('MatrixFormer'):
                method += ' native'
            rows.append(dict(identifier=case['identifier'], method=method, nrmse=score['nrmse']))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'per_case.csv', index=False)
    summary = frame.groupby('method').nrmse.agg(['mean', 'std', 'count']).reset_index()
    summary['ci95_half_width'] = summary.apply(lambda row: context.diagnosis.student_t.ppf(.975, row['count']-1)*row['std']/np.sqrt(row['count']) if row['count']>1 else np.nan, axis=1)
    summary.sort_values('mean').to_csv(output / 'summary.csv', index=False)
    complete = len(frame)==len(protocol['cases'])*8 and frame.groupby('method').identifier.nunique().eq(len(protocol['cases'])).all()
    shared.write_json(output / 'audit.json', dict(records=len(frame), expected=len(protocol['cases'])*8, complete=bool(complete)))
    print(summary.sort_values('mean').to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run', 'knn', 'report'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    context.torch.set_num_threads(2)
    if arguments.action == 'register':
        register(arguments.output)
    elif arguments.action == 'run':
        context.run(arguments.output)
    elif arguments.action == 'knn':
        local_knn(arguments.output)
    else:
        report(arguments.output)
