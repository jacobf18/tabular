"""Stronger nonlinear calibration controls for the frozen sensor-outage benchmark."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR

import evaluate_cylinder_completion as shared
import evaluate_realistic_science as evaluation


METHODS = ('RBF-SVR', 'RBF-SVR + temporal residual', 'Gradient boosting')


def settings(method):
    if method == 'Gradient boosting':
        return [dict(transform=transform, leaves=leaves, regularization=regularization)
                for transform in ('column', 'asinh_column') for leaves in (7, 15)
                for regularization in (1., 10.)]
    return [dict(transform=transform, cost=cost, gamma=gamma)
            for transform in ('column', 'asinh_column') for cost in (1., 10., 100.)
            for gamma in (.01, .1, 1.)]


def predict(matrix, targets, method, setting):
    normalized, inverse = evaluation.transform(matrix, setting['transform'])
    prediction = evaluation.temporal_fill(normalized)
    positions = np.arange(len(matrix))
    for column in targets:
        observed = np.isfinite(normalized[:, column])
        if observed.sum() < 3:
            continue
        predictors = np.delete(normalized, column, axis=1)
        if method == 'Gradient boosting':
            features = predictors
            estimator = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=setting['leaves'],
                                                       l2_regularization=setting['regularization'],
                                                       min_samples_leaf=10, early_stopping=False, random_state=961)
        else:
            features = SimpleImputer(strategy='median', add_indicator=True,
                                     keep_empty_features=True).fit_transform(predictors)
            estimator = SVR(C=setting['cost'], gamma=setting['gamma'], epsilon=.05)
        estimator.fit(features[observed], normalized[observed, column])
        fitted = estimator.predict(features)
        if method == 'RBF-SVR + temporal residual':
            fitted += np.interp(positions, positions[observed], normalized[observed, column] - fitted[observed])
        prediction[:, column] = fitted
    result = inverse(prediction)
    result[np.isfinite(matrix)] = matrix[np.isfinite(matrix)]
    assert np.isfinite(result).all()
    return result


def register(parent, output):
    parent_protocol = json.loads((parent / 'protocol.json').read_text())
    if output.exists():
        raise ValueError('Register controls in a fresh directory')
    output.mkdir(parents=True)
    sources = [Path(__file__), Path(evaluation.__file__), Path(shared.__file__)]
    protocol = dict(created=datetime.now(timezone.utc).isoformat(), parent=str(parent),
                    parent_protocol_sha256=shared.sha256(parent / 'protocol.json'),
                    datasets=parent_protocol['datasets'], cases=parent_protocol['cases'], methods=list(METHODS),
                    source_sha256={str(path): shared.sha256(path) for path in sources},
                    grids={method: settings(method) for method in METHODS},
                    selection='Same recorded observed-validation mask and normalized metric as neural methods; no test selection')
    shared.write_json(output / 'protocol.json', protocol)
    shutil.copyfile(__file__, output / 'source.py')


def run(output, split, datasets):
    protocol = json.loads((output / 'protocol.json').read_text())
    parent = Path(protocol['parent'])
    assert shared.sha256(parent / 'protocol.json') == protocol['parent_protocol_sha256']
    for path, expected in protocol['source_sha256'].items():
        assert shared.sha256(path) == expected
    for case in protocol['cases']:
        if case['split'] != split or case['dataset'] not in datasets:
            continue
        source = parent / f'{case["identifier"]}.npz'
        assert shared.sha256(source) == case['input_sha256']
        arrays = np.load(source)
        truth, missing, validation = [arrays[name] for name in ('truth', 'missing', 'validation')]
        context = np.where(missing, np.nan, truth)
        reduced = np.where(validation, np.nan, context)
        targets = protocol['datasets'][case['dataset']]['targets']
        for method in protocol['methods']:
            destination = output / 'records' / f'{case["identifier"]}_{method.replace(" ", "_")}.json'
            if destination.exists():
                continue
            started = time.perf_counter()
            print('Start', case['identifier'], method, flush=True)
            trials = []
            for setting in protocol['grids'][method]:
                candidate = predict(reduced, targets, method, setting)
                scores = evaluation.metrics(truth, candidate, validation, reduced, targets)
                trials.append(dict(setting=setting, validation_nrmse=scores['nrmse']))
            selected = min(trials, key=lambda record: record['validation_nrmse'])
            prediction = predict(context, targets, method, selected['setting'])
            scores = evaluation.metrics(truth, prediction, missing, context, targets)
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination.with_suffix('.npz'), prediction=prediction.astype(np.float32))
            shared.write_json(destination, dict(identifier=case['identifier'], dataset=case['dataset'], split=split,
                                               method=method, scores=scores, tuning=dict(selected=selected, trials=trials),
                                               seconds=time.perf_counter()-started))
            print('Done', case['identifier'], method, f'{scores["nrmse"]:.5f}', selected['setting'],
                  f'{time.perf_counter()-started:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run'])
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['development', 'confirmation'], default='development')
    parser.add_argument('--datasets', nargs='+', choices=list(evaluation.PLANS), default=list(evaluation.PLANS))
    arguments = parser.parse_args()
    if arguments.action == 'register':
        register(arguments.parent, arguments.output)
    else:
        run(arguments.output, arguments.split, arguments.datasets)
