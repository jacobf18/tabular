"""Linear chemometric and iterative-forest controls using the original validation masks."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import Ridge

import control_realistic_calibration as runner
import evaluate_cylinder_completion as shared
import evaluate_realistic_science as evaluation


METHODS = ('Ridge calibration', 'PLS calibration', 'Iterative random forest')


def settings(method):
    if method == 'Iterative random forest':
        return [dict(transform=transform, leaf=leaf) for transform in ('column', 'asinh_column') for leaf in (1, 3)]
    key, values = ('alpha', (.01, .1, 1., 10., 100.)) if method == 'Ridge calibration' else ('components', (2, 4, 8))
    return [dict(transform=transform, **{key: value}) for transform in ('column', 'asinh_column') for value in values]


def predict(matrix, targets, method, setting):
    normalized, inverse = evaluation.transform(matrix, setting['transform'])
    if method == 'Iterative random forest':
        estimator = RandomForestRegressor(n_estimators=100, min_samples_leaf=setting['leaf'],
                                          max_features=1., n_jobs=1, random_state=961)
        imputer = IterativeImputer(estimator=estimator, max_iter=10, tol=1e-3, skip_complete=True,
                                  initial_strategy='median', keep_empty_features=True, random_state=961)
        prediction = imputer.fit_transform(normalized)
    else:
        prediction = evaluation.temporal_fill(normalized)
        for column in targets:
            observed = np.isfinite(normalized[:, column])
            if observed.sum() < 3:
                continue
            features = SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(np.delete(normalized, column, axis=1))
            if method == 'Ridge calibration':
                estimator = Ridge(alpha=setting['alpha'])
            else:
                count = min(setting['components'], features.shape[1], observed.sum() - 1)
                estimator = PLSRegression(n_components=count, scale=False, max_iter=1000)
            estimator.fit(features[observed], normalized[observed, column])
            prediction[:, column] = estimator.predict(features).reshape(-1)
    restored = inverse(prediction)
    restored[np.isfinite(matrix)] = matrix[np.isfinite(matrix)]
    assert np.isfinite(restored).all()
    return restored


def register(parent, output):
    runner.METHODS = METHODS
    runner.settings = settings
    runner.register(parent, output)
    protocol_path = output / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    protocol['source_sha256'][str(Path(__file__))] = shared.sha256(__file__)
    shared.write_json(protocol_path, protocol)
    (output / 'linear_source.py').write_bytes(Path(__file__).read_bytes())


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
        runner.predict = predict
        runner.run(arguments.output, arguments.split, arguments.datasets)
