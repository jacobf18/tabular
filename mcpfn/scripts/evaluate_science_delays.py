"""Leakage-safe time-delay inputs, selected on recorded validation gaps only."""

import argparse
from datetime import datetime, timezone
from itertools import permutations
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

import evaluate_cylinder_completion as shared
import evaluate_science_panels as panels
from fetch_science_panels import ROOT


METHODS = ('Column mean', 'Temporal interpolation', 'Temporal regression', 'KNN', 'SoftImpute',
           'Random forest', 'Ridge calibration', 'MatrixFormer', 'MatrixFormer-Impute')
BASE_PREDICT = panels.predict
DELAY_GRIDS = ((), (1,), (1, 3), (1, 6, 24))


def delayed_matrix(matrix, targets, delays):
    blocks = [matrix]
    for delay in delays:
        if delay <= 0 or delay >= len(matrix):
            raise ValueError('Delay must be positive and smaller than the window')
        previous = np.full((len(matrix), len(targets)), np.nan)
        following = previous.copy()
        previous[delay:] = matrix[:-delay, targets]
        following[:-delay] = matrix[delay:, targets]
        blocks.extend((previous, following))
    return np.column_stack(blocks)


def settings(method):
    delays = ((),) if method in ('Column mean', 'Temporal interpolation', 'Temporal regression') else DELAY_GRIDS
    return [dict(**setting, delays=list(current)) for current in delays for setting in panels.settings(method)]


def predict(matrix, method, setting, targets, groups, models, distribution):
    parameters = {key: value for key, value in setting.items() if key != 'delays'}
    augmented = delayed_matrix(matrix, targets, setting['delays'])
    prediction, details = BASE_PREDICT(augmented, method, parameters, targets, groups, models, distribution)
    return prediction[:, :matrix.shape[1]], details


def register(parent, prepared, output):
    source = json.loads((parent / 'protocol.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    cases, exclusions = [], []
    for case in source['cases']:
        if case['dataset'] not in ('potomac', 'susquehanna'):
            continue
        shutil.copyfile(parent / f'{case["identifier"]}.npz', output / f'{case["identifier"]}.npz')
        cases.append(dict(case, split='exploratory' if case['split'] == 'confirmation' else 'development'))
    for dataset in ('potomac', 'susquehanna'):
        metadata = source['datasets'][dataset]
        path = prepared / f'{dataset}.csv'
        assert shared.sha256(path) == metadata['prepared_sha256']
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        width = metadata['window']
        used = {case['target_start'] // width for case in source['cases'] if case['dataset'] == dataset}
        attempted = {record['block'] for record in source['exclusions'] if record['dataset'] == dataset}
        for block in range(4, len(frame) // width):
            if block in used or block in attempted:
                continue
            attempts = []
            for donor, validation in permutations(range(3), 2):
                try:
                    arrays, details = panels.make_case(frame, metadata, block * width, donor * width, validation * width)
                    break
                except ValueError as error:
                    attempts.append(str(error))
            else:
                exclusions.append(dict(dataset=dataset, block=block, reasons=attempts))
                continue
            identifier = f'{dataset}_replication_{block:02}'
            destination = output / f'{identifier}.npz'
            np.savez_compressed(destination, **arrays)
            cases.append(dict(identifier=identifier, dataset=dataset, split='replication', **details,
                              input_sha256=shared.sha256(destination)))
    protocol = dict(source, created=datetime.now(timezone.utc).isoformat(), cases=cases, exclusions=exclusions,
                    datasets={key: value for key, value in source['datasets'].items() if key in ('potomac', 'susquehanna')},
                    methods=list(METHODS), screen_methods=list(METHODS), control_methods=[],
                    grids={method: settings(method) for method in METHODS},
                    parent=str(parent), parent_sha256=shared.sha256(parent / 'protocol.json'),
                    adaptation='Append past/future target-channel observations at lags 1; 1,3; or 1,6,24 hours. Include native input as candidate. Every candidate is built AFTER masking. One forward pass; central predictions only.',
                    fairness='Same delay candidates for neural methods, KNN, SoftImpute, forest and ridge. Original temporal methods remain strong direct controls.',
                    confirmation='Previously scored windows labeled exploratory; replication windows unused in initial search, registered before delay scores')
    protocol['source_sha256'] = dict(source['source_sha256'], **{str(Path(__file__)): shared.sha256(__file__)})
    shared.write_json(output / 'protocol.json', protocol)
    shutil.copyfile(__file__, output / 'source.py')
    print(pd.DataFrame(cases)[['identifier', 'split', 'target_date', 'target_cells']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run'])
    parser.add_argument('--parent', type=Path, default=ROOT / 'replay_v1')
    parser.add_argument('--prepared', type=Path, default=ROOT / 'prepared_v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'delays_v1')
    parser.add_argument('--datasets', nargs='+', choices=['potomac', 'susquehanna'], default=['potomac', 'susquehanna'])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--cases', nargs='+')
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    if arguments.action == 'register':
        register(arguments.parent, arguments.prepared, arguments.output)
    else:
        panels.predict = predict
        panels.run(arguments.output, arguments.datasets, 'all', arguments.methods, arguments.cases)
