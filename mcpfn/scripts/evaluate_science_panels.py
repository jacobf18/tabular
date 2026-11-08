"""Registered empirical-mask replay on additional real scientific panels."""

import argparse
from datetime import datetime, timezone
from itertools import permutations
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import torch

import control_realistic_calibration as calibration
import control_realistic_linear as linear
import evaluate_cylinder_completion as shared
import evaluate_realistic_science as original
from fetch_science_panels import ROOT


DATASETS = ('wastewater', 'intel_temperature', 'potomac', 'susquehanna')
SCREEN = ('Column mean', 'Temporal interpolation', 'Temporal regression',
          'Spatial residual interpolation', 'KNN', 'SoftImpute', 'MatrixFormer', 'MatrixFormer-Impute')
CONTROLS = ('Random forest', 'Ridge calibration', 'PLS calibration', 'RBF-SVR', 'Iterative random forest')
METHODS = SCREEN + CONTROLS


def settings(method):
    if method == 'Column mean':
        return [dict(transform='column')]
    if method in linear.METHODS:
        return linear.settings(method)
    if method in calibration.METHODS:
        return calibration.settings(method)
    return original.settings(method)


def interpolate(matrix, positions):
    result = matrix.copy()
    for column in range(matrix.shape[1]):
        observed = np.isfinite(matrix[:, column])
        result[:, column] = np.interp(positions, positions[observed], matrix[observed, column]) if observed.any() else 0.
    return result


def predict(matrix, method, setting, targets, groups, models, distribution):
    if method in linear.METHODS:
        return linear.predict(matrix, targets, method, setting), {}
    if method in calibration.METHODS:
        return calibration.predict(matrix, targets, method, setting), {}
    if method not in ('Column mean', 'Temporal interpolation', 'Temporal regression'):
        return original.predict(matrix, method, setting, targets, groups, models, distribution)
    values, inverse = original.transform(matrix, setting['transform'])
    positions = matrix[:, -1]
    if method == 'Column mean':
        prediction = np.nan_to_num(values, nan=0.)
    else:
        prediction = interpolate(values, positions)
        if method == 'Temporal regression':
            for column in targets:
                observed = np.isfinite(values[:, column])
                features = np.delete(interpolate(values, positions), column, axis=1)
                estimator = Ridge(alpha=setting['alpha'])
                estimator.fit(features[observed], values[observed, column])
                fitted = estimator.predict(features)
                prediction[:, column] = fitted + np.interp(positions, positions[observed],
                                                          values[observed, column] - fitted[observed])
    result = inverse(prediction)
    result[np.isfinite(matrix)] = matrix[np.isfinite(matrix)]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite prediction')
    return result, {}


def make_case(frame, metadata, target_start, donor_start, validation_start):
    width = metadata['window']
    starts = (target_start, donor_start, validation_start)
    if any(abs(first - second) < width for first, second in permutations(starts, 2)):
        raise ValueError('Target and donor windows overlap')
    block = frame.iloc[target_start:target_start + width]
    truth = block.to_numpy(dtype=float)
    donor = frame.iloc[donor_start:donor_start + width].isna().to_numpy()
    validation_donor = frame.iloc[validation_start:validation_start + width].isna().to_numpy()
    if truth.shape != donor.shape or truth.shape != validation_donor.shape or len(truth) != width:
        raise ValueError('Incomplete window')
    missing = donor & np.isfinite(truth)
    validation = validation_donor & np.isfinite(truth) & ~missing
    context = np.where(missing, np.nan, truth)
    reduced = np.where(validation, np.nan, context)
    targets = [column for column in metadata['targets'] if
               np.isfinite(reduced[:, column]).sum() >= 20 and
               np.nanstd(reduced[:, column]) > 1e-8 and missing[:, column].any()]
    if not targets or missing[:, targets].sum() < 4 or validation[:, targets].sum() < 4:
        raise ValueError('Fewer than four test/validation targets or no supported target columns')
    if len(metadata['targets']) > 10 and len(targets) < 10:
        raise ValueError('Fewer than ten scorable sensor channels')
    fraction = float(missing[:, targets].mean())
    if fraction > .5:
        raise ValueError('More than half of active targets would be hidden')
    hours = block.index.hour.to_numpy() + block.index.minute.to_numpy() / 60.
    elapsed = (block.index - block.index[0]).total_seconds().to_numpy()
    covariates = np.column_stack((np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24),
                                  elapsed / max(elapsed[-1], 1.)))
    arrays = dict(truth=np.column_stack((truth, covariates)),
                  missing=np.column_stack((missing, np.zeros_like(covariates, dtype=bool))),
                  validation=np.column_stack((validation, np.zeros_like(covariates, dtype=bool))))
    details = dict(target_start=target_start, donor_start=donor_start, validation_start=validation_start,
                   target_date=str(block.index[0]), target_end=str(block.index[-1]),
                   donor_date=str(frame.index[donor_start]), validation_date=str(frame.index[validation_start]),
                   targets=targets, target_cells=int(missing[:, targets].sum()),
                   validation_target_cells=int(validation[:, targets].sum()),
                   original_missing_fraction=float(block.isna().to_numpy().mean()),
                   donor_missing_fraction=float(donor.mean()), target_hidden_fraction=fraction)
    return arrays, details


def register(prepared, output):
    output.mkdir(parents=True, exist_ok=False)
    datasets, cases, exclusions = {}, [], []
    for dataset in DATASETS:
        metadata = json.loads((prepared / f'{dataset}.json').read_text())
        assert shared.sha256(prepared / f'{dataset}.csv') == metadata['prepared_sha256']
        frame = pd.read_csv(prepared / f'{dataset}.csv', index_col=0, parse_dates=True)
        width = metadata['window']
        block_count = len(frame) // width
        donors = [0, 2] if dataset == 'wastewater' else [0, 1, 2]
        development = 1 if dataset == 'wastewater' else 3
        confirmation = list(range(3, block_count)) if dataset == 'wastewater' else np.linspace(4, block_count - 1, 8, dtype=int).tolist()
        datasets[dataset] = metadata
        for split, blocks in [('development', [development]), ('confirmation', confirmation)]:
            for block in blocks:
                attempts = []
                for donor_block, validation_block in permutations(donors, 2):
                    try:
                        arrays, details = make_case(frame, metadata, block * width, donor_block * width, validation_block * width)
                        break
                    except ValueError as error:
                        attempts.append(str(error))
                else:
                    exclusions.append(dict(dataset=dataset, split=split, block=block, reasons=attempts))
                    continue
                identifier = f'{dataset}_{split}_{block:02}'
                destination = output / f'{identifier}.npz'
                np.savez_compressed(destination, **arrays)
                cases.append(dict(identifier=identifier, dataset=dataset, split=split, **details,
                                  input_sha256=shared.sha256(destination)))
    dependencies = [Path(__file__), Path(original.__file__), Path(linear.__file__), Path(calibration.__file__), Path(shared.__file__),
                    Path(__file__).with_name('evaluate_physics_impute.py'), Path(__file__).with_name('refine_measured_softimpute.py'),
                    Path(__file__).with_name('prepare_science_panels.py')]
    protocol = dict(created=datetime.now(timezone.utc).isoformat(), datasets=datasets, cases=cases, exclusions=exclusions,
                    methods=list(METHODS), screen_methods=list(SCREEN), control_methods=list(CONTROLS),
                    grids={method: settings(method) for method in METHODS},
                    source_sha256={str(path): shared.sha256(path) for path in dependencies},
                    checkpoints={name: dict(path=str(path), sha256=shared.sha256(path), config_sha256=shared.sha256(path.with_name('config.json')))
                                 for name, path in [('MatrixFormer', shared.SINGLE), ('MatrixFormer-Impute', shared.GENERAL)]},
                    primary_metric='Equal-weight target-column RMSE / observed population SD, then equal-weight nonoverlapping target windows',
                    inference='Frozen checkpoints, standard orientation, full local window, no training, ensemble, transpose average or rank refinement',
                    masking='Replay whole multichannel masks from earlier nonoverlapping recorded windows; original NaNs never scored',
                    selection='Select first donor pair meeting fixed coverage rules before predictions; excluded windows retained; not random cell masking',
                    target_coverage='At least 20 reduced-context observations per scored target; at least 4 test and validation cells; Intel at least 10 scored motes; <=50% active targets hidden',
                    tuning='Second empirical mask on remaining observations; refit all scaling after validation masking',
                    temporal='Retrospective imputation; all methods may use future observations in same window; interpolation uses actual elapsed time',
                    limitations='Exploratory dataset search; replay does not validate naturally missing values; local-context baselines, not models fitted to years of history')
    shared.write_json(output / 'protocol.json', protocol)
    shutil.copyfile(__file__, output / 'source.py')
    print(pd.DataFrame(cases)[['identifier', 'target_date', 'target_cells', 'validation_target_cells', 'target_hidden_fraction']].to_string(index=False))
    print('Excluded:', len(exclusions), flush=True)


def run(output, datasets, split, methods, case_ids):
    protocol = json.loads((output / 'protocol.json').read_text())
    for path, expected in protocol['source_sha256'].items():
        assert shared.sha256(path) == expected, path
    methods = methods or protocol['screen_methods']
    neural = [method for method in methods if method.startswith('MatrixFormer')]
    models, distribution = {}, None
    if neural:
        checkpoints = {}
        for method in neural:
            metadata = protocol['checkpoints'][method]
            checkpoint = Path(metadata['path'])
            assert shared.sha256(checkpoint) == metadata['sha256']
            assert shared.sha256(checkpoint.with_name('config.json')) == metadata['config_sha256']
            checkpoints[method] = checkpoint
        models, distribution = shared.load_models(checkpoints)
    for case in protocol['cases']:
        if case['dataset'] not in datasets or (split != 'all' and case['split'] != split):
            continue
        if case_ids and case['identifier'] not in case_ids:
            continue
        source = output / f'{case["identifier"]}.npz'
        assert shared.sha256(source) == case['input_sha256']
        arrays = np.load(source)
        truth, missing, validation = [arrays[name] for name in ('truth', 'missing', 'validation')]
        context = np.where(missing, np.nan, truth)
        reduced = np.where(validation, np.nan, context)
        targets = case['targets']
        groups = protocol['datasets'][case['dataset']]['groups']
        for method in methods:
            destination = output / 'records' / f'{case["identifier"]}_{method.replace(" ", "_")}.json'
            if destination.exists():
                continue
            started = time.perf_counter()
            print('Start', case['identifier'], method, flush=True)
            trials = []
            for setting in protocol['grids'][method]:
                prediction, details = predict(reduced, method, setting, targets, groups, models, distribution)
                scores = original.metrics(truth, prediction, validation, reduced, targets)
                trials.append(dict(setting=setting, validation_nrmse=scores['nrmse'], solver=details))
            selected = min(trials, key=lambda record: record['validation_nrmse'])
            prediction, details = predict(context, method, selected['setting'], targets, groups, models, distribution)
            prediction = prediction.astype(np.float64)
            scores = original.metrics(truth, prediction, missing, context, targets)
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination.with_suffix('.npz'), prediction=prediction)
            shared.write_json(destination, dict(identifier=case['identifier'], dataset=case['dataset'], split=case['split'],
                                               method=method, scores=scores, seconds=time.perf_counter()-started,
                                               tuning=dict(selected=selected, trials=trials, final_solver=details),
                                               input_sha256=case['input_sha256'], prediction_sha256=shared.sha256(destination.with_suffix('.npz')),
                                               protocol_sha256=shared.sha256(output / 'protocol.json')))
            print('Done', case['identifier'], method, f'{scores["nrmse"]:.6f}', f'{time.perf_counter()-started:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run'])
    parser.add_argument('--prepared', type=Path, default=ROOT / 'prepared_v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'replay_v1')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--split', choices=['development', 'confirmation', 'all'], default='all')
    parser.add_argument('--methods', nargs='+', choices=METHODS)
    parser.add_argument('--cases', nargs='+')
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    if arguments.action == 'register':
        register(arguments.prepared, arguments.output)
    else:
        run(arguments.output, arguments.datasets, arguments.split, arguments.methods, arguments.cases)
