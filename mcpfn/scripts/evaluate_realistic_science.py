"""Retrospective sensor-gap reconstruction using independently recorded outage masks."""

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import time
import warnings
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
import torch

import evaluate_cylinder_completion as shared
from evaluate_physics_impute import forward_native
from refine_measured_softimpute import complete as soft_complete


ROOT = Path('/mnt/matrixformer_scratch/realistic_science_20261106')
METHODS = ('Temporal interpolation', 'KNN', 'SoftImpute', 'Random forest',
           'Temporal regression', 'Spatial residual interpolation',
           'MatrixFormer', 'MatrixFormer-Impute')
WINDOW = 336
PLANS = {
    'air_quality': {
        'development': [(2016, 336, 1008), (2688, 672, 1344)],
        'confirmation': [(target, donor, 3696) for target, donor in
                         zip((4704, 5376, 6048, 6720, 7392, 8064), (1008, 672, 1344, 1680, 2016, 4368))],
    },
    'beijing': {
        'development': [(8760, 1344, 7728), (9432, 2352, 8064)],
        'confirmation': [(target, donor, 7728) for target, donor in
                         zip((17520, 18864, 20208, 21552, 22896, 24240), (2016, 2352, 2688, 3024, 3360, 3696))],
    },
}


def load_data(name, data_root):
    if name == 'air_quality':
        with ZipFile(data_root / 'air_quality.zip') as archive:
            frame = pd.read_csv(archive.open('AirQualityUCI.csv'), sep=';', decimal=',')
        frame = frame.dropna(subset=['Date'])
        dates = pd.to_datetime(frame.Date + ' ' + frame.Time, format='%d/%m/%Y %H.%M.%S')
        frame = frame.drop(columns=['Date', 'Time', 'NMHC(GT)']).dropna(axis=1, how='all')
        frame.index = dates
        frame = frame.replace(-200, np.nan)
        targets = [frame.columns.get_loc(name) for name in ('CO(GT)', 'NOx(GT)', 'NO2(GT)')]
        groups = [[column] for column in range(frame.shape[1])]
    else:
        frames = []
        with ZipFile(data_root / 'beijing.zip') as outer:
            with ZipFile(io.BytesIO(outer.read('PRSA2017_Data_20130301-20170228.zip'))) as archive:
                for filename in sorted(archive.namelist()):
                    if not filename.endswith('.csv'):
                        continue
                    source = pd.read_csv(archive.open(filename))
                    source.index = pd.to_datetime(source[['year', 'month', 'day', 'hour']])
                    frames.append(source[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].add_prefix(source.station.iloc[0] + ':'))
        frame = pd.concat(frames, axis=1)
        groups = [[column for column, label in enumerate(frame.columns) if label.endswith(':' + pollutant)]
                  for pollutant in ('PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3')]
        targets = groups[0]
    assert not frame.index.duplicated().any()
    frame = frame.sort_index().reindex(pd.date_range(frame.index.min(), frame.index.max(), freq='h'))
    assert np.isfinite(frame.to_numpy()[frame.notna().to_numpy()]).all()
    return frame, targets, groups


def temporal_fill(matrix):
    result = np.array(matrix, dtype=float, copy=True)
    positions = np.arange(len(matrix))
    for column in range(matrix.shape[1]):
        observed = np.isfinite(matrix[:, column])
        result[:, column] = np.interp(positions, positions[observed], matrix[observed, column]) if observed.any() else 0.
    return result


def transform(matrix, name):
    values = matrix.copy()
    shape_scale = np.ones((1, matrix.shape[1]))
    if name == 'asinh_column':
        shape_scale = np.maximum(np.nanmedian(np.abs(matrix), axis=0, keepdims=True), 1e-6)
        shape_scale = np.where(np.isfinite(shape_scale), shape_scale, 1.)
        values = np.arcsinh(values / shape_scale)
    axis = None if name == 'global' else 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        center = np.nanmean(values, axis=axis, keepdims=True)
        scale = np.nanstd(values, axis=axis, keepdims=True)
    center = np.where(np.isfinite(center), center, 0.)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.)

    def inverse(prediction):
        restored = prediction * scale + center
        return np.sinh(restored) * shape_scale if name == 'asinh_column' else restored

    return (values - center) / scale, inverse


def settings(method):
    transforms = ('column', 'asinh_column')
    if method.startswith('MatrixFormer'):
        return [dict(transform=name) for name in ('global',) + transforms]
    if method == 'SoftImpute':
        return [dict(transform=name, shrinkage=value) for name in transforms for value in (.005, .02, .1)]
    if method == 'KNN':
        return [dict(transform=name, neighbors=count) for name in transforms for count in (3, 10, 30)]
    if method == 'Random forest':
        return [dict(transform=name, leaf=leaf) for name in transforms for leaf in (1, 3)]
    if method == 'Temporal regression':
        return [dict(transform=name, alpha=alpha) for name in transforms for alpha in (.1, 10.)]
    return [dict(transform=name) for name in transforms]


def spatial_residual(matrix, groups):
    result = temporal_fill(matrix)
    positions = np.arange(len(matrix))
    for group in groups:
        for column in group:
            others = [other for other in group if other != column]
            if not others:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                reference = np.nanmedian(matrix[:, others], axis=1)
            reference = temporal_fill(reference[:, None])[:, 0]
            observed = np.isfinite(matrix[:, column])
            if observed.any():
                residual = matrix[observed, column] - reference[observed]
                result[:, column] = reference + np.interp(positions, positions[observed], residual)
    return result


def predict(matrix, method, setting, targets, groups, models, distribution):
    values, inverse = transform(matrix, setting['transform'])
    details = {}
    if method.startswith('MatrixFormer'):
        prediction, features = forward_native(values, models[method], distribution)
        del features
    elif method == 'Temporal interpolation':
        prediction = temporal_fill(values)
    elif method == 'Spatial residual interpolation':
        prediction = spatial_residual(values, groups)
    elif method == 'SoftImpute':
        prediction, details = soft_complete(values, setting['shrinkage'])
    elif method == 'KNN':
        prediction = KNNImputer(n_neighbors=setting['neighbors'], weights='distance', keep_empty_features=True).fit_transform(values)
    else:
        prediction = temporal_fill(values)
        positions = np.arange(len(values))
        for column in targets:
            observed = np.isfinite(values[:, column])
            if observed.sum() < 3:
                continue
            predictors = np.delete(values, column, axis=1)
            if method == 'Random forest':
                imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
                features = imputer.fit_transform(predictors)
                estimator = RandomForestRegressor(n_estimators=100, min_samples_leaf=setting['leaf'],
                                                  max_features=1., n_jobs=2, random_state=961)
            else:
                features = temporal_fill(predictors)
                estimator = Ridge(alpha=setting['alpha'])
            estimator.fit(features[observed], values[observed, column])
            fitted = estimator.predict(features)
            if method == 'Temporal regression':
                fitted += np.interp(positions, positions[observed], values[observed, column] - fitted[observed])
            prediction[:, column] = fitted
    restored = inverse(prediction)
    observed = np.isfinite(matrix)
    restored[observed] = matrix[observed]
    if not np.isfinite(restored).all():
        raise ValueError('Nonfinite prediction')
    return restored, details


def metrics(truth, prediction, missing, context, targets):
    records = []
    all_errors = []
    for column in targets:
        selected = missing[:, column]
        if not selected.any():
            continue
        deviation = float(np.nanstd(context[:, column]))
        if not np.isfinite(deviation) or deviation < 1e-8:
            raise ValueError('Target column has no observed variation')
        error = prediction[selected, column] - truth[selected, column]
        all_errors.append(error)
        records.append(dict(column=int(column), count=int(selected.sum()), observed_sd=deviation,
                            rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(np.abs(error)))))
    if not records:
        raise ValueError('No scored targets')
    pooled = np.concatenate(all_errors)
    return dict(nrmse=float(np.mean([record['rmse'] / record['observed_sd'] for record in records])),
                pooled_rmse=float(np.sqrt(np.mean(pooled ** 2))), pooled_mae=float(np.mean(np.abs(pooled))),
                mean_rmse=float(np.mean([record['rmse'] for record in records])),
                mean_mae=float(np.mean([record['mae'] for record in records])),
                target_columns=len(records), target_cells=sum(record['count'] for record in records),
                per_column=records)


def make_case(frame, targets, starts):
    target_start, donor_start, validation_start = starts
    assert all(abs(first - second) >= WINDOW for first, second in
               ((target_start, donor_start), (target_start, validation_start), (donor_start, validation_start)))
    block = frame.iloc[target_start:target_start + WINDOW]
    truth = block.to_numpy(dtype=float)
    donor = frame.iloc[donor_start:donor_start + WINDOW].isna().to_numpy()
    validation_donor = frame.iloc[validation_start:validation_start + WINDOW].isna().to_numpy()
    valid = np.isfinite(truth)
    missing = donor & valid
    validation = validation_donor & valid & ~missing
    hour = block.index.hour.to_numpy()
    covariates = np.column_stack((np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), np.arange(WINDOW) / WINDOW))
    truth = np.column_stack((truth, covariates))
    missing = np.column_stack((missing, np.zeros_like(covariates, dtype=bool)))
    validation = np.column_stack((validation, np.zeros_like(covariates, dtype=bool)))
    context = np.where(missing, np.nan, truth)
    reduced = np.where(validation, np.nan, context)
    assert not (missing & validation).any()
    for selected, current in [(missing, context), (validation, reduced)]:
        metrics(truth, np.nan_to_num(truth), selected, current, targets)
    details = dict(target_start=target_start, donor_start=donor_start, validation_start=validation_start,
                   target_date=str(block.index[0]), donor_date=str(frame.index[donor_start]),
                   validation_donor_date=str(frame.index[validation_start]),
                   donor_missing_fraction=float(donor.mean()), original_missing_fraction=float((~valid).mean()),
                   artificial_hidden_fraction=float(missing[:, :frame.shape[1]].mean()),
                   target_cells=int(missing[:, targets].sum()), validation_target_cells=int(validation[:, targets].sum()))
    return dict(truth=truth, missing=missing, validation=validation), details


def register(args):
    if args.output.exists():
        raise ValueError('Register in a fresh directory')
    args.output.mkdir(parents=True)
    cases = []
    datasets = {}
    for dataset in args.datasets:
        frame, targets, groups = load_data(dataset, args.data)
        datasets[dataset] = dict(columns=list(frame.columns) + ['hour_sin', 'hour_cos', 'relative_time'],
                                 targets=targets, groups=groups, original_missing_fraction=float(frame.isna().to_numpy().mean()))
        for split, plans in PLANS[dataset].items():
            for case_index, starts in enumerate(plans):
                arrays, details = make_case(frame, targets, starts)
                identifier = f'{dataset}_{split}_{case_index:02}'
                destination = args.output / f'{identifier}.npz'
                np.savez_compressed(destination, **arrays)
                cases.append(dict(identifier=identifier, dataset=dataset, split=split, **details,
                                  input_sha256=shared.sha256(destination)))
    dependencies = [Path(__file__), Path(__file__).with_name('evaluate_physics_impute.py'),
                    Path(shared.__file__), Path(__file__).with_name('refine_measured_softimpute.py')]
    protocol = dict(created=datetime.now(timezone.utc).isoformat(), datasets=datasets, cases=cases,
                    methods=list(METHODS), grids={method: settings(method) for method in METHODS},
                    source_sha256={str(path): shared.sha256(path) for path in dependencies},
                    data_sha256={str(args.data / f'{dataset}.zip'): shared.sha256(args.data / f'{dataset}.zip') for dataset in args.datasets},
                    checkpoints={name: dict(path=str(path), sha256=shared.sha256(path),
                                           config_sha256=shared.sha256(path.with_name('config.json')))
                                 for name, path in [('MatrixFormer', shared.SINGLE), ('MatrixFormer-Impute', shared.GENERAL)]},
                    task='Retrospective reconstruction, not forecasting; future observed values are available to every method',
                    masking='Replay masks from distinct recorded time windows; retain original NaNs; score only originally measured targets',
                    limitation='Mask replay is controlled artificial withholding, not ground truth for originally missing cells or a causal model of failures',
                    tuning='Second recorded mask on observed entries only; transforms refit after validation masking',
                    primary_metric='Mean over target columns of held-out RMSE / that column observed population SD',
                    secondary_metric='PM2.5 RMSE and MAE in micrograms/m^3 for Beijing; per-analyte physical errors for Italy',
                    architecture='Frozen native orientation and full 336-row context; no ensemble, training or low-rank postprocessing')
    shared.write_json(args.output / 'protocol.json', protocol)
    shutil.copyfile(__file__, args.output / 'source.py')
    print(pd.DataFrame(cases)[['identifier', 'target_date', 'donor_missing_fraction', 'target_cells', 'validation_target_cells']].to_string(index=False))


def run(args):
    protocol = json.loads((args.output / 'protocol.json').read_text())
    for section in ('source_sha256', 'data_sha256'):
        for path, expected in protocol[section].items():
            assert shared.sha256(path) == expected, path
    methods = args.methods or protocol['methods']
    checkpoints = {name: Path(record['path']) for name, record in protocol['checkpoints'].items() if name in methods}
    for name, path in checkpoints.items():
        assert shared.sha256(path) == protocol['checkpoints'][name]['sha256']
        assert shared.sha256(path.with_name('config.json')) == protocol['checkpoints'][name]['config_sha256']
    models, distribution = shared.load_models(checkpoints) if checkpoints else ({}, None)
    for case in protocol['cases']:
        if case['split'] != args.split or case['dataset'] not in args.datasets:
            continue
        path = args.output / (case['identifier'] + '.npz')
        assert shared.sha256(path) == case['input_sha256']
        arrays = np.load(path)
        truth, missing, validation = [arrays[name] for name in ('truth', 'missing', 'validation')]
        context = np.where(missing, np.nan, truth)
        reduced = np.where(validation, np.nan, context)
        metadata = protocol['datasets'][case['dataset']]
        for method in methods:
            if method == 'Spatial residual interpolation' and case['dataset'] != 'beijing':
                continue
            destination = args.output / 'records' / f'{case["identifier"]}_{method.replace(" ", "_")}.json'
            if destination.exists():
                continue
            print('Start', case['identifier'], method, flush=True)
            started = time.perf_counter()
            trials = []
            for setting in protocol['grids'][method]:
                prediction, details = predict(reduced, method, setting, metadata['targets'], metadata['groups'], models, distribution)
                scores = metrics(truth, prediction, validation, reduced, metadata['targets'])
                trials.append(dict(setting=setting, validation_nrmse=scores['nrmse'], solver=details))
            selected = min(trials, key=lambda trial: trial['validation_nrmse'])
            prediction, details = predict(context, method, selected['setting'], metadata['targets'], metadata['groups'], models, distribution)
            scores = metrics(truth, prediction, missing, context, metadata['targets'])
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination.with_suffix('.npz'), prediction=prediction.astype(np.float32))
            shared.write_json(destination, dict(identifier=case['identifier'], dataset=case['dataset'], split=case['split'],
                                               method=method, scores=scores, tuning=dict(selected=selected, trials=trials, final_solver=details),
                                               seconds=time.perf_counter() - started))
            print('Done', case['identifier'], method, f'{scores["nrmse"]:.5f}', selected['setting'],
                  f'{time.perf_counter()-started:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--datasets', nargs='+', choices=list(PLANS), default=list(PLANS))
    parser.add_argument('--split', choices=['development', 'confirmation'], default='development')
    parser.add_argument('--methods', nargs='+', choices=METHODS)
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    register(arguments) if arguments.action == 'register' else run(arguments)
