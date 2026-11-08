"""Measured GLODAP alkalinity reconstruction with empirical assay masks and CANYON-B."""

import argparse
from datetime import datetime, timezone
from itertools import permutations
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
import torch

import evaluate_cylinder_completion as shared
import evaluate_realistic_science as original
import evaluate_science_panels as panels
from fetch_science_panels import ROOT


DATASET = 'ocean_alkalinity'
COLUMNS = ['latitude', 'longitude', 'pressure', 'temperature', 'salinity', 'oxygen',
           'nitrate', 'silicate', 'phosphate', 'talk', 'station', 'cast', 'year']
TARGET = COLUMNS.index('talk')
WINDOW = 512
METHODS = ('Column mean', 'Profile interpolation', 'KNN', 'SoftImpute', 'Random forest',
           'Ridge calibration', 'PLS calibration', 'RBF-SVR', 'Iterative random forest',
           'CANYON-B', 'CANYON-B + local calibration', 'MatrixFormer', 'MatrixFormer-Impute')
SCREEN = tuple(method for method in METHODS if method != 'Iterative random forest')
BASE_PREDICT = panels.predict
CANYON_ROOT = ROOT / 'data/canyon_b'


def clean_data(source):
    frame = pd.DataFrame({name: source['G2' + name] for name in COLUMNS + ['cruise', 'month', 'day']})
    frame = frame.replace(-9999., np.nan).replace(-999., np.nan)
    for name in ('salinity', 'oxygen', 'nitrate', 'silicate', 'phosphate', 'talk'):
        frame.loc[source['G2' + name + 'f'] != 2, name] = np.nan
    return frame


def settings(method):
    if method == 'Profile interpolation':
        return [dict(transform='column')]
    if method == 'CANYON-B':
        return [dict(neighbors=10, calibration='none')]
    if method == 'CANYON-B + local calibration':
        return [dict(neighbors=count, calibration=calibration)
                for count in (3, 10) for calibration in ('none', 'offset', 'affine')]
    return panels.settings(method)


def profile_interpolate(matrix):
    prediction = original.temporal_fill(matrix)
    for station, cast in np.unique(matrix[:, [10, 11]], axis=0):
        selected = (matrix[:, 10] == station) & (matrix[:, 11] == cast)
        observed = selected & np.isfinite(matrix[:, TARGET])
        if observed.any():
            order = np.argsort(matrix[observed, 2])
            prediction[selected, TARGET] = np.interp(matrix[selected, 2], matrix[observed, 2][order],
                                                     matrix[observed, TARGET][order])
        else:
            prediction[selected, TARGET] = np.nanmedian(matrix[:, TARGET])
    prediction[np.isfinite(matrix)] = matrix[np.isfinite(matrix)]
    return prediction


def canyon_predict(matrix, setting):
    normalized, inverse = original.transform(matrix, 'column')
    completed = KNNImputer(n_neighbors=setting['neighbors'], weights='distance', keep_empty_features=True).fit_transform(normalized)
    completed = inverse(completed)
    inputs = pd.DataFrame(completed, columns=COLUMNS)
    inputs['date'] = [(pd.Timestamp(year=int(year), month=1, day=1) +
                       pd.Timedelta(days=(year - int(year)) * 365)).strftime('%Y-%m-%d %H:%M') for year in inputs.year]
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        directory = Path(temporary)
        inputs.to_csv(directory / 'input.csv', index=False)
        subprocess.run(['Rscript', str(Path(__file__).with_name('canyon_b_bridge.R')),
                        str(CANYON_ROOT), str(directory / 'input.csv'), str(directory / 'output.csv')],
                       check=True, capture_output=True, text=True, timeout=180)
        estimates = pd.read_csv(directory / 'output.csv').prediction.to_numpy()
    observed = np.isfinite(matrix[:, TARGET])
    if setting['calibration'] == 'offset':
        estimates += np.mean(matrix[observed, TARGET] - estimates[observed])
    elif setting['calibration'] == 'affine':
        estimator = LinearRegression().fit(estimates[observed, None], matrix[observed, TARGET])
        estimates = estimator.predict(estimates[:, None])
    completed[:, TARGET] = estimates
    completed[np.isfinite(matrix)] = matrix[np.isfinite(matrix)]
    if not np.isfinite(completed).all():
        raise ValueError('CANYON-B returned nonfinite estimates')
    return completed


def predict(matrix, method, setting, targets, groups, models, distribution):
    if method == 'Profile interpolation':
        return profile_interpolate(matrix), {}
    if method.startswith('CANYON-B'):
        return canyon_predict(matrix, setting), {}
    return BASE_PREDICT(matrix, method, setting, targets, groups, models, distribution)


def register(data, output):
    output.mkdir(parents=True, exist_ok=False)
    archive = data / 'GLODAPv2.2023_Atlantic_Ocean.mat'
    source = loadmat(archive, squeeze_me=True)
    frame = clean_data(source)
    blocks, rejected = {}, []
    for cruise, current in frame.groupby('cruise', sort=True):
        year = int(current.year.iloc[0])
        if year < 2000:
            continue
        current = current.sort_values(['station', 'cast', 'pressure'], kind='stable').iloc[:WINDOW]
        if len(current) < WINDOW:
            if year >= 2016:
                rejected.append(dict(cruise=int(cruise), year=year, reason='Fewer than 512 bottle records'))
            continue
        if not np.isfinite(current[['latitude', 'longitude', 'pressure', 'station', 'cast', 'year']]).all().all():
            rejected.append(dict(cruise=int(cruise), year=year, reason='Missing positional metadata'))
            continue
        blocks[int(cruise)] = current
    donors = [cruise for cruise, current in blocks.items() if int(current.year.iloc[0]) <= 2013
              and .05 <= current.talk.isna().mean() <= .4][:8]
    cases = []
    for cruise, current in blocks.items():
        year = int(current.year.iloc[0])
        if year < 2016:
            continue
        truth = current[COLUMNS].to_numpy(dtype=float)
        dates = pd.to_datetime(current[['year', 'month', 'day']], errors='coerce')
        if dates.isna().any():
            rejected.append(dict(cruise=cruise, year=year, reason='Invalid sampling dates'))
            continue
        year_starts = pd.to_datetime(current.year.astype(int).astype(str) + '-01-01')
        truth[:, -1] = current.year.to_numpy() + (dates - year_starts).dt.total_seconds().to_numpy() / (365 * 86400)
        if np.isfinite(truth[:, TARGET]).sum() < 200:
            rejected.append(dict(cruise=cruise, year=year, reason='Fewer than 200 measured QC2 alkalinity values in first 512 bottles'))
            continue
        for donor, validation_donor in permutations(donors, 2):
            donor_mask = blocks[donor][COLUMNS].isna().to_numpy()
            validation_mask = blocks[validation_donor][COLUMNS].isna().to_numpy()
            missing = donor_mask & np.isfinite(truth)
            validation = validation_mask & np.isfinite(truth) & ~missing
            context = np.where(missing, np.nan, truth)
            reduced = np.where(validation, np.nan, context)
            if (missing[:, TARGET].sum() >= 20 and validation[:, TARGET].sum() >= 20 and
                    np.isfinite(reduced[:, TARGET]).sum() >= 128 and np.nanstd(reduced[:, TARGET]) > 1e-6):
                break
        else:
            rejected.append(dict(cruise=cruise, year=year, reason='No donor pair meets registered coverage'))
            continue
        split = 'development' if year == 2016 else 'confirmation'
        identifier = f'{DATASET}_{split}_{cruise:04}'
        destination = output / f'{identifier}.npz'
        np.savez_compressed(destination, truth=truth, missing=missing, validation=validation)
        expocode = str(source['G2expocode'][current.index[0]])
        cases.append(dict(identifier=identifier, dataset=DATASET, split=split, cruise=cruise, expocode=expocode,
                          year=year, target_start=len(cases) * WINDOW, targets=[TARGET], input_sha256=shared.sha256(destination),
                          donor_cruise=donor, validation_cruise=validation_donor,
                          target_cells=int(missing[:, TARGET].sum()), validation_target_cells=int(validation[:, TARGET].sum()),
                          natural_missing_fraction=float(np.isnan(truth[:, TARGET]).mean()),
                          hidden_target_fraction=float(missing[:, TARGET].mean()), stations=int(current.station.nunique())))
    dependencies = [Path(__file__), Path(panels.__file__), Path(original.__file__), Path(shared.__file__),
                    Path(__file__).with_name('evaluate_physics_impute.py'), Path(__file__).with_name('control_realistic_linear.py'),
                    Path(__file__).with_name('control_realistic_calibration.py'), Path(__file__).with_name('refine_measured_softimpute.py'),
                    Path(__file__).with_name('canyon_b_bridge.R'), CANYON_ROOT / 'CANYONB.R', CANYON_ROOT / 'wgts_AT.txt']
    protocol = dict(created=datetime.now(timezone.utc).isoformat(), methods=list(METHODS), screen_methods=list(SCREEN),
                    grids={method: settings(method) for method in METHODS}, cases=cases, exclusions=rejected,
                    datasets={DATASET: dict(columns=COLUMNS, groups=[[column] for column in range(len(COLUMNS))],
                                           targets=[TARGET], window=WINDOW, units='Alkalinity: micromol/kg')},
                    source_sha256={str(path): shared.sha256(path) for path in dependencies},
                    data_sha256={str(archive): shared.sha256(archive)},
                    checkpoints={name: dict(path=str(path), sha256=shared.sha256(path), config_sha256=shared.sha256(path.with_name('config.json')))
                                 for name, path in [('MatrixFormer', shared.SINGLE), ('MatrixFormer-Impute', shared.GENERAL)]},
                    source_url='https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0283442/GLODAPv2.2023_Atlantic_Ocean.mat',
                    source_citation='Lauvset et al., GLODAPv2.2023, Earth System Science Data 16, 2047-2072 (2024), doi:10.5194/essd-16-2047-2024',
                    canonical_baseline='Official CANYON-B R, commit a5b1efce24fcffc8b9e6dd3a0d1e54fa384a01d5; Bittig et al. (2018), doi:10.3389/fmars.2018.00328; verified published AT test value',
                    quality='Only WOCE flag 2 for salinity, O2, nutrients and alkalinity; exclude flag 0 interpolated/calculated values. No pH, DIC, fCO2, AOU or density inputs.',
                    primary_metric='Mean cruise-normalized held-out alkalinity RMSE / observed population SD; native micromol/kg RMSE also reported',
                    selection='First 512 bottles by station/cast/pressure; all qualifying 2016 cruises development and >=2017 cruises confirmation; mask donors <=2013, first qualifying pair by ID; no predictions before registration',
                    masking='Replay recorded multivariate assay-availability masks across cruises; preserve original NaNs; score originally measured QC2 alkalinity only',
                    fairness='CANYON-B uses published global prior calibration; local baselines and MatrixFormer use same local observed context; extra CANYON-B variant calibrates offset/affine on local observed targets only',
                    limitations='Exploratory application screening; controlled mask replay, not recovery ground truth for naturally missing carbon measurements; actual sampling date supplied to CANYON-B and decimal year supplied to all methods')
    shared.write_json(output / 'protocol.json', protocol)
    shutil.copyfile(__file__, output / 'source.py')
    print(pd.DataFrame(cases)[['identifier', 'expocode', 'year', 'target_cells', 'validation_target_cells', 'hidden_target_fraction']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run'])
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'ocean_v2')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(SCREEN))
    parser.add_argument('--cases', nargs='+')
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    if arguments.action == 'register':
        register(arguments.data, arguments.output)
    else:
        panels.predict = predict
        panels.run(arguments.output, [DATASET], 'all', arguments.methods, arguments.cases)
