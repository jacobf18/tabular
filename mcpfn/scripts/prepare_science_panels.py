"""Prepare measured wastewater, indoor-sensor and river-quality panels without filling gaps."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

import evaluate_cylinder_completion as shared
from fetch_science_panels import ROOT


WATER_COLUMNS = ('Q-E ZN-E PH-E DBO-E DQO-E SS-E SSV-E SED-E COND-E PH-P DBO-P SS-P '
                 'SSV-P SED-P COND-P PH-D DBO-D DQO-D SS-D SSV-D SED-D COND-D PH-S '
                 'DBO-S DQO-S SS-S SSV-S SED-S COND-S').split()


def water(data):
    with ZipFile(data / 'water_treatment.zip') as archive:
        source = pd.read_csv(archive.open('water-treatment.data'), header=None, na_values='?')
    dates = pd.to_datetime(source[0].str.removeprefix('D-'), format='%d/%m/%y')
    frame = source.iloc[:, 1:30].copy()
    frame.columns = WATER_COLUMNS
    frame.index = pd.DatetimeIndex(dates)
    frame = frame.sort_index()
    targets = [frame.columns.get_loc(name) for name in ('DBO-E', 'DQO-E', 'SS-E', 'DBO-S', 'DQO-S', 'SS-S')]
    return frame, dict(targets=targets, groups=[[column] for column in range(len(frame.columns))], window=64,
                       source='UCI Water Treatment Plant, dataset 106; source names file retained in archive',
                       units='Original archive does not give reliable per-column units; normalized errors only',
                       task='Impute missing raw BOD, COD and suspended-solids measurements at plant input/output',
                       preprocessing='Exclude all nine derived treatment-efficiency columns; retain irregular sampling dates; no value imputation',
                       raw_files=['water_treatment.zip'])


def intel(data):
    source = pd.read_csv(data / 'intel_data.txt.gz', sep=r'\s+', header=None,
                         names=['date', 'time', 'epoch', 'mote', 'temperature', 'humidity', 'light', 'voltage'])
    selected = source.mote.between(1, 54) & source.date.between('2004-03-01', '2004-03-20')
    source = source.loc[selected].copy()
    source['timestamp'] = pd.to_datetime(source.date + ' ' + source.time, format='mixed').dt.floor('5min')
    invalid = ~source.temperature.between(-20., 60.)
    source.loc[invalid, 'temperature'] = np.nan
    frame = source.pivot_table(index='timestamp', columns='mote', values='temperature', aggfunc='median')
    frame = frame.reindex(index=pd.date_range('2004-03-01', '2004-03-20 23:55', freq='5min'), columns=range(1, 55))
    frame.columns = [f'temperature_mote_{column:02}' for column in range(1, 55)]
    targets = np.flatnonzero(frame.iloc[:3 * 288].notna().mean().to_numpy() >= .5).tolist()
    return frame, dict(targets=targets, groups=[list(range(54))], window=288,
                       source='Intel Berkeley Research Lab, https://db.csail.mit.edu/labdata/labdata.html',
                       units='degrees Celsius', task='Reconstruct five-minute median indoor temperatures at 54 deployed motes',
                       preprocessing='March 1-20; discard invalid mote IDs; temperatures outside [-20,60] C set missing before binning; no humidity (temperature-corrected derived channel); target motes need 50% coverage in first three donor days',
                       rejected_temperature_readings=int(invalid.sum()), raw_files=['intel_data.txt.gz', 'intel_description.html'])


def approved_values(values, sentinel):
    records = [record for record in values if 'A' in record['qualifiers'] and 'e' not in record['qualifiers']
               and float(record['value']) != sentinel]
    if not records:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz='UTC'))
    return pd.Series([float(record['value']) for record in records],
                     index=pd.to_datetime([record['dateTime'] for record in records], utc=True))


def river(data, site):
    filename = f'usgs_{site}_2019.json'
    content = json.loads((data / filename).read_text())
    channels = {}
    provenance = {}
    for item in content['value']['timeSeries']:
        code = item['variable']['variableCode'][0]['value']
        retained = []
        methods = []
        for values in item['values']:
            description = values['method'][0]['methodDescription']
            if 'Backup' in description or 'riverbed' in description:
                continue
            series = approved_values(values['value'], item['variable']['noDataValue'])
            if not series.empty:
                retained.append(series)
                methods.append(values['method'][0])
        if retained:
            combined = pd.concat(retained).sort_index()
            if combined.index.duplicated().any():
                raise ValueError(f'Duplicate selected measurement timestamps for {site}/{code}')
            channels[code] = combined.resample('h').mean()
            provenance[code] = dict(name=item['variable']['variableName'], unit=item['variable']['unit'], methods=methods)
    frame = pd.DataFrame(channels)
    start = '2019-01-01'
    frame = frame.reindex(pd.date_range(start, '2019-12-31 23:00', freq='h', tz='UTC'))
    targets = [frame.columns.get_loc('99133')]
    return frame, dict(targets=targets, groups=[[column] for column in range(len(frame.columns))], window=336,
                       source=f'USGS NWIS approved 2019 instantaneous values, site {site}', units='63680: FNU; 99133: mg/L as N',
                       task='Reconstruct hourly mean nitrate-plus-nitrite measurements; turbidity and other water-quality channels are auxiliary',
                       preprocessing='Approved non-estimated observations only; exclude backup and separate-depth sensors; concatenate nonoverlapping sonde versions; hourly mean without interpolation',
                       channels=provenance, raw_files=[filename])


def prepare(data, output):
    output.mkdir(parents=True, exist_ok=False)
    for name, loader in [('wastewater', water), ('intel_temperature', intel),
                         ('potomac', lambda path: river(path, '01646500')),
                         ('susquehanna', lambda path: river(path, '01579550'))]:
        frame, metadata = loader(data)
        if frame.index.duplicated().any() or not frame.index.is_monotonic_increasing:
            raise ValueError('Invalid time axis')
        frame.to_csv(output / f'{name}.csv', index_label='timestamp')
        metadata.update(columns=list(frame.columns), shape=list(frame.shape), start=str(frame.index[0]), end=str(frame.index[-1]),
                        missing_fraction=float(frame.isna().to_numpy().mean()),
                        missing_by_column=frame.isna().mean().to_dict(),
                        prepared_sha256=shared.sha256(output / f'{name}.csv'),
                        source_sha256=shared.sha256(__file__),
                        raw_sha256={filename: shared.sha256(data / filename) for filename in metadata['raw_files']})
        shared.write_json(output / f'{name}.json', metadata)
        print(name, frame.shape, 'missing', metadata['missing_fraction'], 'targets',
              {frame.columns[column]: metadata['missing_by_column'][frame.columns[column]] for column in metadata['targets']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'prepared_v1')
    arguments = parser.parse_args()
    prepare(arguments.data, arguments.output)
