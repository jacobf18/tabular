"""Combine independently audited, nonoverlapping sensor-window evaluations."""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import t as student_t
import numpy as np

from report_realistic_science import compare, plot, summarize


def combine(reports, output):
    output.mkdir(parents=True, exist_ok=True)
    frames, audits, physical = [], [], []
    for report in reports:
        frame = pd.read_csv(report / 'per_case.csv')
        audit = pd.read_json(report / 'audit.json')
        analytes = pd.read_csv(report / 'per_analyte.csv')
        for values in (frame, audit, analytes):
            values['phase'] = report.name
            values['identifier'] = report.name + '__' + values.identifier
        frames.append(frame[frame.split == 'confirmation'])
        audits.append(audit[audit.split == 'confirmation'])
        physical.append(analytes[analytes.split == 'confirmation'])
    frame = pd.concat(frames, ignore_index=True)
    audit = pd.concat(audits, ignore_index=True)
    analytes = pd.concat(physical, ignore_index=True)
    for dataset, group in frame.drop_duplicates(['dataset', 'identifier']).groupby('dataset'):
        dates = sorted(pd.to_datetime(group.target_date))
        assert all((later - earlier).total_seconds() >= 336 * 3600 for earlier, later in zip(dates, dates[1:])), dataset
    frame.to_csv(output / 'per_case.csv', index=False)
    analytes.to_csv(output / 'per_analyte.csv', index=False)
    audit.to_json(output / 'audit.json', orient='records', indent=2)
    summaries = [summarize(frame, metric) for metric in ('nrmse', 'pooled_rmse', 'pooled_mae')]
    summary = pd.concat([table for table in summaries if not table.empty], ignore_index=True)
    summary.to_csv(output / 'summary.csv', index=False)
    comparisons = compare(frame, audit)
    comparisons.to_csv(output / 'comparisons.csv', index=False)
    units = analytes.groupby(['dataset', 'name', 'method']).rmse.agg(mean='mean', sd='std', windows='count').reset_index()
    units['ci95_half_width'] = student_t.ppf(.975, units.windows - 1) * units.sd / np.sqrt(units.windows)
    units.to_csv(output / 'per_analyte_summary.csv', index=False)
    plot(summary, audit, output)
    (output / 'scope.json').write_text(json.dumps(dict(sources=[str(path) for path in reports],
        status='complete' if audit.complete.all() else 'partial; comparisons withheld for incomplete datasets',
        averaging='Equal weight for each nonoverlapping scored time window; development excluded',
        uncertainty='Approximate Student-t intervals across time windows, not independent deployments or independent hours',
        scope='Retrospective local 14-day context; empirical mask replay, not ground truth for natural gaps'), indent=2) + '\n')
    print(comparisons.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    combine(arguments.reports, arguments.output)
