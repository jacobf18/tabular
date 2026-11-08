"""Export audited Pavia confirmation artifacts without copying private checkpoints or raw data."""

import argparse
import json
from pathlib import Path
import shutil

import plot_measured_physics as plotting


def export(report, paper):
    audits = json.loads((report / 'audit.json').read_text())
    if not audits or not all(record['complete'] and record['predictions_audited'] for record in audits):
        raise ValueError('Do not export incomplete or unaudited physics results')
    output = paper / 'results/measured_spectroscopy'
    output.mkdir(parents=True, exist_ok=True)
    for filename in ('scores.csv', 'by_rate.csv', 'overall.csv', 'comparisons.csv', 'audit.json'):
        shutil.copyfile(report / filename, output / filename)
    protocols = output / 'protocols'
    protocols.mkdir(exist_ok=True)
    for audit in audits:
        source = Path(audit['path'])
        shutil.copyfile(source / 'protocol.json', protocols / f'{source.name}.json')
    plotting.run(report, paper / 'figures', 'pavia', hide_specialist=True)
    plotting.run(report, output, 'pavia')
    print(f'Exported audited measured spectroscopy results to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--paper', type=Path, default=Path('/root/MatrixFormerPaper'))
    args = parser.parse_args()
    export(args.report, args.paper)
