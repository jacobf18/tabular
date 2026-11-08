"""Geometry, validation coverage, and numerical-sensitivity diagnostics for science runs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_cylinder_completion as shared
import evaluate_realistic_science as original
from diagnose_science_preprocessing import OUTPUT


def analyze(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    geometry, precision, transforms = [], [], []
    for case in protocol['cases']:
        parent = Path(case['parent'])
        arrays = np.load(parent / f'{case["identifier"]}.npz')
        truth, missing, validation = [arrays[name] for name in ('truth', 'missing', 'validation')]
        context = np.where(missing, np.nan, truth)
        if case['dataset'] != 'ocean_alkalinity':
            for column in case['targets']:
                observed = np.isfinite(context[:, column])
                query = np.flatnonzero(missing[:, column])
                isolated = (query > 0) & (query < len(context) - 1)
                isolated &= observed[np.maximum(query - 1, 0)] & observed[np.minimum(query + 1, len(context) - 1)]
                adjacent = observed[:-1] & observed[1:]
                delta = np.diff(context[:, column])[adjacent]
                deviation = np.nanstd(context[:, column])
                geometry.append(dict(dataset=case['dataset'], split=case['split'], identifier=case['identifier'], column=column,
                                     scored_cells=len(query), isolated_cells=int(isolated.sum()),
                                     adjacent_step_sd=float(np.sqrt(np.mean(delta ** 2)) / deviation) if len(delta) else None,
                                     test_cells=int(missing[:, column].sum()), validation_cells=int(validation[:, column].sum())))
        for model in ('MatrixFormer', 'MatrixFormer-Impute'):
            base = output / 'records' / f'{case["identifier"]}_{model}_clean_column.npz'
            full = output / 'records' / f'{case["identifier"]}_{model}_clean_fp32.npz'
            if base.exists() and full.exists():
                baseline = np.load(base)['prediction']
                current = np.load(full)['prediction']
                change = original.metrics(baseline, current, missing, context, case['targets'])['nrmse']
                before = original.metrics(truth, baseline, missing, context, case['targets'])['nrmse']
                after = original.metrics(truth, current, missing, context, case['targets'])['nrmse']
                precision.append(dict(dataset=case['dataset'], split=case['split'], identifier=case['identifier'], model=model,
                                      bf16_nrmse=before, fp32_nrmse=after, prediction_change_nrmse=change))
            source = parent / 'records' / f'{case["identifier"]}_{model}.json'
            record = json.loads(source.read_text())
            transforms.append(dict(dataset=case['dataset'], split=case['split'], identifier=case['identifier'], model=model,
                                   selected_transform=record['tuning']['selected']['setting']['transform']))
    pd.DataFrame(geometry).to_csv(output / 'gap_geometry.csv', index=False)
    precision = pd.DataFrame(precision)
    precision.to_csv(output / 'precision_per_case.csv', index=False)
    precision.groupby(['dataset', 'split', 'model'])[['bf16_nrmse', 'fp32_nrmse', 'prediction_change_nrmse']].mean().to_csv(output / 'precision_summary.csv')
    pd.DataFrame(transforms).to_csv(output / 'original_transform_choices.csv', index=False)
    configuration = {name: json.loads(Path(record['path']).with_name('config.json').read_text())['model']
                     for name, record in protocol['checkpoints'].items()}
    shared.write_json(output / 'architecture_audit.json', dict(models=configuration,
                         warning='Both models have rotary row/column position encodings. Poor temporal results do NOT mean the architecture has no access to row order.',
                         arithmetic='Original point estimator applies inverse transform to posterior mean; affine inversion is exact, nonlinear asinh inversion is not raw-space expectation. Affine-only ablation tests whether avoiding this solves the failures.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    analyze(arguments.output)
