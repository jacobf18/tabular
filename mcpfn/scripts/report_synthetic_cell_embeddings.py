"""Export audited slice-level scores and predetermined cell-embedding examples."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import sklearn

from recsys_evaluation_common import atomic_npz, sha256, write_json
import synthetic_cell_embeddings as analysis


def export(source, destination):
    audit = json.loads((source / 'audit.json').read_text())
    if audit['cases'] != 120 or audit['clustering_fits'] != 30720 or not audit['rescore_matches']:
        raise ValueError('Require the complete audited 120-case campaign')
    protocol = json.loads((source / 'protocol.json').read_text())
    destination.mkdir(parents=True, exist_ok=True)
    names = ('protocol.json', 'manifest.json', 'audit.json', 'per_slice.csv', 'per_matrix.csv', 'summary.csv', 'paired.csv')
    for name in names:
        if name in audit['saved_artifacts_sha256'] and sha256(source / name) != audit['saved_artifacts_sha256'][name]:
            raise ValueError('Audited score artifact changed')
        shutil.copyfile(source / name, destination / name)
    example = protocol['examples']
    directory = source / 'cases' / example['case_id']
    with np.load(Path(protocol['parent']) / 'cases' / example['case_id'] / 'inputs.npz') as saved:
        inputs = dict(saved)
    arrays = {name: inputs[name] for name in ('row_labels', 'column_labels', 'observed')}
    with np.load(directory / 'grids.npz') as grids, np.load(directory / 'assignments.npz') as assignments:
        for axis in analysis.AXES:
            indices = example[axis]
            opposite_labels = inputs['column_labels' if axis == 'row' else 'row_labels']
            np.testing.assert_array_equal(indices, analysis.predetermined_indices(opposite_labels))
            arrays[f'{axis}_contexts'] = np.array(indices)
            for layer in analysis.LAYERS:
                arrays[f'{axis}_{layer}_features'] = np.stack([
                    analysis.slice_features(grids[f'layer_{layer}'], axis, context_index) for context_index in indices])
                arrays[f'{axis}_{layer}_assignments'] = assignments[f'{axis}_{layer}'][indices]
    atomic_npz(destination / 'fixed_cell_examples.npz', **arrays)
    paired = pd.read_csv(destination / 'paired.csv')
    records = []
    for keys, group in paired.groupby(['noise', 'observed_fraction', 'axis', 'scope']):
        noise, fraction, axis, scope = keys
        values = group.difference.dropna().to_numpy()
        half_width = student_t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
        records.append(dict(noise=noise, observed_fraction=fraction, axis=axis, scope=scope,
                            matrices=len(values), mean=values.mean(), ci95_half_width=half_width,
                            ci95_low=values.mean() - half_width, ci95_high=values.mean() + half_width))
    pd.DataFrame(records).to_csv(destination / 'paired_summary.csv', index=False)
    artifacts = (*names, 'fixed_cell_examples.npz', 'paired_summary.csv')
    write_json(destination / 'export_manifest.json', dict(source=str(source), sklearn=sklearn.__version__,
               example_case_id=example['case_id'], no_pooled_features=True, no_score_based_example_selection=True,
               example_grid_sha256=sha256(directory / 'grids.npz'),
               export_script_sha256=sha256(Path(__file__)),
               artifacts_sha256={name: sha256(destination / name) for name in artifacts}))
    print(f'Exported {len(artifacts)} audited artifacts to {destination}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.destination)
