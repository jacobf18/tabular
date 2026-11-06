"""Retrospective geometry diagnosis on every frozen clustered representation case."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partition_scatter(features, labels):
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    centered = features - features.mean(axis=0, keepdims=True)
    total = np.sum(centered * centered)
    between = 0.
    for label in np.unique(labels):
        members = centered[labels == label]
        centroid = members.mean(axis=0)
        between += len(members) * np.dot(centroid, centroid)
    return float(between), float(total)


def geometry(features, labels, observed):
    labels = np.asarray(labels)
    observed = np.asarray(observed, dtype=bool)
    mask_between, total = partition_scatter(features, observed)
    group_between, _ = partition_scatter(features, labels)
    joint_between, _ = partition_scatter(features, labels * 2 + observed)
    if total <= 0:
        return dict(mask_eta2=np.nan, group_eta2=np.nan, joint_eta2=np.nan,
                    group_after_mask_eta2=np.nan)
    return dict(mask_eta2=mask_between / total, group_eta2=group_between / total,
                joint_eta2=joint_between / total,
                group_after_mask_eta2=(joint_between - mask_between) / total)


def summarize(frame, keys, metrics):
    records = []
    for values, group in frame.groupby(keys):
        record = dict(zip(keys, values))
        record['matrices'] = len(group)
        for metric in metrics:
            samples = group[metric].dropna().to_numpy()
            half = t.ppf(.975, len(samples) - 1) * samples.std(ddof=1) / np.sqrt(len(samples))
            record[metric] = samples.mean()
            record[f'{metric}_ci95_half'] = half
        records.append(record)
    return pd.DataFrame(records)


def run(source, pooled, output):
    if output.exists():
        raise ValueError('Use a fresh output directory')
    protocol = json.loads((source / 'protocol.json').read_text())
    audit = json.loads((source / 'audit.json').read_text())
    source_manifest = json.loads((source / 'manifest.json').read_text())
    if audit['cases'] != 120 or not audit['rescore_matches']:
        raise ValueError('The complete audited cell study is required')
    output.mkdir(parents=True)
    pooled_scores_path = pooled / 'clustering_by_matrix.csv'
    cell_scores_path = source / 'per_matrix.csv'
    assert digest(cell_scores_path) == audit['saved_artifacts_sha256']['per_matrix.csv']
    frozen = dict(kind='Retrospective diagnostic, not a preregistered confirmatory study',
                  cases=protocol['cases'], example=protocol['examples']['case_id'],
                  source=str(source), pooled=str(pooled), layers=[0, 8], axes=['row', 'column'],
                  design='Every fixed row/column, every noise/fraction/seed; no new fitting or model selection',
                  variance='Between-partition squared Euclidean scatter divided by total scatter; full raw vectors',
                  partitions=['observed/masked', 'true group', 'joint true group and observation status'],
                  group_after_mask='(joint between scatter - mask between scatter) / total scatter; not additive R-squared',
                  interpretation='Label-informed descriptive diagnostics, not labels supplied to clustering or inference',
                  intervals='95% Student-t over ten matrix means; contexts averaged within each matrix first',
                  selection='Same seed-zero, noise .25, observed .4 example; no cleaner noise level selected',
                  source_sha256={str(path): digest(path) for path in
                                 (Path(__file__), source / 'protocol.json', source / 'audit.json',
                                  source / 'manifest.json', cell_scores_path, pooled_scores_path)})
    (output / 'protocol.json').write_text(json.dumps(frozen, indent=2) + '\n')
    records = []
    for index, case in enumerate(protocol['cases']):
        case_id = case['case_id']
        directory = source / 'cases' / case_id
        complete = json.loads((directory / 'complete.json').read_text())
        assert digest(directory / 'grids.npz') == complete['sha256']['grids.npz']
        inputs = Path(protocol['parent']) / 'cases' / case_id / 'inputs.npz'
        assert digest(inputs) == source_manifest['parent_inputs_sha256'][case_id]['inputs.npz']
        with np.load(inputs) as data, np.load(directory / 'grids.npz') as grids:
            for layer in (0, 8):
                grid = grids[f'layer_{layer}']
                for axis in ('row', 'column'):
                    oriented = grid if axis == 'column' else grid.transpose(1, 0, 2)
                    masks = data['observed'] if axis == 'column' else data['observed'].T
                    for context, (features, observed) in enumerate(zip(oriented, masks)):
                        records.append(dict(case_id=case_id, noise=case['noise'],
                                            observed_fraction=case['observed_fraction'], seed=case['seed'],
                                            layer=layer, axis=axis, context_index=context,
                                            **geometry(features, data[f'{axis}_labels'], observed)))
        if (index + 1) % 10 == 0:
            print(f'{index + 1}/120 matrices diagnosed', flush=True)
    slices = pd.DataFrame(records)
    metrics = ['mask_eta2', 'group_eta2', 'joint_eta2', 'group_after_mask_eta2']
    keys = ['case_id', 'noise', 'observed_fraction', 'seed', 'axis', 'layer']
    matrices = slices.groupby(keys, as_index=False)[metrics].mean()
    summaries = summarize(matrices, ['noise', 'observed_fraction', 'axis', 'layer'], metrics)
    cells = pd.read_csv(cell_scores_path)
    cells = cells[cells.scope == 'all'][keys + ['ari']].assign(representation='single_cell')
    pooled_scores = pd.read_csv(pooled_scores_path)
    pooled_scores = pooled_scores[pooled_scores.layer.isin([0, 8]) & pooled_scores.representation.isin(['mean', 'profile'])]
    comparison = pd.concat([cells, pooled_scores[keys + ['ari', 'representation']]], ignore_index=True)
    compared = summarize(comparison, ['noise', 'observed_fraction', 'axis', 'layer', 'representation'], ['ari'])
    pair_keys = ['case_id', 'noise', 'observed_fraction', 'seed', 'axis', 'representation']
    paired = comparison.pivot(index=pair_keys, columns='layer', values='ari').reset_index()
    paired['difference'] = paired[8] - paired[0]
    paired_summary = summarize(paired, ['noise', 'observed_fraction', 'axis', 'representation'], ['difference'])
    noise_pairs = comparison[comparison.layer == 8].pivot(
        index=['seed', 'observed_fraction', 'axis', 'representation'], columns='noise', values='ari').reset_index()
    noise_pairs['zero_minus_quarter'] = noise_pairs[0.] - noise_pairs[.25]
    noise_summary = summarize(noise_pairs, ['observed_fraction', 'axis', 'representation'], ['zero_minus_quarter'])
    outputs = {'geometry_per_slice.csv': slices, 'geometry_per_matrix.csv': matrices,
               'geometry_summary.csv': summaries, 'comparison_per_matrix.csv': comparison,
               'comparison_summary.csv': compared, 'paired.csv': paired,
               'paired_summary.csv': paired_summary, 'noise_pairs.csv': noise_pairs,
               'noise_summary.csv': noise_summary}
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)
    for path, expected in frozen['source_sha256'].items():
        assert digest(Path(path)) == expected
    assert len(slices) == 30720 and len(comparison) == 1440
    manifest = dict(cases=120, contexts=30720, no_refitting=True, no_new_inference=True,
                    inputs_unchanged=True, artifacts_sha256={name: digest(output / name) for name in
                    (*outputs, 'protocol.json')})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(summaries[(summaries.noise == .25) & (summaries.observed_fraction == .4)].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/mnt/matrixformer_scratch/cell_embeddings_20261105'))
    parser.add_argument('--pooled', type=Path, default=Path('/root/MatrixFormerPaper/results/synthetic_representations'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.pooled, args.output)
