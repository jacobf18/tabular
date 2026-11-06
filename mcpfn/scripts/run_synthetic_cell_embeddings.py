"""Freeze, extract and audit unpooled representations on all 120 clustered cases."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits
import torch

from recsys_evaluation_common import atomic_npz, sha256, write_csv, write_json
from run_matrixformer_synthetic_study import CHECKPOINT, CHECKPOINT_SHA, Predictor, ROOT, source_paths
from run_synthetic_cluster_representations import load_inputs
import synthetic_cell_embeddings as analysis


PARENT = Path('/mnt/matrixformer_scratch/synthetic_capabilities_20261103')


def register(output):
    if output.exists():
        raise ValueError('Choose a new output directory')
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Frozen checkpoint changed')
    cases = [case for case in json.loads((PARENT / 'protocol.json').read_text())['cases']
             if case['family'] == 'clusters']
    if len(cases) != 120:
        raise ValueError('Expected all 120 clustered cases')
    example = load_inputs(PARENT, next(case for case in cases if case['case_id'] == analysis.EXAMPLE_CASE))
    output.mkdir(parents=True)
    protocol = dict(cases=cases, parent=str(PARENT), layers=list(analysis.LAYERS), axes=list(analysis.AXES),
                    seed=analysis.CLUSTER_SEED, clusters=4, initializations=20,
                    features='Raw 768-dimensional cell vectors; no axis pooling, feature rescaling or PCA before K-means',
                    scope='Every column: cluster its 64 row-cell embeddings; every row: cluster its 64 column-cell embeddings',
                    primary='All-cell ARI, averaged equally over 64 slices within each matrix, then over ten matrices per condition',
                    masked='Secondary ARI restricted to masked cells, using the same all-cell fitted assignments; require at least two cells and two true groups',
                    uncertainty='Two-sided 95% Student-t intervals over ten matrix means; paired final-minus-encoder differences use matrix means; never treat slices or cells as independent replicates',
                    examples=dict(case_id=analysis.EXAMPLE_CASE,
                                  row=analysis.predetermined_indices(example['column_labels']),
                                  column=analysis.predetermined_indices(example['row_labels']),
                                  rule='First original-order column/row in each true opposite-axis group; selected before fitting, only for illustration'),
                    pca='Independent label-blind two-dimensional PCA per slice and layer, no jitter or selected components; ARI uses full features',
                    labels='True groups enter only metric evaluation, display, and prespecified representative-slice selection; never model input or K-means',
                    checkpoint_selection=False, training=False, standard_orientation=True)
    write_json(output / 'protocol.json', protocol)
    sources = {}
    dependencies = source_paths() + [Path(__file__).resolve(), Path(analysis.__file__).resolve(),
                                    ROOT / 'scripts/run_synthetic_cluster_representations.py',
                                    ROOT / 'scripts/synthetic_cluster_representations.py']
    for path in dependencies:
        relative = path.relative_to(ROOT)
        target = output / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        sources[str(relative)] = sha256(path)
    inputs = {case['case_id']: {name: sha256(PARENT / 'cases' / case['case_id'] / name)
                              for name in ('inputs.npz', 'predictions.npz')} for case in cases}
    write_json(output / 'manifest.json', dict(created=datetime.now(timezone.utc).isoformat(),
               checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA,
               config_sha256=sha256(CHECKPOINT.parent / 'config.json'), source_sha256=sources,
               protocol_sha256=sha256(output / 'protocol.json'), parent_inputs_sha256=inputs,
               torch=torch.__version__, numpy=np.__version__))
    print(json.dumps(dict(cases=len(cases), examples=protocol['examples'], output=str(output))), flush=True)


def verify(output):
    manifest = json.loads((output / 'manifest.json').read_text())
    if sha256(output / 'protocol.json') != manifest['protocol_sha256']:
        raise ValueError('Registered protocol changed')
    if sha256(CHECKPOINT) != manifest['checkpoint_sha256'] or sha256(CHECKPOINT.parent / 'config.json') != manifest['config_sha256']:
        raise ValueError('Model changed')
    for relative, expected in manifest['source_sha256'].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f'Source changed: {relative}')
    for case_id, files in manifest['parent_inputs_sha256'].items():
        for name, expected in files.items():
            if sha256(PARENT / 'cases' / case_id / name) != expected:
                raise ValueError('Frozen input changed')
    return json.loads((output / 'protocol.json').read_text())


def execute(output, limit, jobs):
    protocol = verify(output)
    torch.set_num_threads(2)
    predictor = Predictor()
    completed = 0
    with threadpool_limits(limits=1):
        for index, case in enumerate(protocol['cases']):
            directory = output / 'cases' / case['case_id']
            if (directory / 'complete.json').exists():
                continue
            start = time.perf_counter()
            directory.mkdir(parents=True, exist_ok=True)
            data = load_inputs(PARENT, case)
            context = data['context']
            means, scales = np.nanmean(context.astype(float), 0), np.nanstd(context.astype(float), 0)
            fresh = predictor.predict(context, means, scales)
            prediction, grids = analysis.capture(predictor, context, means, scales)
            np.testing.assert_array_equal(fresh, prediction)
            with np.load(PARENT / 'cases' / case['case_id'] / 'predictions.npz') as saved:
                np.testing.assert_array_equal(prediction, saved['matrixformer'])
            records, assignments = [], {}
            for layer, grid in grids.items():
                for axis in analysis.AXES:
                    results = Parallel(n_jobs=jobs, prefer='threads')(
                        delayed(analysis.evaluate_slice)(grid, data['observed'], data[f'{axis}_labels'], axis, context_index)
                        for context_index in range(64))
                    assignments[f'{axis}_{layer}'] = np.stack([result[0] for result in results])
                    for context_index, (_, scores) in enumerate(results):
                        records.extend(dict(**case, axis=axis, layer=layer, context_index=context_index, **score)
                                       for score in scores)
            atomic_npz(directory / 'grids.npz', **{f'layer_{layer}': grid for layer, grid in grids.items()})
            atomic_npz(directory / 'assignments.npz', **assignments)
            write_csv(directory / 'metrics.csv', records)
            predictor.check_unchanged()
            write_json(directory / 'complete.json', dict(capture_bit_identical=True, parent_bit_identical=True,
                       checkpoint_unchanged=True, seconds=time.perf_counter() - start,
                       sha256={path.name: sha256(path) for path in directory.iterdir() if path.is_file()}))
            write_json(output / 'status.json', dict(completed=index + 1, total=120, last=case))
            print(json.dumps(dict(index=index + 1, case=case['case_id'], seconds=round(time.perf_counter() - start, 2))), flush=True)
            completed += 1
            if limit and completed >= limit:
                break


def summarize(frame):
    keys = ['case_id', 'noise', 'observed_fraction', 'seed', 'axis', 'layer', 'scope']
    per_matrix = frame.groupby(keys, as_index=False).agg(ari=('ari', 'mean'), slices=('ari', 'count'))
    primary = per_matrix[per_matrix.scope == 'all']
    if not (primary.slices == 64).all():
        raise ValueError('Every primary matrix score must average all 64 slices')
    records = []
    for keys, group in per_matrix.groupby(['noise', 'observed_fraction', 'axis', 'layer', 'scope']):
        noise, fraction, axis, layer, scope = keys
        values = group.ari.dropna().to_numpy()
        half_width = student_t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
        records.append(dict(noise=noise, observed_fraction=fraction, axis=axis, layer=layer, scope=scope,
                            matrices=len(values), mean=values.mean(), ci95_half_width=half_width,
                            ci95_low=values.mean() - half_width, ci95_high=values.mean() + half_width))
    paired = per_matrix.pivot(index=['case_id', 'noise', 'observed_fraction', 'seed', 'axis', 'scope'],
                              columns='layer', values='ari').reset_index()
    paired['difference'] = paired[8] - paired[0]
    return per_matrix, pd.DataFrame(records), paired


def audit(output):
    protocol = verify(output)
    frames = []
    for case in protocol['cases']:
        directory = output / 'cases' / case['case_id']
        complete = json.loads((directory / 'complete.json').read_text())
        assert all(complete[key] for key in ('capture_bit_identical', 'parent_bit_identical', 'checkpoint_unchanged'))
        for name, expected in complete['sha256'].items():
            if sha256(directory / name) != expected:
                raise ValueError('Case artifact changed')
        data = load_inputs(PARENT, case)
        records = []
        with np.load(directory / 'assignments.npz') as assignments:
            for layer in analysis.LAYERS:
                for axis in analysis.AXES:
                    for context_index, fitted in enumerate(assignments[f'{axis}_{layer}']):
                        observed = data['observed'][:, context_index] if axis == 'row' else data['observed'][context_index, :]
                        records.extend(dict(**case, axis=axis, layer=layer, context_index=context_index, **score)
                                       for score in analysis.score_slice(fitted, data[f'{axis}_labels'], observed))
        actual = pd.read_csv(directory / 'metrics.csv')
        pd.testing.assert_frame_equal(actual, pd.DataFrame(records), check_dtype=False, rtol=1e-12, atol=1e-12)
        frames.append(actual)
    combined = pd.concat(frames, ignore_index=True)
    per_matrix, summary, paired = summarize(combined)
    combined.to_csv(output / 'per_slice.csv', index=False)
    per_matrix.to_csv(output / 'per_matrix.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    paired.to_csv(output / 'paired.csv', index=False)
    write_json(output / 'audit.json', dict(cases=len(frames), slice_score_rows=len(combined),
               clustering_fits=len(combined) // 2, all_capture_bit_identical=True, all_parent_bit_identical=True,
               checkpoint_unchanged=True, rescore_matches=True, all_primary_matrices_use_64_slices=True,
               saved_artifacts_sha256={name: sha256(output / name) for name in
                                      ('per_slice.csv', 'per_matrix.csv', 'summary.csv', 'paired.csv')}))
    print(summary[(summary.noise == .25) & (summary.observed_fraction == .4)].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run', 'audit'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.action == 'plan':
        register(args.output)
    elif args.action == 'run':
        execute(args.output, args.limit, args.jobs)
    else:
        audit(args.output)


if __name__ == '__main__':
    main()
