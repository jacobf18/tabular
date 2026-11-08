"""Paired, retrospective preprocessing ablations without changing frozen evaluations."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import torch

import evaluate_cylinder_completion as shared
import evaluate_realistic_science as original
from fetch_science_panels import ROOT


OUTPUT = ROOT / 'diagnosis_v1'
MODELS = ('MatrixFormer', 'MatrixFormer-Impute')
VARIANTS = ('column', 'clean_column', 'clean_fp32', 'clean_transpose',
            'clean_local', 'physical_salinity', 'physical_residual')


def prepare(matrix, dataset, columns, variant):
    values = matrix.copy()
    keep = np.ones(matrix.shape[1], dtype=bool)
    if variant != 'column':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            keep = (np.isfinite(values).sum(axis=0) >= 8) & (np.nanstd(values, axis=0) > 1e-8)
        if dataset == 'ocean_alkalinity':
            keep &= np.array([name not in ('station', 'cast', 'year') for name in columns])
    target = columns.index('talk') if dataset == 'ocean_alkalinity' else None
    slope, intercept = 0., 0.
    salinity = None
    if variant.startswith('physical_'):
        salinity = matrix[:, columns.index('salinity')].copy()
        salinity[~np.isfinite(salinity)] = np.nanmedian(salinity)
        if variant == 'physical_salinity':
            if (salinity <= .1).any():
                raise ValueError('Salinity normalization is undefined near freshwater')
            values[:, target] *= 35. / salinity
        else:
            observed = np.isfinite(matrix[:, target])
            design = np.column_stack((salinity[observed], np.ones(observed.sum())))
            slope, intercept = np.linalg.lstsq(design, matrix[observed, target], rcond=None)[0]
            values[:, target] -= slope * salinity + intercept
    normalized, inverse = original.transform(values[:, keep], 'column')

    def restore(prediction):
        result = original.temporal_fill(matrix)
        result[:, keep] = inverse(prediction)
        if variant == 'physical_salinity':
            result[:, target] *= salinity / 35.
        elif variant == 'physical_residual':
            result[:, target] += slope * salinity + intercept
        result[np.isfinite(matrix)] = matrix[np.isfinite(matrix)]
        return result

    return normalized, restore, dict(kept_columns=np.flatnonzero(keep).tolist(),
                                     removed_columns=[name for name, retained in zip(columns, keep) if not retained],
                                     salinity_slope=float(slope), salinity_intercept=float(intercept))


@torch.inference_mode()
def forward(matrix, model, distribution, full_precision=False, return_diagnostics=False):
    normalized, center, scale = shared.normalize_observed(matrix)
    values = torch.as_tensor(normalized, dtype=torch.float32, device='cuda')[None]
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=not full_precision):
        embedded = model(values, return_embeddings=True)[0]
    features = embedded.reshape(-1, embedded.shape[-1])
    mean = np.empty(matrix.size, dtype=np.float64)
    median = np.empty(matrix.size, dtype=np.float64)
    tail = np.empty(matrix.size, dtype=np.float64)
    for start in range(0, matrix.size, 2048):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=not full_precision):
            logits = model.decoder(features[start:start + 2048])
        logits = logits.float()
        stop = start + len(logits)
        mean[start:stop] = distribution.mean(logits).cpu().numpy()
        if return_diagnostics:
            median[start:stop] = distribution.median(logits).cpu().numpy()
            probabilities = logits.softmax(-1)
            tail[start:stop] = (probabilities[:, 0] + probabilities[:, -1]).cpu().numpy()
    del features, embedded
    result = mean.reshape(matrix.shape) * scale + center
    diagnostics = {}
    if return_diagnostics:
        diagnostics = dict(median=median.reshape(matrix.shape) * scale + center,
                           tail_mass=tail.reshape(matrix.shape), global_sd=float(scale))
    return result, diagnostics


def predict(matrix, model, distribution, dataset, columns, variant):
    normalized, restore, details = prepare(matrix, dataset, columns, variant)
    if variant == 'clean_local':
        output = np.empty_like(normalized)
        for start in range(0, len(matrix), 32):
            stop = min(start + 32, len(matrix))
            center = (start + stop) // 2
            left = max(0, min(center - 64, len(matrix) - 128))
            right = min(left + 128, len(matrix))
            local, inverse = original.transform(normalized[left:right], 'column')
            prediction, _ = forward(local, model, distribution)
            output[start:stop] = inverse(prediction)[start-left:stop-left]
    else:
        transpose = variant == 'clean_transpose'
        output, diagnostics = forward(normalized.T if transpose else normalized, model, distribution,
                                      full_precision=variant == 'clean_fp32', return_diagnostics=variant == 'column')
        if transpose:
            output = output.T
        if variant == 'column':
            details['mean_tail_mass'] = float(diagnostics['tail_mass'][np.isnan(normalized)].mean())
            details['max_tail_mass'] = float(diagnostics['tail_mass'][np.isnan(normalized)].max())
            details['mean_median_gap'] = float(np.mean(np.abs(output[np.isnan(normalized)] - diagnostics['median'][np.isnan(normalized)])))
    return restore(output), details


def register(output):
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for name in ('replay_v1', 'ocean_v2'):
        parent = ROOT / name
        protocol = json.loads((parent / 'protocol.json').read_text())
        for case in protocol['cases']:
            columns = protocol['datasets'][case['dataset']]['columns']
            if name == 'replay_v1':
                columns = columns + ['hour_sin', 'hour_cos', 'relative_time']
            cases.append(dict(case, parent=str(parent), columns=columns,
                              parent_protocol_sha256=shared.sha256(parent / 'protocol.json')))
    sources = [Path(__file__), Path(original.__file__), Path(shared.__file__),
               Path(__file__).with_name('evaluate_physics_impute.py')]
    configuration = dict(created=datetime.now(timezone.utc).isoformat(), cases=cases, variants=list(VARIANTS),
                         source_sha256={str(path): shared.sha256(path) for path in sources},
                         checkpoints={name: dict(path=str(path), sha256=shared.sha256(path), config_sha256=shared.sha256(path.with_name('config.json')))
                                      for name, path in [('MatrixFormer', shared.SINGLE), ('MatrixFormer-Impute', shared.GENERAL)]},
                         task='Retrospective paired diagnosis on the SAME previously scored matrices; not new independent confirmation',
                         selection='Frozen ablations; also choose a deployable candidate on recorded validation only; test-best variants are never selected',
                         physical='Observed-only salinity median for missing salinity; normalize alkalinity to S=35 or subtract observed OLS salinity trend',
                         cleanup='Remove unsupported/constant columns using current observations; ocean also removes station/cast IDs and date as an explicit ablation',
                         local='128-row moving context, central 32 rows retained; both normalizations use current observed context only')
    shared.write_json(output / 'protocol.json', configuration)
    print('Registered', len(cases), 'cases', flush=True)


def run(output, datasets, variants):
    protocol = json.loads((output / 'protocol.json').read_text())
    for path, expected in protocol['source_sha256'].items():
        assert shared.sha256(path) == expected
    checkpoints = {}
    for name, record in protocol['checkpoints'].items():
        checkpoint = Path(record['path'])
        assert shared.sha256(checkpoint) == record['sha256']
        assert shared.sha256(checkpoint.with_name('config.json')) == record['config_sha256']
        checkpoints[name] = checkpoint
    models, distribution = shared.load_models(checkpoints)
    for case in protocol['cases']:
        if datasets and case['dataset'] not in datasets:
            continue
        parent = Path(case['parent'])
        assert shared.sha256(parent / 'protocol.json') == case['parent_protocol_sha256']
        path = parent / f'{case["identifier"]}.npz'
        assert shared.sha256(path) == case['input_sha256']
        arrays = np.load(path)
        truth, missing, validation = [arrays[name] for name in ('truth', 'missing', 'validation')]
        context = np.where(missing, np.nan, truth)
        reduced = np.where(validation, np.nan, context)
        for variant in variants or protocol['variants']:
            if variant.startswith('physical_') and case['dataset'] != 'ocean_alkalinity':
                continue
            for name, model in models.items():
                destination = output / 'records' / f'{case["identifier"]}_{name}_{variant}.json'
                if destination.exists():
                    continue
                started = time.perf_counter()
                candidate, validation_details = predict(reduced, model, distribution, case['dataset'], case['columns'], variant)
                validation_scores = original.metrics(truth, candidate, validation, reduced, case['targets'])
                prediction, details = predict(context, model, distribution, case['dataset'], case['columns'], variant)
                scores = original.metrics(truth, prediction, missing, context, case['targets'])
                assert np.isfinite(prediction).all()
                np.testing.assert_array_equal(prediction[np.isfinite(context)], context[np.isfinite(context)])
                destination.parent.mkdir(exist_ok=True, parents=True)
                np.savez_compressed(destination.with_suffix('.npz'), prediction=prediction, validation_prediction=candidate)
                shared.write_json(destination, dict(identifier=case['identifier'], dataset=case['dataset'], split=case['split'],
                                                   model=name, variant=variant, scores=scores, validation_scores=validation_scores,
                                                   details=details, validation_details=validation_details,
                                                   seconds=time.perf_counter()-started,
                                                   prediction_sha256=shared.sha256(destination.with_suffix('.npz')),
                                                   protocol_sha256=shared.sha256(output / 'protocol.json')))
                print(case['identifier'], name, variant, f'val={validation_scores["nrmse"]:.5f}',
                      f'test={scores["nrmse"]:.5f}', flush=True)


def summarize(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    rows, diagnostics = [], []
    for case in protocol['cases']:
        arrays = np.load(Path(case['parent']) / f'{case["identifier"]}.npz')
        truth, missing, validation = [arrays[name] for name in ('truth', 'missing', 'validation')]
        context = np.where(missing, np.nan, truth)
        reduced = np.where(validation, np.nan, context)
        for model in MODELS:
            reference = json.loads((Path(case['parent']) / 'records' / f'{case["identifier"]}_{model}.json').read_text())
            rows.append(dict(dataset=case['dataset'], split=case['split'], identifier=case['identifier'], model=model,
                             variant='original_selected', nrmse=reference['scores']['nrmse'], validation=reference['tuning']['selected']['validation_nrmse']))
            expected = [variant for variant in VARIANTS if not variant.startswith('physical_') or case['dataset'] == 'ocean_alkalinity']
            candidates = []
            for variant in expected:
                path = output / 'records' / f'{case["identifier"]}_{model}_{variant}.json'
                if not path.exists():
                    continue
                record = json.loads(path.read_text())
                assert shared.sha256(path.with_suffix('.npz')) == record['prediction_sha256']
                assert shared.sha256(output / 'protocol.json') == record['protocol_sha256']
                saved = np.load(path.with_suffix('.npz'))
                for key, mask, current, field in [('prediction', missing, context, 'scores'), ('validation_prediction', validation, reduced, 'validation_scores')]:
                    np.testing.assert_array_equal(saved[key][np.isfinite(current)], current[np.isfinite(current)])
                    score = original.metrics(truth, saved[key], mask, current, case['targets'])
                    np.testing.assert_allclose(score['nrmse'], record[field]['nrmse'], rtol=1e-12)
                row = dict(dataset=case['dataset'], split=case['split'], identifier=case['identifier'], model=model,
                           variant=variant, nrmse=record['scores']['nrmse'], validation=record['validation_scores']['nrmse'])
                candidates.append(row)
                rows.append(row)
            if len(candidates) == len(expected):
                selected = min(candidates, key=lambda row: row['validation'])
                rows.append(dict(selected, variant='validation_selected', selected_variant=selected['variant']))
        diagnostics.append(dict(dataset=case['dataset'], identifier=case['identifier'], test_cells=int(missing[:, case['targets']].sum()),
                                validation_cells=int(validation[:, case['targets']].sum()),
                                unvalidated_test_columns=sum(validation[:, column].sum() == 0 for column in case['targets']),
                                all_missing_columns=int((np.isfinite(context).sum(axis=0) == 0).sum())))
    frame = pd.DataFrame(rows)
    records = []
    for key, group in frame.groupby(['dataset', 'split', 'model', 'variant']):
        values = group.nrmse.to_numpy()
        half = float(student_t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
        records.append(dict(zip(['dataset', 'split', 'model', 'variant'], key), mean=float(values.mean()), ci95_half_width=half, cases=len(values)))
    summary = pd.DataFrame(records)
    frame.to_csv(output / 'per_case.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(output / 'coverage_diagnostics.csv', index=False)
    print(summary[summary.split == 'confirmation'].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run', 'report'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS)
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    if arguments.action == 'register':
        register(arguments.output)
    elif arguments.action == 'run':
        run(arguments.output, arguments.datasets, arguments.variants)
    else:
        summarize(arguments.output)
