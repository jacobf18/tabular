"""Register, run, and independently rescore clustered representation diagnostics."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import sklearn
import torch

from recsys_evaluation_common import atomic_npz, sha256, write_csv, write_json
from run_matrixformer_synthetic_study import CHECKPOINT, CHECKPOINT_SHA, Predictor, ROOT, source_paths
import synthetic_cluster_representations as analysis
import synthetic_matrixformer_study as study


DEFAULT_PARENT = Path('/mnt/matrixformer_scratch/synthetic_capabilities_20261103')


def example_case(case):
    return case['seed'] == 0 and case['observed_fraction'] == .4


def register(output, parent, profile):
    if output.exists():
        raise ValueError('Registration requires a new directory')
    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Checkpoint hash mismatch')
    parent_protocol = json.loads((parent / 'protocol.json').read_text())
    cases = [case for case in parent_protocol['cases'] if case['family'] == 'clusters']
    if profile == 'smoke':
        cases = [case for case in cases if example_case(case) and case['noise'] == .25]
    output.mkdir(parents=True)
    protocol = dict(cases=cases, profile=profile, parent=str(parent),
                    primary='Adjusted Rand index of K-means with supplied K=4, 20 initializations',
                    seed=analysis.CLUSTER_SEED, layers=list(range(9)), axes=['row', 'column'],
                    poolings=list(analysis.POOLINGS), baselines=list(analysis.BASELINES),
                    pooling='Mean over opposite data axis; and complete cell profiles with an exact '
                            'distance-preserving Gram embedding. Exclude all special tokens. No feature rescaling.',
                    oracle='Complete noisy values are privileged; only oracle_complete receives hidden values.',
                    examples='Seed 0, 40% observations, every noise level; first pre-existing sampled missing query',
                    maps='Every head and layer; one fixed context and mean over all 64 actual contexts. '
                         'Read actual production Q/K projection outputs, apply original RoPE, softmax in FP32; '
                         'no re-evaluation of projections with a changed BF16 batch shape. '
                         'Unconditional probabilities, retaining special keys in saved arrays.',
                    pca='Two-dimensional label-blind illustration only; clustering uses full representation',
                    null='Mask-only features and 20 fixed label permutations per case/axis/representation',
                    uncertainty='95% Student-t intervals over ten matrices within each condition, '
                                'never over cells/heads/layers; conditions share seeds and are not independent',
                    no_model_selection=True, standard_orientation=True)
    write_json(output / 'protocol.json', protocol)
    sources = {}
    for path in source_paths() + [Path(__file__), Path(analysis.__file__)]:
        relative = path.relative_to(ROOT)
        target = output / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        sources[str(relative)] = sha256(path)
    inputs = {}
    for case in cases:
        directory = parent / 'cases' / case['case_id']
        names = ['inputs.npz', 'predictions.npz']
        if example_case(case):
            names.append('attention.npz')
        inputs[case['case_id']] = {name: sha256(directory / name) for name in names}
    write_json(output / 'manifest.json', dict(created=datetime.now(timezone.utc).isoformat(),
               checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA,
               config_sha256=sha256(CHECKPOINT.parent / 'config.json'), source_sha256=sources,
               protocol_sha256=sha256(output / 'protocol.json'), parent_inputs_sha256=inputs,
               parent_protocol_sha256=sha256(parent / 'protocol.json'),
               torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__))
    print(json.dumps(dict(cases=len(cases), output=str(output))), flush=True)


def verify(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    manifest = json.loads((output / 'manifest.json').read_text())
    if sha256(output / 'protocol.json') != manifest['protocol_sha256']:
        raise ValueError('Protocol changed')
    if sha256(CHECKPOINT) != manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    if sha256(CHECKPOINT.parent / 'config.json') != manifest['config_sha256']:
        raise ValueError('Model config changed')
    for relative, expected in manifest['source_sha256'].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f'Source changed: {relative}')
    parent = Path(protocol['parent'])
    if sha256(parent / 'protocol.json') != manifest['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    for case_id, files in manifest['parent_inputs_sha256'].items():
        for name, expected in files.items():
            if sha256(parent / 'cases' / case_id / name) != expected:
                raise ValueError(f'Parent artifact changed: {case_id}/{name}')
    return protocol


def load_inputs(parent, case):
    with np.load(parent / 'cases' / case['case_id'] / 'inputs.npz') as saved:
        data = dict(saved)
    for name, expected in study.generate(case).items():
        np.testing.assert_array_equal(data[name], expected)
    return data


def execute(output):
    protocol = verify(output)
    parent = Path(protocol['parent'])
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError(f'GPU occupied: {active}')
    torch.set_num_threads(2)
    predictor = Predictor()
    for index, case in enumerate(protocol['cases']):
        directory = output / 'cases' / case['case_id']
        if (directory / 'complete.json').exists():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        data = load_inputs(parent, case)
        context = data['context']
        means, scales = np.nanmean(context.astype(float), 0), np.nanstd(context.astype(float), 0)
        target = None
        if example_case(case):
            with np.load(parent / 'cases' / case['case_id'] / 'attention.npz') as previous:
                target = previous['targets'][0]
        baseline = predictor.predict(context, means, scales)
        prediction, features, attention = analysis.capture(predictor, context, means, scales, target)
        np.testing.assert_array_equal(prediction, baseline)
        with np.load(parent / 'cases' / case['case_id'] / 'predictions.npz') as previous:
            np.testing.assert_array_equal(prediction, previous['matrixformer'])
        features.update(analysis.baseline_features(context, prediction, data['truth'], means, scales))
        assignments, records = {}, []
        for name, values in features.items():
            representation, axis, layer = name.rsplit('_', 2)
            labels, projection, variance = analysis.fit_clusters(values)
            assignments[name] = labels
            assignments[f'{name}_pca'] = projection
            assignments[f'{name}_pca_variance'] = variance
            records.append(dict(**case, representation=representation, axis=axis, layer=int(layer),
                                **analysis.cluster_metrics(values, labels, data[f'{axis}_labels'])))
        atomic_npz(directory / 'features.npz', **features)
        atomic_npz(directory / 'clustering.npz', **assignments)
        write_csv(directory / 'metrics.csv', records)
        if attention:
            for values in attention.values():
                np.testing.assert_allclose(values.sum(-1), 1, atol=1e-6)
            atomic_npz(directory / 'attention.npz', target=target, **attention)
        predictor.check_unchanged()
        write_json(directory / 'complete.json', dict(capture_bit_identical=True, parent_bit_identical=True,
                   checkpoint_unchanged=True, seconds=time.perf_counter() - start,
                   sha256={path.name: sha256(path) for path in directory.iterdir() if path.is_file()}))
        write_json(output / 'status.json', dict(completed=index + 1, total=len(protocol['cases']), last=case))
        print(json.dumps(dict(index=index + 1, total=len(protocol['cases']), case=case['case_id'],
                              seconds=round(time.perf_counter() - start, 2))), flush=True)


def audit(output):
    protocol = verify(output)
    parent = Path(protocol['parent'])
    frames = []
    expected_keys = {f'{pool}_{axis}_{layer}' for pool in analysis.POOLINGS
                     for axis in ('row', 'column') for layer in range(9)}
    expected_keys |= {f'{method}_{axis}_-1' for method in analysis.BASELINES for axis in ('row', 'column')}
    for case in protocol['cases']:
        directory = output / 'cases' / case['case_id']
        completion = json.loads((directory / 'complete.json').read_text())
        if not all(completion[key] for key in ('capture_bit_identical', 'parent_bit_identical', 'checkpoint_unchanged')):
            raise ValueError('Capture parity audit failed')
        for name, expected in completion['sha256'].items():
            if sha256(directory / name) != expected:
                raise ValueError(f'Changed artifact: {name}')
        data = load_inputs(parent, case)
        records = []
        with np.load(directory / 'features.npz') as features, np.load(directory / 'clustering.npz') as clusters:
            if set(features.files) != expected_keys:
                raise ValueError('Incomplete representation inventory')
            for name in sorted(features.files):
                representation, axis, layer = name.rsplit('_', 2)
                records.append(dict(**case, representation=representation, axis=axis, layer=int(layer),
                                    **analysis.cluster_metrics(features[name], clusters[name], data[f'{axis}_labels'])))
        expected = pd.DataFrame(records)
        actual = pd.read_csv(directory / 'metrics.csv').sort_values(['representation', 'axis', 'layer']).reset_index(drop=True)
        expected = expected.sort_values(['representation', 'axis', 'layer']).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
        if example_case(case):
            with np.load(directory / 'attention.npz') as attention:
                if len(attention.files) != 33:
                    raise ValueError('Incomplete attention map inventory')
                for name in attention.files:
                    if name != 'target':
                        values = attention[name]
                        if values.shape != (16, 76, 76) or not np.isfinite(values).all() or values.min() < 0:
                            raise ValueError('Invalid attention map')
                        np.testing.assert_allclose(values.sum(-1), 1, atol=1e-6)
        frames.append(actual)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output / 'metrics.csv', index=False)
    write_json(output / 'audit.json', dict(cases=len(frames), clustering_results=len(combined),
               all_capture_bit_identical=True, all_parent_bit_identical=True,
               rescore_matches=True, complete_inventory=True))
    print((output / 'audit.json').read_text(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run', 'audit'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    parser.add_argument('--profile', choices=['smoke', 'main'], default='main')
    args = parser.parse_args()
    if args.action == 'plan':
        register(args.output, args.parent, args.profile)
    elif args.action == 'run':
        execute(args.output)
    else:
        audit(args.output)


if __name__ == '__main__':
    main()
