"""Matched SoftImpute and combined-MatrixFormer controls for flow refinement."""

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import shutil
import time

from fancyimpute import SoftImpute
import numpy as np
import pandas as pd
import torch

import evaluate_physics_transforms as experiment


SHRINKAGES = (.005, .02, .1)


def soft_prediction(matrix, shrinkage, seed):
    supported = np.isfinite(matrix).any(axis=0)
    prediction = np.zeros_like(matrix)
    prediction[:, supported] = experiment.shared.fit_baseline(matrix[:, supported], 'SoftImpute', shrinkage, seed)
    return prediction


def select_soft(matrix, validation, seed, archived_shrinkage):
    reduced = matrix.copy()
    reduced[validation] = np.nan
    trials = []
    for transform in experiment.FLOW_TRANSFORMS:
        for shrinkage in SHRINKAGES:
            initial = experiment.predict_flow(reduced, transform, lambda values: soft_prediction(values, shrinkage, seed))
            for rank in (None, *experiment.RANKS):
                prediction = experiment.low_rank_refine(reduced, initial, rank) if rank else initial
                trials.append(dict(transform=transform, shrinkage=shrinkage, rank=rank,
                                   validation_rmse=experiment.validation_error(prediction, matrix, validation)))
    selectors = {
        'SoftImpute + refinement': lambda trial: trial['transform'] == 'global' and trial['shrinkage'] == archived_shrinkage,
        'SoftImpute + joint refinement': lambda trial: trial['transform'] == 'global',
        'SoftImpute preprocessing': lambda trial: trial['rank'] is None,
        'SoftImpute preprocessing + refinement': lambda trial: True,
    }
    selected = {method: min([trial for trial in trials if predicate(trial)], key=lambda trial: trial['validation_rmse'])
                for method, predicate in selectors.items()}
    cache, predictions = {}, {}
    for method, trial in selected.items():
        key = (trial['transform'], trial['shrinkage'])
        if key not in cache:
            cache[key] = experiment.predict_flow(matrix, trial['transform'], lambda values: soft_prediction(values, trial['shrinkage'], seed))
        initial = cache[key]
        predictions[method] = experiment.low_rank_refine(matrix, initial, trial['rank']) if trial['rank'] else initial
    predictions['SoftImpute replay'] = cache.get(('global', archived_shrinkage))
    if predictions['SoftImpute replay'] is None:
        predictions['SoftImpute replay'] = soft_prediction(matrix, archived_shrinkage, seed)
    return predictions, dict(trials=trials, selected=selected)


def register(output):
    if (output / 'protocol.json').exists():
        raise FileExistsError('Use a fresh output directory')
    sources = [Path(__file__), Path(experiment.__file__), Path(experiment.shared.__file__),
               Path(__file__).with_name('evaluate_physics_impute.py'), Path(inspect.getfile(SoftImpute))]
    inputs = [path for path in experiment.CYLINDER.rglob('*') if path.suffix in ('.json', '.npz', '.csv')]
    inputs.append(experiment.shared.SINGLE.with_name('config.json'))
    (output / 'source_snapshot').mkdir(parents=True, exist_ok=True)
    for source in sources:
        shutil.copy2(source, output / 'source_snapshot' / source.name)
    experiment.shared.write_json(output / 'protocol.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(), cases=18, benchmark='cylinder',
        checkpoint=str(experiment.shared.SINGLE), checkpoint_sha256=experiment.shared.sha256(experiment.shared.SINGLE),
        source_sha256={str(path.resolve()): experiment.shared.sha256(path) for path in sources},
        input_sha256={str(path.resolve()): experiment.shared.sha256(path) for path in inputs},
        transforms=experiment.FLOW_TRANSFORMS, ranks=experiment.RANKS, steps=experiment.PROJECTION_STEPS,
        shrinkages=SHRINKAGES, selection='Original 10% observed-cell baseline validation mask, no outer-target selection',
        direct_control='Original selected SoftImpute shrinkage, then select rank or no refinement',
        matched_control='Same transforms and ranks as neural method, additionally select SoftImpute shrinkage',
        scope='Same 18 original cylinder cases, no MRI; exploratory follow-up, not independent confirmation'))


def run(output, method):
    protocol = json.loads((output / 'protocol.json').read_text())
    for section in ('source_sha256', 'input_sha256'):
        for name, digest in protocol[section].items():
            assert experiment.shared.sha256(name) == digest, name
    directory = output / method
    (directory / 'cases').mkdir(parents=True, exist_ok=True)
    if method == 'matrixformer':
        assert experiment.shared.sha256(protocol['checkpoint']) == protocol['checkpoint_sha256']
        torch.set_num_threads(4)
        torch.manual_seed(20261106)
        torch.backends.cuda.enable_cudnn_sdp(False)
        models, distribution = experiment.shared.load_models({'combined': experiment.shared.SINGLE})
        model = models['combined']
        versions = [parameter._version for parameter in model.parameters()]

        def predict(matrix):
            prediction, features = experiment.forward_native(matrix, model, distribution)
            del features
            return prediction

    with np.load(experiment.CYLINDER / 'dataset.npz') as saved:
        truth = saved['truth']
    for parent_path in sorted((experiment.CYLINDER / 'cases').glob('*.json')):
        parent = json.loads(parent_path.read_text())
        destination = directory / 'cases' / parent_path.name
        if destination.exists():
            saved = json.loads(destination.read_text())
            assert experiment.shared.sha256(destination.with_suffix('.npz')) == saved['arrays_sha256']
            continue
        experiment.shared.write_json(directory / 'status.json', dict(status='running', case_id=parent['case_id']))
        print(f"Running {method}/{parent['case_id']}", flush=True)
        started = time.perf_counter()
        with np.load(experiment.CYLINDER / 'predictions' / destination.with_suffix('.npz').name) as saved:
            matrix = np.where(saved['missing'], np.nan, truth)
            validation = saved['validation']
            reference = saved['prediction_3' if method == 'softimpute' else 'prediction_4']
        if method == 'softimpute':
            shrinkage = parent['baseline_tuning']['SoftImpute']['selected']['parameter']
            predictions, decisions = select_soft(matrix, validation, parent['seed'], shrinkage)
            replay = predictions['SoftImpute replay']
            np.testing.assert_allclose(replay.astype(reference.dtype), reference, rtol=1e-6, atol=1e-7)
        else:
            predictions, decisions = experiment.select_flow(matrix, validation, predict)
            predictions = {name.replace('MF-Impute', 'MatrixFormer'): values for name, values in predictions.items()}
            decisions['selected'] = {name.replace('MF-Impute', 'MatrixFormer'): trial for name, trial in decisions['selected'].items()}
            replay = predictions['MatrixFormer input: global']
            np.testing.assert_allclose(replay, reference, rtol=1e-6, atol=1e-7)
            assert not model.training and versions == [parameter._version for parameter in model.parameters()]
        _, _, scale = experiment.shared.normalize_observed(matrix)
        metrics = {name: experiment.shared.point_metrics(truth, values, ~np.isfinite(matrix), scale) for name, values in predictions.items()}
        keys = {name: f'prediction_{index}' for index, name in enumerate(predictions)}
        experiment.atomic_npz(destination.with_suffix('.npz'), matrix=matrix, validation=validation,
                              **{keys[name]: values for name, values in predictions.items()})
        metadata = {key: parent[key] for key in ('case_id', 'pattern', 'rate', 'seed')}
        experiment.shared.write_json(destination, dict(metadata=metadata, metrics=metrics, prediction_keys=keys, decisions=decisions,
                                     seconds=time.perf_counter() - started, original_prediction_replay_passed=True,
                                     arrays_sha256=experiment.shared.sha256(destination.with_suffix('.npz'))))
        print(json.dumps(dict(case=parent['case_id'], nrmse={name: values['nrmse'] for name, values in metrics.items()})), flush=True)
    records = [json.loads(path.read_text()) for path in sorted((directory / 'cases').glob('*.json'))]
    frame = pd.DataFrame([dict(**record['metadata'], method=name, **metrics)
                          for record in records for name, metrics in record['metrics'].items()])
    frame.to_csv(directory / 'per_case.csv', index=False)
    frame.groupby('method').nrmse.agg(['mean', 'std', 'count']).to_csv(directory / 'overall.csv')
    experiment.shared.write_json(directory / 'status.json', dict(status='complete', cases=len(records), expected=18))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['softimpute', 'matrixformer'])
    args = parser.parse_args()
    if args.action == 'plan':
        register(args.output)
    else:
        if args.method is None:
            parser.error('run requires --method')
        run(args.output, args.method)
