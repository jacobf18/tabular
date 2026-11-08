"""Complete a realified Pauli moment matrix with frozen MatrixFormer-Impute."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

import evaluate_physics_transforms as experiment


def pauli_products(qubits):
    count = 4 ** qubits
    indices = np.arange(count)
    products = np.bitwise_xor(indices[:, None], indices[None, :])
    local_phases = np.array([[1, 1, 1, 1], [1, 1, 1j, -1j],
                             [1, -1j, 1, 1j], [1, 1j, -1j, 1]])
    phases = np.ones((count, count), dtype=complex)
    for shift in range(qubits):
        digits = (indices >> (2 * shift)) & 3
        phases *= local_phases[digits[:, None], digits[None, :]]
    return products, phases


def lift_coefficients(coefficients, products, phases):
    values = coefficients.ravel()[products]
    real = np.where(phases.real != 0, phases.real * values, 0.)
    imaginary = np.where(phases.imag != 0, phases.imag * values, 0.)
    return np.block([[real, -imaginary], [imaginary, real]])


def unlift_coefficients(prediction, products, phases, shape):
    count = len(products)
    real = (prediction[:count, :count] + prediction[count:, count:]) / 2
    imaginary = (prediction[count:, :count] - prediction[:count, count:]) / 2
    estimates = real * phases.real + imaginary * phases.imag
    sums = np.bincount(products.ravel(), weights=estimates.ravel(), minlength=count)
    counts = np.bincount(products.ravel(), minlength=count)
    return (sums / counts).reshape(shape)


def predict_moments(matrix, predict, products, phases):
    lifted = lift_coefficients(matrix, products, phases)
    prediction = predict(lifted)
    return unlift_coefficients(prediction, products, phases, matrix.shape)


def register(output):
    if (output / 'protocol.json').exists():
        raise FileExistsError('Use a fresh output directory')
    sources = [Path(__file__), Path(experiment.__file__), Path(experiment.shared.__file__),
               Path(experiment.quantum.__file__), Path(__file__).with_name('evaluate_physics_impute.py')]
    inputs = [path for path in experiment.QUANTUM.rglob('*') if path.suffix in ('.json', '.npz', '.csv')]
    inputs.append(experiment.shared.GENERAL.with_name('config.json'))
    (output / 'source_snapshot').mkdir(parents=True, exist_ok=True)
    for source in sources:
        shutil.copy2(source, output / 'source_snapshot' / source.name)
    experiment.shared.write_json(output / 'protocol.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(), cases=18, benchmark='quantum',
        checkpoint=str(experiment.shared.GENERAL), checkpoint_sha256=experiment.shared.sha256(experiment.shared.GENERAL),
        source_sha256={str(path.resolve()): experiment.shared.sha256(path) for path in sources},
        input_sha256={str(path.resolve()): experiment.shared.sha256(path) for path in inputs},
        transform='G_ab = Tr(rho P_a P_b); realify to 512x512; average signed repeated coefficients on inverse',
        strengths=experiment.STRENGTHS, validation='10% measured nonidentity coefficients; seed 20261106 + state seed',
        selection='Observed-only validation coefficient RMSE, no fidelity or hidden-target selection',
        caveat='Second exploratory representation experiment; all 18 original quantum cases retained; no MRI'))


def run(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    for section in ('source_sha256', 'input_sha256'):
        for name, digest in protocol[section].items():
            assert experiment.shared.sha256(name) == digest, name
    assert experiment.shared.sha256(protocol['checkpoint']) == protocol['checkpoint_sha256']
    torch.set_num_threads(4)
    torch.manual_seed(20261106)
    torch.backends.cuda.enable_cudnn_sdp(False)
    models, distribution = experiment.shared.load_models({'impute': experiment.shared.GENERAL})
    model = models['impute']
    versions = [parameter._version for parameter in model.parameters()]
    products, phases = pauli_products(4)
    _, operators = experiment.quantum.pauli_basis(4)
    (output / 'cases').mkdir(exist_ok=True)

    def predict(values):
        prediction, features = experiment.forward_native(values, model, distribution)
        del features
        return prediction

    for parent_path in sorted((experiment.QUANTUM / 'cases').glob('*.json')):
        parent = json.loads(parent_path.read_text())
        destination = output / 'cases' / parent_path.name
        if destination.exists():
            record = json.loads(destination.read_text())
            assert experiment.shared.sha256(destination.with_suffix('.npz')) == record['arrays_sha256']
            continue
        print(f"Running moment lift/{parent['case_id']}", flush=True)
        experiment.shared.write_json(output / 'status.json', dict(status='running', case_id=parent['case_id']))
        started = time.perf_counter()
        with np.load(parent_path.with_suffix('.npz')) as saved:
            matrix = saved['matrix']
        observed = np.flatnonzero(np.isfinite(matrix.ravel()))
        observed = observed[observed != 0]
        indices = np.random.default_rng(20261106 + parent['seed']).choice(observed, size=max(1, len(observed) // 10), replace=False)
        validation = np.zeros_like(matrix, dtype=bool)
        validation.ravel()[indices] = True
        reduced = matrix.copy()
        reduced[validation] = np.nan
        reduced_prediction = predict_moments(reduced, predict, products, phases)
        trials = []
        for strength in experiment.STRENGTHS:
            density, diagnostics = experiment.physical_refine(reduced, np.clip(reduced_prediction, -1, 1), strength, operators)
            loss = experiment.validation_error(experiment.quantum.pauli_coefficients(density, operators), matrix, validation)
            trials.append(dict(strength=strength, validation_rmse=loss, solver=diagnostics))
        selected = min(trials, key=lambda trial: trial['validation_rmse'])
        prediction = predict_moments(matrix, predict, products, phases)
        completed = experiment.quantum.complete_coefficients(matrix, prediction)
        raw_density = experiment.quantum.project_density(experiment.quantum.density_from_coefficients(completed, operators))
        refined, solver = experiment.physical_refine(matrix, np.clip(prediction, -1, 1), selected['strength'], operators)
        with np.load(experiment.QUANTUM / 'states' / f"{parent['state_id']}.npz") as saved:
            truth, truth_density = saved['coefficients'], saved['density']
        predictions = {'MF-Impute moment lift': raw_density, 'MF-Impute moment lift + PSD': refined}
        metrics = {method: experiment.quantum.reconstruction_metrics(truth_density, truth, density, matrix, operators)
                   for method, density in predictions.items()}
        assert not model.training and versions == [parameter._version for parameter in model.parameters()]
        experiment.atomic_npz(destination.with_suffix('.npz'), matrix=matrix, validation=validation,
                              reduced_prediction=reduced_prediction, prediction=prediction,
                              density=raw_density, refined_density=refined)
        metadata = {key: parent[key] for key in ('case_id', 'seed', 'rate', 'family', 'state_id')}
        experiment.shared.write_json(destination, dict(metadata=metadata, metrics=metrics, trials=trials,
                                     selected=selected, refit_solver=solver, seconds=time.perf_counter() - started,
                                     arrays_sha256=experiment.shared.sha256(destination.with_suffix('.npz'))))
        print(json.dumps(dict(case=parent['case_id'], fidelity={method: scores['fidelity'] for method, scores in metrics.items()},
                              strength=selected['strength'])), flush=True)
    report(output)


def report(output):
    rows = []
    _, operators = experiment.quantum.pauli_basis(4)
    paths = sorted((output / 'cases').glob('*.json'))
    assert {path.name for path in paths} == {path.name for path in (experiment.QUANTUM / 'cases').glob('*.json')}
    for path in paths:
        record = json.loads(path.read_text())
        assert experiment.shared.sha256(path.with_suffix('.npz')) == record['arrays_sha256']
        assert record['selected'] == min(record['trials'], key=lambda trial: trial['validation_rmse'])
        with np.load(experiment.QUANTUM / 'states' / f"{record['metadata']['state_id']}.npz") as state:
            truth, density = state['coefficients'], state['density']
        with np.load(path.with_suffix('.npz')) as arrays, np.load(experiment.QUANTUM / 'cases' / path.with_suffix('.npz').name) as parent:
            np.testing.assert_array_equal(arrays['matrix'], parent['matrix'])
            assert not (arrays['validation'] & ~np.isfinite(arrays['matrix'])).any()
            for method, key in (('MF-Impute moment lift', 'density'), ('MF-Impute moment lift + PSD', 'refined_density')):
                metrics = experiment.quantum.reconstruction_metrics(density, truth, arrays[key], arrays['matrix'], operators)
                for name, value in metrics.items():
                    np.testing.assert_allclose(value, record['metrics'][method][name], atol=1e-12, rtol=1e-10)
                rows.append(dict(**record['metadata'], method=method, **metrics))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'per_case.csv', index=False)
    frame.groupby('method').fidelity.agg(['mean', 'std', 'count']).to_csv(output / 'overall.csv')
    frame.groupby(['family', 'rate', 'method']).fidelity.agg(['mean', 'std', 'count']).to_csv(output / 'by_condition.csv')
    experiment.shared.write_json(output / 'audit.json', dict(status='passed', cases=18, metric_values_checked=len(rows) * len(metrics)))
    experiment.shared.write_json(output / 'status.json', dict(status='complete', cases=18, expected=18, audit='passed'))
    print(frame.groupby('method').fidelity.mean().to_string(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run', 'report'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'plan': register, 'run': run, 'report': report}[args.action](args.output)
