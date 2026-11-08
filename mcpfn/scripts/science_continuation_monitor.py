"""Seven-application development monitoring without changing historical evaluators."""

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import torch

import evaluate_cylinder_completion as shared
import evaluate_realistic_science as original
import evaluate_scientific_continuation as previous


PILOT = Path('/mnt/matrixformer_scratch/scientific_prior_training_20261106/pilot_v1')


def register(output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((PILOT / 'real_benchmarks/protocol.json').read_text())
    protocol.update(
        checkpoint_selection='These previously inspected real cases are now development monitors for checkpoint selection, with independent old-benchmark retention gates',
        source_sha256={str(Path(module.__file__).resolve()): shared.sha256(module.__file__) for module in (original, previous, shared)},
        source_monitor_sha256=shared.sha256(__file__),
        science_rule='Every original confirmation-cohort dataset mean must be no worse than the original combined model; minimize geometric mean of seven ratios',
    )
    shared.write_json(output / 'protocol.json', protocol)


def science_score(summary):
    if len(summary) != 7 or {record['dataset'] for record in summary} != {
        'wastewater', 'intel_temperature', 'potomac', 'susquehanna', 'ocean_alkalinity', 'air_quality', 'beijing'
    }:
        raise ValueError('All seven applications are required')
    ratios = [record['candidate'] / record['parent'] for record in summary]
    if not np.isfinite(ratios).all() or min(ratios) <= 0:
        raise ValueError('Invalid science ratios')
    return dict(score=float(np.exp(np.log(ratios).mean())), all_not_worse=bool(max(ratios) <= 1.),
                improved_datasets=int(sum(ratio < 1 for ratio in ratios)), worst_ratio=float(max(ratios)))


def evaluate(checkpoint, registry, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'predictions').mkdir()
    protocol = json.loads((registry / 'protocol.json').read_text())
    assert protocol['source_monitor_sha256'] == shared.sha256(__file__)
    for path, expected in protocol['source_sha256'].items():
        assert shared.sha256(path) == expected
    for path, expected in protocol['control_sha256'].items():
        assert shared.sha256(path) == expected
    shared.write_json(output / 'manifest.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=shared.sha256(checkpoint),
                      registry_sha256=shared.sha256(registry / 'protocol.json')))
    models, distribution = shared.load_models({'MatrixFormer-science': checkpoint})
    rows = []
    for case in protocol['cases']:
        assert shared.sha256(case['input_path']) == case['input_sha256']
        arrays = np.load(case['input_path'])
        current = np.where(arrays['missing'], np.nan, arrays['truth'])
        reduced = np.where(arrays['validation'], np.nan, current)
        controls = previous.cached_controls(case, arrays)
        trials = []
        for setting in original.settings('MatrixFormer'):
            prediction, _ = original.predict(reduced, 'MatrixFormer-science', setting, case['targets'], [], models, distribution)
            scores = original.metrics(arrays['truth'], prediction, arrays['validation'], reduced, case['targets'])
            trials.append(dict(setting=setting, validation_nrmse=scores['nrmse']))
        selected = min(trials, key=lambda trial: trial['validation_nrmse'])
        prediction, _ = original.predict(current, 'MatrixFormer-science', selected['setting'], case['targets'], [], models, distribution)
        scores = original.metrics(arrays['truth'], prediction, arrays['missing'], current, case['targets'])
        np.testing.assert_array_equal(prediction[np.isfinite(current)], current[np.isfinite(current)])
        path = output / 'predictions' / f'{case["suite"]}_{case["identifier"]}.npz'
        np.savez_compressed(path, prediction=prediction)
        row = dict(suite=case['suite'], identifier=case['identifier'], dataset=case['dataset'], split=case['split'],
                   candidate=scores['nrmse'], parent=next(record['nrmse'] for record in controls if record['method'] == 'MatrixFormer'),
                   prediction_sha256=shared.sha256(path), selected=selected, trials=trials, controls=controls)
        rows.append(row)
        shared.write_json(path.with_suffix('.json'), row)
    assert len(rows) == 72
    frame = pd.DataFrame(rows)
    frame.drop(columns=['trials', 'controls', 'selected']).to_csv(output / 'per_case.csv', index=False)
    summary = []
    for dataset, group in frame[frame.split == 'confirmation'].groupby('dataset'):
        difference = group.parent - group.candidate
        summary.append(dict(dataset=dataset, candidate=float(group.candidate.mean()), parent=float(group.parent.mean()),
                            windows=len(group), improvement=float(difference.mean()),
                            paired_ci95_half_width=float(student_t.ppf(.975, len(group)-1) * difference.std(ddof=1) / np.sqrt(len(group)))))
    pd.DataFrame(summary).to_csv(output / 'summary.csv', index=False)
    result = dict(complete=True, cases=len(rows), **science_score(summary))
    shared.write_json(output / 'gate.json', result)
    del models, distribution
    gc.collect()
    torch.cuda.empty_cache()
    return result
