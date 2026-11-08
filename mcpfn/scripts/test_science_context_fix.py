"""Retrospective local-context trial, selected only with target-covered validation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import diagnose_science_preprocessing as diagnosis
import evaluate_cylinder_completion as shared
import evaluate_realistic_science as original
from fetch_science_panels import ROOT
import science_context_preprocessing as preprocessing


OUTPUT = ROOT / 'context_fix_v2'
SETTINGS = [dict(rows=rows, delays=delays) for rows in (32, 64, 128) for delays in ([], [1, 3])]
DATASETS = ('intel_temperature', 'potomac', 'susquehanna')


def register(output):
    parent = diagnosis.OUTPUT
    protocol = json.loads((parent / 'protocol.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for case in protocol['cases']:
        if case['dataset'] not in DATASETS:
            continue
        arrays = np.load(Path(case['parent']) / f'{case["identifier"]}.npz')
        context = np.where(arrays['missing'], np.nan, arrays['truth'])
        validation = preprocessing.cover_validation_targets(context, arrays['validation'], case['targets'])
        destination = output / f'{case["identifier"]}.npz'
        np.savez_compressed(destination, truth=arrays['truth'], missing=arrays['missing'], validation=validation)
        cases.append(dict(case, modified_input_sha256=shared.sha256(destination)))
    sources = [Path(__file__), Path(preprocessing.__file__), Path(diagnosis.__file__), Path(original.__file__)]
    shared.write_json(output / 'protocol.json', dict(created=datetime.now(timezone.utc).isoformat(), cases=cases,
                      settings=SETTINGS, checkpoints=protocol['checkpoints'],
                      source_sha256={str(path): shared.sha256(path) for path in sources},
                      inference='32/64/128-row moving windows, 16-row central predictions; optional +/-1,+/-3 target delays built only after masking; global observed column scaling then local observed scaling, retaining the global observed mean as fallback for unsupported local columns',
                      validation='Original recorded auxiliary masks retained; supplement each supported scored target to min(12, observed_count//4) validation observations using short blocks and fixed RNG, never test truth',
                      status='Retrospective diagnostic rerun on unchanged test masks; not independent confirmation'))


def run(output):
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
        path = output / f'{case["identifier"]}.npz'
        assert shared.sha256(path) == case['modified_input_sha256']
        arrays = np.load(path)
        truth, missing, validation = [arrays[name] for name in ('truth', 'missing', 'validation')]
        context = np.where(missing, np.nan, truth)
        reduced = np.where(validation, np.nan, context)
        for name, model in models.items():
            destination = output / 'records' / f'{case["identifier"]}_{name}.json'
            if destination.exists():
                continue
            trials = []
            for setting in protocol['settings']:
                candidate = preprocessing.temporal_context_predict(reduced, case['targets'], model, distribution,
                                                                   diagnosis.forward, **setting)
                score = original.metrics(truth, candidate, validation, reduced, case['targets'])
                trials.append(dict(setting=setting, validation_nrmse=score['nrmse']))
            selected = min(trials, key=lambda record: record['validation_nrmse'])
            prediction = preprocessing.temporal_context_predict(context, case['targets'], model, distribution,
                                                                 diagnosis.forward, **selected['setting'])
            scores = original.metrics(truth, prediction, missing, context, case['targets'])
            destination.parent.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(destination.with_suffix('.npz'), prediction=prediction)
            shared.write_json(destination, dict(identifier=case['identifier'], dataset=case['dataset'], split=case['split'],
                                               model=name, scores=scores, tuning=dict(selected=selected, trials=trials),
                                               prediction_sha256=shared.sha256(destination.with_suffix('.npz')),
                                               protocol_sha256=shared.sha256(output / 'protocol.json')))
            print(case['identifier'], name, f'{scores["nrmse"]:.5f}', selected['setting'], flush=True)


def report(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    rows = []
    for case in protocol['cases']:
        arrays = np.load(output / f'{case["identifier"]}.npz')
        context = np.where(arrays['missing'], np.nan, arrays['truth'])
        assert not (arrays['missing'] & arrays['validation']).any()
        for model in diagnosis.MODELS:
            path = output / 'records' / f'{case["identifier"]}_{model}.json'
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            assert record['protocol_sha256'] == shared.sha256(output / 'protocol.json')
            assert record['prediction_sha256'] == shared.sha256(path.with_suffix('.npz'))
            assert [trial['setting'] for trial in record['tuning']['trials']] == protocol['settings']
            assert record['tuning']['selected'] == min(record['tuning']['trials'], key=lambda trial: trial['validation_nrmse'])
            prediction = np.load(path.with_suffix('.npz'))['prediction']
            np.testing.assert_array_equal(prediction[np.isfinite(context)], context[np.isfinite(context)])
            scores = original.metrics(arrays['truth'], prediction, arrays['missing'], context, case['targets'])
            np.testing.assert_allclose(scores['nrmse'], record['scores']['nrmse'])
            rows.append(dict(dataset=case['dataset'], split=case['split'], identifier=case['identifier'], model=model,
                             nrmse=scores['nrmse'], **record['tuning']['selected']['setting']))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'per_case.csv', index=False)
    summary = frame.groupby(['dataset', 'split', 'model']).nrmse.agg(['mean', 'std', 'count']).reset_index()
    summary['ci95_half_width'] = summary.apply(lambda row: diagnosis.student_t.ppf(.975, row['count']-1)*row['std']/np.sqrt(row['count']) if row['count']>1 else np.nan, axis=1)
    summary.to_csv(output / 'summary.csv', index=False)
    shared.write_json(output / 'audit.json', dict(records=len(frame), expected=len(protocol['cases'])*2,
                                                 complete=len(frame)==len(protocol['cases'])*2))
    print(summary[summary.split == 'confirmation'].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run', 'report'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    if arguments.action == 'register':
        register(arguments.output)
    elif arguments.action == 'run':
        run(arguments.output)
    else:
        report(arguments.output)
