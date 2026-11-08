"""Audit saved preprocessing experiments and export paired fresh-day comparisons."""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import evaluate_cylinder_completion as shared
import evaluate_realistic_science as original
from fetch_science_panels import ROOT


RUNNERS = {
    'diagnosis_v1': 'diagnose_science_preprocessing.py',
    'context_fix_v2': 'test_science_context_fix.py',
    'context_confirmation_v1': 'confirm_science_context.py',
    'decoding_diagnosis_v1': 'probe_science_decoding.py',
    'local_scaling_ablation_v1': 'ablate_science_local_scaling.py',
}


def run():
    output = ROOT / 'diagnosis_report_v1'
    output.mkdir(exist_ok=True)
    protocols = {}
    checked_sources = set()
    for directory, runner in RUNNERS.items():
        protocol = json.loads((ROOT / directory / 'protocol.json').read_text())
        protocols[directory] = protocol
        sources = protocol['source_sha256']
        if isinstance(sources, str):
            sources = {str(Path(__file__).with_name(runner)): sources}
        for path, expected in sources.items():
            assert shared.sha256(path) == expected, path
            checked_sources.add(path)
        for case in protocol.get('cases', []):
            if directory in ('context_fix_v2', 'context_confirmation_v1'):
                path = ROOT / directory / f'{case["identifier"]}.npz'
                expected = case['modified_input_sha256']
            else:
                path = Path(case['parent']) / f'{case["identifier"]}.npz'
                expected = case['input_sha256']
                assert shared.sha256(path.parent / 'protocol.json') == case['parent_protocol_sha256']
            assert shared.sha256(path) == expected, path
        for path in (ROOT / directory / 'records').glob('*.json'):
            record = json.loads(path.read_text())
            assert shared.sha256(path.with_suffix('.npz')) == record['prediction_sha256']

    parent = protocols['diagnosis_v1']
    for checkpoint in parent['checkpoints'].values():
        assert shared.sha256(checkpoint['path']) == checkpoint['sha256']
        assert shared.sha256(Path(checkpoint['path']).with_name('config.json')) == checkpoint['config_sha256']
    diagnosis = pd.read_csv(ROOT / 'diagnosis_v1/per_case.csv')
    actual = diagnosis[~diagnosis.variant.isin(['original_selected', 'validation_selected'])]
    assert len(actual) == 448
    assert not actual.duplicated(['identifier', 'model', 'variant']).any()
    assert len(list((ROOT / 'diagnosis_v1/records').glob('*.json'))) == 448

    original_days = {case['target_start'] for case in parent['cases'] if case['dataset'] == 'intel_temperature'}
    new_days = {case['target_start'] for case in protocols['context_confirmation_v1']['cases']}
    assert not original_days & new_days
    assert len(new_days) == 8
    for directory, expected in [('context_fix_v2', 44), ('context_confirmation_v1', 64)]:
        audit = json.loads((ROOT / directory / 'audit.json').read_text())
        assert audit['complete'] and audit['records'] == audit['expected'] == expected

    cases = {case['identifier']: case for case in parent['cases']}
    counts = {}
    for directory, expected in [('decoding_diagnosis_v1', 240), ('local_scaling_ablation_v1', 54)]:
        frame = pd.read_csv(ROOT / directory / 'per_case.csv')
        assert len(frame) == expected
        variant_key = 'variant' if directory == 'decoding_diagnosis_v1' else 'mode'
        assert not frame.duplicated(['identifier', 'model', variant_key]).any()
        for record in frame.to_dict('records'):
            case = cases[record['identifier']]
            arrays = np.load(Path(case['parent']) / f'{case["identifier"]}.npz')
            current = np.where(arrays['missing'], np.nan, arrays['truth'])
            basename = f'{case["identifier"]}_{record["model"]}'
            if directory == 'decoding_diagnosis_v1':
                predictions = np.load(ROOT / directory / f'{basename}.npz')
                prediction = predictions[record['variant']]
                reduced = np.where(arrays['validation'], np.nan, current)
                validation_prediction = predictions['validation_' + record['variant']]
                score = original.metrics(arrays['truth'], validation_prediction, arrays['validation'], reduced, case['targets'])
                np.testing.assert_allclose(score['nrmse'], record['validation_nrmse'])
            else:
                prediction = np.load(ROOT / directory / f'{basename}_{record["mode"]}.npz')['prediction']
            np.testing.assert_array_equal(prediction[np.isfinite(current)], current[np.isfinite(current)])
            score = original.metrics(arrays['truth'], prediction, arrays['missing'], current, case['targets'])
            np.testing.assert_allclose(score['nrmse'], record['nrmse'])
        counts[directory] = len(frame)
    completion = json.loads((ROOT / 'decoding_diagnosis_v1/completion.json').read_text())
    assert completion['complete']
    for name, expected in completion['predictions_sha256'].items():
        assert shared.sha256(ROOT / 'decoding_diagnosis_v1' / name) == expected

    fresh = pd.read_csv(ROOT / 'context_confirmation_v1/per_case.csv')
    wide = fresh.pivot(index='identifier', columns='method', values='nrmse')
    assert wide.shape == (8, 8) and not wide.isna().any().any()
    rows = []
    for model in parent['checkpoints']:
        references = [model + ' native', 'Temporal regression', 'Temporal interpolation', 'Local KNN', 'KNN']
        for reference in references:
            improvement = wide[reference] - wide[model]
            half_width = student_t.ppf(.975, len(improvement) - 1) * improvement.std(ddof=1) / np.sqrt(len(improvement))
            rows.append(dict(model=model, reference=reference, windows=len(improvement),
                             mean_improvement=improvement.mean(), ci95_half_width=half_width,
                             improved_windows=int((improvement > 0).sum()),
                             relative_mean_improvement=improvement.mean() / wide[reference].mean()))
    paired = pd.DataFrame(rows)
    paired.to_csv(output / 'fresh_day_paired_comparisons.csv', index=False)
    shared.write_json(output / 'audit.json', dict(
        created=datetime.now(timezone.utc).isoformat(), complete=True,
        registered_source_files_checked=len(checked_sources),
        protocol_sha256={name: shared.sha256(ROOT / name / 'protocol.json') for name in RUNNERS},
        report_script_sha256=shared.sha256(__file__),
        records=dict(diagnosis=448, context_fix=44, fresh_confirmation=64, **counts),
        fresh_days_disjoint_from_original=True,
        paired_interval='95% Student-t across eight days, positive difference favors local MatrixFormer; same deployment, residual temporal dependence possible',
        excluded='context_fix_v1 was stopped; no results from it are included',
    ))
    print(paired.to_string(index=False))
    print('Audit and paired comparisons saved to', output)


if __name__ == '__main__':
    run()
