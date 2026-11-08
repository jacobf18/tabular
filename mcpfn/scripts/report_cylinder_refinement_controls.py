"""Audit and export matched cylinder-flow refinement controls for the paper."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

import evaluate_cylinder_refinement_controls as controls


experiment = controls.experiment
PAPER_METHODS = ('Iterative gappy POD', 'SoftImpute', 'SoftImpute + refinement',
                 'SoftImpute preprocessing + refinement', 'MatrixFormer',
                 'MatrixFormer + refinement', 'MatrixFormer + low-rank refinement')


def seed_intervals(frame, groups):
    per_seed = frame.groupby([*groups, 'seed']).nrmse.mean().reset_index()
    summary = per_seed.groupby(groups).nrmse.agg(['mean', 'std', 'count']).reset_index()
    assert (summary['count'] == 3).all()
    summary['ci95_half_width'] = student_t.ppf(.975, summary['count'] - 1) * summary['std'] / np.sqrt(summary['count'])
    summary['ci95_low'] = summary['mean'] - summary.ci95_half_width
    summary['ci95_high'] = summary['mean'] + summary.ci95_half_width
    return summary


def audit(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    for section in ('input_sha256', 'source_sha256'):
        for name, digest in protocol[section].items():
            assert experiment.shared.sha256(name) == digest, name
    assert experiment.shared.sha256(protocol['checkpoint']) == protocol['checkpoint_sha256']
    expected = {path.name for path in (experiment.CYLINDER / 'cases').glob('*.json')}
    with np.load(experiment.CYLINDER / 'dataset.npz') as saved:
        truth = saved['truth']
    frames, derived, settings = [], [], []
    checked = 0
    for method in ('softimpute', 'matrixformer'):
        paths = sorted((output / method / 'cases').glob('*.json'))
        assert {path.name for path in paths} == expected
        rows = []
        for path in paths:
            record = json.loads(path.read_text())
            parent = json.loads((experiment.CYLINDER / 'cases' / path.name).read_text())
            assert record['original_prediction_replay_passed']
            assert experiment.shared.sha256(path.with_suffix('.npz')) == record['arrays_sha256']
            with np.load(path.with_suffix('.npz')) as arrays, np.load(experiment.CYLINDER / 'predictions' / path.with_suffix('.npz').name) as original:
                matrix, validation = arrays['matrix'], arrays['validation']
                np.testing.assert_array_equal(matrix, np.where(original['missing'], np.nan, truth))
                np.testing.assert_array_equal(validation, original['validation'])
                assert not (validation & ~np.isfinite(matrix)).any()
                _, _, scale = experiment.shared.normalize_observed(matrix)
                for name, key in record['prediction_keys'].items():
                    metrics = experiment.shared.point_metrics(truth, arrays[key], ~np.isfinite(matrix), scale)
                    assert set(metrics) == set(record['metrics'][name])
                    for metric, value in metrics.items():
                        np.testing.assert_allclose(value, record['metrics'][name][metric], rtol=1e-10, atol=1e-12)
                        checked += 1
                    rows.append(dict(**record['metadata'], method=name, **metrics))
                trials = record['decisions']['trials']
                if method == 'matrixformer':
                    eligible = [trial for trial in trials if trial['transform'] == 'global']
                    selected = min(eligible, key=lambda trial: trial['validation_rmse'])
                    initial = arrays[record['prediction_keys']['MatrixFormer input: global']]
                    prediction = experiment.low_rank_refine(matrix, initial, selected['rank']) if selected['rank'] else initial
                    metrics = experiment.shared.point_metrics(truth, prediction, ~np.isfinite(matrix), scale)
                    derived.append(dict(**record['metadata'], method='MatrixFormer + refinement', **metrics))
                    directory = output / 'matrixformer_direct'
                    directory.mkdir(exist_ok=True)
                    artifact = directory / path.with_suffix('.npz').name
                    experiment.atomic_npz(artifact, prediction=prediction)
                    experiment.shared.write_json(directory / path.name, dict(metadata=record['metadata'], selected=selected,
                                                 metrics=metrics, arrays_sha256=experiment.shared.sha256(artifact)))
                    settings.append(dict(**record['metadata'], method='MatrixFormer + refinement', **selected))
                for name, selected in record['decisions']['selected'].items():
                    if method == 'softimpute':
                        eligible = trials
                        if name == 'SoftImpute + refinement':
                            shrinkage = parent['baseline_tuning']['SoftImpute']['selected']['parameter']
                            eligible = [trial for trial in trials if trial['transform'] == 'global' and trial['shrinkage'] == shrinkage]
                        elif name == 'SoftImpute + joint refinement':
                            eligible = [trial for trial in trials if trial['transform'] == 'global']
                        elif name == 'SoftImpute preprocessing':
                            eligible = [trial for trial in trials if trial['rank'] is None]
                    else:
                        eligible = [trial for trial in trials if (trial['transform'] == 'row_mean_control') == (name == 'Low-rank refinement control')]
                        if name == 'MatrixFormer input transform':
                            eligible = [trial for trial in eligible if trial['rank'] is None]
                    assert selected == min(eligible, key=lambda trial: trial['validation_rmse'])
                    settings.append(dict(**record['metadata'], method=name, **selected))
        frame = pd.DataFrame(rows)
        saved = pd.read_csv(output / method / 'per_case.csv')
        order = ['case_id', 'method']
        pd.testing.assert_frame_equal(frame.sort_values(order).reset_index(drop=True),
                                      saved.sort_values(order).reset_index(drop=True), check_dtype=False)
        frames.append(frame)
    original = pd.read_csv(experiment.CYLINDER / 'per_case.csv')
    columns = ['case_id', 'pattern', 'rate', 'seed', 'method', 'rmse', 'nrmse', 'mae', 'r2']
    frames.append(original[columns])
    frames.append(pd.DataFrame(derived))
    comparison = pd.concat(frames, ignore_index=True)
    assert not comparison.duplicated(['case_id', 'method']).any()
    comparison.to_csv(output / 'comparison_per_case.csv', index=False)
    pd.DataFrame(settings).to_csv(output / 'selected_settings.csv', index=False)
    experiment.shared.write_json(output / 'audit.json', dict(status='passed', cases=36, original_cases=18,
                                 metric_values_rescored=checked, source_input_checkpoint_hashes_verified=True,
                                 original_prediction_replays_passed=True, validation_masks_and_argmins_verified=True,
                                 direct_matrixformer='Derived from already evaluated global-input rank validation grid',
                                 report_sha256=experiment.shared.sha256(Path(__file__))))
    return comparison


def report(output):
    comparison = audit(output)
    paper = comparison[comparison.method.isin(PAPER_METHODS)].copy()
    assert len(paper) == 18 * len(PAPER_METHODS)
    paper.to_csv(output / 'paper_per_case.csv', index=False)
    overall = seed_intervals(comparison, ['method']).sort_values('mean')
    overall.to_csv(output / 'overall.csv', index=False)
    patterns = seed_intervals(paper, ['pattern', 'method'])
    patterns.to_csv(output / 'patterns.csv', index=False)
    conditions = seed_intervals(paper, ['pattern', 'rate', 'method'])
    conditions.to_csv(output / 'conditions.csv', index=False)
    means = overall.set_index('method')['mean']
    matched = paper.pivot(index='case_id', columns='method', values='nrmse')
    wins = int((matched['MatrixFormer + low-rank refinement'] < matched['SoftImpute preprocessing + refinement']).sum())
    text = [
        '# Cylinder flow: matched refinement controls', '',
        'All methods use the same 18 cases and original observed-cell validation masks. '
        'The primary neural checkpoint is the combined MatrixFormer, not MatrixFormer-Impute.', '',
        '| Method | Mean normalized RMSE | 95% mask-seed CI half-width |',
        '| --- | ---: | ---: |',
    ]
    for _, row in overall.iterrows():
        text.append(f"| {row['method']} | {row['mean']:.6f} | {row['ci95_half_width']:.6f} |")
    text.extend(['', 'The CI averages the six fixed pattern/rate conditions within each seed first, then '
                 'uses a two-sided Student-t interval across the three seed means. It is not a CI '
                 'over cells, 18 independent flows, or new flow regimes.', '',
                 'Direct SoftImpute refinement keeps its original validation-selected shrinkage and '
                 'selects rank (or no refinement). Joint refinement also reselects shrinkage. The '
                 'matched preprocessing comparison gives SoftImpute all four neural input transforms '
                 'and the same rank grid, plus its three shrinkage settings. All refits use every '
                 'outer observation. Validation never uses outer hidden targets.', '',
                 f'The combined MatrixFormer matched hybrid beats the matched SoftImpute hybrid on {wins}/18 cases.', '',
                 'Gappy POD is the traditional flow-reconstruction baseline family (Everson and Sirovich, '
                 '1995; Venturi and Karniadakis, 2004). Our archived comparator is a simple iterative '
                 'POD/SVD completion implementation, not a reproduction of their weighted gappy-POD '
                 'procedures. SoftImpute is the general matrix-completion comparator (Mazumder et al., 2010).', '',
                 'These are exploratory, conditional results on one noiseless trajectory. Rank refinement '
                 'is an additional hybrid algorithm, not the network alone. No MRI is included.', ''])
    (output / 'RESULTS.md').write_text('\n'.join(text))
    findings = (
        f"Across the 18 fixed cases, SoftImpute's mean normalized RMSE changes from {means['SoftImpute']:.4f} "
        f"to {means['SoftImpute + refinement']:.4f} after refinement; MatrixFormer changes from "
        f"{means['MatrixFormer']:.4f} to {means['MatrixFormer + refinement']:.4f}. "
        f"With preprocessing and refinement selected from the matched grids, the respective errors are "
        f"{means['SoftImpute preprocessing + refinement']:.4f} and {means['MatrixFormer + low-rank refinement']:.4f}; "
        f"the neural-initialized hybrid has lower error in {wins} of 18 cases. "
        "These comparisons separate initialization from post-processing; refined results belong to the "
        "hybrid pipelines, not the unmodified network. Condition-level results, including regressions, "
        "are retained in \\cref{fig:physics-cylinder-conditions}.\n")
    (output / 'findings.tex').write_text(findings)
    experiment.shared.write_json(output / 'status.json', dict(status='complete', cases=18, methods=len(PAPER_METHODS), audit='passed'))
    print(overall[['method', 'mean', 'ci95_half_width']].to_string(index=False), flush=True)
    print(f'Matched hybrid wins: {wins}/18', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    report(parser.parse_args().output)
