"""Audit the repeated evaluation and render paper-style previews outside the paper."""

import argparse
from collections import Counter
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from seaborn.algorithms import bootstrap

from distributed_imputation import digest, read, write
from repeated_imputation_statistics import summarize


BASE = Path('/mnt/matrixformer_scratch/repeated_imputation_20261114/tabimpute_paper_v7')
FINAL = Path('/mnt/matrixformer_scratch/MatrixFormer-final_20261118')
PAPER = Path('/root/MatrixFormerPaper')
OUTPUT = Path('/root/tabular/mcpfn/runs/benchmark_evaluations/repeated_protocol_preview_20261123')
LABELS = dict(matrixformer='MatrixFormer', missforest='MissForest', knn='K-NN',
              hyperimpute='HyperImpute', mice='MICE', ice='ICE', softimpute='SoftImpute',
              miwae='MIWAE', gain='GAIN', notmiwae='not-MIWAE')
PANELS = ('Overall', 'MCAR', 'MAR', 'MNAR')
VALUE_DECIMALS = 3
VALUE_LABEL_SIZE = 7.3


def paper_snapshot(paper):
    names = subprocess.check_output(['git', '-C', str(paper), 'ls-files', '-z']).decode().split('\0')
    return {name: digest(paper / name) for name in names if name and (paper / name).is_file()}


def validate_protocols(base, final):
    original = read(base / 'manifest.json')
    latest = read(final / 'tabular/manifest.json')
    campaign = read(final / 'manifest.json')
    if original['cases'] != latest['cases'] or original['datasets'] != latest['datasets']:
        raise ValueError('Final checkpoint and baselines use different evaluation cases')
    if latest['methods'] != ['matrixformer'] or campaign['tabular_parent_sha256'] != digest(base / 'manifest.json'):
        raise ValueError('Unexpected final-checkpoint campaign provenance')
    for field in ('scoring', 'uncertainty', 'inference_seed', 'mask_repeats', 'deterministic_repeats'):
        if original[field] != latest[field]:
            raise ValueError(f'Protocol mismatch: {field}')
    mask_hashes = {}
    for case in original['cases']:
        relative = Path('masks') / (case['case_id'] + '.npz')
        actual = digest(base / relative)
        for root in (base, final / 'tabular'):
            metadata = read((root / relative).with_suffix('.json'))
            if digest(root / relative) != actual or metadata['mask_sha256'] != actual:
                raise ValueError(f'Mask mismatch: {case["case_id"]}')
        mask_hashes[case['case_id']] = actual
    for dataset in original['datasets']:
        if digest(Path(dataset['truth_path'])) != dataset['truth_sha256']:
            raise ValueError('Truth data changed')
    for key in ('checkpoint', 'config'):
        model = latest['models']['matrixformer']
        if digest(Path(model[key])) != model[key + '_sha256']:
            raise ValueError('Final model artifact changed')
    return original, latest, mask_hashes


def load_records(root, method, manifest, mask_hashes):
    expected = {case['case_id']: case for case in manifest['cases']}
    manifest_hash = digest(root / 'manifest.json')
    results = []
    for path in (root / 'results' / method / 'cases').rglob('*.json'):
        record = read(path)
        case_id = record['case_id']
        if case_id not in expected or record['method'] != method:
            raise ValueError(f'Unexpected result: {path}')
        if record['manifest_sha256'] != manifest_hash or record['mask_sha256'] != mask_hashes[case_id]:
            raise ValueError(f'Result protocol mismatch: {path}')
        if any(record[key] != value for key, value in expected[case_id].items()):
            raise ValueError(f'Result identity mismatch: {path}')
        if record['status'] == 'ok' and not np.isfinite(record['nrmse']):
            raise ValueError(f'Nonfinite score: {path}')
        results.append(record)
    if len(results) != len(expected) or len({row['case_id'] for row in results}) != len(expected):
        raise ValueError(f'Incomplete or duplicated case inventory: {method}')
    return results


def recompute_score(truth, mask, missing_predictions, suite):
    if not mask.any() or not (~mask).any() or not np.isfinite(missing_predictions).all():
        raise ValueError('Invalid scoring input')
    prediction = truth.copy()
    prediction[mask] = missing_predictions
    threshold = 1e-3 if suite == 'missbench' else 1e-4
    values = []
    for column in range(truth.shape[1]):
        selected = mask[:, column]
        if not selected.any():
            continue
        error = prediction[selected, column] - truth[selected, column]
        rmse = np.sqrt(np.mean(error ** 2))
        scale = np.std(truth[selected, column])
        if scale >= threshold:
            values.append(rmse / scale)
        elif rmse < threshold:
            values.append(0.)
    if not values:
        raise ValueError('No scorable held-out columns')
    return float(np.mean(values)), float(np.sqrt(np.mean((prediction[mask] - truth[mask]) ** 2)))


def check_prediction(root, method, row):
    path = root / 'results' / method / 'predictions' / (row['case_id'] + '.npz')
    if digest(path) != row['prediction_sha256']:
        raise ValueError('Saved prediction hash mismatch')
    truth = np.load(row['truth_path'], allow_pickle=False).astype(np.float64)
    with np.load(root / 'masks' / (row['case_id'] + '.npz'), allow_pickle=False) as packed:
        shape = tuple(packed['shape'])
        mask = np.unpackbits(packed['packed'], count=int(np.prod(shape))).reshape(shape).astype(bool)
    with np.load(path, allow_pickle=False) as saved:
        if tuple(saved['shape']) != truth.shape or mask.shape != truth.shape:
            raise ValueError('Prediction or mask shape mismatch')
        normalized, raw = recompute_score(truth, mask, saved['missing_values'], row['suite'])
    np.testing.assert_allclose([normalized, raw], [row['nrmse'], row['rmse']], rtol=1e-10, atol=1e-10)
    return dict(method=method, case_id=row['case_id'], nrmse=normalized)


def spot_check_predictions(root, method, records):
    checks = []
    for suite in ('missbench', 'uci'):
        successful = sorted([row for row in records if row['suite'] == suite and row['status'] == 'ok'],
                            key=lambda row: row['case_id'])
        indices = np.random.default_rng(0).choice(len(successful), size=min(4, len(successful)), replace=False)
        for index in indices:
            checks.append(check_prediction(root, method, successful[index]))
    return checks


def overlay_mice_retry(base, retry, original, masks, records):
    manifest = read(retry / 'manifest.json')
    lineage = manifest['timeout_retry']
    if (Path(lineage['parent']).resolve() != base.resolve()
            or lineage['parent_manifest_sha256'] != digest(base / 'manifest.json')):
        raise ValueError('Retry parent mismatch')
    changed_fields = {'created_utc', 'cases', 'datasets', 'methods', 'models', 'timeout_seconds', 'source_sha256'}
    for key, value in original.items():
        if key not in changed_fields and manifest.get(key) != value:
            raise ValueError(f'Retry protocol mismatch: {key}')
    if manifest['methods'] != ['mice'] or manifest['timeout_seconds'] <= original['timeout_seconds']:
        raise ValueError('Expected a MICE timeout-only retry')
    expected = {case['case_id']: case for case in original['cases']}
    identifiers = [case['case_id'] for case in manifest['cases']]
    if set(identifiers) != set(lineage['case_ids']) or len(set(identifiers)) != len(identifiers):
        raise ValueError('Retry inventory mismatch')
    for case in manifest['cases']:
        selected_missingness = lineage.get('selected_missingness', .4)
        if (case != expected.get(case['case_id']) or case['suite'] != 'uci'
                or (selected_missingness is not None and case['missingness'] != selected_missingness)):
            raise ValueError('Retry case changed')
    retry_records = load_records(retry, 'mice', manifest, masks)
    if any(row['status'] != 'ok' for row in retry_records):
        raise ValueError('Retry has unsuccessful cases')
    originals = {row['case_id']: row for row in records if row['method'] == 'mice'}
    checks, replacements = [], {}
    for row in retry_records:
        case_id = row['case_id']
        path = base / 'results/mice/cases' / (case_id + '.json')
        if originals[case_id]['status'] != 'timeout':
            raise ValueError('Retry must replace only an original timeout')
        if digest(path) != lineage['parent_result_sha256'].get(str(path)):
            raise ValueError('Original timeout record changed')
        mask = retry / 'masks' / (case_id + '.npz')
        if digest(mask) != masks[case_id] or digest(Path(row['truth_path'])) != row['truth_sha256']:
            raise ValueError('Retry mask or truth changed')
        checks.append(check_prediction(retry, 'mice', row))
        replacements[case_id] = dict(row, retry_source=str(retry), original_timeout_sha256=digest(path))
    updated = [replacements.get(row['case_id'], row) if row['method'] == 'mice' else row for row in records]
    receipt = dict(root=str(retry), manifest_sha256=digest(retry / 'manifest.json'),
                   replaced_cases=identifiers, original_records_unchanged=True,
                   predictions_verified=len(checks), timeout_seconds=manifest['timeout_seconds'])
    return updated, checks, receipt


def bootstrap_bounds(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Need at least two finite dataset scores')
    draws = bootstrap(values, func=np.mean, n_boot=1000, seed=0)
    return tuple(float(value) for value in np.percentile(draws, [2.5, 97.5]))


def format_value_labels(figure, decimals=VALUE_DECIMALS, fontsize=VALUE_LABEL_SIZE):
    for axis in figure.axes:
        for label in axis.texts:
            text = label.get_text()
            number = text.rstrip('†‡')
            try:
                float(number)
            except ValueError:
                continue
            position = label.get_position()[1]
            bars = [bar for bar in axis.patches
                    if np.isclose(bar.get_y() + bar.get_height() / 2, position, rtol=0, atol=1e-10)]
            if len(bars) != 1:
                raise ValueError('A numeric value label must match exactly one bar')
            value = bars[0].get_width()
            label.set_text(f'{value:.{decimals}f}' + text[len(number):])
            label.set_fontsize(fontsize)


def select_rate(cases, records, missingness):
    selected = [case for case in cases if np.isclose(case['missingness'], missingness, rtol=0, atol=1e-12)]
    if {case['suite'] for case in selected} != {case['suite'] for case in cases}:
        raise ValueError('Requested missingness rate is not available in every suite')
    identifiers = {case['case_id'] for case in selected}
    return selected, [record for record in records if record['case_id'] in identifiers]


def plot_values(scores, summaries, coverage):
    frame = pd.DataFrame(scores)
    summaries = pd.DataFrame(summaries)
    rows = []
    for row in summaries[summaries.panel.isin(PANELS)].itertuples():
        selected = frame[(frame.suite == row.suite) & (frame.method == row.method) & (frame.panel == row.panel)]
        selected = selected.sort_values('dataset')
        complete = row.datasets == row.expected_datasets
        lower, upper = bootstrap_bounds(selected.nrmse) if complete else (np.nan, np.nan)
        rows.append(dict(suite=row.suite, method=row.method, label=LABELS[row.method],
                         pattern='overall' if row.panel == 'Overall' else row.panel,
                         nrmse=row.mean if complete else np.nan,
                         ci95_low=lower, ci95_high=upper, datasets=row.datasets,
                         expected_datasets=row.expected_datasets, available=complete,
                         incomplete=not complete, valid_cases=int(selected.cases.sum())))
    data = pd.DataFrame(rows)
    if len(data) != 2 * len(LABELS) * len(PANELS):
        raise ValueError('Missing plot panel/method rows')
    return data


def compare_protocols(paper, old_scores, final_scores):
    archived = pd.read_csv(paper / 'results/imputation_plots/values.csv')
    archived = archived[archived.method == 'matrixformer_single']
    rows = []
    for suite in ('missbench', 'uci'):
        for pattern in ('overall', 'MCAR', 'MAR', 'MNAR'):
            panel = 'Overall' if pattern == 'overall' else pattern
            old = old_scores[(old_scores.suite == suite) & (old_scores.panel == panel)].nrmse.mean()
            final = final_scores[(final_scores.suite == suite) & (final_scores.panel == panel)].nrmse.mean()
            original = archived[(archived.suite == suite) & (archived.pattern == pattern)].iloc[0]
            rows.append(dict(suite=suite, panel=panel, paper_checkpoint_old_protocol=original.nrmse,
                             paper_checkpoint_new_protocol=old, final_checkpoint_new_protocol=final,
                             checkpoint_change_percent=100 * (final / old - 1),
                             same_pattern_scope_as_paper=not (suite == 'uci' and pattern == 'overall')))
    for scores, name in ((old_scores, 'paper_checkpoint'), (final_scores, 'final_checkpoint')):
        selected = scores[(scores.suite == 'uci') & scores.panel.isin(('MCAR', 'MAR', 'MNAR'))]
        per_dataset = selected.groupby('dataset').nrmse.mean()
        if len(per_dataset) != 12 or len(selected) != 36:
            raise ValueError('Incomplete three-pattern UCI comparison')
        rows.append(dict(suite='uci', panel='Common three-pattern Overall', comparison_model=name,
                         new_protocol_mean=float(per_dataset.mean()),
                         paper_checkpoint_old_protocol=float(archived[(archived.suite == 'uci')
                                                                      & (archived.pattern == 'overall')].nrmse.iloc[0])))
    return pd.DataFrame(rows)


def assemble_preview(output, missingness=None):
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib.font_manager import FontProperties, findfont

    images = [Image.open(output / f'{suite}_imputation.png').convert('RGB') for suite in ('missbench', 'uci')]
    title_font = ImageFont.truetype(findfont(FontProperties(family='STIXGeneral', weight='bold')), 26)
    note_font = ImageFont.truetype(findfont(FontProperties(family='STIXGeneral')), 22)
    width = max(image.width for image in images)
    canvas = Image.new('RGB', (width, sum(image.height for image in images) + 224), 'white')
    draw = ImageDraw.Draw(canvas)
    position = 12
    headings = ['MissBench · 41 independent datasets · 40% missing',
                'UCI · 12 datasets · 10–50% missing']
    if missingness is not None:
        headings = [f'MissBench · 41 independent datasets · {missingness:.0%} missing',
                    f'UCI · 12 datasets · {missingness:.0%} missing']
    for heading, image in zip(headings, images):
        draw.text((18, position), heading, font=title_font, fill='black')
        canvas.paste(image, (0, position + 38))
        position += image.height + 67
    draw.text((18, position), 'Overall: all 11 patterns.  Whiskers: 95% dataset-bootstrap intervals.', font=note_font, fill='#333333')
    draw.text((18, position + 28), '— = incomplete UCI coverage for MICE; those partial means are not plotted.', font=note_font, fill='#333333')
    canvas.save(output / 'imputation_combined_preview.png')


def generate(base, final, paper, output, missingness=None, mice_retry=None):
    protected_roots = (paper, base, final) + ((mice_retry,) if mice_retry else ())
    if any(output.is_relative_to(protected) for protected in protected_roots):
        raise ValueError('Preview output must not modify the paper or evaluation archives')
    before = paper_snapshot(paper)
    original, latest, masks = validate_protocols(base, final)
    lineage = read(paper / 'results/single_model/sources.json')
    if lineage['checkpoint_sha256'] != original['models']['matrixformer']['checkpoint_sha256']:
        raise ValueError('The old-checkpoint control is not the model shown in the paper')
    records, checks = [], []
    for method in LABELS:
        root, manifest = (final / 'tabular', latest) if method == 'matrixformer' else (base, original)
        selected = load_records(root, method, manifest, masks)
        records.extend(selected)
        checks.extend(spot_check_predictions(root, method, selected))
    old_records = load_records(base, 'matrixformer', original, masks)
    checks.extend(spot_check_predictions(base, 'matrixformer', old_records))
    scores, coverage, summaries, paired = summarize(original['cases'], records, list(LABELS), shared_methods=True)
    for source, method in ((base, None), (final / 'tabular', 'matrixformer')):
        archived = pd.read_csv(source / 'summary/summary.csv')
        for row in summaries:
            if (method is None and row['method'] == 'matrixformer') or (method and row['method'] != method):
                continue
            expected = archived[(archived.suite == row['suite']) & (archived.method == row['method'])
                                & (archived.panel == row['panel'])]
            np.testing.assert_allclose(row['mean'], expected['mean'].item(), rtol=0, atol=1e-12)
    retry_receipt = None
    if mice_retry is not None:
        records, retry_checks, retry_receipt = overlay_mice_retry(base, mice_retry, original, masks, records)
        checks.extend(retry_checks)
        scores, coverage, summaries, paired = summarize(original['cases'], records, list(LABELS), shared_methods=True)
    cases = original['cases']
    if missingness is not None:
        cases, records = select_rate(original['cases'], records, missingness)
        unused_cases, old_records = select_rate(original['cases'], old_records, missingness)
        scores, coverage, summaries, paired = summarize(cases, records, list(LABELS), shared_methods=True)
    old_scores, unused_coverage, old_summaries, unused_tests = summarize(cases, old_records, ['matrixformer'], shared_methods=True)
    data = plot_values(scores, summaries, coverage)
    score_frame = pd.DataFrame(scores)
    comparisons = compare_protocols(paper, pd.DataFrame(old_scores), score_frame[score_frame.method == 'matrixformer'])
    output.mkdir(parents=True, exist_ok=True)
    for filename, frame in [('values', data), ('dataset_scores', score_frame), ('coverage', pd.DataFrame(coverage)),
                            ('summary', pd.DataFrame(summaries)), ('paired_tests', pd.DataFrame(paired)),
                            ('protocol_comparison', comparisons), ('prediction_spot_checks', pd.DataFrame(checks))]:
        frame.to_csv(output / f'{filename}.csv', index=False)
    failures = pd.DataFrame([row for row in records if row['status'] != 'ok'])
    failures.to_csv(output / 'failed_cases.csv', index=False)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(paper / 'scripts'))
    plotting = importlib.import_module('plot_imputation_results')
    plotting.NAMES = LABELS.copy()
    plotting.COLORS = {'matrixformer': plotting.style.BLUE}
    plotting.style.apply()
    orders = {}
    save_figure = plotting.style.save

    def save_preview(figure, destination, extensions=('pdf', 'png')):
        format_value_labels(figure)
        save_figure(figure, destination, extensions=extensions)

    plotting.style.save = save_preview
    try:
        for suite in ('missbench', 'uci'):
            selected = data[(data.suite == suite) & (data.pattern == 'overall')].copy()
            selected['ordering'] = selected.nrmse.round(6)
            order = selected.sort_values(['ordering', 'label'], na_position='last').method.tolist()
            orders[suite] = order
            plotting.plot_panels(data, order, suite, ('overall', 'MCAR', 'MAR', 'MNAR'), output / f'{suite}_imputation')
    finally:
        plotting.style.save = save_figure
    assemble_preview(output, missingness)
    if paper_snapshot(paper) != before:
        raise ValueError('Paper files changed during preview generation')
    verification = dict(paper_unchanged=True, cases_per_method=len(cases),
                        value_decimals=VALUE_DECIMALS, value_label_size_points=VALUE_LABEL_SIZE,
                        source_cases_per_method=len(original['cases']), selected_missingness=missingness,
                        final_model=latest['models']['matrixformer'], case_and_mask_identity_verified=True,
                        prediction_spot_checks=len(checks), full_source_summary_means_match_archives=True,
                        source_manifests={str(root / 'manifest.json'): digest(root / 'manifest.json')
                                          for root in (base, final / 'tabular')},
                        missing_case_policy='Incomplete dataset-panels have no plotted mean; no failed-case dropping',
                        metric=original['scoring'], aggregation='seed -> rate -> pattern -> dataset',
                        uncertainty='1000 seed-0 percentile bootstrap resamples of complete dataset means, matching paper figures',
                        orders=orders, displayed_methods=LABELS, failed_cases=len(failures),
                        case_statuses=dict(Counter(row['status'] for row in records)), mice_retry=retry_receipt,
                        source_sha256={str(Path(__file__)): digest(Path(__file__)),
                                       str(Path(plotting.__file__)): digest(Path(plotting.__file__)),
                                       str(Path(plotting.style.__file__)): digest(Path(plotting.style.__file__))})
    write(output / 'validation.json', verification)
    rate_note = ('MissBench uses 40% missingness; UCI averages rates 10–50%.' if missingness is None
                 else f'Both suites use only the registered {missingness:.0%} missingness cases; other rates are excluded before aggregation.')
    failed_uci = sum(row['failed'] for row in coverage if row['suite'] == 'uci' and row['method'] == 'mice')
    mice_note = (f'MICE (the MICE-Forest adapter) has {failed_uci} failed UCI cases in this selection. '
                 'Incomplete panels are shown as dashes, not partial-cohort bars.' if failed_uci else
                 'MICE (the MICE-Forest adapter) has complete UCI coverage in this selection, including every plotted panel.')
    if retry_receipt:
        mice_note += (f" The {len(retry_receipt['replaced_cases'])} registered original UCI timeouts are replaced "
                      f"by verified successful retries with a {retry_receipt['timeout_seconds']}-second budget; "
                      'the original result files remain unchanged.')
    lines = ['# Repeated-protocol imputation preview', '',
             'Local preview only. The paper, its figures and its archived results were not changed.', '',
             'The blue bars use MatrixFormer-final, the selected additional-250,000-matrix checkpoint, not the older checkpoint stored alongside the baselines.',
             'Overall averages all 11 mechanisms in both suites. MissBench has 41 independent datasets after removal of the duplicate appendicitis alias; UCI has 12.',
             'Stochastic mechanisms use five mask seeds; deterministic polarization uses one. ' + rate_note,
             'RMSE follows the historical mean-columnwise RMSE / held-out-truth SD scorer, including its constant-column rules. Observed-SD NRMSE is not substituted.',
             'Intervals resample dataset means, not cells, rates or mask seeds. Seeds, rates and patterns receive hierarchical equal weighting.',
             mice_note, '',
             '## Checkpoint comparison on identical new-protocol cases', '',
             '| Suite | Previous checkpoint | Final checkpoint | Change |', '|---|---:|---:|---:|']
    for row in comparisons[comparisons.panel == 'Overall'].itertuples():
        lines.append(f'| {row.suite} | {row.paper_checkpoint_new_protocol:.4f} | {row.final_checkpoint_new_protocol:.4f} | {row.checkpoint_change_percent:+.2f}% |')
    common = comparisons[(comparisons.panel == 'Common three-pattern Overall') & (comparisons.comparison_model == 'final_checkpoint')].iloc[0]
    lines.extend(['', '## Historical comparison caveat', '',
                  f"The paper's UCI Overall is {common.paper_checkpoint_old_protocol:.4f} over three mechanisms. "
                  f"For those same three mechanisms, the final checkpoint under the selected new-protocol cases has mean {common.new_protocol_mean:.4f}. "
                  'Do not compare the new eleven-mechanism Overall directly with the old three-mechanism Overall. '
                  'Historical versus new-protocol values also differ in masks, repeats and the deduplicated MissBench cohort; a fixed-rate selection also differs from the historical UCI rate sweep. Only the old-versus-final checkpoint comparison under the new protocol is paired on identical cases.', '',
                  'See values.csv, coverage.csv, failed_cases.csv, protocol_comparison.csv, paired_tests.csv and validation.json for exact values and audit details.'])
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--final', type=Path, default=FINAL)
    parser.add_argument('--paper', type=Path, default=PAPER)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--missingness', type=float, help='Restrict both suites to this registered missingness rate before aggregation')
    parser.add_argument('--mice-retry', type=Path, help='Verified successful UCI timeout retry campaign to overlay without changing original results')
    arguments = parser.parse_args()
    output = arguments.output or (OUTPUT if arguments.missingness is None else
                                  OUTPUT.with_name(f'{OUTPUT.name}_missing{arguments.missingness * 100:g}'))
    generate(*(path.resolve() for path in (arguments.base, arguments.final, arguments.paper, output)),
             missingness=arguments.missingness,
             mice_retry=arguments.mice_retry.resolve() if arguments.mice_retry else None)


if __name__ == '__main__':
    main()
